"""
SM100 (Blackwell / B300, sm_100a & sm_103a) Chunk-Gated-Delta-Rule prefill op.

Derived from FlashInfer's `flashinfer/gdn_prefill.py` (Apache-2.0, (c) 2025 FlashInfer
team) — the chunk-parallel single-pass scan + fused Triton correction launcher
(workspace "launcher_v23_oddsplit.py"). Ported into atrex as a *self-contained* op:
the underlying CuTe kernel is the locally optimized `gdn_prefill_sm100` package
(based on flashinfer 0.6.9 `gdn_kernels/blackwell`, pure cutlass.*); all flashinfer
imports are stripped and the SM90/Hopper path is dropped (this is SM100-only).

Original copyright:
  Copyright (c) 2025 by FlashInfer team. Licensed under the Apache License 2.0.
"""

import math
import os
from typing import Optional, Union, Tuple
import torch

# ---------------------------------------------------------------------------
# Vendored CuTe kernel (flashinfer 0.6.9 base plus documented local changes).
# Dual import: the atrex package path first, then a flat path for standalone use.
# ---------------------------------------------------------------------------
try:
    from atrex.src.cutedsl.gdn_prefill_sm100 import (
        chunk_gated_delta_rule_sm100 as _vendor_blackwell_cgdr,
    )
except ImportError:  # standalone (staging dir on PYTHONPATH)
    try:
        from gdn_prefill_sm100 import (
            chunk_gated_delta_rule_sm100 as _vendor_blackwell_cgdr,
        )
    except ImportError:
        _vendor_blackwell_cgdr = None

_has_blackwell_prefill = _vendor_blackwell_cgdr is not None


def chunk_gated_delta_rule_sm100(
    q,
    k,
    v,
    g,
    beta,
    output,
    cu_seqlens,
    initial_state,
    output_state,
    scale,
    mn_mode=False,
    m_seg=None,
):
    """Thin pass-through onto the vendored SM100 Blackwell adapter.

    The vendored adapter writes `output` / `output_state` IN PLACE (returns None)
    and natively takes:
      * 3D (total,H,D) q/k/v and 3D (total,Ho,D) output (no B=1 reshape needed),
      * the forget gate `g` in LINEAR space -- it takes log() internally -- which
        matches this launcher's linear-g convention, so NO log() wrapper here
        (unlike the previous 0.6.7 kernel, which consumed log-space gates),
      * fp32 gate/beta.
    """
    _vendor_blackwell_cgdr(
        q,
        k,
        v,
        g.float(),
        beta.float(),
        output,
        cu_seqlens,
        initial_state,
        output_state,
        scale,
        mn_mode=mn_mode,
        m_seg=m_seg,
    )
    return output


# ---------------------------------------------------------------------------
# Chunk-parallel single-pass scan (SM100): shorten the serial chunk-recurrence
# critical path for long sequences. Split each long segment into P sub-segments
# that run in parallel in ONE heavy kernel pass (zero init), then recover the
# exact result analytically with a cheap correction -- no second heavy pass.
#
# GDN's inter-segment map is affine with scalar decay:
#     H_t = cumdecay_t * H_in_seg + H_t^local      (state at token t)
#   so   o_t = q_t @ H_t = o_local_t + cumdecay_t * (q_t @ H_in_seg)
#   cumdecay_t[h] = prod_{s<=t in seg} g_s[h]      (inclusive cumulative gate)
#   H_in_seg      = true entry state from the serial inter-segment scan.
# Env: GDN_CP_P (sub-segments, default 8 cap), GDN_CP_MIN (seg must be strictly
# longer than this to split, default 2048), GDN_CP_DISABLE=1 to force off,
# GDN_CP_FORCE=1 uses cap verbatim.
# ---------------------------------------------------------------------------
_GDN_CHUNK = 64  # kernel chunk size (BT)
_gdn_cp_cache = {}  # reusable scratch buffers, keyed on (device, nfine, H, D)


def _gdn_cp_buffers(device, nfine, H, D):
    key = (device, nfine, H, D)
    buf = _gdn_cp_cache.get(key)
    if buf is None:
        buf = {
            "zero_init": torch.zeros(nfine, H, D, D, dtype=torch.float32, device=device),
            "state1": torch.empty(nfine, H, D, D, dtype=torch.float32, device=device),
            "Hin": torch.empty(nfine, H, D, D, dtype=torch.float32, device=device),
        }
        _gdn_cp_cache[key] = buf
    return buf


# ---------------------------------------------------------------------------
# EXACT chunk-parallel carry (Stage 1). Instead of the scalar-decay carry
# approximation (H_in_j = (prod alpha)*I @ H_in_{j-1} + N_{j-1}), carry the
# FULL K x K per-fine-segment transition M_seg. GDN's per-token map is affine:
#   S_t = (alpha_t I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T = M_t S_{t-1} + c_t
# so a fine segment is S_end = M_seg @ S_start + N_seg, EXACT. M_seg is obtained
# for free from the existing (already-exact) vendor kernel via the v=0 / identity-
# init trick: with v=0 the additive term vanishes, so final_state(S_start=I) = M_seg.
# Then the cross-segment recurrence H_in_j = M_{j-1} @ H_in_{j-1} + N_{j-1} is an
# exact K x K matmul chain (P<=8 steps), and exact per-token outputs come from
# re-running the kernel with initial_state = H_in (no scalar output correction).
# Validated bit-exact vs the serial reference at all alpha incl. carried init
# (MLFlow-B300-Sync/_cp_stage0.py). Gated by env GDN_CP_EXACT=1 (scalar path kept
# as fallback); when exact, the alpha accuracy gate is unnecessary.
# ---------------------------------------------------------------------------
_gdn_cp_exact_cache = {}


def _gdn_cp_exact_buffers(device, nfine, H, D, total, Hv, Ho, io_dtype):
    key = (device, nfine, H, D, total, Hv, Ho, io_dtype)
    buf = _gdn_cp_exact_cache.get(key)
    if buf is None:
        eye = torch.eye(D, dtype=torch.float32, device=device)
        buf = {
            "zero_init": torch.zeros(nfine, H, D, D, dtype=torch.float32, device=device),
            "eye_init": eye.expand(nfine, H, D, D).contiguous(),
            "Nseg": torch.empty(nfine, H, D, D, dtype=torch.float32, device=device),
            "Mseg": torch.empty(nfine, H, D, D, dtype=torch.float32, device=device),
            "Hin": torch.empty(nfine, H, D, D, dtype=torch.float32, device=device),
            # discard buffers for the M/N passes (token outputs unused)
            "vzero": torch.zeros(total, Hv, D, dtype=io_dtype, device=device),
            "scratch": torch.empty(total, Ho, D, dtype=io_dtype, device=device),
        }
        _gdn_cp_exact_cache[key] = buf
    return buf


_gdn_exact_scan_idx_cache = {}


def _exact_scan_idx(groups, device):
    """Cache the (gstart, glen) int32 device tensors for the fused scan kernel so the
    tiny H2D copy happens once per plan, not per call."""
    key = (device, tuple((g[0], len(g)) for g in groups))
    v = _gdn_exact_scan_idx_cache.get(key)
    if v is None:
        gs = torch.tensor([g[0] for g in groups], dtype=torch.int32, device=device)
        gl = torch.tensor([len(g) for g in groups], dtype=torch.int32, device=device)
        v = (gs, gl)
        _gdn_exact_scan_idx_cache[key] = v
    return v


def _gdn_exact_scan(Hin, Mseg, Nseg, groups, has_init, output_state=None):
    """Exact cross-segment carry with the full K x K transition.
    Hin[idxs[0]] must already hold the group's true entry state (initial_state or 0).
      H_in_{j} = M_{j-1} @ H_in_{j-1} + N_{j-1}
    and (if requested) output_state[g] = M_last @ H_in_last + N_last (group final state).
    Orientation (M @ H) is validated against no-CP; flip to H @ M if bit-exact fails.

    GDN_CP_EXACT_SCAN_FUSE=1 runs the serial matrix scan in ONE Triton launch, BUT it
    is ~200x SLOWER (12.8ms vs 66us): one program per head loses torch.bmm's batched
    tensor-core matmul, and fp32 ieee tl.dot disables tensor cores. Default OFF — the
    batched torch.bmm carry (7 serial tensor-core launches) is far better here."""
    if _HAS_TRITON and os.environ.get("GDN_CP_EXACT_SCAN_FUSE", "0") == "1":
        H, D = Hin.shape[1], Hin.shape[2]
        gs, gl = _exact_scan_idx(groups, Hin.device)
        os_t = output_state if output_state is not None else Hin
        _gdn_exact_scan_kernel[(len(groups), H)](
            Hin, Mseg, Nseg, os_t, gs, gl,
            output_state is not None,
            Hin.stride(0), Hin.stride(1), Hin.stride(2), Hin.stride(3),
            os_t.stride(0), os_t.stride(1), os_t.stride(2), os_t.stride(3),
            D,
        )
        return
    for g, idxs in enumerate(groups):
        prev = idxs[0]
        for t in range(1, len(idxs)):
            cur = idxs[t]
            # [H,D,D] batched matmul over heads
            Hin[cur] = torch.bmm(Mseg[prev], Hin[prev]) + Nseg[prev]
            prev = cur
        if output_state is not None:
            output_state[g] = torch.bmm(Mseg[prev], Hin[prev]) + Nseg[prev]


def _gdn_plan_split(cu_list, P, min_len, chunk=_GDN_CHUNK):
    """Return (fine_cu_list, groups). groups = list of fine-seg index lists,
    one per original segment (so groups[i] are the fine segs of original seg i)."""
    fine = [cu_list[0]]
    groups = []
    for i in range(len(cu_list) - 1):
        s0, s1 = cu_list[i], cu_list[i + 1]
        L = s1 - s0
        nchunks = (L + chunk - 1) // chunk
        # pick the largest p <= P that divides nchunks evenly, so every split uses
        # the fast equal-length batched-cumsum path (uneven splits use the slower
        # per-segment cumdecay loop).
        p = 1
        # CP needs a segment strictly LONGER than min_len (default 2048). The
        # analytic carry-correction's scalar-decay approximation only crosses the
        # 0.02 out-threshold at the smallest CP size (L=2048) under a large
        # nonzero initial_state, so keeping L<=2048 on the exact vendor path
        # removes that corner at zero prod cost (prod segments are >=6912).
        if L > min_len:
            for cand in range(min(P, nchunks), 1, -1):
                if nchunks % cand == 0:
                    p = cand
                    break
            # [v23] odd/prime nchunks has no even divisor -> the loop above leaves
            # p=1 (no split, vendor speed). A balanced UNEQUAL 2-way split (ceil/
            # floor chunks, e.g. 63+62) still halves the serial pass1 critical
            # path; the vendor varlen kernel handles unequal segments natively and
            # only cumdecay falls to the cheap 2-iter slow path.
            if p == 1 and nchunks >= 2 and P >= 2:
                p = 2
        base, rem = nchunks // p, nchunks % p
        idxs, pos = [], s0
        for j in range(p):
            cj = base + (1 if j < rem else 0)
            pos = min(pos + cj * chunk, s1)
            idxs.append(len(fine) - 1)
            fine.append(pos)
        groups.append(idxs)
    return fine, groups


# Fused Triton correction: output[t,h,:] += scale * cumdecay[t,h] * (q[t,h//grp] @ H_in[h]^T)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gdn_corr_kernel(
        q_ptr, out_ptr, cd_ptr, H_ptr,
        seg_starts_ptr, seg_ids_ptr, seg_lens_ptr, grp, scale,
        sq_t, sq_h, sq_k, so_t, so_h, so_v, scd_seg, scd_h, scd_l,
        sH_seg, sH_h, sH_v, sH_k,
        NTB_SB: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        EPS: tl.constexpr,
    ):
        sb = tl.program_id(0)
        h = tl.program_id(1)
        ci = tl.program_id(2)
        seg_start = tl.load(seg_starts_ptr + ci)
        seg_idx = tl.load(seg_ids_ptr + ci)
        seg_len = tl.load(seg_lens_ptr + ci)
        offs_k = tl.arange(0, K)
        offs_v = tl.arange(0, V)
        # [fast-decay skip] scalar-CP only runs on fast-decay layers, so cumdecay
        # (= prod of gates, monotonically DECREASING in t) collapses toward 0 after
        # the first ~1/(1-alpha) tokens. block-0 holds this program's LARGEST cumdecay;
        # if it's already < EPS, the whole token range contributes < EPS*(q@Hin), so we
        # leave o_local untouched (correct to EPS) and skip the q-load / dot / output
        # RMW entirely. Cuts ~85-95% of blocks on fast-decay. EPS=0 disables the skip.
        t0f = (sb * NTB_SB) * BT + tl.arange(0, BT)
        cd0 = tl.load(cd_ptr + seg_idx * scd_seg + h * scd_h + t0f * scd_l,
                      mask=t0f < seg_len, other=0.0)
        if EPS == 0.0 or tl.max(cd0) >= EPS:
            hk = h // grp
            Ht = tl.load(H_ptr + seg_idx * sH_seg + h * sH_h
                         + offs_v[:, None] * sH_v + offs_k[None, :] * sH_k).to(tl.bfloat16)
            HtT = tl.trans(Ht)
            for j in tl.static_range(NTB_SB):
                t0 = (sb * NTB_SB + j) * BT
                offs_t = t0 + tl.arange(0, BT)
                mask_t = offs_t < seg_len
                tok = seg_start + offs_t
                qb = tl.load(q_ptr + tok[:, None] * sq_t + hk * sq_h + offs_k[None, :] * sq_k,
                             mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
                qH = tl.dot(qb, HtT)
                # cumdecay in (nfine, H, L) layout; index by segment + LOCAL token offs_t
                cd = tl.load(cd_ptr + seg_idx * scd_seg + h * scd_h + offs_t * scd_l,
                             mask=mask_t, other=0.0)
                corr = (scale * cd)[:, None] * qH
                op = out_ptr + tok[:, None] * so_t + h * so_h + offs_v[None, :] * so_v
                ov = tl.load(op, mask=mask_t[:, None], other=0.0).to(tl.float32)
                tl.store(op, (ov + corr).to(out_ptr.dtype.element_ty), mask=mask_t[:, None])

    @triton.jit
    def _gdn_scan_kernel(
        Hin_ptr, B_ptr, Gamma_ptr, gstart_ptr, glen_ptr, OS_ptr,
        HAS_INIT: tl.constexpr, HAS_OUT: tl.constexpr,
        sH_seg, sH_h, sH_d, sG_seg, sG_h,
        sO_seg, sO_h, sO_d, D: tl.constexpr,
    ):
        # One program per (group, head, d-row); internal sequential loop over the
        # group's fine-segment chain replaces the per-step python scan launches.
        #   Hin[base+t] = Gamma[base+t-1]*Hin[base+t-1] + B[base+t-1]
        g = tl.program_id(0)
        h = tl.program_id(1)
        dr = tl.program_id(2)
        base = tl.load(gstart_ptr + g)
        ln = tl.load(glen_ptr + g)
        ov = tl.arange(0, D)
        off0 = base * sH_seg + h * sH_h + dr * sH_d + ov
        if HAS_INIT:
            hin_prev = tl.load(Hin_ptr + off0).to(tl.float32)
        else:
            hin_prev = tl.zeros([D], dtype=tl.float32)
            tl.store(Hin_ptr + off0, hin_prev)
        for t in range(1, ln):
            pseg = base + t - 1
            gam = tl.load(Gamma_ptr + pseg * sG_seg + h * sG_h)
            b = tl.load(B_ptr + pseg * sH_seg + h * sH_h + dr * sH_d + ov).to(tl.float32)
            hin_prev = gam * hin_prev + b
            tl.store(Hin_ptr + (pseg + 1) * sH_seg + h * sH_h + dr * sH_d + ov, hin_prev)
        if HAS_OUT:
            lastseg = base + ln - 1
            gam = tl.load(Gamma_ptr + lastseg * sG_seg + h * sG_h)
            b = tl.load(B_ptr + lastseg * sH_seg + h * sH_h + dr * sH_d + ov).to(tl.float32)
            fin = gam * hin_prev + b
            tl.store(OS_ptr + g * sO_seg + h * sO_h + dr * sO_d + ov,
                     fin.to(OS_ptr.dtype.element_ty))

    @triton.jit
    def _gdn_exact_scan_kernel(
        Hin_ptr, M_ptr, N_ptr, OS_ptr, gstart_ptr, glen_ptr,
        HAS_OUT: tl.constexpr,
        sH_seg, sH_h, sH_r, sH_c, sO_seg, sO_h, sO_r, sO_c,
        D: tl.constexpr,
    ):
        # One program per (group, head): serial EXACT affine matrix carry over the
        # group's fine-segment chain, replacing the ~P per-step torch.bmm launches.
        #   Hin[base+t] = M[base+t-1] @ Hin[base+t-1] + N[base+t-1]   ([D,D] @ [D,D])
        # Hin[base] must already hold the group's true entry state (host-set).
        g = tl.program_id(0)
        h = tl.program_id(1)
        base = tl.load(gstart_ptr + g)
        ln = tl.load(glen_ptr + g)
        rr = tl.arange(0, D)[:, None]
        cc = tl.arange(0, D)[None, :]
        tile = rr * sH_r + cc * sH_c
        hin_prev = tl.load(Hin_ptr + base * sH_seg + h * sH_h + tile).to(tl.float32)
        for t in range(1, ln):
            pseg = base + t - 1
            offp = pseg * sH_seg + h * sH_h
            m = tl.load(M_ptr + offp + tile).to(tl.float32)
            n = tl.load(N_ptr + offp + tile).to(tl.float32)
            hin_prev = tl.dot(m, hin_prev, input_precision="ieee") + n
            tl.store(Hin_ptr + (pseg + 1) * sH_seg + h * sH_h + tile,
                     hin_prev.to(Hin_ptr.dtype.element_ty))
        if HAS_OUT:
            lastseg = base + ln - 1
            offl = lastseg * sH_seg + h * sH_h
            m = tl.load(M_ptr + offl + tile).to(tl.float32)
            n = tl.load(N_ptr + offl + tile).to(tl.float32)
            fin = tl.dot(m, hin_prev, input_precision="ieee") + n
            tl.store(OS_ptr + g * sO_seg + h * sO_h + rr * sO_r + cc * sO_c,
                     fin.to(OS_ptr.dtype.element_ty))

    @triton.jit
    def _cumdecay_kernel(
        logg_ptr, cd_ptr, gamma_ptr, seg_len,
        s_lg_t, s_lg_h, s_cd_seg, s_cd_h, s_cd_l, s_g_seg, s_g_h,
        BL: tl.constexpr,
    ):
        # One program per (segment, head): inclusive cumprod of gates in log space.
        #   cumdecay[seg,h,t] = exp(sum_{s<=t in seg} log g_s[h]) ; Gamma = last t.
        seg = tl.program_id(0)
        h = tl.program_id(1)
        base = seg * seg_len
        carry = 0.0
        n_tiles = tl.cdiv(seg_len, BL)
        for it in range(n_tiles):
            offs = it * BL + tl.arange(0, BL)
            mask = offs < seg_len
            lg = tl.load(logg_ptr + (base + offs) * s_lg_t + h * s_lg_h,
                         mask=mask, other=0.0)
            cs = tl.cumsum(lg, axis=0) + carry
            tl.store(cd_ptr + seg * s_cd_seg + h * s_cd_h + offs * s_cd_l,
                     tl.exp(cs), mask=mask)
            carry = carry + tl.sum(lg, axis=0)
        tl.store(gamma_ptr + seg * s_g_seg + h * s_g_h, tl.exp(carry))

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _gdn_cumdecay_launch(logg_full, nfine, L, BL=2048):
    """Custom cumdecay for equal-length segments: returns (nfine,H,L) cumdecay and
    (nfine,H) Gamma in one launch. logg_full is (total, H) contiguous."""
    H = logg_full.shape[1]
    device = logg_full.device
    cd = torch.empty(nfine, H, L, dtype=torch.float32, device=device)
    Gamma = torch.empty(nfine, H, dtype=torch.float32, device=device)
    _cumdecay_kernel[(nfine, H)](
        logg_full, cd, Gamma, L,
        logg_full.stride(0), logg_full.stride(1),
        cd.stride(0), cd.stride(1), cd.stride(2),
        Gamma.stride(0), Gamma.stride(1), BL=BL,
    )
    return cd, Gamma


def _gdn_scan_launch(Hin, B, Gamma, plan, has_init, output_state=None):
    """Fused inter-segment scan: one kernel launch replaces the (P-1)-step python
    recurrence loop (and optionally the per-group final-state computation)."""
    H, D = Hin.shape[1:3]
    n_groups = plan["n_groups"]
    grid = (n_groups, H, D)
    has_out = output_state is not None
    os_strides = (
        (output_state.stride(0), output_state.stride(1), output_state.stride(2))
        if has_out
        else (0, 0, 0)
    )
    _gdn_scan_kernel[grid](
        Hin, B, Gamma, plan["group_starts"], plan["group_lens"],
        output_state if has_out else Hin, has_init, has_out,
        Hin.stride(0), Hin.stride(1), Hin.stride(2),
        Gamma.stride(0), Gamma.stride(1),
        os_strides[0], os_strides[1], os_strides[2], D,
    )


def _gdn_corr_launch(
    output, q, cumdecay, H_in, plan, scale, BT=_GDN_CHUNK, NTB_SB=2
):
    """plan provides the cached corrected-segment tensors (no per-call allocation)."""
    # [tuning] NTB_SB amortizes the per-program Hin (K x V) reload over more tokens:
    # each program reloads Hin once, so nsb=cdiv(cdiv(max_len,BT),NTB_SB) reloads per
    # (head,seg). Higher NTB_SB -> fewer L2-thrashing Hin reloads, fewer programs.
    BT = int(os.environ.get("GDN_CORR_BT", str(BT)))
    NTB_SB = int(os.environ.get("GDN_CORR_NTB", str(NTB_SB)))
    _nw = int(os.environ.get("GDN_CORR_NW", "8"))
    _eps = float(os.environ.get("GDN_CORR_EPS", "1e-4"))  # skip cumdecay<eps blocks
    num_state_heads = H_in.shape[1]
    num_q_heads = q.shape[1]
    group_size = (
        num_state_heads // num_q_heads if num_state_heads >= num_q_heads else 1
    )
    num_output_heads = output.shape[1]
    K = q.shape[2]
    V = output.shape[2]
    n_corr = plan["n_corr"]
    nsb = triton.cdiv(triton.cdiv(plan["max_len"], BT), NTB_SB)
    grid = (nsb, num_output_heads, n_corr)
    _gdn_corr_kernel[grid](
        q, output, cumdecay, H_in,
        plan["seg_starts"], plan["seg_ids"], plan["seg_lens"], group_size, scale,
        q.stride(0), q.stride(1), q.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        cumdecay.stride(0), cumdecay.stride(1), cumdecay.stride(2),
        H_in.stride(0), H_in.stride(1), H_in.stride(2), H_in.stride(3),
        NTB_SB, K, V, BT, _eps,
        num_warps=_nw,
    )


# Cache the device-capability query. SM100 check costs ~7us per call; device caps
# never change, so memoize per device index.
_sm100a_cache = {}


def _is_sm100a_cached(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    v = _sm100a_cache.get(idx)
    if v is None:
        v = torch.cuda.get_device_capability(device)[0] == 10
        _sm100a_cache[idx] = v
    return v


# Cache the SM count per device (used by the occupancy-adaptive P selector).
_sm_count_cache = {}


def _gdn_sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    v = _sm_count_cache.get(idx)
    if v is None:
        v = torch.cuda.get_device_properties(device).multi_processor_count
        _sm_count_cache[idx] = v
    return v


# Cache the int32 cu_seqlens conversion on the ORIGINAL cu_seqlens object so the
# copy + the cu_list .tolist() happen once per forward, not once per layer.
def _cu_seqlens_i32(cu_seqlens):
    ci = getattr(cu_seqlens, "_gdn_i32_cache", None)
    if ci is None:
        ci = cu_seqlens.to(torch.int32)
        try:
            cu_seqlens._gdn_i32_cache = ci
        except Exception:
            pass
    return ci


# Cache the per-shape plan (host lists + GPU index tensors) so the hot path makes
# no per-call host->device tensor copies. Device is part of the key because each
# plan owns CUDA tensors.
_gdn_plan_cache = {}


def _gdn_get_plan(cu_list, P, min_len, initial_state_present, device):
    key = (tuple(cu_list), P, min_len, initial_state_present, device)
    plan = _gdn_plan_cache.get(key)
    if plan is not None:
        return plan
    fine, groups = _gdn_plan_split(cu_list, P, min_len)
    if all(len(g) == 1 for g in groups):
        plan = {"trivial": True}
        _gdn_plan_cache[key] = plan
        return plan
    nfine = len(fine) - 1
    seg_lens_host = [fine[f + 1] - fine[f] for f in range(nfine)]
    corr = [
        fidx
        for idxs in groups
        for t, fidx in enumerate(idxs)
        if not (t == 0 and not initial_state_present)
    ]
    plan = {
        "trivial": False,
        "fine": fine,
        "groups": groups,
        "nfine": nfine,
        "equal": (
            len(set(seg_lens_host)) == 1
            and nfine * seg_lens_host[0] == fine[-1]
        ),
        "L": seg_lens_host[0],
        "fine_cu": torch.tensor(fine, dtype=torch.int32, device=device),
        "n_corr": len(corr),
        "max_len": max((fine[c + 1] - fine[c] for c in corr), default=0),
        "seg_starts": torch.tensor(
            [fine[c] for c in corr], dtype=torch.int32, device=device
        ),
        "seg_ids": torch.tensor(corr, dtype=torch.int32, device=device),
        "seg_lens": torch.tensor(
            [fine[c + 1] - fine[c] for c in corr],
            dtype=torch.int32,
            device=device,
        ),
        # contiguous fine-seg chain per group, for the fused scan kernel
        "n_groups": len(groups),
        "group_starts": torch.tensor(
            [g[0] for g in groups], dtype=torch.int32, device=device
        ),
        "group_lens": torch.tensor(
            [len(g) for g in groups], dtype=torch.int32, device=device
        ),
        "max_fine_len": max(seg_lens_host),
    }
    _gdn_plan_cache[key] = plan
    return plan


def _try_gdn_chunk_parallel(
    q,
    k,
    v,
    gate,
    beta,
    output,
    cu_seqlens_i32,
    initial_state,
    output_state,
    scale,
    P,
    min_len,
    g_log=None,
    exact=False,
):
    """Single-pass chunk-parallel execution with analytic correction. Writes
    `output` and (if not None) `output_state` in the ORIGINAL [num_seqs, ...]
    layout. Returns True if a split was applied, False if nothing was eligible.

    exact=True uses the full K x K transition carry (3 kernel passes, accurate at
    all alpha); exact=False uses the legacy scalar-decay carry (1 heavy pass +
    analytic correction, needs the alpha accuracy gate)."""
    # Avoid a blocking D2H sync (.tolist()) on every GDN layer: all GDN layers in
    # one forward pass share the SAME cu_seqlens tensor object, so cache the host
    # list as an attribute on it (recomputed next forward on a fresh slice object).
    cu_list = getattr(cu_seqlens_i32, "_gdn_cu_list_cache", None)
    if cu_list is None:
        cu_list = cu_seqlens_i32.tolist()  # blocking D2H, once per forward
        try:
            cu_seqlens_i32._gdn_cu_list_cache = cu_list
        except Exception:
            pass
    device = q.device
    plan = _gdn_get_plan(cu_list, P, min_len, initial_state is not None, device)
    if plan["trivial"]:
        return False  # nothing long enough -> let caller do the single call

    fine, groups, nfine = plan["fine"], plan["groups"], plan["nfine"]
    H = output.size(1)
    D = q.size(2)

    if exact:
        Hv = v.size(1)
        Ho = output.size(1)
        total = q.size(0)
        eb = _gdn_cp_exact_buffers(device, nfine, H, D, total, Hv, Ho, output.dtype)
        zero_init, eye_init = eb["zero_init"], eb["eye_init"]
        Nseg, Mseg, Hin = eb["Nseg"], eb["Mseg"], eb["Hin"]
        vzero, scratch = eb["vzero"], eb["scratch"]
        _prof = os.environ.get("GDN_CP_PROFILE", "0") == "1"
        if _prof:
            import sys as _sys
            _p0, _p1, _p2, _p3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
            torch.cuda.synchronize(); _p0.record()
        # [STAGE3] Fused MN: ONE heavy pass produces BOTH N_seg (zero init, real v)
        # and M_seg (kernel identity-inits the homogeneous state internally, v=0),
        # sharing the per-chunk WY-inverse/decay work -> 3 heavy passes drop to 2.
        # GDN_CP_FUSE_MN=0 falls back to the original two separate passes.
        if os.environ.get("GDN_CP_FUSE_MN", "1") == "1":
            chunk_gated_delta_rule_sm100(
                q, k, v, gate, beta, scratch, plan["fine_cu"],
                zero_init, Nseg, scale, mn_mode=True, m_seg=Mseg,
            )
        else:
            # Pass N: zero init -> per-fine-seg additive state N_seg (token outputs discarded).
            chunk_gated_delta_rule_sm100(
                q, k, v, gate, beta, scratch, plan["fine_cu"],
                zero_init, Nseg, scale,
            )
            # Pass M: v=0, identity init -> per-fine-seg transition M_seg (= final_state).
            chunk_gated_delta_rule_sm100(
                q, k, vzero, gate, beta, scratch, plan["fine_cu"],
                eye_init, Mseg, scale,
            )
        if _prof: _p1.record()
        # exact cross-segment carry: true entry state per fine seg + per-group final state.
        if initial_state is not None:
            for i, idxs in enumerate(groups):
                Hin[idxs[0]] = initial_state[i].to(torch.float32)
        else:
            for idxs in groups:
                Hin[idxs[0]].zero_()
        _gdn_exact_scan(Hin, Mseg, Nseg, groups, initial_state is not None,
                        output_state=output_state)
        if _prof: _p2.record()
        # exact outputs: re-run each fine seg with its true entry state.
        chunk_gated_delta_rule_sm100(
            q, k, v, gate, beta, output, plan["fine_cu"],
            Hin, None, scale,
        )
        if _prof:
            _p3.record(); torch.cuda.synchronize()
            print(f"[CP_PROFILE] mn={_p0.elapsed_time(_p1)*1e3:.0f}us "
                  f"carry={_p1.elapsed_time(_p2)*1e3:.0f}us "
                  f"out={_p2.elapsed_time(_p3)*1e3:.0f}us", file=_sys.stderr)
        return True

    buf = _gdn_cp_buffers(device, nfine, H, D)
    zero_init, state1, Hin = buf["zero_init"], buf["state1"], buf["Hin"]

    # pass 1 (the only heavy pass): all sub-segments from zero init.
    # [opt] Pass initial_state=None instead of a zeros tensor so the kernel compiles
    # use_initial_state=False -> valid_state=False on EACH segment's first chunk
    # (is_first_chunk is per-segment), skipping the wasted zero-state load + q*state /
    # K@S GEMMs on chunk 0. Mathematically identical (zero init == None init).
    # GDN_CP_PASS1_NOINIT=0 restores the zeros-tensor path.
    _p1_init = None if os.environ.get("GDN_CP_PASS1_NOINIT", "1") == "1" else zero_init
    chunk_gated_delta_rule_sm100(
        q, k, v, gate, beta, output, plan["fine_cu"],
        _p1_init, state1, scale,
    )
    segment_updates = state1

    # cumdecay_t[h] = prod_{s<=t in seg} g_s[h]. Kept in (nfine, H, L) layout.
    # Prefer the caller's log-space gate (dispatch passes it through) to avoid the
    # exp()->log() round-trip: the SM100 op takes LINEAR g, but the CP cumdecay
    # needs log g, so re-logging the exp'd gate is pure waste. Fall back to the
    # local log when no g_log was supplied (direct packed-op callers).
    if g_log is not None:
        logg_full = g_log                                          # (total, H) log-space
    else:
        logg_full = torch.log(gate.clamp_min(1e-30))               # (total, H)
    if plan["equal"] and _HAS_TRITON:
        L = plan["L"]
        cumdecay, Gamma = _gdn_cumdecay_launch(logg_full, nfine, L)
    elif plan["equal"]:
        L = plan["L"]
        cumdecay = torch.exp(
            torch.cumsum(logg_full.view(nfine, L, H).transpose(1, 2), dim=2))  # (nfine,H,L)
        Gamma = cumdecay[:, :, L - 1].contiguous()                 # (nfine, H)
    else:
        maxL = plan["max_fine_len"]
        cumdecay = torch.empty(nfine, H, maxL, dtype=torch.float32, device=device)
        Gamma = torch.empty(nfine, H, dtype=torch.float32, device=device)
        for f in range(nfine):
            s0, s1 = fine[f], fine[f + 1]
            seg = torch.exp(torch.cumsum(logg_full[s0:s1].t(), dim=1))  # (H, Lf)
            cumdecay[f, :, : s1 - s0] = seg
            Gamma[f] = seg[:, -1]

    # serial scan within each original segment -> true entry state H_in_j.
    if initial_state is not None:
        for i, idxs in enumerate(groups):
            Hin[idxs[0]] = initial_state[i].to(torch.float32)
    # fused scan ALSO writes the per-group final state (one extra recurrence step)
    _gdn_scan_launch(Hin, segment_updates, Gamma, plan, initial_state is not None,
                     output_state=output_state)

    # analytic correction (fused Triton): o_t += scale * cumdecay_t * (q_t @ H_in^T)
    if plan["n_corr"]:
        _gdn_corr_launch(output, q, cumdecay, Hin, plan, scale)
    return True


def can_use_chunk_gdn_prefill_sm100(
    q: torch.Tensor,
    v: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
) -> bool:
    """Arch + shape gate for the SM100 packed-varlen GDN prefill op.

    ``output_final_state`` is retained for compatibility with the original
    public gate; it does not change kernel support.

    True iff: the vendored CuTe kernel loaded, q/v are compatible tensors on a
    CUDA SM100 device (capability major == 10, i.e. sm_100a/sm_103a), both head
    sizes are 128, and cu_seqlens is supplied on the same device.
    """
    if not _has_blackwell_prefill:
        return False
    del output_final_state
    try:
        if not q.is_cuda:
            return False
        if not _is_sm100a_cached(q.device):
            return False
        if v is None or cu_seqlens is None:
            return False
        if q.device != v.device or cu_seqlens.device != q.device:
            return False
        if q.dtype != v.dtype:
            return False
        if q.ndim not in (3, 4) or v.ndim != q.ndim:
            return False
        if q.shape[:-2] != v.shape[:-2]:
            return False
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return False
        num_q_heads = q.shape[-2]
        num_v_heads = v.shape[-2]
        if min(num_q_heads, num_v_heads) <= 0:
            return False
        if max(num_q_heads, num_v_heads) % min(num_q_heads, num_v_heads) != 0:
            return False
    except Exception:
        return False
    return True


def chunk_gated_delta_rule_sm100_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    g_log: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""SM100 packed-varlen Chunk-Gated-Delta-Rule prefill.

    Packed (single batch dim folded into the token axis) convention:
      q : ``[total_seq_len, num_q_heads, head_size]`` (bf16, contiguous, CUDA)
      k : ``[total_seq_len, num_k_heads, head_size]``
      v : ``[total_seq_len, num_v_heads, head_size]``
      g : ``[total_seq_len, num_sab_heads]`` LINEAR forget gate (alpha in (0,1]),
          float32; ``num_sab_heads = max(num_q_heads, num_v_heads)``.
      g_log : optional ``[total_seq_len, num_sab_heads]`` float32 LOG-space gate
          (``log(g)``). When supplied, the chunk-parallel cumdecay uses it directly
          instead of re-computing ``log(g)``, eliding an exp()->log() round-trip on
          the dispatch path. Purely an optimization; must equal ``log(g)`` if given.
      beta : ``[total_seq_len, num_sab_heads]`` float32.
      cu_seqlens : ``[num_seqs + 1]`` (required, varlen).
      initial_state / output_state : ``[num_seqs, num_sab_heads, head_size, head_size]`` fp32.

    Returns ``output`` (``[total_seq_len, num_o_heads, head_size]``) or
    ``(output, final_state)`` if ``output_final_state``.

    Requires SM100 (Blackwell, sm_100a/sm_103a) and head_size == 128. The
    chunk-parallel single-pass scan is env-gated (GDN_CP_*); falls back to a
    single vendor kernel call when no split is eligible.
    """
    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"
    if not (_has_blackwell_prefill and _is_sm100a_cached(q.device)):
        raise RuntimeError(
            "chunk_gated_delta_rule_sm100_packed requires an SM100 (Blackwell) "
            "device and the vendored gdn_prefill_sm100 CuTe kernel."
        )

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    device = q.device
    assert head_size == 128, (
        f"SM100 GDN prefill requires head_size=128, got {head_size}"
    )

    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_size), dtype=q.dtype, device=device
        )
    if output_final_state and output_state is None:
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32, device=device,
        )

    _scale = scale if scale is not None else 1.0 / math.sqrt(head_size)
    _g = (
        g if g is not None
        else torch.ones(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
    )
    _beta = (
        beta if beta is not None
        else torch.ones(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
    )

    _cu_i32 = _cu_seqlens_i32(cu_seqlens)

    # Occupancy-adaptive P (power-of-two grid fill): p = floor_pow2(SM // (B*Hv)),
    # so grid B*Hv*p <= SM (<= 1 wave). Override: GDN_CP_P caps P; GDN_CP_FORCE=1
    # uses the cap verbatim; GDN_CP_DISABLE=1 = single vendor pass.
    _cp_cap = int(os.environ.get("GDN_CP_P", "8"))
    _cp_min = int(os.environ.get("GDN_CP_MIN", "2048"))
    _cp_off = os.environ.get("GDN_CP_DISABLE", "0") == "1"
    _cp_force = os.environ.get("GDN_CP_FORCE", "0") == "1"
    # Exact K x K transition carry (Stage 1). GDN_CP_EXACT enables a 3-way dispatch:
    # fast-decay -> scalar CP (fastest, accurate there); slow-decay -> EXACT CP
    # (accurate at all alpha) instead of the no-CP fallback, provided its 3-pass cost
    # still beats no-CP (only for small state-head counts, GDN_CP_EXACT_MAX_HEADS).
    # With GDN_CP_FORCE=1 also set, exact CP is forced on every CP call (testing).
    _cp_exact = os.environ.get("GDN_CP_EXACT", "0") == "1"
    _cp_exact_max_heads = int(os.environ.get("GDN_CP_EXACT_MAX_HEADS", "32"))
    # Alpha-based CP skip: the scalar-cumdecay CP approximation loses precision on
    # slow-decay tokens (alpha close to 1). Empirically on real Qwen3.5-Plus:
    #   * Layer 0 has 67% of (token,head) alpha > 0.99 -> scalar-CP rel_l2 up to 90%.
    #   * Layers 6/13/20/26 have 0.01% alpha > 0.99  -> scalar-CP rel_l2 ~3e-3.
    # The dispatcher gates on the FRACTION of slow-decay (alpha > 0.99) pairs; if the
    # fraction exceeds GDN_CP_ALPHA_MAX_FRAC (default 0.5%), scalar CP is skipped and
    # the exact single-pass kernel is used.
    #
    # Tunables:
    #   GDN_CP_ALPHA_SLOW_THRESH (default 0.99): what counts as "slow decay" alpha.
    #   GDN_CP_ALPHA_MAX_FRAC (default 0.005):  max acceptable slow-decay fraction.
    #   Set GDN_CP_ALPHA_MAX_FRAC < 0 to disable the gate entirely.
    _cp_alpha_slow = float(os.environ.get("GDN_CP_ALPHA_SLOW_THRESH", "0.99"))
    _cp_alpha_max_frac = float(os.environ.get("GDN_CP_ALPHA_MAX_FRAC", "0.005"))
    _grid_base = num_seqs * num_sab_heads
    if _cp_force:
        _cp_P = _cp_cap
    else:
        _p_fit = (_gdn_sm_count(device) // _grid_base) if _grid_base > 0 else 0
        _p_pow2 = (1 << (_p_fit.bit_length() - 1)) if _p_fit >= 1 else 1
        _cp_P = min(_cp_cap, _p_pow2)

    # Pre-decide whether CP is viable BEFORE trying chunk parallelism, so we don't
    # do the plan-lookup / buffer alloc when the alpha guard rejects. Alpha check is
    # two element-wise reductions on _g plus one D2H sync (~3-6 us on B300); only pay
    # it when we would otherwise take the CP path.
    _cp_use = not _cp_off and _cp_P > 1 and _HAS_TRITON
    _this_exact = False
    if _cp_use and _cp_force:
        _this_exact = _cp_exact  # forced CP; exact only if explicitly combined (testing)
    elif _cp_use and _cp_alpha_max_frac >= 0.0:
        # _g is the LINEAR forget gate (alpha). Fraction of slow-decay (token,head).
        _alpha_frac_slow = float((_g > _cp_alpha_slow).float().mean())  # D2H sync
        if _alpha_frac_slow > _cp_alpha_max_frac:
            # slow-decay: scalar CP is inaccurate. Prefer EXACT CP (accurate at all
            # alpha) when enabled and its 3-pass cost still beats no-CP (small state
            # heads); else fall back to the exact single-pass vendor kernel (no-CP).
            if _cp_exact and num_sab_heads <= _cp_exact_max_heads:
                _this_exact = True
            else:
                _cp_use = False
        # fast-decay: keep scalar CP (accurate + fastest); _this_exact stays False
        if os.environ.get("GDN_CP_DISPATCH_TRACE", "0") == "1":
            _route = ("exact-CP" if _this_exact else "scalar-CP" if _cp_use else "no-CP")
            print(f"[GDN dispatch] frac(alpha>{_cp_alpha_slow})={_alpha_frac_slow:.4%} "
                  f"max_frac={_cp_alpha_max_frac:.4%} -> {_route}", flush=True)

    if _cp_use and _try_gdn_chunk_parallel(
        q, k, v, _g, _beta, output,
        _cu_i32, initial_state,
        output_state if output_final_state else None,
        _scale, _cp_P, _cp_min,
        g_log=g_log, exact=_this_exact,
    ):
        pass  # chunk-parallel single-pass handled it
    else:
        chunk_gated_delta_rule_sm100(
            q, k, v, _g, _beta, output, _cu_i32,
            initial_state, output_state if output_final_state else None, _scale,
        )

    if output_final_state:
        return output, output_state
    return output


# Convenience alias (the packed op is the public SM100 entry point).
chunk_gated_delta_rule = chunk_gated_delta_rule_sm100_packed
