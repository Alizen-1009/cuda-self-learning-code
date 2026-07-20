"""CuTeDSL Chunk-GDN forward API for NVIDIA SM120.

This wraps the SM120-tuned 3-kernel GDN chunk-forward implementation:
K0 preprocess + K_inv Neumann + K1 fused chunk_h+chunk_o.
"""

import math
from typing import Optional, Tuple

import torch

BT = 32
K_DIM = 128
V_DIM = 128
BV = 16


def chunk_gdn_fwd_cutedsl_final_state_layout() -> str:
    return "B_HV_V_K"


def _to_public_final_state_layout(
    final_state: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if final_state is None:
        return None
    return final_state.transpose(-1, -2).contiguous()


def _check_sm120_device(tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError("CuTeDSL Chunk-GDN requires CUDA tensors")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    if (major, minor) != (12, 0):
        raise RuntimeError(
            "CuTeDSL Chunk-GDN is tuned for SM120; "
            f"current CUDA capability is sm_{major}{minor}"
        )


def _check_sm120_current_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CuTeDSL Chunk-GDN requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(
            "CuTeDSL Chunk-GDN is tuned for SM120; "
            f"current CUDA capability is sm_{major}{minor}"
        )


def _validate_inputs(ctx: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor) -> None:
    if q.ndim != 4:
        raise ValueError(f"q must be 4D [B, T, H, K], got shape {tuple(q.shape)}")
    b, t, h, k_dim = q.shape
    expected_t = ctx.get("T")
    if expected_t is not None and t != expected_t:
        raise ValueError(f"T shape mismatch: ctx was initialized for T={expected_t}, got T={t}")
    runtime_t = t if expected_t is None else expected_t
    expected_q = (ctx["B"], runtime_t, ctx["H"], ctx["K"])
    expected_v = (ctx["B"], runtime_t, ctx["HV"], ctx["V"])
    expected_gate = (ctx["B"], runtime_t, ctx["HV"])
    expected = {
        "q": expected_q,
        "k": expected_q,
        "v": expected_v,
        "g": expected_gate,
        "beta": expected_gate,
    }
    actual = {name: tuple(tensor.shape) for name, tensor in (
        ("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)
    )}
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {actual[name]}")
    if (b, h, k_dim) != (ctx["B"], ctx["H"], ctx["K"]):
        raise ValueError(
            "q shape config mismatch: "
            f"expected B/H/K={(ctx['B'], ctx['H'], ctx['K'])}, got {(b, h, k_dim)}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
    for name, tensor in (("g", g), ("beta", beta)):
        if not _gate_dtype_supported(tensor):
            raise TypeError(
                f"{name} dtype mismatch: expected torch.bfloat16 or "
                f"torch.float32, got {tensor.dtype}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")

    _check_sm120_device(q)


def _gate_dtype_supported(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.bfloat16, torch.float32)


def _to_kernel_gate_dtype(
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if g.dtype != torch.bfloat16:
        g = g.to(torch.bfloat16)
    if beta.dtype != torch.bfloat16:
        beta = beta.to(torch.bfloat16)
    return g, beta


def _is_supported_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: Optional[torch.Tensor],
    cp_context,
    transpose_state_layout: bool,
    kwargs: dict,
) -> bool:
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        return False
    if g.device != q.device or beta.device != q.device:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if not _gate_dtype_supported(g) or not _gate_dtype_supported(beta):
        return False
    if q.shape != k.shape or q.ndim != 4 or v.ndim != 4:
        return False
    b, t, h, k_dim = q.shape
    bv, tv, hv, v_dim = v.shape
    # Accept Qwen3.5 GDN shapes:
    #   dense: H=16, HV=48 (h_per_hv=3)
    #   TP1:   H=16, HV=64 (h_per_hv=4)
    #   TP2:   H=8,  HV=32 (h_per_hv=4)
    #   V113:  H=16, HV=32 (h_per_hv=2)
    if (b, tv, k_dim, v_dim) != (1, t, 128, 128):
        return False
    if h not in (8, 16) or hv not in (32, 48, 64):
        return False
    if hv % h != 0:
        return False
    h_per_hv = hv // h
    if h_per_hv not in (2, 3, 4):
        return False
    if g.shape != (1, t, hv) or beta.shape != (1, t, hv):
        return False
    if initial_state is not None or cp_context is not None or transpose_state_layout:
        return False
    if not use_qk_l2norm_in_kernel:
        return False
    if kwargs.get("head_first", False):
        return False
    if kwargs.get("use_gate_in_kernel", False):
        return False
    if cu_seqlens is not None and cu_seqlens.numel() != 2:
        return False
    return True


def _can_use_direct_runtime_t(
    t: int,
    hv: int,
    output_final_state: bool,
) -> bool:
    if t <= 0:
        return False
    if output_final_state:
        # Tail final-state direct kernels have shown wider numerical drift on
        # real GDN-style inputs. Serving passes cu_seqlens and uses the varlen
        # prefill kernel for packed prompt batches.
        if t % BT != 0:
            return hv == 48 and t >= 4096 and t < 32768
        return not (t >= 65536 and t % BT == 0)
    return t < 32768


def _packed_cu_values(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    total_t: int,
) -> list[int]:
    # [nosync] cache the D2H copy on the cu_seqlens tensor: all GDN layers in one
    # forward share the same cu_seqlens object, so do the .cpu()/.tolist() once.
    if cu_seqlens_cpu is None:
        _c = getattr(cu_seqlens, "_gdn_packed_cache", None)
        if _c is not None and _c[1] == total_t:
            return _c[0]
    meta = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.detach().cpu()
    values = [int(x) for x in meta.tolist()]
    if len(values) != int(cu_seqlens.numel()):
        raise ValueError(
            "cu_seqlens_cpu size mismatch: "
            f"expected {cu_seqlens.numel()}, got {len(values)}")
    if len(values) < 2:
        raise ValueError("cu_seqlens must contain at least two entries")
    if values[0] != 0 or values[-1] != int(total_t):
        raise ValueError(
            "packed cu_seqlens must start at 0 and end at total sequence "
            f"length {total_t}, got {values[0]}..{values[-1]}")
    prev = values[0]
    for cur in values[1:]:
        if cur <= prev:
            raise ValueError("packed cu_seqlens must be strictly increasing")
        prev = cur
    if cu_seqlens_cpu is None:
        try:
            cu_seqlens._gdn_packed_cache = (values, total_t)
        except Exception:
            pass
    return values


def _can_use_cu_seqlens_runtime(
    q: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    output_final_state: bool,
) -> bool:
    if q.shape[0] != 1:
        return False
    if not output_final_state:
        return False
    if q.shape[-1] != K_DIM or v.shape[-1] != V_DIM:
        return False
    if int(cu_seqlens.numel()) < 2:
        return False
    _packed_cu_values(cu_seqlens, cu_seqlens_cpu, int(q.shape[1]))
    try:
        from atrex.src.cutedsl.gdn_delta_rule_varlen_sm120 import (
            _has_sm120_delta_rule_dsl,
            delta_rule_prefill_dsl_sm120,
        )
    except Exception:
        return False
    if not _has_sm120_delta_rule_dsl or delta_rule_prefill_dsl_sm120 is None:
        return False
    return True


def can_use_chunk_gdn_fwd_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> bool:
    """Return whether ATREX should handle this GDN chunk call.

    vLLM uses this as the routing boundary. Unsupported calls should fall back
    before entering ATREX, matching the NVFP4 ``can_use_*`` dispatch pattern.
    """
    del scale
    try:
        # SM100 (Blackwell / B300): route to the self-contained packed-varlen
        # prefill op. vLLM drives the trio API with single [1, T, H, K] tensors
        # and cu_seqlens=None; synthesize [0, T] so the SM100 gate accepts it.
        if q.is_cuda:
            from atrex.api.chunk_gdn_prefill_sm100 import (
                _is_sm100a_cached,
                can_use_chunk_gdn_prefill_sm100,
            )
            if _is_sm100a_cached(q.device):
                cs = cu_seqlens
                if cs is None:
                    cs = torch.tensor(
                        [0, int(q.shape[1])],
                        dtype=torch.int32,
                        device=q.device,
                    )
                return can_use_chunk_gdn_prefill_sm100(q, v, cs)
        has_cu_seqlens = cu_seqlens is not None
        if not _is_supported_fast_path(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            use_qk_l2norm_in_kernel,
            None if has_cu_seqlens else cu_seqlens,
            cp_context,
            transpose_state_layout,
            kwargs,
        ):
            return False
        t = int(q.shape[1])
        hv = int(v.shape[2])
        if has_cu_seqlens:
            return _can_use_cu_seqlens_runtime(
                q,
                v,
                cu_seqlens,
                cu_seqlens_cpu,
                bool(output_final_state),
            )
        if _can_use_direct_runtime_t(t, hv, bool(output_final_state)):
            return True
        return False
    except Exception:
        return False


def chunk_gdn_fwd_cutedsl_build(
    B: int = 1,
    H: int = 16,
    HV: int = 32,
    K: int = K_DIM,
    V: int = V_DIM,
    warmup: bool = False,
    seq_len: Optional[int] = None,
    output_final_state: bool = False,
    scale: Optional[float] = None,
) -> dict:
    """Build the SM120 CuTeDSL Chunk-GDN forward context.

    This is intended to be called once during model initialization. Sequence
    length can remain dynamic by leaving ``seq_len=None``. For inference with a
    known static length, pass ``seq_len`` so the CuTeDSL kernels compile and warm
    during initialization instead of during the first forward call. ``warmup`` is
    kept for backward compatibility and requires ``seq_len`` when enabled.
    """
    # SM100 (Blackwell / B300): the packed-varlen prefill op is stateless and
    # needs no kernel precompile, so return a lightweight arch-tagged context
    # instead of running the SM120 device gate / static initialization below.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        init_scale = float(scale) if scale is not None else 1.0 / math.sqrt(K)
        return {
            "arch": "sm100",
            "B": int(B),
            "H": int(H),
            "HV": int(HV),
            "K": int(K),
            "V": int(V),
            "scale": init_scale,
            "output_final_state": bool(output_final_state),
        }
    _check_sm120_current_device()
    if B != 1:
        raise ValueError(f"only B=1 is supported, got B={B}")
    if K != K_DIM or V != V_DIM:
        raise ValueError(f"only K={K_DIM}, V={V_DIM} are supported, got K={K}, V={V}")
    if HV % H != 0:
        raise ValueError(f"HV must be divisible by H, got HV={HV}, H={H}")
    if V % BV != 0:
        raise ValueError(f"V must be divisible by {BV}, got {V}")
    if seq_len is not None and seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if warmup and seq_len is None:
        raise ValueError("warmup=True requires seq_len so kernels can be initialized statically")

    init_scale = float(scale) if scale is not None else 1.0 / math.sqrt(K)

    ctx = {
        "B": int(B),
        "H": int(H),
        "HV": int(HV),
        "K": int(K),
        "V": int(V),
        "BT": BT,
        "BV": BV,
        "T": int(seq_len) if seq_len is not None else None,
        "scale": init_scale if seq_len is not None else None,
        "output_final_state": bool(output_final_state) if seq_len is not None else None,
        "static_initialized": False,
    }

    if seq_len is not None:
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import initialize_3kernel

        initialize_3kernel(
            int(B),
            int(H),
            int(HV),
            int(K),
            int(V),
            int(seq_len),
            init_scale,
            output_final_state=bool(output_final_state),
        )
        ctx["static_initialized"] = True

    return ctx


def chunk_gdn_fwd_cutedsl(
    ctx: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    qk_l2norm_already_applied: bool = False,
    gate_is_exp: bool = False,
    output: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SM120 CuTeDSL Chunk-GDN forward.

    ``output`` (alias ``out``) is the caller's pre-allocated result buffer. This
    is the trio-API forward that vLLM actually drives, so threading it here (not
    just on the high-level ``chunk_gated_delta_rule_cutedsl`` wrapper) is what
    lets the SM100 path write in place and elide the per-layer output D2D copy.
    Both spellings are accepted since the caller convention may use either.

    Args:
        ctx: Context from :func:`chunk_gdn_fwd_cutedsl_build`.
        q: [B, T, H, K=128] bf16.
        k: [B, T, H, K=128] bf16.
        v: [B, T, HV, V=128] bf16.
        g: [B, T, HV] bf16/fp32 log-decay gate.
        beta: [B, T, HV] bf16/fp32 mixing weight.
        scale: Optional QK scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Whether to return the final recurrent state.

    Returns:
        ``(o, final_state)``. ``o`` is [B, T, HV, V] bf16. ``final_state`` is
        [B, HV, V, K] fp32 when requested, otherwise ``None``.
    """
    requested_scale = scale
    if requested_scale is None:
        requested_scale = ctx["scale"] if ctx.get("scale") is not None else 1.0 / math.sqrt(ctx["K"])
    requested_scale = float(requested_scale)

    # SM100 (Blackwell / B300): dispatch to the packed-varlen prefill op. The
    # ctx was arch-tagged by chunk_gdn_fwd_cutedsl_build; _dispatch_sm100 adapts
    # the [1, T, H, K] FLA-convention inputs and returns the public layout.
    if ctx.get("arch") == "sm100":
        cs = cu_seqlens
        if cs is None:
            cs = torch.tensor(
                [0, int(q.shape[1])],
                dtype=torch.int32,
                device=q.device,
            )
        return _dispatch_sm100(
            q,
            k,
            v,
            g,
            beta,
            float(requested_scale),
            None,
            bool(output_final_state),
            cs,
            True,
            output=output if output is not None else out,
        )

    if cu_seqlens is not None:
        return _chunk_gdn_fwd_cutedsl_cu_seqlens(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            scale=requested_scale,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            qk_l2norm_already_applied=qk_l2norm_already_applied,
            gate_is_exp=gate_is_exp,
        )

    g, beta = _to_kernel_gate_dtype(g, beta)
    cache_validation = ctx.get("T") is not None
    if cache_validation and ctx.get("_static_fast_ready"):
        if bool(output_final_state) != bool(ctx["output_final_state"]):
            raise ValueError(
                "output_final_state mismatch: ctx was initialized for "
                f"{ctx['output_final_state']}, got {output_final_state}"
            )
        if requested_scale != float(ctx["scale"]):
            raise ValueError(f"scale mismatch: ctx was initialized for {ctx['scale']}, got {requested_scale}")
        o, final_state = ctx["_chunk_gdn_static_impl"](
            q,
            k,
            v,
            g,
            beta,
            requested_scale,
        )
        if output_final_state:
            return o, _to_public_final_state_layout(final_state)
        return o, None
    validation_key = None
    if cache_validation:
        validation_key = (
            id(q), id(k), id(v), id(g), id(beta),
            requested_scale, bool(output_final_state),
        )
        if ctx.get("_validated_input_key") == validation_key and "_chunk_gdn_static_impl" in ctx:
            o, final_state = ctx["_chunk_gdn_static_impl"](
                q,
                k,
                v,
                g,
                beta,
                requested_scale,
            )
            if output_final_state:
                return o, _to_public_final_state_layout(final_state)
            return o, None
    if not cache_validation or ctx.get("_validated_input_key") != validation_key:
        _validate_inputs(ctx, q, k, v, g, beta)
    if ctx.get("static_initialized"):
        if bool(output_final_state) != bool(ctx["output_final_state"]):
            raise ValueError(
                "output_final_state mismatch: ctx was initialized for "
                f"{ctx['output_final_state']}, got {output_final_state}"
            )
        if requested_scale != float(ctx["scale"]):
            raise ValueError(f"scale mismatch: ctx was initialized for {ctx['scale']}, got {requested_scale}")

    from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import chunk_gated_delta_rule
    if cache_validation:
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _run_3kernel_v31_final_state
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _run_3kernel_v31_final_state_contiguous
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _run_3kernel_v31_final_state_direct_tail_contiguous
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _run_3kernel_v31_final_state_tail
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _run_3kernel_v31_final_state_tail_contiguous
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _alloc_3kernel_v31_workspace
        from atrex.src.cutedsl.gdn_chunk_fwd_sm120 import _use_split_state_output

        static_t = int(ctx["T"])
        all_contiguous = q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and g.is_contiguous() and beta.is_contiguous()
        static_impl = None
        workspace_tail_direct = False
        if static_t % BT != 0:
            if all_contiguous:
                if not _use_split_state_output(static_t):
                    static_impl = _run_3kernel_v31_final_state_direct_tail_contiguous
                    workspace_tail_direct = True
                else:
                    static_impl = _run_3kernel_v31_final_state_tail_contiguous
            else:
                static_impl = _run_3kernel_v31_final_state_tail
                workspace_tail_direct = not _use_split_state_output(static_t)
        elif static_t < 32768:
            if all_contiguous:
                static_impl = _run_3kernel_v31_final_state_contiguous
            else:
                static_impl = _run_3kernel_v31_final_state
        if static_impl is not None:
            if static_t % BT == 0:
                workspace_key = (
                    static_t, q.device.index, q.dtype, v.dtype,
                    tuple(q.shape), tuple(v.shape), False,
                )
                if ctx.get("_workspace_key") != workspace_key:
                    ctx["_workspace"] = _alloc_3kernel_v31_workspace(
                        q, v, tail_direct=False,
                    )
                    ctx["_workspace_key"] = workspace_key
                workspace = ctx["_workspace"]

                def static_call(q, k, v, g, beta, scale, _impl=static_impl, _workspace=workspace):
                    return _impl(q, k, v, g, beta, scale, workspace=_workspace)

                ctx["_chunk_gdn_static_impl"] = static_call
            else:
                ctx.pop("_workspace", None)
                ctx.pop("_workspace_key", None)
                ctx["_chunk_gdn_static_impl"] = static_impl
            ctx["_validated_input_key"] = validation_key
            if ctx.get("static_initialized"):
                ctx["_static_fast_ready"] = True

    o, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=requested_scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
    )
    if output_final_state:
        return o, _to_public_final_state_layout(final_state)
    return o, None


def _l2_normalize_last_dim_bf16(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inv_norm = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + 1e-6)
    return (xf * inv_norm).to(x.dtype)


def _chunk_gdn_fwd_cutedsl_cu_seqlens(
    ctx: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    qk_l2norm_already_applied: bool = False,
    gate_is_exp: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _validate_inputs(
        {**ctx, "T": int(q.shape[1])},
        q,
        k,
        v,
        g,
        beta,
    )
    if q.shape[0] != 1:
        raise ValueError(
            "ATREX cu_seqlens GDN prefill expects packed B=1 tensors, "
            f"got B={q.shape[0]}")
    if not output_final_state:
        raise ValueError("ATREX cu_seqlens GDN prefill requires output_final_state=True")

    cu_values = _packed_cu_values(cu_seqlens, cu_seqlens_cpu, int(q.shape[1]))
    num_seqs = len(cu_values) - 1
    total_t = int(q.shape[1])
    hv = int(v.shape[2])
    k_dim = int(q.shape[3])
    v_dim = int(v.shape[3])
    aligned_full_blocks = all(
        ((end - start) % 64) == 0
        for start, end in zip(cu_values[:-1], cu_values[1:])
    )

    from atrex.src.cutedsl.gdn_delta_rule_varlen_sm120 import (
        delta_rule_prefill_dsl_sm120,
    )

    if delta_rule_prefill_dsl_sm120 is None:
        raise RuntimeError("ATREX SM120 cu_seqlens GDN prefill kernel is unavailable")

    q_flat = q.squeeze(0).contiguous()
    k_flat = k.squeeze(0).contiguous()
    if not qk_l2norm_already_applied:
        q_flat = _l2_normalize_last_dim_bf16(q_flat)
        k_flat = _l2_normalize_last_dim_bf16(k_flat)
    v_flat = v.squeeze(0).contiguous()
    gate = g.squeeze(0).float()
    if not gate_is_exp:
        gate = torch.exp(gate).contiguous()
    beta_flat = beta.squeeze(0).float()
    cu_i64 = cu_seqlens.to(torch.int64).contiguous()

    output = torch.empty(
        (total_t, hv, v_dim),
        dtype=v.dtype,
        device=v.device,
    )
    final_state = torch.empty(
        (num_seqs, hv, v_dim, k_dim),
        dtype=torch.float32,
        device=v.device,
    )

    delta_rule_prefill_dsl_sm120(
        output,
        final_state,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate,
        beta_flat,
        cu_i64,
        float(scale),
        split_v_parts=2,
        aligned_full_blocks=aligned_full_blocks,
    )

    return output.unsqueeze(0), final_state


def _dispatch_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapt the FLA-convention call onto the packed SM100 GDN prefill op.

    atrex FLA convention -> packed-varlen convention:
      * q/k/v ``[1, total, H, D]`` -> ``[total, H, D]``
      * ``g`` is log-space here; kept as ``g_log`` for the CP cumdecay AND ``exp()``'d
        to linear for the vendor kernel (which takes linear g). Passing both elides
        an exp()->log() round-trip inside the op.
      * ``beta`` -> float32; q/k l2-normalized when ``use_qk_l2norm_in_kernel``
      * ``output`` (caller's ``[1, total, Ho, D]`` buffer, optional): squeezed to a
        ``[total, Ho, D]`` VIEW and written in place, eliding a per-layer D2D copy.
      * initial_state/output_state stay ``[N, HV, V, K]`` fp32 (= atrex public layout)
    Returns ``(o [1, total, Ho, D], final_state [N, HV, V, K] | None)``.
    """
    from atrex.api.chunk_gdn_prefill_sm100 import (
        chunk_gated_delta_rule_sm100_packed,
    )

    q2 = q.squeeze(0)
    k2 = k.squeeze(0)
    v2 = v.squeeze(0).contiguous()
    if use_qk_l2norm_in_kernel:
        q2 = _l2_normalize_last_dim_bf16(q2)
        k2 = _l2_normalize_last_dim_bf16(k2)
    q2 = q2.contiguous()
    k2 = k2.contiguous()
    g_log2 = g.squeeze(0).float().contiguous()        # keep log-space for CP cumdecay
    g2 = g_log2.exp().contiguous()                    # linear g for the vendor kernel
    beta2 = beta.squeeze(0).float().contiguous()

    # Pass the caller's output buffer straight down so the op writes in place
    # (elides a per-layer device-to-device copy). squeeze(0) of a contiguous
    # [1, total, Ho, D] is a contiguous VIEW, so the in-place write aliases it.
    out2 = output.squeeze(0) if (output is not None and output.dim() == 4) else output

    result = chunk_gated_delta_rule_sm100_packed(
        q2,
        k2,
        v2,
        g=g2,
        beta=beta2,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output=out2,
        g_log=g_log2,
    )
    if output_final_state:
        o, final_state = result
    else:
        o, final_state = result, None
    o = o.unsqueeze(0)                                 # [1, total, Ho, D]
    return o, final_state


def chunk_gated_delta_rule_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    output: Optional[torch.Tensor] = None,
    **kwargs,
):
    """FLA-compatible CuTeDSL Chunk-GDN wrapper (arch-dispatched SM100 / SM120).

    ``output`` (optional) is the caller's pre-allocated result buffer
    ``[1, total, Ho, D]``; on the SM100 path it is threaded down and written in
    place to elide a per-layer device-to-device copy. Ignored on fallback paths
    (result is returned normally). Kept after the FLA-signature params so the
    signature-compat test still sees the FLA prefix.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(k.shape[-1])

    # ---- SM100 (Blackwell / B300, sm_100a/sm_103a) path --------------------
    # A separate self-contained vendored op (no flashinfer dependency), selected
    # by architecture BEFORE the SM120 gate. Packed-varlen prefill (cu_seqlens)
    # only; dense and cp_context/transpose_state cases fall through to the
    # SM120/FLA logic below.
    if (
        (not head_first)
        and cu_seqlens is not None
        and cp_context is None
        and not transpose_state_layout
    ):
        try:
            from atrex.api.chunk_gdn_prefill_sm100 import (
                can_use_chunk_gdn_prefill_sm100,
            )

            _use_sm100 = can_use_chunk_gdn_prefill_sm100(q, v, cu_seqlens)
        except Exception:
            _use_sm100 = False
        if _use_sm100:
            return _dispatch_sm100(
                q,
                k,
                v,
                g,
                beta,
                scale,
                initial_state,
                output_final_state,
                cu_seqlens,
                use_qk_l2norm_in_kernel,
                output=output,
            )

    can_use_fast_path = (not head_first) and can_use_chunk_gdn_fwd_cutedsl(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cp_context=cp_context,
        transpose_state_layout=transpose_state_layout,
        head_first=head_first,
        **kwargs,
    )

    if not can_use_fast_path:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

    if cu_seqlens is None:
        g, beta = _to_kernel_gate_dtype(g, beta)
    _check_sm120_device(q)

    ctx = chunk_gdn_fwd_cutedsl_build(
        B=int(q.shape[0]),
        H=int(q.shape[2]),
        HV=int(v.shape[2]),
        K=int(q.shape[3]),
        V=int(v.shape[3]),
        output_final_state=output_final_state,
        scale=float(scale),
    )
    o, final_state = chunk_gdn_fwd_cutedsl(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale=float(scale),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return o, final_state
