"""Fused MoE down-projection (w2 GEMM) + topk-weight + expert reduction.

Specialized for the Qwen3.5-397B fixed prefill shape on H20 TP=8:
    A (act)   : [num_tokens*topk, K=128]   FP8 e4m3, block-quant [128,128]
    w2        : [E=512, N=4096, K=128]      FP8 e4m3, block-quant [128,128]
    output    : [num_tokens, N=4096]        bf16   (topk already reduced)

Folds three ops the stock path runs separately into one kernel:
  1. down GEMM   (was fused_moe_kernel on w2 -> intermediate_cache3 [M,topk,N])
  2. topk-weight multiply
  3. expert reduction over topk (was the separate ops.moe_sum)
via tl.atomic_add into output[token//topk]. This removes the moe_sum kernel
(~1.2 ms/call here) AND the 2.68 GB intermediate_cache3 write+read round trip.

Correctness-critical difference from the abandoned fused_moe_fused_reduce.py:
  BLOCK_SIZE_M is an EXPLICIT meta-param that MUST equal the BLOCK_SIZE_M used by
  moe_align_block_size to build sorted_token_ids/expert_ids. The old kernel let
  @triton.autotune choose BM independently -> expert_ids[pid_m] out-of-bounds ->
  CUDA illegal memory access. We never autotune BM; the caller passes it.

Atomic accumulator dtype:
  - bf16 output -> bf16 atomics (CAS-emulated on Hopper; contention is low, ~topk
    contributions per output row scattered across experts).
  - fp32 output -> native fp32 atomics (faster RMW), caller converts to bf16 after.
Both are exposed; the launcher picks via the `output` tensor dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_down_reduce_kernel(
    a_ptr, b_ptr, out_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_outm, stride_outn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    # block-quant group sizes
    group_n: tl.constexpr, group_k: tl.constexpr,
    # meta
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A: [num_tokens*topk, K], indexed by the expanded-row id directly.
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    # block-quant scale pointers (group_n=group_k=128 for this model).
    a_scale_ptrs = a_scale_ptr + offs_token * stride_asm
    offs_bsn = offs_bn // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        offs_ks = (k * BLOCK_SIZE_K) // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # topk-weight multiply (was MUL_ROUTED_WEIGHT in the down GEMM epilogue)
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    # reduce over topk: scatter-add into output[token // topk]
    actual_token_ids = offs_token // top_k
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_outm * actual_token_ids[:, None] + stride_outn * offs_cn[None, :]
    out_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(out_ptrs, accumulator, mask=out_mask, sem="relaxed")


@triton.jit
def fused_moe_down_reduce_v2_kernel(
    # Fast path for K == group_k (single scale group) and EVEN_K.
    # Hoists the block-quant scale OUT of the K-loop and uses fp8 fast-accum, which
    # cuts registers 195->119/thread -> occupancy 12.5%->25% -> ~32% faster than v1.
    a_ptr, b_ptr, out_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_outm, stride_outn,
    stride_asm,
    stride_bse, stride_bsn,
    group_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # V1: drop int64 casts. For this fixed deployment all offsets fit in i32
    # (max offs_token=327680, max byte offset under 270 MB).
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        return

    # V9: drop dead `% N` (max offs_bn = (N/BN-1)*BN + (BN-1) = N-1 < N always).
    # V10: hint alignment + contiguity for the BN dim — enables wider vector loads on w2/output.
    offs_bn = tl.max_contiguous(tl.multiple_of(
        pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    # V7: hoist small scale/weight loads BEFORE the K-loop so they overlap with wgmma.
    a_scale = tl.load(a_scale_ptr + offs_token * stride_asm, mask=token_mask, other=0.0)
    offs_bsn = offs_bn // group_n
    b_scale = tl.load(b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn)
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)

    # V8: hoist output address computation pre-loop too (none depend on accumulator).
    actual_token_ids = offs_token // top_k
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_outm * actual_token_ids[:, None] + stride_outn * offs_cn[None, :]
    out_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    # fp8 fast-accum GEMM over full K, NO per-iter scaling, NO K-mask (EVEN_K).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # iter5 V2: A reused across 32 N-tiles per m_tile under GM=8 -> evict_last
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0,
                    eviction_policy="evict_last")
        # V1: w2 is reused across M-blocks (8x via GROUP_SIZE_M=8 swizzle) -> L2 evict_last
        # iter5 V3: .cg cache_modifier tested neutral (Hopper L1 already not caching dense w2)
        b = tl.load(b_ptrs, eviction_policy="evict_last")
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # V7: fold three scalars in one expression (compiler emits single multiply chain).
    accumulator = (accumulator * (a_scale * moe_weight)[:, None] * b_scale[None, :]).to(compute_type)
    tl.atomic_add(out_ptrs, accumulator, mask=out_mask, sem="relaxed")


def fused_moe_down_reduce(
    A: torch.Tensor,            # [num_tokens*topk, K] fp8
    w2: torch.Tensor,           # [E, N, K] fp8
    a_scale: torch.Tensor,      # [num_tokens*topk, K//group_k] fp32
    b_scale: torch.Tensor,      # [E, N//group_n, K//group_k] fp32
    topk_weights: torch.Tensor, # [num_tokens, topk]
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    num_tokens: int,
    config: dict,               # BLOCK_SIZE_M (== moe_align BM) / N / K, GROUP_SIZE_M, num_warps, num_stages
    block_shape: list,          # [group_n, group_k] = [128,128]
    output: torch.Tensor | None = None,   # [num_tokens, N], will be zeroed
    fp32_acc: bool = False,     # accumulate in an fp32 scratch then convert to bf16
    compute_type=tl.bfloat16,
) -> torch.Tensor:
    E, N, K = w2.shape
    EM = sorted_token_ids.shape[0]
    group_n, group_k = block_shape

    if output is None:
        output = torch.empty((num_tokens, N), dtype=torch.bfloat16, device=A.device)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Fast path: K == group_k (single scale group) + EVEN_K. Hoists scale out of the
    # K-loop + fp8 fast-accum -> ~32% faster (ncu: regs 195->119, occ 12.5%->25%).
    if (not fp32_acc) and K == group_k and (K % config["BLOCK_SIZE_K"] == 0):
        output.zero_()
        fused_moe_down_reduce_v2_kernel[grid](
            A, w2, output,
            a_scale, b_scale,
            topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, EM, num_tokens * top_k,
            A.stride(0), A.stride(1),
            w2.stride(0), w2.stride(2), w2.stride(1),
            output.stride(0), output.stride(1),
            a_scale.stride(0),
            b_scale.stride(0), b_scale.stride(1),   # bse, bsn (N-block stride = dim1)
            group_n=group_n,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            top_k=top_k, compute_type=compute_type,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 3),
        )
        return output

    # General fallback (multi-group K, or fp32 accumulation).
    if fp32_acc:
        acc = torch.zeros((num_tokens, N), dtype=torch.float32, device=A.device)
        out_buf = acc
        atomic_compute = tl.float32
    else:
        output.zero_()
        out_buf = output
        atomic_compute = compute_type

    fused_moe_down_reduce_kernel[grid](
        A, w2, out_buf,
        a_scale, b_scale,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_tokens * top_k,
        A.stride(0), A.stride(1),
        w2.stride(0), w2.stride(2), w2.stride(1),   # be, bk(=K stride), bn(=N stride)
        out_buf.stride(0), out_buf.stride(1),
        a_scale.stride(0), a_scale.stride(1) if a_scale.ndim > 1 else 0,
        b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),  # bse, bsk, bsn(=N-block stride)
        group_n=group_n, group_k=group_k,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        top_k=top_k, compute_type=atomic_compute,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 3),
    )

    if fp32_acc:
        output.copy_(acc)
    return output
