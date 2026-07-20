"""Synthetic inputs at the fixed deployment shape (no real model weights needed).

Shape: M=32768 tokens, E=512 experts, topk=10, hidden=4096, per-GPU
intermediate=128, FP8 e4m3 block-quant [128,128]. One H20 GPU is enough.
Shared by test_accuracy.py and bench_perf.py.
"""
import os

os.environ.setdefault("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")
os.environ.setdefault("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()

M = 32768
E = 512
TOPK = 10
HIDDEN = 4096
SHARD_INTERMEDIATE = 256        # = 2 * per-GPU intermediate (128)
BLOCK_SHAPE = [128, 128]


def build_inputs(device="cuda"):
    torch.manual_seed(0)
    dt = torch.bfloat16
    x = torch.randn(M, HIDDEN, dtype=dt, device=device) / 10
    w1 = (torch.randn(E, SHARD_INTERMEDIATE, HIDDEN, dtype=dt, device=device) / 10).to(FP8_DTYPE)
    w2 = (torch.randn(E, HIDDEN, SHARD_INTERMEDIATE // 2, dtype=dt, device=device) / 10).to(FP8_DTYPE)
    block_n, block_k = BLOCK_SHAPE
    N, K, f = SHARD_INTERMEDIATE // 2, HIDDEN, 1e-2
    w1_scale = torch.rand(E, (2 * N + block_n - 1) // block_n, (K + block_k - 1) // block_k,
                          dtype=torch.float32, device=device) * f
    w2_scale = torch.rand(E, (K + block_n - 1) // block_n, (N + block_k - 1) // block_k,
                          dtype=torch.float32, device=device) * f
    a1_scale = torch.randn(1, dtype=torch.float32, device=device)
    a2_scale = torch.randn(1, dtype=torch.float32, device=device)
    gating = torch.randn(M, E, dtype=torch.float32, device=device)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_scale, w2_scale=w2_scale,
        a1_scale=a1_scale, a2_scale=a2_scale,
        block_shape=BLOCK_SHAPE,
    )
    topk_weights, topk_ids, *_ = fused_topk(x, gating, TOPK, renormalize=True)
    return dict(x=x, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids,
                quant_config=quant_config)
