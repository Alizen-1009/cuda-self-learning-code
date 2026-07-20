"""Correctness gate: compare stock causal_conv1d_fn vs conv_fast fast-path on the exact
production prefill call (single seq, fresh prefill, channel-last bf16, dim=12288/w=4).

Checks BOTH the conv output AND the conv_states cache writeback. Run with CONV1D_FAST unset
so `causal_conv1d_fn` resolves to stock; the fast path is called directly via conv_fast_ext.
Pass criterion: out rel_l2 < 1e-2 and conv_state rel_l2 < 1e-2.
"""
import os, sys
os.environ.pop("CONV1D_FAST", None)   # force stock in causal_conv1d_fn
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.mamba.ops import conv_fast_ext as cfe

dim = int(os.environ.get("DIM", 12288)); T = int(os.environ.get("T", 8192)); width = 4
torch.manual_seed(0)
xtok = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.1
x = xtok.transpose(0, 1)                                   # (dim,T) channel-last (== prod)
weight = (torch.randn(dim, width, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1
nlines = 4
# production conv_state is dim-contiguous: stride(1)==1. Allocate (nlines, w-1, dim) then
# transpose to (nlines, dim, w-1) so dim is the fastest axis (matches stock's assert).
cs_stock = torch.zeros(nlines, width - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
cs_fast = torch.zeros(nlines, width - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
cache_idx = torch.tensor([1], dtype=torch.int32, device="cuda")
has_init = torch.tensor([False], device="cuda")

out_stock = causal_conv1d_fn(x, weight, bias, activation="silu", conv_states=cs_stock,
                             has_initial_state=has_init, cache_indices=cache_idx,
                             query_start_loc=qsl, metadata=None)
out_fast = cfe.try_fast_prefill(x, weight, bias, cs_fast, qsl, cache_idx, has_init, "silu")
torch.cuda.synchronize()

if out_fast is None:
    print("FAIL: fast path returned None (guard rejected the production call)"); sys.exit(1)

rel_out = ((out_fast.float() - out_stock.float()).norm() / out_stock.float().norm()).item()
rel_cs = ((cs_fast.float() - cs_stock.float()).norm() /
          (cs_stock.float().norm() + 1e-9)).item()
print(f"out rel_l2       = {rel_out:.3e}")
print(f"conv_state rel_l2= {rel_cs:.3e}")
print("cs_stock[1] last-3 tok, chan0:", cs_stock[1, 0, :].float().cpu().tolist())
print("cs_fast [1] last-3 tok, chan0:", cs_fast[1, 0, :].float().cpu().tolist())
ok = rel_out < 1e-2 and rel_cs < 1e-2
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
