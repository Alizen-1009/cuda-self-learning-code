"""Correctness: [conv_split + l2norm] vs stock [causal_conv1d_fn + fused_conv_split_l2norm_rearrange]
on the production prefill call (single seq, fresh prefill, channel-last bf16, dim=12288)."""
import os, sys
os.environ.pop("CONV_SPLIT_FUSE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.models.qwen3_next import fused_conv_split_l2norm_rearrange
import conv_split_ext as cse

NHQK, NHV, HD, WD = 16, 64, 128, 4
KEYDIM = NHQK * HD; DIM = 2 * KEYDIM + NHV * HD
T = int(os.environ.get("T", 8192))
torch.manual_seed(0)
xtok = torch.randn(T, DIM, device="cuda", dtype=torch.bfloat16) * 0.05
x_T = xtok.transpose(0, 1)                                   # (dim,T) channel-last, = mixed_qkv_non_spec_T
weight = (torch.randn(DIM, WD, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
bias = torch.randn(DIM, device="cuda", dtype=torch.bfloat16) * 0.1
cs = torch.zeros(4, WD - 1, DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)  # dim-contig
qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
cidx = torch.tensor([1], dtype=torch.int32, device="cuda")
hinit = torch.tensor([False], device="cuda")

# stock: conv -> (T,dim) -> fused split+l2norm+rearrange
conv_out = causal_conv1d_fn(x_T, weight, bias, activation="silu", conv_states=cs.clone(),
                            has_initial_state=hinit, cache_indices=cidx,
                            query_start_loc=qsl, metadata=None).transpose(0, 1)
rq, rk, rv = fused_conv_split_l2norm_rearrange(conv_out, NHQK, NHV, HD, HD)

# fused: gate -> conv_split + l2norm
assert cse.gate(weight, qsl, hinit, x_T), "gate rejected production input"
q, k, v = cse.fused_conv_split(x_T, weight, bias, NHQK, NHV, HD)
torch.cuda.synchronize()

rel = lambda a, r: (a.float() - r.float()).norm().item() / (r.float().norm().item() + 1e-9)
print(f"out rel  q={rel(q,rq):.3e} k={rel(k,rk):.3e} v={rel(v,rv):.3e}")
ok = rel(q, rq) < 1e-2 and rel(k, rk) < 1e-2 and rel(v, rv) < 1e-2
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
