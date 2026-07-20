"""Before/after (stock Triton vs conv_fast) sweep, T=2k..16k at production dim=12288.
This range is kernel-bound, so the wall-clock ratio is a real kernel-level speedup.
"""
import os
os.environ.pop("CONV1D_FAST", None)
import torch, triton
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd

WD = 4
DIM = int(os.environ.get("DIM", 12288))


def make(dim, T):
    torch.manual_seed(0)
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.02).transpose(0, 1)
    w = (torch.randn(dim, WD, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    b = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.02
    cs = torch.zeros(4, WD - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    cidx = torch.tensor([1], dtype=torch.int32, device="cuda")
    hinit = torch.tensor([False], device="cuda")
    return x, w, b, cs, qsl, cidx, hinit


print(f"dim={DIM}")
print(f"{'T':>6} {'stock_us':>9} {'fast_us':>8} {'copy_us':>8} {'speedup':>8} {'fast_%copy':>10}")
for T in [2048, 4096, 8192, 12288, 16384]:
    x, w, b, cs, qsl, cidx, hinit = make(DIM, T)
    stock = lambda: causal_conv1d_fn(x, w, b, activation="silu", conv_states=cs,
                                     has_initial_state=hinit, cache_indices=cidx,
                                     query_start_loc=qsl, metadata=None)
    fast = lambda: causal_conv1d_fwd(x, w, b)
    dst = torch.empty_like(x)
    us_s = triton.testing.do_bench(stock, warmup=25, rep=100, return_mode="median") * 1e3
    us_f = triton.testing.do_bench(fast, warmup=25, rep=100, return_mode="median") * 1e3
    cp = triton.testing.do_bench(lambda: dst.copy_(x), warmup=25, rep=100, return_mode="median") * 1e3
    print(f"{T:>6} {us_s:>9.2f} {us_f:>8.2f} {cp:>8.2f} {us_s/us_f:>7.2f}x {100*cp/us_f:>9.1f}%")
