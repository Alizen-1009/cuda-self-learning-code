"""Sweep causal_conv1d_fwd across shapes: report us, %SOL (nominal 8TB/s), and % of the
per-shape torch-copy ceiling (achievable HBM BW), plus correctness rel_l2.

Run on B300: CUDA_VISIBLE_DEVICES=<free> python bench_shapes.py
"""
import os, torch, torch.nn.functional as F, triton
from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd

WD = 4
NOMINAL_TBps = 8.0  # B300 HBM nominal


def bench(dim, T):
    torch.manual_seed(0)
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.1).transpose(0, 1)
    w = (torch.randn(dim, WD, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    b = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1
    # correctness
    seq = F.pad(x.unsqueeze(0).float(), (WD - 1, 0))
    ref = F.silu(F.conv1d(seq, w.unsqueeze(1).float(), b.float(), groups=dim)[:, :, -T:]).squeeze(0)
    act = causal_conv1d_fwd(x, w, b)
    torch.cuda.synchronize()
    rel = (act.float() - ref).norm().item() / ref.norm().item()
    # timing
    us = triton.testing.do_bench(lambda: causal_conv1d_fwd(x, w, b), warmup=25, rep=100,
                                 return_mode="median") * 1e3
    # per-shape copy ceiling (read x + write out, same traffic)
    dst = torch.empty_like(x)
    cp = triton.testing.do_bench(lambda: dst.copy_(x), warmup=25, rep=100,
                                 return_mode="median") * 1e3
    bytes_moved = 2 * dim * T * 2  # read x + write out, bf16
    sol_us = bytes_moved / (NOMINAL_TBps * 1e12) * 1e6
    pct_sol = 100 * sol_us / us
    pct_copy = 100 * cp / us  # copy_ moves same 2*dim*T*2 bytes -> fair ceiling
    return rel, us, pct_sol, cp, pct_copy


print(f"{'dim':>6} {'T':>6} {'rel_l2':>9} {'conv_us':>9} {'copy_us':>9} {'%SOL(nom)':>10} {'%of_copy':>9}")
DIM = 12288
for T in [512, 1024, 2048, 4096, 8192, 16384]:
    rel, us, psol, cp, pcopy = bench(DIM, T)
    print(f"{DIM:>6} {T:>6} {rel:>9.2e} {us:>9.2f} {cp:>9.2f} {psol:>10.1f} {pcopy:>9.1f}")
print("--- dim sensitivity @ T=8192 ---")
for dim in [4096, 8192, 12288, 16384]:
    rel, us, psol, cp, pcopy = bench(dim, 8192)
    print(f"{dim:>6} {8192:>6} {rel:>9.2e} {us:>9.2f} {cp:>9.2f} {psol:>10.1f} {pcopy:>9.1f}")
