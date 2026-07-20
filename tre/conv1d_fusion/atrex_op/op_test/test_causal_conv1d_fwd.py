"""Correctness + performance test for the causal_conv1d_fwd CUDA op (SM100).

Compares against a PyTorch reference silu(conv1d) on the production shape (channel-contiguous
bf16, dim=12288, T=8192, width=4) and reports pct-of-SOL vs the 8 TB/s nominal HBM bandwidth.
Pass: rel_l2 < 1e-2.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd, can_use_causal_conv1d_fwd


def reference(x, weight, bias):
    # x: (dim, T) -> conv1d expects (N, C, L); depthwise groups=dim, left-pad width-1.
    dim, T = x.shape
    wd = weight.shape[1]
    seq = F.pad(x.unsqueeze(0).float(), (wd - 1, 0))
    out = F.conv1d(seq, weight.unsqueeze(1).float(), bias.float(), groups=dim)[:, :, -T:]
    return F.silu(out).squeeze(0)


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 12288
    wd = 4
    torch.manual_seed(0)
    # production layout: channel-contiguous (dim, T) with dim stride 1
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.1).transpose(0, 1)
    weight = (torch.randn(dim, wd, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1

    assert can_use_causal_conv1d_fwd(x, weight, bias), "guard rejected the production input"

    ref = reference(x, weight, bias)
    act = causal_conv1d_fwd(x, weight, bias)
    torch.cuda.synchronize()
    rel = (act.float() - ref).norm().item() / ref.norm().item()

    import triton
    us = triton.testing.do_bench(lambda: causal_conv1d_fwd(x, weight, bias),
                                 warmup=25, rep=100, return_mode="median") * 1e3
    sol = 2 * dim * T * 2 / 8e12 * 1e6
    print(f"causal_conv1d_fwd dim={dim} T={T} rel_l2={rel:.2e} "
          f"us={us:.2f} SOL={sol:.1f}us pct_SOL={100 * sol / us:.1f}%")
    assert rel < 1e-2, f"correctness FAIL: rel_l2={rel:.2e}"
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
