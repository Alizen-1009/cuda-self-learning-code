"""Bench causal_conv1d_fwd on the actual shapes.json shapes (prefill only, token_count>1),
in the production CHANNEL-CONTIGUOUS layout (shapes.json's input.py uses token-contiguous,
which is the harness layout and the wrong one to measure on). Decode shapes (tokens=1) are
the update-kernel path and skipped.
"""
import json, os, torch, torch.nn.functional as F, triton
from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd

WD = 4
HERE = os.path.dirname(os.path.abspath(__file__))
shapes = json.load(open(os.path.join(HERE, "..", "shapes.json")))


def bench(dim, T):
    torch.manual_seed(0)
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.02).transpose(0, 1)  # channel-contig
    w = (torch.randn(dim, WD, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    b = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.02
    seq = F.pad(x.unsqueeze(0).float(), (WD - 1, 0))
    ref = F.silu(F.conv1d(seq, w.unsqueeze(1).float(), b.float(), groups=dim)[:, :, -T:]).squeeze(0)
    act = causal_conv1d_fwd(x, w, b)
    torch.cuda.synchronize()
    rel = (act.float() - ref).norm().item() / ref.norm().item()
    us = triton.testing.do_bench(lambda: causal_conv1d_fwd(x, w, b), warmup=25, rep=100, return_mode="median") * 1e3
    dst = torch.empty_like(x)
    cp = triton.testing.do_bench(lambda: dst.copy_(x), warmup=25, rep=100, return_mode="median") * 1e3
    sol_us = 2 * dim * T * 2 / 8e12 * 1e6
    return rel, us, cp, 100 * sol_us / us, 100 * cp / us


print(f"{'#':>2} {'model':>16} {'tokens':>6} {'dim':>5} {'rel_l2':>9} {'conv_us':>8} {'copy_us':>8} {'%SOL':>6} {'%copy':>6}")
for k, v in shapes.items():
    ik = v["input_kwargs"]; T = ik["token_count"]; dim = ik["dim"]
    model = v.get("shape_metadata", {}).get("source_model", "?")
    if T == 1:
        print(f"{k:>2} {model:>16} {T:>6} {dim:>5}   -- decode (causal_conv1d_update, not this fwd op) --")
        continue
    rel, us, cp, psol, pcopy = bench(dim, T)
    print(f"{k:>2} {model:>16} {T:>6} {dim:>5} {rel:>9.2e} {us:>8.2f} {cp:>8.2f} {psol:>6.1f} {pcopy:>6.1f}")
