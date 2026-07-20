"""Overlay the causal_conv1d_fwd op into the installed atrex, then run correctness+bench.

Run on B300: CUDA_VISIBLE_DEVICES=<free> python overlay_and_test.py
Backs up the installed optCompilerConfig.json to .convfastorig (idempotent).
"""
import importlib.util, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATREX = os.path.dirname(importlib.util.find_spec("atrex").origin)
print("installed atrex:", ATREX)

# 1. copy op sources into site-packages atrex
dst = os.path.join(ATREX, "src/cuda/causal_conv1d_fwd/include")
os.makedirs(dst, exist_ok=True)
srcd = os.path.join(HERE, "src/cuda/causal_conv1d_fwd")
shutil.copy2(os.path.join(srcd, "causal_conv1d_fwd.cu"), os.path.dirname(dst))
shutil.copy2(os.path.join(srcd, "causal_conv1d_fwd_pybind.cu"), os.path.dirname(dst))
shutil.copy2(os.path.join(srcd, "include/causal_conv1d_fwd.h"), dst)
shutil.copy2(os.path.join(HERE, "api/causal_conv1d_fwd.py"), os.path.join(ATREX, "api"))
print("copied op sources + api wrapper")

# 2. add manifest entry (from pristine backup, idempotent)
cfg = os.path.join(ATREX, "core/optCompilerConfig.json")
bak = cfg + ".convfastorig"
if not os.path.exists(bak):
    shutil.copy2(cfg, bak)
d = json.load(open(bak))
d["_causal_conv1d_fwd_kernel"] = {
    "srcs": [
        "f'{src_dir}/cuda/causal_conv1d_fwd/causal_conv1d_fwd.cu'",
        "f'{src_dir}/cuda/causal_conv1d_fwd/causal_conv1d_fwd_pybind.cu'",
    ],
    "flags_extra_cc": [],
    "flags_extra_hip": ["'--use_fast_math'", "f'-gencode=arch=compute_100a,code=sm_100a'"],
    "extra_include": ["f'{src_dir}/cuda/causal_conv1d_fwd/include'"],
    "verbose": "False",
}
json.dump(d, open(cfg, "w"), indent=4)
print("manifest entry present:", "_causal_conv1d_fwd_kernel" in json.load(open(cfg)))

# 3. correctness + bench against PyTorch reference (production shape)
import torch, torch.nn.functional as F
from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd, can_use_causal_conv1d_fwd

T = int(os.environ.get("T", 8192)); dim = int(os.environ.get("DIM", 12288)); wd = 4
torch.manual_seed(0)
x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.1).transpose(0, 1)
weight = (torch.randn(dim, wd, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1
assert can_use_causal_conv1d_fwd(x, weight, bias), "guard rejected production input"

seq = F.pad(x.unsqueeze(0).float(), (wd - 1, 0))
ref = F.silu(F.conv1d(seq, weight.unsqueeze(1).float(), bias.float(), groups=dim)[:, :, -T:]).squeeze(0)
act = causal_conv1d_fwd(x, weight, bias)
torch.cuda.synchronize()
rel = (act.float() - ref).norm().item() / ref.norm().item()

import triton
us = triton.testing.do_bench(lambda: causal_conv1d_fwd(x, weight, bias),
                             warmup=25, rep=100, return_mode="median") * 1e3
sol = 2 * dim * T * 2 / 8e12 * 1e6
print(f"causal_conv1d_fwd dim={dim} T={T} rel_l2={rel:.2e} us={us:.2f} pct_SOL={100*sol/us:.1f}%")
print("RESULT:", "PASS" if rel < 1e-2 else "FAIL")
sys.exit(0 if rel < 1e-2 else 1)
