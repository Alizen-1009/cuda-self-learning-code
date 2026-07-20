"""Before/after per shapes.json prefill shape:
  before = vLLM stock Triton `causal_conv1d_fn` (current live kernel; already has the SiLU-tanh
           patch, so vs the truly-original exp-SiLU it is ~10% faster already)
  after  = atrex causal_conv1d_fwd (conv_fast)
Both on channel-contiguous layout, single-seq fresh prefill. do_bench wall-clock includes each
kernel's host-side prep (heavier for the varlen Triton fn), so treat the large-T rows as the
meaningful ones and the in-model 1.79x @ dim=12288/T=8192 as the authoritative anchor.
"""
import os
os.environ.pop("CONV1D_FAST", None)  # ensure stock triton path
import json, torch, triton
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from atrex.api.causal_conv1d_fwd import causal_conv1d_fwd

WD = 4
HERE = os.path.dirname(os.path.abspath(__file__))
shapes = json.load(open(os.path.join(HERE, "..", "shapes.json")))


def make(dim, T):
    torch.manual_seed(0)
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.02).transpose(0, 1)  # channel-contig
    w = (torch.randn(dim, WD, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    b = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.02
    cs = torch.zeros(4, WD - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)  # dim-contig
    qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    cidx = torch.tensor([1], dtype=torch.int32, device="cuda")
    hinit = torch.tensor([False], device="cuda")
    return x, w, b, cs, qsl, cidx, hinit


def bench(dim, T):
    x, w, b, cs, qsl, cidx, hinit = make(dim, T)
    stock = lambda: causal_conv1d_fn(x, w, b, activation="silu", conv_states=cs,
                                     has_initial_state=hinit, cache_indices=cidx,
                                     query_start_loc=qsl, metadata=None)
    fast = lambda: causal_conv1d_fwd(x, w, b)
    us_s = triton.testing.do_bench(stock, warmup=25, rep=100, return_mode="median") * 1e3
    us_f = triton.testing.do_bench(fast, warmup=25, rep=100, return_mode="median") * 1e3
    return us_s, us_f


print(f"{'#':>2} {'model':>16} {'tokens':>6} {'dim':>5} {'stock_us':>9} {'fast_us':>8} {'speedup':>8}")
for k, v in shapes.items():
    ik = v["input_kwargs"]; T = ik["token_count"]; dim = ik["dim"]
    model = v.get("shape_metadata", {}).get("source_model", "?")
    if T == 1:
        print(f"{k:>2} {model:>16} {T:>6} {dim:>5}   -- decode (update kernel, skipped) --")
        continue
    us_s, us_f = bench(dim, T)
    print(f"{k:>2} {model:>16} {T:>6} {dim:>5} {us_s:>9.2f} {us_f:>8.2f} {us_s/us_f:>7.2f}x")
