#!/usr/bin/env python3
"""性能对比(triton do_bench):MoE 层 fused_experts,开关 OFF vs ON。

真实的替换是「两个核(down GEMM + moe_sum)-> 一个融合核」。在 fused_experts
这一层对比 OFF/ON,delta 正好隔离出 down+reduce 这步的收益(up GEMM/激活不变),
是最贴近部署的可信数字。量化在 build_inputs 里一次性完成,不计入计时。

用法:  CUDA_VISIBLE_DEVICES=0 python bench_perf.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.triton_utils import triton
from _inputs import build_inputs, M, TOPK, HIDDEN, SHARD_INTERMEDIATE


def make_thunk(flag, inp):
    os.environ["VLLM_MOE_USE_FUSED_REDUCE"] = flag

    def run():
        fm.fused_experts(inp["x"], inp["w1"], inp["w2"], inp["topk_weights"],
                         inp["topk_ids"], inplace=False,
                         quant_config=inp["quant_config"])
    return run


def bench(flag, inp):
    fn = make_thunk(flag, inp)
    fn()  # warmup / JIT autotune
    torch.cuda.synchronize()
    # do_bench 默认在 rep 间冲 L2,取中位数避免抖动
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def main():
    inp = build_inputs()
    off = bench("0", inp)   # 原路径:down GEMM + moe_sum
    on = bench("1", inp)    # 融合核

    delta = off - on
    print(f"fixed shape: M={M}, topk={TOPK}, hidden={HIDDEN}, "
          f"intermediate(2x)={SHARD_INTERMEDIATE}")
    print(f"MoE fused_experts (median over 100 reps):")
    print(f"  OFF  (down GEMM + moe_sum) : {off:8.3f} ms")
    print(f"  ON   (fused down-reduce)   : {on:8.3f} ms")
    print(f"  delta (down+reduce 收益)   : {delta:8.3f} ms  "
          f"({100 * delta / off:+.1f}% of the whole MoE layer)")


if __name__ == "__main__":
    main()
