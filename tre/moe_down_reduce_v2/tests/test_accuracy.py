#!/usr/bin/env python3
"""精度门禁:融合核 vs 原路径(down GEMM + moe_sum)的单算子对比。

同一份合成输入,跑完整 fused_experts 两遍 —— 开关 OFF(原路径)与 ON(融合核),
比较规约后的 [M, hidden] 输出。参照物 = 它要替换的原路径(不是 fp32 真值):
我们要证的是「实现等价」,落在 FP8 量化自身噪声以内即放行。

融合核改了规约顺序 + 累加器(bf16 原子加替代 fp32 moe_sum),所以预期会有
~1e-2 级相对差异 —— 这是预期,不是 bug。判据是 < FP8 量化噪声(THRESH)。

用法:  CUDA_VISIBLE_DEVICES=0 python test_accuracy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import vllm.model_executor.layers.fused_moe.fused_moe as fm
from _inputs import build_inputs

THRESH = 2e-2  # FP8 e4m3 块量化自身误差量级;融合差异须低于此


def run(flag, inp):
    # 开关在 fused_experts_impl 内每次调用时读取 -> 同进程切换即可
    os.environ["VLLM_MOE_USE_FUSED_REDUCE"] = flag
    out = fm.fused_experts(inp["x"], inp["w1"], inp["w2"], inp["topk_weights"],
                           inp["topk_ids"], inplace=False,
                           quant_config=inp["quant_config"])
    torch.cuda.synchronize()
    return out.float()


def main():
    inp = build_inputs()
    stock = run("0", inp)   # 原路径(参照)
    fused = run("1", inp)   # 融合核

    d = (stock - fused).abs()
    max_rel = d.max().item() / (stock.abs().max().item() + 1e-9)
    mean_rel = d.mean().item() / (stock.abs().mean().item() + 1e-9)

    print(f"output shape           : {tuple(stock.shape)}")
    print(f"max_rel (vs stock)     : {max_rel:.3e}")
    print(f"mean_rel (vs stock)    : {mean_rel:.3e}")
    print(f"threshold              : {THRESH:.1e}")
    ok = max_rel < THRESH
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"max_rel {'<' if ok else '>='} {THRESH:.1e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
