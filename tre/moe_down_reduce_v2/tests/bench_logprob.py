#!/usr/bin/env python3
"""(可选)端到端兜底:开关 off vs on 的 PPL / 逐 token logprob 对比。

对一组固定文本用 /v1/completions 的 prompt_logprobs(teacher-forced,确定性、
无采样)取逐 token logprob,算 PPL 与逐 token 差异。off 与 on 用同一段文本 ->
同样的 token 序列 -> 可逐位置精确比较。delta≈噪声即说明跨层累积误差无害。

任务准确率(MMLU/GSM8K)对 1e-2 级扰动不敏感,故这里用 PPL/logprob(敏感探针)。

用法(服务需先起好,两次分别在 VLLM_MOE_USE_FUSED_REDUCE=0/1 下起):
    python bench_logprob.py collect --tag off    # 在 off 服务上跑
    python bench_logprob.py collect --tag on     # 在 on  服务上跑
    python bench_logprob.py compare              # 比较两份结果
"""
import argparse
import json
import math
import os
import sys
import urllib.request

PORT = int(os.getenv("PORT", "8000"))
MODEL = os.getenv("MODEL_PATH", os.path.expanduser("~/models/Qwen3.5-397B-A17B-FP8"))
OUT = "/tmp/logprob_{tag}.json"

# 固定文本(确定性);多段不同体裁,够长以让 PPL 有意义。
PROMPTS = [
    "The transformer architecture relies on self-attention to model dependencies "
    "between tokens regardless of their distance in the sequence. Each layer "
    "refines the representation by mixing information across positions, and the "
    "feed-forward sublayer applies a position-wise nonlinearity. Residual "
    "connections and layer normalization keep the optimization stable as depth "
    "grows into the dozens of layers used by modern large language models.",
    "Mixture-of-experts models route each token to a small subset of experts, so "
    "the number of parameters can grow enormously while the compute per token "
    "stays bounded. A gating network produces a distribution over experts and the "
    "top-k are selected; their outputs are combined by a weighted sum. The main "
    "engineering challenge is balancing the load across experts and hiding the "
    "communication cost of dispatching tokens to the right devices.",
    "在大规模语言模型的推理中,首个 token 的延迟往往由两部分主导:一是注意力对长序列"
    "的平方级开销,二是稀疏专家层在张量并行下的通信与访存。优化的关键在于辨别每个算子"
    "到底是受限于计算、带宽,还是延迟,然后据此选择对应的手段,而不是盲目地堆叠技巧。",
    "Quantization reduces the memory footprint and bandwidth pressure of neural "
    "networks by representing weights and activations with fewer bits. Block-wise "
    "FP8 keeps a separate scale for each tile, which preserves accuracy far better "
    "than a single global scale while remaining cheap to apply. The trade-off is a "
    "small, bounded numerical error that, in a well-designed pipeline, stays well "
    "below the noise floor of the task being solved.",
]


def post(path, payload):
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def collect(tag):
    all_lp = []
    for i, p in enumerate(PROMPTS):
        resp = post("/v1/completions", {
            "model": MODEL, "prompt": p,
            "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0,   # 只返回 prompt 各 token 自身的 logprob
        })
        plp = resp["choices"][0]["prompt_logprobs"]
        lps = []
        for pos in plp:
            if pos is None:        # 第一个 token 无条件分布
                continue
            # prompt_logprobs=0 时每个位置只含实际 token 一项
            entry = next(iter(pos.values()))
            lps.append(entry["logprob"])
        all_lp.append(lps)
        print(f"  prompt[{i}]: {len(lps)} tokens, "
              f"mean logprob={sum(lps)/len(lps):.4f}")
    with open(OUT.format(tag=tag), "w") as f:
        json.dump(all_lp, f)
    flat = [x for lps in all_lp for x in lps]
    ppl = math.exp(-sum(flat) / len(flat))
    print(f"[{tag}] total {len(flat)} tokens, PPL={ppl:.4f} -> {OUT.format(tag=tag)}")


def compare():
    a = json.load(open(OUT.format(tag="off")))
    b = json.load(open(OUT.format(tag="on")))
    fa = [x for lps in a for x in lps]
    fb = [x for lps in b for x in lps]
    assert len(fa) == len(fb), f"token count mismatch {len(fa)} vs {len(fb)}"
    ppl_a = math.exp(-sum(fa) / len(fa))
    ppl_b = math.exp(-sum(fb) / len(fb))
    diffs = [abs(x - y) for x, y in zip(fa, fb)]
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    print(f"tokens compared      : {len(fa)}")
    print(f"PPL  off (stock)     : {ppl_a:.5f}")
    print(f"PPL  on  (fused)     : {ppl_b:.5f}")
    print(f"PPL  delta           : {ppl_b - ppl_a:+.5f}  "
          f"({100 * (ppl_b - ppl_a) / ppl_a:+.3f}%)")
    print(f"per-token logprob |diff|  max={max_d:.3e}  mean={mean_d:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--tag", required=True)
    sub.add_parser("compare")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.tag)
    else:
        compare()
