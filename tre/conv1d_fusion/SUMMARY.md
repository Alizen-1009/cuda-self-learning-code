# causal_conv1d (Qwen3.5 GDN) B300 优化 — 收尾总结

**任务**:优化 vLLM `_causal_conv1d_fwd_kernel`(Qwen3.5-plus GDN 的 causal depthwise conv1d + SiLU)。
**生产 shape**:dim=12288, width=4, bf16, 单序列, channel-contiguous(x=(dim,T), dim 轴 stride=1)。
**状态**:✅ 收官(不做融合)。standalone + in-model 均已验证并回退干净。

## 最终结果

| 阶段 | 时间/call | SOL | vs stock |
|---|---|---|---|
| stock(Triton,含 SiLU-tanh)| 121µs standalone / 126.4µs in-model | 40% | — |
| **conv_fast** standalone | **74.7µs** | **67.4%** | 1.6× |
| **conv_fast** in-model(真实 serve+trace)| **70.5µs** | — | **1.79×** |

正确性:standalone rel_l2 2.4e-3;in-model 对 stock out rel_l2 2.9e-3、conv_state rel_l2 0(精确)。

## 结论

- **1.79× 端到端加速已在真实 Qwen3.5-plus 网络里实锤**(45 个 GDN 层 call,prefill T=8192)。
- conv_fast 的 67% SOL 已是**本卡该 shape 物理极限的 ~91%**:实测纯 torch copy 上限就只有
  ~73% SOL(5.85 TB/s,非标称 8),80% 物理不可达。standalone 到顶,无进一步空间。
- 破 44%→67% 墙的根因是 **occupancy**:小 block + `__launch_bounds__` + **向量化 float2 load AND store**
  + bf16x2 `__hfma2` + cp.async ring + tanh.approx SiLU。**不需要 CuTe/TMA**(TMA read 4.35 TB/s 反而更慢)。
- 更大收益只剩融合(conv⊕split⊕l2norm,理论 ~2×)——**本次明确不做**。

## 目录

```
SUMMARY.md              <- 本文件(唯一入口)
handoff_result/         <- standalone 最优交付
  conv_fast.py            最快 kernel(70.5µs/67% SOL)★ 核心产物
  conv_cute.py            CuTe TMA 版(44%,死锁已修,备查)
  README.md               standalone 优化全过程 + 为何 80% 不可达
integration/            <- vLLM 集成 + in-model 验证 ★
  conv_fast_ext.py        证过的 kernel + 窄 guard 的 try_fast_prefill(否则落回 stock)
  conv_fast_patch.py      env CONV1D_FAST=1 gate 的 apply/revert
  test_integration.py     stock vs fast 正确性门(out + conv_state)
  results/profiler_*.txt  fast/stock 两份 trace 汇总(1.79× 证据)
  README.md               集成 how-to + 复现步骤
candidates/ reference.py test_kernel.py input.py shapes.json  <- 正确性/bench harness
experiments/            <- 所有走过的死胡同(cp.async/纯 CUDA/融合/widen/TMA/timing sweep 等),备查
  old_integration/        早期集成尝试
  handoff_early/          早期 handoff 快照(已被 handoff_result/ + integration/ 取代)
```

## 落地/环境注记

- 已交付净收益进 pod chuanwu vllm(site-packages,可回退):**SiLU tanh.approx**(~10%,备份 `.siluorig`)。
  conv_fast 集成为 **env-gated 试验**,已 `conv_fast_patch.py revert` 干净回退(仅移除 gate,SiLU 保留)。
- pod 现状:serve 已停、GPU 释放、patch 已回退、无残留 gate 行。
- 布局坑:atrex-bench harness 是 token-contiguous,生产是 channel-contiguous,务必在 channel-contiguous
  上测(详见 memory `causal-conv1d-layout-pitfall`)。
