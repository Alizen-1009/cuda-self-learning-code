# causal_conv1d SM100/B300 优化 — Handoff

目标:把 vLLM `_causal_conv1d_fwd_kernel`(Qwen3.5 GDN 的 causal depthwise conv1d + SiLU)
在 **standalone 单算子**上冲到 ~80% SOL / 耗时减半(~2×,mentor 做到过)。

生产 shape:dim=12288, width=4, bf16, 单序列, **channel-contiguous**(x=(dim,T), dim 轴 stride=1)。
SOL = 2·dim·T·2B / 8TB/s = 50µs（T=8192）。B300 HBM 8.0TB/s。

## 现状（最优 = `conv_fast.py`,67% SOL,1.5×）

| 版本 | 文件 | 状态 | 数字 |
|---|---|---|---|
| **最快(occ+向量化+bf16x2)** | `conv_fast.py` | ✅ **正确,全 9 shape 过 harness** | **74.75µs / 67% SOL** ← 最优 |
| SiLU tanh.approx（已落地净收益）| `silu_patch.py` | ✅ 已打进 pod vllm | 136.7→125.7µs (~10%) |
| CUDA cp.async | `conv_async.py` | ✅ 正确+可跑 | 114.8µs / 44% SOL |
| CuTe DSL + TMA | `conv_cute.py` | ✅ 死锁已修好,正确+可跑 | 113µs / 44% SOL |

**44% → 67% 怎么破的(全程 ncu profile 驱动):** 两个 44% 基线(cp.async、TMA)其实都**占用率被卡死**
(~17-20%,寄存器/SMEM 限制)→ 在飞内存请求太少 → DRAM 只有 ~42%。修法(见 `conv_fast.py` 头注释):
①小 block(128)+ `__launch_bounds__(128,MINB)` 强制多 resident block 提占用率;②**向量化 float2 load+store**
(之前的高占用率尝试用了标量 store = LSU 受限);③cp.async ring(DEPTH=6)提供 MLP;④**bf16x2 `__hfma2`**
减半 conv FMA、消除 bf16↔f32 转换、减半寄存器 → 再提占用率。rel_l2~2.4e-3。

**为什么到不了 80%(定量证据,别再试):** 在本卡上,**同结构的纯 streaming copy 只有 72%、torch copy 73%**
(68.7µs = 5.85 TB/s = 8 TB/s 标称的 73%)——这就是本卡 bf16 该访存模式的**可达 HBM 带宽上限**,与 kernel 无关。
`conv_fast.py` 的 67% 已是该 copy 上限的 **91%**;把 SiLU 换成恒等只升到 69%(SiLU 仅占 ~2%)。
**80% 高于本卡纯 copy 的可达带宽(73%),对该 shape 物理上不可达。** README 顶部"80%/2×"的目标应理解为
"~73%/接近 copy 上限";`conv_fast.py` 的 67%(1.5× vs 44% 基线、1.6× vs stock 121µs)已接近该硬件极限。

## CuTe TMA 死锁——已定位并修复(2026-07-15,更正原诊断)

**原 README 的诊断是错的。** 之前判定"TMA copy/descriptor 没传输"——不对。做了一个最小
2D copy 隔离测试(`/tmp/tma_diag2.py`:TMA load 一个 tile 后延时再读 SMEM),**SMEM 里就是
x 的准确值,毒值全被覆盖 → TMA 描述符/分区完全正确,数据传输完美。**

**真正根因:warp 结构 bug。** 原版单 warpgroup 里 **TMA 生产者线程(tidx0)和消费者线程同处
warp 0**(tidx0 load、所有 128 线程 compute+wait)。非生产者线程对同一个 barrier 做阻塞
`mbarrier_wait` 会和 TMA 完成信号死锁。手写 mbarrier 和 `PipelineTmaAsync` **一模一样死锁**——
正因为都是这个结构问题,才让人误以为"barrier 不是问题"。而 **`cute.printf` 在 hang 的 kernel 里
永不 flush**(device printf 只在 kernel 结束/sync 时刷出),所以 `LOAD_DONE` 从没打印,看着
像 TMA 没跑。逐步隔离证明:①tidx0 单独 wait 能过;②tidx1(非生产者)单独 wait 就 hang;
③所有线程一起 wait 也 hang。

**修法:warp 专用化**(cuLA/fwd_o.py 的结构)。见 `conv_cute.py` 文件头注释:
- **独立的 load warp**(warp 4)发 TMA;**独立的消费者 warp**(0-3,128 线程)compute。生产者 warp ≠ 消费者 warp。
- load warp 必须**整个 warp**执行 `acquire_and_advance`+`cute.copy`(cute.copy 自动 elect 发射线程);
  **不能**套 `if tidx==elect`——那会让 load warp 内部 divergence,在后续 block barrier 上又死锁。
- 多级 TMA load 流水(load warp 预取 tile n+K)隐藏读延迟。

**现状:** 正确(rel~1.7e-3),113µs / **44% SOL**——追平 cp.async。

**关于 80% 目标——实测证据(结论:TMA 不是这条路,80% 在本卡多半不现实):**
逐项隔离(去 exp / 去 store / 纯 read):
- 纯 read(TMA load 流水)= 46µs;+标量 store(去 exp)= 91µs(**读写串行,不重叠**);+SiLU = 113µs(exp 暴露 +22µs)。
- **TMA G2S read 硬顶 ~4.35 TB/s**——与 TILE_T(64~256)、stages(2~4)、BLOCK_C(128/256/512)、占用率**全无关**。
- **torch 向量化 copy 同一张量 = 68.7µs = 5.85 TB/s = 73% SOL**(读写重叠)。这才是本卡实际内存上限。
- ⇒ **TMA read(4.35)比向量化 copy(5.85)还慢**。"用 TMA 打满带宽冲 80%"在本卡**不成立**;TMA 对这个带宽瓶颈的流式 conv 是错的工具。
- 试过独立 store warp / 5-warp 自发异步 TMA-store,都因 sO SMEM staging 把占用率腰斩而**退化到 12~30%**(ncu:占用率仅 19.6%,受 SMEM 限制)。

**80% 高于 torch copy 自己的 73%,基本不现实。** 现实最优 ≈ 73%(~68µs)。**若继续冲:放弃 TMA**,
去优化向量化 cp.async 路(`conv_async.py`,44%)——目标是读写重叠 + 高占用率 + 隐藏 SiLU(tanh.approx),
**不是**加大 SMEM tile。ncu 已确认占用率是主限制。

原始死锁版备份在 `conv_cute.py.bak_deadlock`;实验脚本在 `/tmp/tma_*.py`、`/tmp/conv_ws*.py`。

## 怎么跑

环境:pod chuanwu conda（`/root/miniconda3/envs/chuanwu/bin/python`）,cutlass-dsl **4.5.2**,
CUDA_VISIBLE_DEVICES 挑空卡（别用 GPU3,常有别人的 serve）。

```bash
# CuTe（WIP,会 hang——加了 LOAD_DONE/STORE_DONE printf 定位）
CUDA_VISIBLE_DEVICES=<free> python conv_cute.py 8192
# cp.async（能跑,44%）
CUDA_VISIBLE_DEVICES=<free> python conv_async.py 8192
# 正确性 harness（对 PyTorch reference,全 9 shape）— 需 vllm
CUDA_VISIBLE_DEVICES=<free> python test_kernel.py --candidate candidates/v0_vllm.py --no-bench
```

正确性判据:rel_l2 vs `reference.py`（PyTorch silu(conv)）< 1e-2。conv_cute/conv_async 的 `__main__`
自带对 F.conv1d+silu 参考的 rel 检查 + do_bench + pct_SOL。

## 布局坑（务必记住）
atrex-bench harness 的 `input.py` 造的 x 是 **token-contiguous**;vLLM 生产是 **channel-contiguous**
（`qwen3_next.py` 的 `mixed_qkv.transpose(0,1)`）。两者相反。**一定在 channel-contiguous 上测**
（conv_cute/conv_async 已用 `(T,dim).transpose(0,1)` 复刻生产布局）。

## SiLU 补丁（已在 pod,可回退）
`silu_patch.py`:把 vllm causal_conv1d.py 的 SiLU `acc/(1+exp(-acc))` 换成
`0.5*acc*(1+tanh.approx.f32(0.5*acc))`（inline PTX）。备份 `.siluorig`。`python silu_patch.py revert` 回退。
