# GDN prefill(chunk_gated_delta_rule)B300 SM100 优化 — 交付说明

针对 **Gated Delta Net(GDN,linear-attention/SSM)prefill** 算子在 **B300(SM100 /
sm_103,Blackwell Ultra)** 上的 CuteDSL kernel 优化。核基于 FlashInfer 0.6.9 的
`blackwell` GDN prefill,作为**维护 fork** 持续优化(见 `kernel/PROVENANCE.md`),
并在 atrex 启动器上加了一条 **exact chunk-parallel(CP)** 路径。

**生产 shape**:packed-varlen 单序列,head_dim=128,bf16,真实 Qwen3-Next / Qwen3.5-Plus
的 Hk16/Hv64(profile 用 S=8192/16384)。

---

## 一、成果总览

| 优化 | 内容 | 结果 |
|---|---|---|
| **核 fork + 本地优化** | V3 coalesced recurrent-state GMEM 传输 + v21 state-input R2T hoist | **112.8 TFLOPS / 2018 GB/s**(B300),与 FlashInfer-main 打平 exact 路径 |
| **exact-CP(Stage 1,已部署)** | 精确 K×K transition carry(3 pass)+ 3 路 auto dispatch | slow-decay 层 **~7% 端到端**(exact 580µs vs no-CP 627µs @ Hk16/Hv32);fast-decay 层走 scalar-CP |
| **fused MN-precompute(Stage 3)** | 把 exact-CP 的 M-pass + N-pass 融进**一趟 dual-state scan**,3 pass → 2 pass | fused MN pass **303.7µs(1.53× single)**,2-pass 等效 502.8µs(vs 3-pass 597µs);M/N 对参考 **rel_l2 0.0** |
| **nosync D2H cache** | `cu_seqlens` 的 `.cpu()/.tolist()` 每层重复 → 一次 forward 内**缓存一次** | 消除 per-layer GPU→CPU 同步 bubble(45 个 GDN 层)|

### 头对头(B300,fast-decay,S=16384,详见仓库 `gdn_headtohead_bench.md`)

| kernel | Hk4/Hv32 µs/call | Hk16/Hv64 µs/call |
|---|---|---|
| **atrex(CP auto)** | **310.7(2.06× vs FI-main)** | **499.6** |
| FlashInfer-main | 639.9(baseline) | 644.4 |
| vLLM SM100 CuTe | 598.0 | 895.5 |
| FLA(Triton) | 916.3 | — |

**atrex 独有的 lead = SM100 上的 exact-CP 路径**(FlashInfer-main 的 CP 只有 SM90/SM120,
SM100 上 `use_cp=auto` 退回 no-CP)。exact 无 CP 路径 atrex 与 FI-main 打平(同源 fork)。

---

## 二、交付内容

```
kernel/                                   ← CuteDSL kernel 包(= atrex src/cutedsl/gdn_prefill_sm100/)
  gated_delta_net_chunked.py                主 kernel:chunked GDN + Stage3 fused-MN(mn_mode)
  gdn_prefill.py                            CuTe host adapter(compile-once-cache-replay + mn_mode 接线)
  gated_delta_net_tile_scheduler.py         persistent-CTA tile scheduler(upstream verbatim)
  __init__.py                               guarded 导出 chunk_gated_delta_rule_sm100
  PROVENANCE.md                             fork 来源(FI 0.6.9)+ 本地改动清单 + gate 约定
api/                                      ← 启动器(= atrex python/atrex/api/)
  chunk_gdn_prefill_sm100.py                chunk-parallel 单趟 scan + fused Triton correction + 3 路 CP dispatch
  chunk_gdn_cutedsl.py                      CuteDSL 入口;含 [nosync] cu_seqlens D2H 缓存(消 per-layer 同步)
op_test/
  test_gated_delta_net.py                   正确性 + bench(对 FlashInfer / no-CP 参考)
```

> 目录结构对齐 atrex 仓库(`gdn-exact-cp-stage3` 分支),回落时可原位覆盖:
> `kernel/*` → `src/cutedsl/gdn_prefill_sm100/`,`api/*` → `python/atrex/api/`,
> `op_test/*` → `op_test/`。

---

## 三、关键结论(勿重新推导,代价高)

1. **CP dispatch 是核心杠杆。** 3 路 auto(`GDN_CP_EXACT=1`):fast-decay → scalar-CP;
   slow-decay(`num_sab_heads ≤ GDN_CP_EXACT_MAX_HEADS=32)→ exact-CP;否则 no-CP。
   exact-CP 对 no-CP **bit-exact**(value-split 是同一算术的重新分区)。
2. **瓶颈是串行关键路径延迟,不是 SM 数。** no-CP 核 ncu:~128 SM busy 但仅 ~38.9% 吞吐,
   occ=1;填满空闲 SM 只有 ~15% 天花板,真正的收益在每个 busy SM 内被串行 stall 吃掉的 ~60%。
3. **occ=2 不可达**(CG0 alone ≈87% of occ=2 register 预算);BV-split / register-carry 的价值
   在**关键路径延迟 + 去 spill**,不是 occupancy。
4. **Stage 3 fused-MN** 把 exact-CP 从 3 heavy pass 降到 2 pass:一趟 scan 同时 carry
   `S_part`(真 v → N_seg)与 `S_hom`(identity-init,v=0 → M_seg = transition)。目标
   slow-decay 层 **~1.5-1.8× GDN prefill**。

---

## 四、开关 / 复现

```bash
# 环境:B300 pod chuanwu conda env(system python 的 cutlass-dsl 4.5.2 会在
# tcgen05.make_tmem_copy 上 ICE;chuanwu 同版本可编)
export PYTHONPATH=<atrex>/python:<atrex>/src/cutedsl

# 正确性 + bench
python op_test/test_gated_delta_net.py

# CP dispatch
GDN_CP_EXACT=1 GDN_CP_DISPATCH_TRACE=1 <起 serve>     # 看每层 frac(alpha>0.99) → CP/exact/no-CP
GDN_CP_DISABLE=1 <...>                                # 强制 no-CP exact(bit-exact 基准)
```

**gate 约定**:核吃 **linear** 空间 forget gate(内部算 `cumsumlog = sum log(gate)`),
启动器直接传 linear gate,**不要**外面再包 `log()`(与旧 0.6.7 核相反)。

---

## 五、范围 / 注意

- **exact-CP 仅对 `num_sab_heads ≤ 32` 部署**(大头数即使 2-pass 也不划算)。
- Stage 3 fused-MN 是 all-or-nothing:失败模式**静默**(错数或死锁),改动后须逐步 bit-check
  (ncu 看 pipeline,cuda-gdb 看 hang)。保留 `mn_mode` on/off 两份 JIT 特化,off 路径需 byte-identical。
- **vLLM 独立 SM100 CuTe 核在小头(Hk4/Hv32)exact 路径快 ~7%**(598 vs 630µs),但大头
  scale 更差(895µs)。是 atrex no-CP fallback 的一个 kernel 级追赶目标。
- 源与全部证据:sync 仓库 `MLFlow-B300-Sync/`(`gdn_headtohead_bench.md`、`STAGE3_HANDOFF.md`、
  `gdn_stage3_mn_fusion_design.md`、`GDN_HANDOFF.md`);atrex 分支 `gdn-exact-cp-stage3`。
