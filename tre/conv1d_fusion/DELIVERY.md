# causal_conv1d (Qwen3.5 GDN) B300 优化 — 交付文档

**范围**:Qwen3.5-plus GDN prefill 的 causal depthwise conv1d(width-4)+ SiLU,及其下游 split/rearrange 融合。
**硬件**:B300(L20D,148 SM,HBM 标称 8 TB/s,实测该访存模式 copy 上限 ~5.85 TB/s)。
**生产 shape**:dim=12288(q 2048 | k 2048 | v 8192),num_k_heads=16,num_v_heads=64,head_dim=128,width=4,bf16,**channel-contiguous**(x=(dim,T),dim 轴 stride=1),单序列 prefill。

---

## 1. 成果总览

| 优化 | 内容 | 结果 |
|---|---|---|
| **SiLU tanh.approx** | conv 内 SiLU 用 `tanh.approx.f32` 恒等式替 exp/除法 | ~10%(136.7→125.7µs in-model),精度无损,**已落 pod 生产 vllm** |
| **conv_fast**(单算子)| 重写 conv+SiLU,占用率+向量化 load/store+bf16x2+cp.async | **74.7µs / 67% SOL**,standalone 1.6×,**in-model 1.79×**(126.4→70.5µs/call) |
| **conv_split**(融合)| conv+SiLU+split+rearrange 一个 read-once kernel | 前处理 conv→q/k/v 从 198.7µs → **~100µs(~2×)**,in-model 验证、无 bubble |

物理上限说明:该 shape 纯 copy 上限 ~73% SOL(68.7µs),conv_fast 的 67% 已是 copy 上限的 91%;80% 物理不可达。

---

## 2. atrex 修改(CR 已提)

**分支**:`conv1d_fwd_sm100`(起于 `origin/master`,author `chuanwu <zhuangwu.zyh@alibaba-inc.com>`)
**CR**:https://code.alibaba-inc.com/alibaba/atrex/codereview/new?from=master&to=conv1d_fwd_sm100

新增一个一等 CUDA op `causal_conv1d_fwd`(= conv_fast kernel),照 `nvfp4_rmsnorm_quant` 模板:

| 文件 | 内容 |
|---|---|
| `src/cuda/causal_conv1d_fwd/causal_conv1d_fwd.cu` | `__global__ conv_fwd<...>` kernel + host launcher。float2 向量化 load/store、bf16x2 `__hfma2`、cp.async ring(DEPTH=6)、`__launch_bounds__(128,8)` 提占用、`tanh.approx.f32` SiLU |
| `src/cuda/causal_conv1d_fwd/causal_conv1d_fwd_pybind.cu` | pybind11 绑定 `_causal_conv1d_fwd_kernel → causal_conv1d_fwd` |
| `src/cuda/causal_conv1d_fwd/include/causal_conv1d_fwd.h` | launcher 声明 |
| `python/atrex/api/causal_conv1d_fwd.py` | `@cuda_kernel()` stub + 公开 `causal_conv1d_fwd(x,weight,bias,initial_state=None)` + `can_use_causal_conv1d_fwd()` 守卫(SM100/bf16/width4/channel-last)|
| `python/atrex/core/optCompilerConfig.json` | 加 `_causal_conv1d_fwd_kernel` build 条目(srcs + `--use_fast_math` + `-gencode=arch=compute_100a,code=sm_100a` + include)|
| `op_test/test_causal_conv1d_fwd.py` | 对 `F.silu(F.conv1d)` 参考的正确性 + bench(rel<1e-2)|

**构建/验证**:`@cuda_kernel` 首调 JIT 编译(`compile_cu`);已在 B300 chuanwu env 跑通:**rel_l2 2.44e-3,74.7µs / 67.4% SOL**。
**附带 CR**:`chore-drop-hardcoded-author` —— 删掉 atrex CLAUDE.md 里硬编码 commit author 那行。

---

## 3. vLLM(pai-vllm)修改

全部 **env-gated + 备份 + 可 revert**;除 SiLU tanh 外,验证后均已 revert 干净,生产未受影响。

### 3.1 SiLU tanh.approx(已落地,~10%)
- 改 `causal_conv1d.py` 的 SiLU:`acc/(1+exp(-acc))` → `0.5·acc·(1+tanh.approx.f32(0.5·acc))`(inline PTX)。fwd+update 两处,备份 `.siluorig`,`python silu_patch.py revert` 回退。

### 3.2 conv_fast(单算子 1.79×,env `CONV1D_FAST=1`)
件在 `_conv1d_opt/integration/`:
- `conv_fast_ext.py` — kernel(load_inline)+ `try_fast_prefill()` 窄 guard(单序列/fresh prefill/bf16/channel-last/width4,否则 return None 落回 stock)+ conv_states 写回。
- `conv_fast_patch.py` — 在 `causal_conv1d_fn` 开头插 `CONV1D_FAST` gate;`apply`/`revert`/`status`,备份 `.convfastorig`。
- `test_integration.py` — 对 stock 校验(out rel 2.9e-3、conv_state rel 0)。
- **in-model**:stock `_causal_conv1d_fwd_kernel` 126.4µs → `conv_fwd` 70.5µs/call = **1.79×**。

### 3.3 conv_split 融合(conv+split ~2×,env `CONV_SPLIT_FUSE=1`)
件在 `_conv1d_opt/integration2/`:
- `conv_split_ext.py` — conv_split kernel(conv+SiLU+split+rearrange,按通道范围路由写 q/k/v)+ `fused_conv_split(x_T,w,b,...,do_l2norm)` + `gate()`(**纯 shape 判断,无 GPU→CPU 同步**)+ `writeback()`(**`index_copy_`,无 `.item()` 同步**)。
- `conv_split_patch.py` — **3 处 patch** `qwen3_next.py`,`apply`/`revert`/`status`,备份 `.convsplitorig`:
  - **A(conv 调用处)**:gated+eligible 时**跳过** `causal_conv1d_fn`(保留 pre-conv mixed_qkv,做 conv_states 写回),置 `_cs_fuse`。
  - **B(split 处)**:`_cs_fuse` 时用 `conv_split(...,do_l2norm=False)` 替 `fused_conv_split_l2norm_rearrange`。
  - **C(gating 之后)**:`_cs_fuse` 时对 q/k 调 `l2norm_fwd`(**l2norm 保持在 gating 之后,算子顺序不变**)。
- `test_integration2.py` — 对 stock [causal_conv1d_fn + fused_conv_split_l2norm_rearrange] 校验(rel q/k 3.2e-3)。
- **in-model(flashinfer GDN backend)**:`conv_split` 79µs + l2norm_fwd 20.7µs = ~100µs,替 stock 的 conv 126 + split_l2norm 72.7 = 198.7µs = **~2×**,顺序对齐、**无 bubble**(0.3µs 间隙)。

**关键坑(已解决)**:per-layer 的 `.item()/.any()` 会触发 GPU→CPU 同步 → 每层 ~380µs bubble;必须用纯 shape 判断 + `index_copy_` 全 GPU 端写回。

---

## 4. 如何启用 / 复现

```bash
# atrex op(pod chuanwu env,@cuda_kernel 首调 JIT)
python op_test/test_causal_conv1d_fwd.py            # 正确性+bench

# vllm 单算子 conv_fast
cd _conv1d_opt/integration && python conv_fast_patch.py apply
CONV1D_FAST=1  <启动 serve>                          # gate 生效;python conv_fast_patch.py revert 回退

# vllm 融合 conv_split(flashinfer GDN)
cd _conv1d_opt/integration2 && python conv_split_patch.py apply
CONV_SPLIT_FUSE=1 VLLM_GDN_PREFILL_BACKEND=flashinfer  <启动 serve>
# 回退:python conv_split_patch.py revert
```
trace 在 `_conv1d_opt/traces/`(`.json.gz` 拖进 https://ui.perfetto.dev)。最终对比 trace:`trace_prefill_fused_flashinfer_20260717.json.gz`。

---

## 5. 范围 / 注意

- **仅覆盖 prefill 单序列 fresh 场景**(profile/生产主路径)。chunked-continuation 的 initial_state、decode(`causal_conv1d_update`)、多序列 varlen 均**落回 stock**(guard 拒绝)。
- **gating(a/b 支路)不在本次范围**:它是 elementwise 门控,吃 a/b(来自 `in_proj_ba`),和 conv/l2norm 数据不相干、融不进 conv;其天然融合对象是 `in_proj_ba` 的 GEMM epilogue。
- l2norm 保持**独立算子**且在 **gating 之后**(与生产/参考 trace 顺序一致),用 vllm 高效 `l2norm_fwd`(非 torch)。
- conv_split 更进一步的 conv⊕split⊕l2norm 全融(冲 copy floor 69µs)需 cutedsl/cutlass warpgroup-async,是独立立项(hand-CUDA 撞 per-token 归约墙 ~62µs)。

件与全部证据:sync 仓库 `MLFlow-B300-Sync/_conv1d_opt/`(本地 `/Users/alizen/B300_sync/`,B300 `/root/workspace/chuanwu/`)。
