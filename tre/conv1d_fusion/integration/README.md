# conv_fast → vLLM 集成 + in-model 验证

把 standalone 最优 `conv_fast`(handoff_result/conv_fast.py,70.5µs/67% SOL)接进真实
Qwen3.5-plus serve,用 torch profiler trace 确认端到端加速成立。

## 结果(同会话 apples-to-apples,prefill T=8192 单序列,45 个 GDN 层 call)

| conv kernel | per-call | |
|---|---|---|
| stock `_causal_conv1d_fwd_kernel`(Triton,含 SiLU-tanh) | **126.4µs** | baseline |
| `conv_fwd<4,128,64,6,8>`(conv_fast,`CONV1D_FAST=1`) | **70.5µs** | **1.79×** |

正确性:`test_integration.py` 对 stock 校验 out rel_l2=2.9e-3、conv_state rel_l2=0(精确)。
证据:`results/profiler_{fast,stock}.txt`(trace.json.gz 留在 pod /tmp/trace_{fast,stock}/)。

## 文件

- `conv_fast_ext.py` — 证过的 CUDA kernel(load_inline)+ `try_fast_prefill()` 窄 guard 包装。
  guard 通过条件:bf16、channel-last(`x.stride(0)==1`)、width==4、activation silu、
  单序列(`query_start_loc==[0,T]`)、fresh prefill(`has_initial_state` 全 False)。
  否则 return None → 调用方落回 stock。附带 conv_states 缓存写回(尾 width-1 个输入 token)。
- `conv_fast_patch.py` — `apply` / `revert` / `status`。env `CONV1D_FAST=1` gate 插进
  `causal_conv1d_fn` 开头,备份 `.convfastorig`(=SiLU-patched stock)。幂等。
- `test_integration.py` — stock vs fast 正确性门(out + conv_state)。跑前 `CONV1D_FAST` 不设。

## 复现

```bash
# 1. 装 gate（在 pod chuanwu env）
python conv_fast_patch.py apply
# 2. 正确性门（GPU 空卡）
CUDA_VISIBLE_DEVICES=<free> python test_integration.py      # 期望 RESULT: PASS
# 3. serve + trace（fast）
CONV1D_FAST=1 CUDA_VISIBLE_DEVICES=<free> PORT=8000 TRACE_DIR=/tmp/trace_fast \
  bash .../run_qwen35plus_pai_profile.sh prefill            # 后台 setsid,轮询 ready
BASE=http://localhost:8000 TRACE_DIR=/tmp/trace_fast python .../profile_run.py prefill
grep conv_fwd /tmp/trace_fast/profiler_out_0.txt
# 4. baseline：不设 CONV1D_FAST 重跑，grep _causal_conv1d_fwd_kernel
# 5. 收尾
python conv_fast_patch.py revert                            # 干净回退（SiLU-tanh 保留）
```

结论:standalone 1.6× 在真实网络里成立(实测 1.79×,in-model stock 有额外开销)。
conv_fast 已是该 shape 物理极限(~91% copy 上限)。更大收益只剩融合(conv⊕split⊕l2norm)。
