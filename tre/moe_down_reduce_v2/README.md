# Fused MoE down-projection kernel — 交付说明

针对 FP8 block-quant MoE 的**下投影(w2)**做的一个专用 Triton 算子:把
**down-GEMM + topk 权重乘 + expert 规约(原 `ops.moe_sum`)** 三步融合进一个核,
并且不再物化 `[M, topk, N]` 的中间张量。通过环境变量开关启用,**默认关闭、不满足
前置条件自动回退原实现**,对其它路径零影响。

适用形状:fp8_w8a8 + 设了 `block_shape` 的 MoE(本部署 E=512, topk=10, 每卡 K=128,
N=4096)。其它配置一律走原 GEMM + `moe_sum`。

---

## 一、交付内容(3 个文件)

```
fused_moe_down_reduce.py    ← 新增算子(整文件,直接放进去)
fused_moe.py.patch          ← 对 fused_moe.py 的改动(调度分支)
configs/E=512,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
                            ← UP/DOWN 拆分后的配置(覆盖同名文件)
tests/_inputs.py            ← 测试用合成输入(固定形状,无需真实权重)
tests/test_accuracy.py      ← 精度门禁:融合核 vs 原路径(单卡,合成输入)
tests/bench_perf.py         ← 性能对比(triton do_bench):MoE 层 OFF vs ON(单卡)
tests/bench_logprob.py      ← 端到端兜底:真实模型 PPL/logprob,开关 off vs on
```

下文 `<FUSED_MOE>` phi-vllm 安装里的目录:
`.../vllm/model_executor/layers/fused_moe/`

---

## 二、安装步骤(3 步 + 1 个开关)

### 1. 放入新算子(新增,无冲突)
把 `fused_moe_down_reduce.py` 复制到 `<FUSED_MOE>/`。

### 2. 改 `fused_moe.py`(调度分支)

#### 2.1 自动安装
推荐用补丁(补丁以社区 vllm 原版 `fused_moe.py` 为基准):
```bash
cd <FUSED_MOE>
patch -p1 < /path/to/fused_moe.py.patch      # 或:git apply /path/to/fused_moe.py.patch
```

#### 2.2 手动安装
若 `fused_moe.py` 版本不同、补丁打不上,按下面**两处手改**(都在
`fused_experts_impl` 函数内):

**改动① — 函数开头附近(读开关,原 ~L2552):**
在拿到 `config` 之后、进入分块循环之前,加:
```python
import os as _os
_use_fused_reduce = bool(int(_os.getenv("VLLM_MOE_USE_FUSED_REDUCE", "0")))
```

**改动② — 下投影那一段(原 ~L2867):**
找到下投影现有的这段(`invoke_quant_moe_kernel(... config.get("DOWN", config) ...)`
紧跟着 `intermediate_cache3.to(...)` 和 `ops.moe_sum(...)`),把它整段包成
`if 走新核 / else 走原逻辑`:
```python
        _can_fuse_down_reduce = (
            _use_fused_reduce
            and use_fp8_w8a8
            and expert_map is None
            and not apply_router_weight_on_input
            and block_shape is not None
            and w2_bias is None
            and w2_zp is None
            and not per_channel_quant
        )
        if _can_fuse_down_reduce:
            from vllm.model_executor.layers.fused_moe.fused_moe_down_reduce import (
                fused_moe_down_reduce,
            )
            # BLOCK_SIZE_M 必须等于 moe_align_block_size 用的 BM,否则
            # expert_ids[pid_m] 越界。
            down_cfg = dict(config.get("DOWN", config))
            down_cfg["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]
            fused_moe_down_reduce(
                A=qintermediate_cache2,
                w2=w2,
                a_scale=a2q_scale,
                b_scale=w2_scale,
                topk_weights=curr_topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                top_k=top_k_num,
                num_tokens=tokens_in_chunk,
                config=down_cfg,
                block_shape=block_shape,
                output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                fp32_acc=False,
            )
        else:
            # ↓↓↓ 这里放原来的下投影逻辑(原封不动缩进进来):
            #   if expert_map is not None: intermediate_cache3.zero_()
            #   invoke_quant_moe_kernel(... config.get("DOWN", config) ...)
            #   intermediate_cache3 = intermediate_cache3.to(hidden_states.dtype)
            #   ops.moe_sum(intermediate_cache3..., out_hidden_states[...])
            ...
```
> 关键约束:`down_cfg["BLOCK_SIZE_M"]` 必须 = 顶层 `config["BLOCK_SIZE_M"]`
> (`moe_align_block_size` 用它给每个 expert 的 token 数做 padding 并据此索引
> `expert_ids`)。补丁里已处理,手改时务必保留这一行。

### 3. 覆盖配置 JSON
把 `configs/E=512,N=128,...H20...block_shape=[128,128].json` 复制到
`<FUSED_MOE>/configs/`,覆盖同名文件。
- 这一版把 UP/DOWN 拆开:**DOWN = `BLOCK_SIZE_K=128, num_stages=2, GROUP_SIZE_M=8`**
  (K=128 是单个 block-quant group,单次 K 迭代 + 双缓冲预取优于多迭代);UP 维持原值。
- 机制是已有的 `config.get("UP"/"DOWN", config)`:不拆分时两者回退同一份扁平配置,
  所以这个 JSON 改动对其它设备/形状无影响。

### 4. 打开开关
启动 vllm 前设环境变量:
```bash
export VLLM_MOE_USE_FUSED_REDUCE=1
```

---

## 三、开关与生效条件

- `VLLM_MOE_USE_FUSED_REDUCE`
  - `0`(默认)→ 完全走原实现,等于没装这个东西。
  - `1` → **且**满足全部前置条件时,下投影走新核;否则**自动回退**原 GEMM + `moe_sum`。
- 生效前置条件(任一不满足即回退,安全):
  fp8_w8a8 · 设了 `block_shape` · 无 `expert_map` · 非 router-weight-on-input ·
  无 w2 bias · 无 w2 zero-point · 非 per-channel 量化。

---

## 四、自带测试(性能 + 精度)

装完(放新核 + 打补丁 + 覆盖 JSON)后,在**一张 H20** 上即可跑。输入是合成随机
数据(固定部署形状 M=32768/E=512/topk=10/K=128/N=4096),不需要真实模型权重。

```bash
cd tests
CUDA_VISIBLE_DEVICES=0 python test_accuracy.py    # 精度门禁
CUDA_VISIBLE_DEVICES=0 python bench_perf.py       # 性能对比
```

**精度门禁(`test_accuracy.py`)** —— 单算子对比,主门禁:
- 同一份输入跑完整 `fused_experts` 两遍(开关 OFF=原路径 / ON=融合核),比规约后输出。
- 参照物 = **它要替换的原路径**(down GEMM + `moe_sum`),不是 fp32 真值——证「实现等价」。
- 融合核改了规约顺序 + 累加器(bf16 原子加替代 fp32 `moe_sum`),**预期**有 ~1e-2 级
  相对差异(这是预期,非 bug)。判据:`max_rel < 2e-2`(FP8 块量化自身误差量级)。
- 退出码 0=PASS / 1=FAIL,可进 CI。本机实测:`max_rel 9.6e-3 / mean_rel 4.0e-3` → PASS。

**性能对比(`bench_perf.py`)** —— triton `do_bench`:
- 真实替换是「两个核(down GEMM + moe_sum)→ 一个融合核」,故在 `fused_experts` 这层对比
  OFF/ON,delta 正好隔离 down+reduce 的收益(up GEMM/激活不变),量化不计入计时。
- `do_bench(warmup=25, rep=100, return_mode="median")`,rep 间默认冲 L2。
- 本机实测:OFF 8.71 ms → ON 5.90 ms,**delta 2.81 ms(占整层 −32%)**。

**端到端兜底(`bench_logprob.py`)** —— 真实模型,PPL/逐 token logprob:
- 单算子门禁只保证「核正确」,不保证「1e-2 误差跨 60 层累积后不伤质量」。这一层用
  **PPL / 逐 token logprob 对比(同模型、同输入,开关 off vs on)**——任务准确率
  (MMLU/GSM8K)对 1e-2 扰动不敏感、噪声大,容易给虚假安心,故不用。
- 需真实模型 + 多卡。两次分别在 `VLLM_MOE_USE_FUSED_REDUCE=0/1` 下起服务:
  ```bash
  # 起 off 服务后:  python bench_logprob.py collect --tag off
  # 起 on  服务后:  python bench_logprob.py collect --tag on
  python bench_logprob.py compare
  ```
- 用 `/v1/completions` 的 `prompt_logprobs`(teacher-forced,确定性、无采样),
  off/on 同文本 → 同 token 序列 → 逐位置精确比。
- **本机实测(8×H20,真实 Qwen3.5-397B,297 tokens):**
  PPL 8.6953(off)→ 8.6397(on),**delta −0.64%**;逐 token |diff| mean 0.086 / max 1.43。
- **关键对照**:融合核用 bf16 原子加,跨 CTA 规约顺序不确定 → **同开关重跑也有抖动**。
  实测 on-vs-on 噪声地板:PPL 摆动 0.6%、|diff| mean 0.069 / max 0.51。off-vs-on 的
  差异与此**同量级** → 换核**无系统性质量变化**,差异即原子加的运行间噪声。
  > 副作用:融合核因此**不是逐位 bit 可复现**(质量无碍;若要严格 bitwise 复现,
  > 用 `VLLM_MOE_USE_FUSED_REDUCE=0` 即可)。

## 五、回滚

- 最快:`export VLLM_MOE_USE_FUSED_REDUCE=0`(或 unset)——无需改任何代码,立即回到原路径。
- 彻底:还原 `fused_moe.py`(反向打补丁 `patch -R -p1 < fused_moe.py.patch`)、删掉
  `fused_moe_down_reduce.py`、还原 JSON。
