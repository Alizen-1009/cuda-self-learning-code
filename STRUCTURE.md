# 项目目录结构

> 本目录 = **教学工作区**（你的学习状态） + `repos/`（你 clone 的参考仓库）。
> 相关文档：[MISSION.md](./MISSION.md) · [RESOURCES.md](./RESOURCES.md) · [NOTES.md](./NOTES.md)

## 顶层总览

```
self-learing/
├── README.md                  # 项目首页：简介、快速开始、学习主线
├── CLAUDE.md                 # 你最初写给我的需求（原始记录）
├── MISSION.md                # 🎯 学习使命：为什么学 / 成功标准 / 边界
├── NOTES.md                  # 教学偏好 & 待确认项
├── RESOURCES.md              # 高信任知识源清单（本地仓库 + 权威外部 + 社区）
├── STRUCTURE.md              # 本文件：目录导航
├── learning-records/         # 学习记录（像 ADR，决定下一步教什么）
│   └── 0001-initial-assessment-and-path.md
├── reference/                # 📖 参考文档（常读常新的精华）
│   └── glossary.html         # 术语表（每课都遵循）
├── lessons/                  # 🎓 一节节的课（教学主体）
│   └── 0001-cuda-core-to-tensor-core.html
├── assets/                   # 各课共享组件
│   └── style.css             # 统一样式表
└── repos/                    # 📦 你 clone 的 8 个参考仓库（见下）
```

### 教学工作区各部分的分工

| 目录/文件 | 作用 | 生命周期 |
|---|---|---|
| `MISSION.md` | 指南针——决定教什么 | 目标变化时更新 |
| `learning-records/` | 记录你已掌握的，算 ZPD（下一步难度） | 只增，偶尔标记 superseded |
| `RESOURCES.md` | 弹药库——知识/社区来源 | 持续增删 |
| `lessons/` | 一次性教学，编号递增 | 学完很少回看 |
| `reference/` | 压缩后的精华，反复回看 | 长期沉淀 |
| `assets/` | 跨课复用的样式/组件 | 随课程增长 |

---

## `repos/` — 参考仓库

8 个仓库。按用途分三类：**知识主脊**、**动手素材**、**练习题库**。

### 🌟 知识主脊

```
repos/how-to-optim-algorithm-in-cuda/     # 成体系中文笔记，最贴合 MoE 主线
├── cutlass/          # cute / wgmma / tma / instructions / gemm / swizzle  ← 阶段2-5 核心
├── cuda-mode/        # GPU MODE 系统课讲义（lectures/）                    ← 全程
├── large-language-model/   # moe / flash-attention / sglang / trt-llm / vllm ← 主线
├── ptx-isa/          # PTX ISA 8.5 PDF + 笔记                             ← PTX 查证
├── cuda-kernels/     # 各类算子实现
├── triton/           # Triton 相关
├── papers/  ml-engineering/  pytorch/  tools/  deprecated/
```

```
repos/modern-gpu-programming-for-mlsys/    # MLC/CMU 书：Blackwell 心智模型 + 图解（TIRx DSL，概念透镜）
├── chapter_tensor_cores/ chapter_tmem/ chapter_tma/ chapter_async_barriers/ chapter_clc/  ← 阶段3-4 Blackwell 核心
├── chapter_data_layout/ chapter_layout_generations/ chapter_performance/   ← 阶段0/2 布局·swizzle·roofline
├── chapter_gemm_{basics,async,advanced}/  chapter_flash_attention/         ← 阶段3-5 SOTA GEMM/Attn 结构参照
├── img/              # tcgen05_ldst / tmem_grid / mma_cg*/ mma_block_scaled … 高质量 SVG（课上直接引用）
├── chapter_intro_tirx/ chapter_tirx_layout_api/ tirx_guide/   # TIRx 语言（第二对照透镜，非 PTX/CUTLASS）
└── zh/              # 中文版（目前多为 TODO 占位，以英文为准）
```

```
repos/LeetCUDA/                            # 可读可改的实战 kernel
├── kernels/          # hgemm / flash-attn / swizzle / ws-hgemm / openai-triton / softmax … ← 每阶段动手
├── HGEMM/            # 高性能 HGEMM 专项
├── ffpa-attn/        # FlashAttention 变体
├── docs/  slides/  others/  third-party/
```

### 🛠 动手素材 & 官方教程

```
repos/accelerated-computing-hub/           # NVIDIA 官方教程（权威对照）
├── tutorials/        # cuda-cpp / cuda-tile / warp / accelerated-python …
├── docs/  resources/  Accelerated_Python_User_Guide/  brev/  events/

repos/CUDA_Kernel_Samples/                 # 干净的逐步优化样例
├── sgemm/   reduce/   transpose/   gemv/   elementwise/   example/
```

### 🧩 练习题库（检索式复习 / interleaving）

```
repos/Cuda-Tutorials/     # 编号 .cu 单文件练习（从 vec-add 到 tensor core）
repos/LeetGPU/            # 算法题式 GPU 练手：Conv / Attention / Prefix Sum / Softmax …
repos/interview_code/     # 面试题：cuda / quant / cpp / python / cf
```

---

## 学习路径 ↔ 仓库对照

| 阶段 | 主要用到的仓库路径 |
|---|---|
| 0 性能心智模型 | `cuda-mode/lectures`(L8/L1) · `CUDA_Kernel_Samples/reduce` · `modern-gpu-.../chapter_performance`(roofline/overlap) |
| **1 Tensor Core 入门** | `cutlass/instructions`(ldmatrix) · `LeetCUDA/kernels/hgemm`(wmma) · `ptx-isa` |
| 2 GEMM + CuTe | `cutlass/cute` · `CUDA_Kernel_Samples/sgemm` · `LeetCUDA/kernels/{sgemm,swizzle}` · `modern-gpu-.../chapter_data_layout`(swizzle) |
| 3 Hopper | `cutlass/{wgmma,tma}` · `LeetCUDA/kernels/ws-hgemm` · `modern-gpu-.../chapter_{tma,async_barriers,gemm_async,gemm_advanced}` |
| 4 Blackwell | `ptx-isa` · CUTLASS Blackwell 示例（外部） · `modern-gpu-.../chapter_{tensor_cores,tmem,clc}`+`img/`(心智模型+图解) |
| 5 MoE 毕业作品 | `large-language-model/moe` · `cutlass/gemm`(量化 GEMM) · `modern-gpu-.../chapter_flash_attention`(FA4 结构) |
| Triton / TIRx 对照透镜 | `cuda-mode/lectures`(L14/L29) · `LeetCUDA/kernels/openai-triton` · `modern-gpu-.../chapter_intro_tirx`+`tirx_guide`(TIRx DSL) |
