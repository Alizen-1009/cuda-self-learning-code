# CUDA Tensor Core 自学工作区

> 面向 CUDA Core 到 Tensor Core / CUTLASS / MoE Kernel 的长期学习工作区：用课程、术语表、学习记录和参考子模块，把 GPU kernel 优化从“黑盒调参”推进到能读懂、能质疑、能指方向。

## 简介

这个仓库不是一个单一软件项目，而是一个围绕 GPU kernel 工程能力搭建的学习系统。目标是从熟练 CUDA Core 算子出发，逐步吃透 `mma.sync`、`wgmma`、Blackwell `tcgen05` / TMMA、CUTLASS 3.x / CuTe，以及 MoE 推理计算侧 kernel。

最终标准不是“能闭门手写一个 toy kernel”，而是能在真实推理框架项目里与 AI Agent 平等协作：看懂 Agent 生成的 CUDA / CUTLASS kernel，判断它的访存、线程组织、流水线和量化 GEMM 方案是否合理，并能用 `ncu` 与 PTX / SASS 证据定位问题。

## 快速开始

首次克隆时建议一并拉取子模块：

```bash
git clone --recurse-submodules git@github.com:Alizen-1009/cuda-self-learning-code.git
cd cuda-self-learning-code
```

如果已经克隆过父仓库，再初始化参考仓库：

```bash
git submodule update --init --recursive
```

当前课程和参考文档是静态 HTML，直接用浏览器打开即可：

```bash
open lessons/0001-cuda-core-to-tensor-core.html
open reference/glossary.html
```

> 注意：`.gitmodules` 使用 GitHub SSH URL。克隆者需要先配置 GitHub SSH key，或者把子模块 URL 改成 HTTPS。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `MISSION.md` | 学习使命、成功标准和边界 |
| `NOTES.md` | 学习者画像、教学偏好和待确认问题 |
| `STRUCTURE.md` | 更完整的目录导航 |
| `RESOURCES.md` | 高信任知识源清单 |
| `learning-records/` | 学习记录，用来记录掌握状态和下一步难度 |
| `lessons/` | 编号课程，当前以静态 HTML 为主 |
| `reference/` | 常读常新的参考材料，例如术语表 |
| `assets/` | 课程和参考页共享样式 |
| `repos/` | 8 个参考仓库，以 Git submodule 方式链接 |

## 学习主线

1. 性能心智模型：内存层次、roofline、profiling、occupancy。
2. Tensor Core 入门：`mma.sync`、fragment 布局、`ldmatrix`。
3. GEMM 优化与 CuTe：tiling、swizzle、`cp.async`、pipelining、TiledMMA / TiledCopy。
4. Hopper：`wgmma`、TMA、mbarrier、warp specialization。
5. Blackwell：`tcgen05` / TMMA、TMEM、FP8 / FP4 block-scaled MMA。
6. MoE 计算侧毕业作品：grouped / segmented GEMM、permute / unpermute、W4A16 / W8A8 / FP8 量化 GEMM。

Triton 与 TIRx 作为两条对照透镜穿插使用，用来快速建立 baseline、从 IR/DSL 视角三角验证并反推 CUDA / CUTLASS 设计。

## 参考子模块

| 子模块 | 用途 |
|---|---|
| `repos/how-to-optim-algorithm-in-cuda` | 中文体系化 CUDA / CUTLASS / MoE 笔记，是当前知识主脊 |
| `repos/modern-gpu-programming-for-mlsys` | MLC/CMU 书，Blackwell 心智模型 + 高质量图解（TIRx DSL，第二对照透镜；英文为准） |
| `repos/LeetCUDA` | 可读可改的实战 kernel，覆盖 HGEMM、FlashAttention、swizzle、Triton 等 |
| `repos/accelerated-computing-hub` | NVIDIA 官方教程，用作权威对照 |
| `repos/CUDA_Kernel_Samples` | 干净的逐步优化样例，适合打 GEMM / reduce / transpose 基础 |
| `repos/Cuda-Tutorials` | 编号 `.cu` 单文件练习 |
| `repos/LeetGPU` | 算法题式 GPU 练习，适合检索式复习 |
| `repos/interview_code` | 面试题与工程题素材 |

子模块会锁定到父仓库记录的具体提交。更新参考仓库后，如需让父仓库记录新版本，需要在父仓库中提交新的 submodule pointer。

## 当前进度

- 初始目标、边界和学习路径已记录在 `MISSION.md` 与 `learning-records/0001-initial-assessment-and-path.md`。
- 第 1 课已完成：`lessons/0001-cuda-core-to-tensor-core.html`。
- 术语表已建立：`reference/glossary.html`。
- 下一课方向：手写 `mma.sync` + `ldmatrix`，拆开 WMMA，图解 fragment 映射。

## 工作约定

- 每节课只锁定一个 tangible win，优先建立“能看懂 / 能质疑”的判断力。
- 新术语沉淀到 `reference/glossary.html`。
- 非平凡的掌握、纠误、目标变化写入 `learning-records/`。
- 参考仓库只通过 submodule 链接，不把外部仓库源码复制进父仓库历史。
