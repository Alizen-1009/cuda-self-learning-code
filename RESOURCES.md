# Tensor Core / CUTLASS / MoE Kernel Resources

> 已 clone 的本地仓库是主力知识源（成体系、中文、贴合 MoE 主线）。外部主要作 primary source（权威、可引用）。

## Knowledge — 本地仓库（已 clone）

- **repos/how-to-optim-algorithm-in-cuda/** — 本课程的**知识主脊**。
  - `cutlass/cute/` (杨远航、reed 的 CuTe 笔记)：Use for — Layout 代数、TiledMMA/Copy 抽象入门。
  - `cutlass/instructions/cuda的ldmatrix指令的详细解释.md`：Use for — Lesson 1~2，理解 fragment 加载。
  - `cutlass/wgmma/`、`cutlass/tma/`：Use for — Hopper 阶段（wgmma + TMA）。
  - `cutlass/gemm/` (TRT-LLM Mixed/Quantization GEMM 讲解)：Use for — MoE 量化 GEMM 主线。
  - `large-language-model/moe/` (DeepEP、moe_align_block_size、EP 优化)：Use for — 毕业作品阶段。
  - `ptx-isa/ptx_isa_8.5.pdf`：Use for — PTX 逐字段查证。
- **repos/how-to-optim-algorithm-in-cuda/cuda-mode/lectures/** — GPU MODE 系统课中文笔记。
  Use for — L8 Performance Checklist、L1 profiling、L12 Flash Attention、L15 CUTLASS、L14/L29 Triton。
- **repos/LeetCUDA/kernels/** — 可读可改的实战 kernel（hgemm 各版本、flash-attn、swizzle、ws-hgemm、openai-triton）。Use for — 每阶段的 hands-on 读改练习。
- **repos/CUDA_Kernel_Samples/sgemm/** — 干净的 SGEMM 逐步优化样例。Use for — GEMM tiling/pipelining 打地基。
- **repos/accelerated-computing-hub/tutorials/** — NVIDIA 官方教程（cuda-cpp、cuda-tile）。Use for — 官方权威对照。
- **Cuda-Tutorials/**、**LeetGPU/**、**interview_code/** — 从易到难的练手题与面试题。Use for — 检索式复习、interleaving 练习。

## Knowledge — 外部 primary sources（权威）

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 执行模型/内存层次的权威定义。
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — `mma`/`wgmma`/`tcgen05`/TMA 指令的**唯一权威**语义来源。逐字段查这里。
- [CUTLASS GitHub + media/docs](https://github.com/NVIDIA/cutlass) — CuTe、TiledMMA、Blackwell 示例的源头。
- [Colfax Research — CUTLASS/GPU tutorials](https://research.colfax-intl.com/blog/) — TMA、WGMMA、Hopper/Blackwell GEMM 的最佳深度英文教程（本地笔记多为其翻译）。
- [Lei Mao's Blog](https://leimao.github.io/) — CUDA 概念（内存、profiling、tensor core）清晰讲解。
- [NVIDIA Nsight Compute (ncu) 文档](https://docs.nvidia.com/nsight-compute/) — 指令级 profiling 与 roofline。

## Wisdom (Communities)

- [GPU MODE Discord](https://discord.gg/gpumode)（原 CUDA MODE）— 高信号，kernel 优化圈核心社区，有 leaderboard 打榜。Use for — 提交 kernel 求 review、追新架构讨论。
- [r/CUDA](https://reddit.com/r/CUDA) — Use for — 概念澄清、报错排查。
- [NVIDIA Developer Forums — CUDA / CUTLASS](https://forums.developer.nvidia.com/) — Use for — 官方工程师会答的深水区问题。
- CUTLASS GitHub Issues/Discussions — Use for — Blackwell tcgen05 等新特性的一手答疑。

## Gaps（待补）
- Blackwell `tcgen05`/TMEM/FP4 的成体系中文教程尚少；本地仓库覆盖薄，需靠 PTX ISA + CUTLASS Blackwell 示例 + Colfax 新文补齐。
- 真实项目所用框架未定，暂无法锁定最贴合的参考 kernel（见 NOTES.md 待确认）。
