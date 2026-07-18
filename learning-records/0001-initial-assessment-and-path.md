# 初始评估：起点、目标与学习路径已确立

通过 7 轮访谈确立了学习者画像与 mission，为后续 ZPD 计算设基线。

## 已确立的先验知识
- 熟练 CUDA core 算子编写；强算法背景（ACM），多段推理框架实习。抽象/工程能力强，**无需从 CUDA/C++ 基础教起**。
- **Tensor core 零基础**——这是当前能力天花板，也是第一阶段的起点。
- Hopper/Blackwell 架构特性不了解。

## 已确立的目标（详见 [[MISSION.md]]）
- 终极效果不是"闭门手写算子"，而是"深度到能与 AI Agent 平等协作、看穿黑盒"——读懂/质疑/指方向/纠错。
- 深度终点：**PTX 指令层**（SASS 仅随 ncu 顺带）。
- 主力 kernel：**MoE 计算侧**（grouped GEMM / permute / 量化 GEMM），Hopper + Blackwell 双目标。
- 通信层（NVSHMEM/all2all/DeepEP）明确排除出主线。

## 规划的学习路径（阶段）
0. 性能心智模型 & 内存层次/roofline（补 Agent 常用词汇）→ H20
1. **Tensor Core 入门：`mma.sync` m16n8k16 + ldmatrix**（最大缺口，第一阶段起点）→ H20
2. GEMM 优化全景（tiling/cp.async/swizzle/pipelining）+ CuTe 入门 → H20
3. Hopper：`wgmma` + TMA + warp specialization → H20
4. Blackwell：`tcgen05`/TMMA + TMEM + FP4/FP8 → B200 / B300
5. 毕业作品：MoE 计算侧（grouped/量化 GEMM）→ H20 + B200 / B300
- Triton 作为对照透镜穿插各阶段。

## Implications
- 第一课锁定"从 CUDA core 到 tensor core 的心智跃迁"——理解一条 `mma.sync` 是 warp 协作指令、fragment 如何分布在 32 线程寄存器里。这是 wgmma/tcgen05/CUTLASS 的共同地基。
- 待确认项见 [[NOTES.md]]（真实项目框架、Triton 权重、精度顺序）。
