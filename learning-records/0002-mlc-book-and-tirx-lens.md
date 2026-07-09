# 新增知识源：MLC《Modern GPU Programming for MLSys》与 TIRx 第二透镜

加入 `repos/modern-gpu-programming-for-mlsys`（MLC-AI / CMU MLSys 课程配套书）作为 **Blackwell 阶段的知识主脊与图解来源**，并确立 **TIRx 为第二条对照透镜**（与 Triton 并列）。补上原 [[RESOURCES.md]] 标注的 Blackwell 空缺。

## 为什么加它
- 全书以 **Blackwell 为第一目标**，成体系讲 `tcgen05` / TMEM / TMA / mbarrier / CLC / 2-CTA cluster / block-scaled FP4·FP8 / Flash Attention 4——正是路径阶段 3~4 覆盖最薄的部分。
- 配套高质量 SVG 图解（`img/` 下 tcgen05_ldst、tmem_grid、mma_cg1/cg2 shape、mma_block_scaled、smem_descriptor、swizzle_conflict），可在课上直接引用，省去自绘。
- 出自 Tianqi Chen / MLC 团队，信任度高，含 CMU MLSys 课程脉络。

## 它的边界（关键，防止误用）
- 该书用 **TIRx（TVM 系 Python IR-level DSL）** 写 kernel，**不是裸 CUDA/PTX/CUTLASS**。因此它是**概念心智模型 + 结构/图解参照**，**不替代**本 mission 的深度终点——PTX `mma`/`wgmma`/`tcgen05` 逐字段仍走 PTX ISA + CUTLASS。
- 定位为 **第二对照透镜**：Triton 与 TIRx 都是"IR/DSL 视角看同一硬件概念"，用来三角验证理解，而非主线实操路径。
- `zh/` 中文版目前多为 TODO 占位，**以英文正文为准**（约 5.4k 行）。

## Implications
- 进入阶段 3（Hopper）/ 4（Blackwell）时，先用本书建立异步引擎与 tensor core 的心智模型 + 图，再落到 PTX/CUTLASS 裸指令。
- 未来可设计一节"同一个 tile MMA，PTX vs TIRx vs Triton 三视角对照"的 interleaving 课，强化 storage strength。
- 待确认项（Triton 权重）现可扩展为 Triton/TIRx 两条透镜的相对权重，见 [[NOTES.md]]。
