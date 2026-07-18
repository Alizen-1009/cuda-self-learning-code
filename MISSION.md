# Mission: 精通 Tensor Core 指令与 CUTLASS，成为能与 AI Agent 平等协作的 GPU kernel 工程师

## Why
我在真实推理框架项目中和 AI Agent 一起优化 kernel，但目前只会写 CUDA core 算子，Agent 一谈 tensor core / Hopper / Blackwell 就变成黑盒。我要把底层吃透到**能读懂、能质疑、能指方向、能发现 Agent 说错的地方**——让优化过程不再是黑盒，而是我能完全参与、主导判断的协作。

## Success looks like
- 能读懂并逐字段解释一条 tensor core 指令的 PTX：`mma.sync`(Ada) / `wgmma`(Hopper) / `tcgen05` 即 TMMA(Blackwell)。
- 能读写 CUTLASS 3.x / CuTe 抽象（Layout 代数、TiledMMA、TiledCopy、TMA），看穿 Agent 生成的 CUTLASS kernel 在做什么。
- 对内存层次（global→L2→SMEM→register→TMEM）、线程/warp 编排、swizzle、pipelining 的权衡得心应手，能判断 Agent 的优化建议对不对。
- 能深度参与项目主力 **MoE 计算侧 kernel**：grouped/segmented GEMM、permute/unpermute、量化 GEMM（W4A16/W8A8），在 Hopper + Blackwell 上都能看穿。
- 能用 `ncu` 定位到指令级瓶颈，并读懂对应 SASS 的关键片段。

## Constraints
- 高强度投入：>15h/周，可排陡峭路径。
- 硬件齐全：H20(Hopper) + B200 / B300(Blackwell)，全部为数据中心真机开发环境。
- 强算法背景（ACM 获奖）、多段推理框架实习——概念抽象与工程能力强，无需从编程基础讲起。
- 起点：只写过 CUDA core 算子，**tensor core 零基础**，Hopper/Blackwell 特性不了解。

## Out of scope（当前）
- MoE **通信层**（NVSHMEM / IBGDA / all2all / DeepEP）——另一套网络/RDMA 技能，作为后期可选模块，不进主线。
- 训练侧 kernel、稀疏、图优化编译器内部——除非与主线交叉。
- 从零学 C++/CUDA 语法基础——已具备。
