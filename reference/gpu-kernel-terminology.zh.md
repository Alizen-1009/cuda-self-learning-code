# GPU Kernel 术语表

最后更新：2026-07-09

这份术语表服务于当前学习主线：Tensor Core 指令、CUTLASS/CuTe、Hopper/Blackwell GEMM、MoE 计算侧 kernel，以及基于 `ncu` 的性能诊断。

## 对原始表格的快速校正

你的原始表格方向是对的。需要特别记住的修正点：

- `WMMA` 不是 `WGMMA`：`WMMA` 是 CUDA C++ 的 warp 级 API；`WGMMA` 是 Hopper 的 warpgroup 级 PTX 指令。
- `UMMA` 不适合作为单独的“官方术语”使用。它更像社区/库里的简称，写文档时建议同时标注官方 PTX 名称 `tcgen05.mma`，或 CUTLASS 里的 SM100 MMA 类型名。
- `TMA` 从 Hopper 引入，但 Blackwell 仍然重要。不要把它理解成只属于 Hopper 的概念。
- `TMEM` 是 Blackwell 上服务 Tensor Core 累加器流的 Tensor Memory。不要和 SMEM、TMA、普通 shared memory 混成一个概念。
- `Stall` 要结合 profiler 的具体子类型看，不是单一根因。`memory dependency`、`barrier`、`not selected`、`pipe throttle` 对应的优化方向不同。

## 线程与执行模型

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| Thread | CUDA thread | CUDA 最小执行实例。 | 标量 CUDA 执行的一条 lane；单独看 thread 对理解 Tensor Core 不够。 |
| Lane | Warp lane | warp 内线程编号，通常是 `lane_id = threadIdx.x % 32`。 | 读 `ldmatrix`、`mma.sync`、warp shuffle、fragment 映射时必备。 |
| Warp | 32-thread execution group | NVIDIA GPU 上一起调度的 32 个线程 SIMT 执行组。 | `mma.sync`、`ldmatrix`、warp 级规约、访存合并的基本单位。 |
| Warpgroup | 4 contiguous warps, 128 threads | 4 个连续 warp，共 128 线程；Hopper WGMMA 的基本协作粒度。 | Hopper GEMM/attention mainloop 中常见 producer/consumer warpgroup 分工。 |
| CTA / Thread block | Cooperative Thread Array | CUDA thread block；同一 CTA 内线程可同步并共享 SMEM。 | kernel tiling 通常从 CTA tile size 开始设计。 |
| CTA cluster / Thread block cluster | CTA 集群 | 多个 CTA 组成的 cluster，可使用 cluster 级特性。 | Hopper/Blackwell 的 cluster launch、TMA multicast、distributed shared memory、persistent 调度都会遇到。 |
| Persistent kernel | 持久化 CTA/cluster 调度 | CTA 或 cluster 常驻 SM，从调度器/任务队列中持续取活。 | grouped GEMM、MoE、decode、负载不均任务常用；注意 occupancy、队列开销、公平性。 |
| CLC | Cluster Launch Control | Blackwell 时代的动态工作重分配机制；block/cluster 可取消尚未启动的任务并接过来做。 | 用于 persistent 或不规则 kernel 的负载均衡。 |
| PDL | Programmatic Dependent Launch | CUDA 层机制：有依赖的后续 kernel 可在前一个 kernel 完全结束前提前启动，前提是依赖被显式 signal。 | 当 kernel B 有不依赖 A 结果的 prologue，后续才需要 A 结果时有用。 |
| GDC | Grid Dependency Control | 与 dependent grid execution 相关的更底层 PTX/device-side 控制机制。 | 注释里要写清具体 CUDA/PTX 术语，因为 PDL 和 GDC 所在抽象层不同。 |

## Tensor Core 指令层

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| Tensor Core | Tensor matrix engine | SM 内专用矩阵乘加硬件单元。 | `mma.sync`、`wgmma.mma_async`、`tcgen05.mma` 背后的硬件。 |
| MMA | Matrix Multiply-Accumulate | 矩阵乘加指令族的通称。 | PTX/SASS、CUTLASS/CuTe、Triton lowering、架构文档里常见。 |
| `mma.sync` | Warp-level MMA PTX | warp 级同步 PTX 指令，例如 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。 | Ada/Ampere 路线的 Tensor Core 地基；先在这里学 fragment 布局。 |
| WMMA | Warp Matrix Multiply Accumulate | `nvcuda::wmma` 中的 CUDA C++ API，封装 Tensor Core fragment 和 `mma_sync`。 | 适合教学入门；隐藏 fragment 布局，控制力通常低于 inline PTX/CuTe。 |
| WGMMA | Warpgroup Matrix Multiply-Accumulate | Hopper warpgroup 级异步 MMA PTX，例如 `wgmma.mma_async`。 | Hopper/H20 GEMM 核心路径；常和 TMA、mbarrier、warp specialization 搭配。 |
| `tcgen05.mma` | Tensor Core generation 5 MMA | Blackwell SM100 第五代 Tensor Core PTX 指令族。 | B200/SM100 GEMM 核心路径；和 TMEM、FP8/FP4/block scaling 强相关。 |
| TMMA | Tensor Memory MMA / Blackwell MMA 非正式简称 | 社区可能用来指 Blackwell Tensor Core + TMEM MMA 路线的简称。 | 建议同时写 `tcgen05.mma`，避免歧义。 |
| UMMA | Unified MMA / 库或社区简称 | 常见于社区/库中，描述 Blackwell 统一 MMA 路线的简称。 | 不要当成独立 PTX opcode；应和 `tcgen05.mma` 或 CUTLASS SM100 MMA 类型名一起使用。 |
| HMMA | Half-precision MMA SASS family | SASS 层 Tensor Core 指令名，常见于 Volta/Ampere/Ada 反汇编。 | 在 `nvdisasm`、`cuobjdump` 中确认 Tensor Core lowering 时会看到。 |
| GMMA | Generic / warpgroup MMA SASS-family naming | Hopper WGMMA 路径附近常见的 SASS 层命名。 | 用于把 PTX `wgmma.mma_async` 和最终机器码对应起来。 |
| Fragment | 每线程寄存器切片 | 一条 collective MMA 指令中，每个 lane 持有的 A/B/C/D 局部数据。 | 解释为什么 `mma.sync` 操作数是寄存器列表，而不是矩阵指针。 |
| Accumulator | C/D fragment | 存放部分和的寄存器，或 Blackwell TMEM 位置。 | 影响精度、寄存器压力、epilogue 复杂度、spill 风险。 |
| Instruction tile | MMA atom shape | 单条硬件 MMA 指令消费的 tile 形状，例如 `m16n8k16`。 | 构建 warp tile、CTA tile、CUTLASS `TiledMMA` 的基础单位。 |
| `row.col` | A/B 布局 PTX modifier | 在 `mma.sync` 中说明 A 和 B fragment 如何被解释。 | 决定 B 是否要转置加载、预转置，或用特殊 `ldmatrix` 方式加载。 |

## 内存与数据搬运

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| GMEM / HBM | Global memory | 设备 DRAM，容量最大、延迟高。 | 低算术强度 kernel 的主要带宽瓶颈。 |
| L2 | Level-2 cache | 芯片级共享 cache。 | 跨 CTA 复用、persistent kernel、cache residency 调优时重要。 |
| L1 / SMEM | L1 cache / Shared memory | 靠近每个 SM 的片上内存；SMEM 由程序显式管理。 | 大多数 Tensor Core GEMM 会把 A/B tile stage 到 SMEM。 |
| Register file | 寄存器文件 | 每线程寄存器存储，普通可编程存储里最快但有限。 | register pressure 会影响 occupancy 和 spill。 |
| Local memory | 本地内存 | 每线程私有但实际由 global memory 支撑；常用于 spill 或过大的 per-thread 数组。 | 如果 `ncu` 看到 local memory traffic，要检查寄存器压力和索引方式。 |
| TMEM | Tensor Memory | Blackwell 上用于 Tensor Core/tcgen05 累加器流的片上内存。 | B200 专属心智模型；与 register 和 SMEM 分开看。 |
| DSMEM | Distributed Shared Memory | cluster 内跨 CTA shared memory 寻址能力。 | cluster 级算法、Hopper/Blackwell 协作 kernel 中会遇到。 |
| `cp.async` | Async copy global to shared | Ampere+ 的 PTX 异步拷贝路径，把 GMEM stage 到 SMEM，避免普通寄存器中转。 | Hopper TMA 之前 multistage software pipeline 的基础。 |
| TMA | Tensor Memory Accelerator | 使用 descriptor 的硬件 tensor copy 路径，通常在 GMEM 和 SMEM 间搬多维 tile。 | Hopper/Blackwell GEMM 和 attention 常把 TMA、mbarrier、WGMMA 组合使用。 |
| `cp.async.bulk.tensor` | TMA PTX copy 指令族 | 很多 TMA 操作背后的 PTX tensor copy 指令族。 | 检查代码是否真的走 TMA 时看这个。 |
| Tensor map / `CUtensorMap` | TMA tensor descriptor | 编码多维 layout、stride、box shape、swizzle 等 TMA 元数据。 | 读 CUTLASS/CuTe TMA 代码或写 raw TMA 示例时需要。 |
| `ldmatrix` | Load matrix | warp 级 PTX 指令，把 SMEM 矩阵数据加载到 MMA fragment 寄存器。 | Ampere/Ada 路径关键桥梁：`SMEM -> registers -> mma.sync`。 |
| Swizzle | 地址置换 | 重排 SMEM layout，常用 XOR，减少 bank conflict 并匹配 Tensor Core 访问模式。 | `ldmatrix`、CUTLASS shared layout、MoE/GEMM tile staging 的核心概念。 |
| Bank conflict | 共享内存 bank 冲突 | 多个 lane 访问冲突的 SMEM bank，导致串行化。 | 常在 `ldmatrix`、transpose、错误 shared layout/swizzle 中诊断。 |
| Coalescing | 访存合并 | warp 的内存访问被合并成高效 memory transaction。 | 深挖 Tensor Core 前，先检查 GMEM load/store 是否高效。 |
| Vectorized load/store | 向量化访存 | 单条指令加载/存储多个元素，例如 `int4`、`float4`、128-bit copy。 | 对齐满足时可降低指令数、提高带宽利用。 |

## 同步与流水线

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| `mbarrier` | Memory / async barrier | shared memory 中的 barrier 对象，用于协调 TMA、WGMMA 等异步操作。 | Hopper/Blackwell mainloop 的 producer/consumer 同步核心。 |
| `bar.sync` / `__syncthreads()` | CTA barrier | 同步一个 CTA 内所有线程。 | 比 mbarrier 简单，但对高性能异步 pipeline 往往太粗。 |
| Pipeline stage | 缓冲 stage | multistage mainloop 中一个 SMEM buffer slot。 | stage 数在 SMEM 占用和延迟隐藏之间做权衡。 |
| Double buffering | 双缓冲 | 当前 tile 计算和下一个 tile 加载交替进行。 | 最小可用的计算/搬运重叠模式。 |
| Multistage pipeline | 多级流水 | 多个预取 tile 同时在路上。 | 高性能 GEMM 常见；受 SMEM 和寄存器压力限制。 |
| Warp specialization | producer/consumer warp 分工 | 部分 warp/warpgroup 负责搬数据，另一部分负责计算。 | Hopper/Blackwell 高性能 GEMM/attention 的标准结构。 |
| `wgmma.fence` | WGMMA ordering primitive | 保证 WGMMA 发射异步 MMA 前能正确看到 operand/register 状态。 | raw WGMMA 流程中需要；CUTLASS/CuTe wrapper 通常会隐藏。 |
| `wgmma.commit_group` | WGMMA async group commit | 把已发射的 WGMMA 操作提交成异步 group。 | 心智模型类似 `cp.async.commit_group`，但作用于 WGMMA。 |
| `wgmma.wait_group` | WGMMA async group wait | 等待直到未完成的 WGMMA group 数不超过指定界限。 | 控制 accumulator 何时可以安全消费。 |

## CUTLASS / CuTe 抽象层

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | NVIDIA 的 GEMM/conv 等高性能模板库。 | 生产级 Tensor Core kernel 的重要参考。 |
| CuTe | CUTLASS Tensor Expressions | CUTLASS 3.x 的核心 layout/tensor 元编程层。 | 读现代 CUTLASS kernel 和 Agent 生成代码时绕不开。 |
| Layout | `Shape + Stride` 映射 | 把逻辑坐标映射到线性 offset。 | CuTe 基础；解释 swizzle、tiling、thread-value layout 的关键。 |
| Tensor | 指针 + Layout | CuTe 中表示“数据指针 + 坐标映射”的对象。 | copy/MMA 可以基于 layout 操作，而不是手写 raw indexing。 |
| Shape | 形状元组 | tensor/layout 的逻辑维度。 | 用于 tile shape、atom shape、CTA shape、instruction shape。 |
| Stride | 步长元组 | 逻辑坐标每变化一维时，线性 offset 如何变化。 | 决定 row-major、col-major 和自定义 layout。 |
| MMA Atom | 硬件 MMA 操作 wrapper | CuTe/CUTLASS 对单个 MMA 指令族的封装。 | 可包装 `mma.sync`、`wgmma`、SM100 MMA 等。 |
| Copy Atom | 硬件 copy 操作 wrapper | CuTe 对 `ldmatrix`、`cp.async`、TMA 等 copy 指令的封装。 | 连接 global-to-shared、shared-to-register 等数据搬运。 |
| TiledMMA | MMA atom 的 tiled 组合 | 把一个硬件 MMA atom 映射到 threads/warps/warpgroups 上，覆盖更大 tile。 | 读 CUTLASS Tensor Core compute 的主抽象。 |
| TiledCopy | Copy atom 的 tiled 组合 | 把硬件 copy atom 映射到 threads 上，完成 tile 搬运。 | 用于 GMEM->SMEM、SMEM->register、TMA、epilogue。 |
| CTA tile | thread-block 输出 tile | 一个 CTA 负责计算的矩阵 tile。 | occupancy、数据复用、scheduler 粒度的调优轴。 |
| Warp tile | 每 warp 的 tile | CTA tile 中由一个 warp 负责的部分。 | Ampere/Ada `mma.sync` kernel 中重要。 |
| Warpgroup tile | 每 warpgroup 的 tile | CTA tile 中由一个 warpgroup 负责的部分。 | Hopper WGMMA kernel 中重要。 |
| Epilogue | 输出后处理 | 把 accumulator fragment 转成最终输出：store、bias、activation、scale、quantize 等。 | LLM kernel 中常因量化和融合操作变得复杂。 |

## 精度与量化

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| FP16 / `.f16` | IEEE half precision | 16-bit float，5 位 exponent、10 位 mantissa。 | `mma.sync` 常见 A/B 输入类型；很多 fragment 中以 `.b32` packed pair 承载。 |
| BF16 / `.bf16` | bfloat16 | 16-bit float，exponent 宽度接近 FP32，mantissa 更少。 | 训练/推理常见精度；不要和 `.f16` 混用。 |
| TF32 / `.tf32` | TensorFloat-32 | 类 FP32 的 Tensor Core 格式，通过 32-bit 路径存放/操作。 | Ampere+ 上加速 FP32 风格 GEMM。 |
| FP8 E4M3 / E5M2 | 8-bit floating point formats | FP8 的两种常见变体，exponent/mantissa 权衡不同。 | Hopper/Blackwell 推理和量化 GEMM 中常见。 |
| FP4 / MXFP4 | 4-bit floating point / microscaling FP4 | Blackwell 低精度路径，通常结合 block scaling。 | B200 MoE 和新一代量化 GEMM 重点。 |
| Block scaling | 分块 scale factor | 一小组值共享 scale metadata。 | 用于恢复 FP8/FP4 等低精度格式的动态范围。 |
| W4A16 / W8A8 | 权重/激活精度 shorthand | 量化简写，例如 4-bit weights + 16-bit activations。 | LLM inference GEMM 讨论中常见。 |
| Dequantization | 反量化 | 在计算前或计算中应用 scale/zero-point 等元数据，把量化值转成计算格式。 | 可在 mainloop、epilogue 或独立 kernel 中完成。 |

## 性能诊断

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| SASS | NVIDIA native machine code | 编译后 GPU 最终执行的机器指令。 | 用 `cuobjdump`、`nvdisasm`、Nsight Compute source/SASS view 查看。 |
| PTX | Parallel Thread Execution | NVIDIA 虚拟 ISA，位于高级代码和最终机器码之间。 | 适合查指令语义；不总是和 SASS 一一对应。 |
| Stall | 调度器未发射指令的原因 | profiler 中表示 warp 为什么无法发射下一条指令的类别。 | 必须看子类型，不要只看 “stall” 这个词。 |
| Memory dependency | stall 子类型 | warp 等待之前的 memory operation。 | 检查延迟隐藏、cache 行为、memory-level parallelism、pipeline depth。 |
| Barrier stall | stall 子类型 | warp 等待同步点。 | 检查 `__syncthreads`、mbarrier、WGMMA wait、producer/consumer 失衡。 |
| Not selected | eligible warp 未被选中 | warp 可以发射，但 scheduler 选择了其他 warp。 | 不一定是问题；要结合 eligible warp 数和 issue rate 看。 |
| Pipe throttle | 执行管线压力 | 某条 instruction pipe 饱和或受限。 | 可能是 tensor、math、memory、LSU 等 pipe 成为瓶颈。 |
| Spill | 寄存器 spill 到 local memory | 寄存器压力过高，编译器把值放到 local memory。 | 关注 local load/store traffic 和 register count；可能要降低 tile size/stage 或简化 epilogue。 |
| Occupancy | 活跃 warps/CTAs 占硬件上限比例 | 受寄存器、SMEM、block size 影响的容量指标。 | Tensor Core kernel 不一定追求最高 occupancy，目标是足够隐藏延迟。 |
| Register pressure | 寄存器压力 | 每线程寄存器需求量。 | MMA mainloop 和 epilogue 的主要调优轴。 |
| Arithmetic intensity | 算术强度 | FLOPs per byte。 | 配合 roofline 判断 memory-bound 还是 compute-bound。 |
| Roofline | 性能上界模型 | 比较算术强度、内存带宽上界、计算峰值。 | 指令级微调前的第一轮判断工具。 |
| SOL | Speed of Light | Nsight Compute 中把性能和硬件峰值对比的高层指标。 | 可做总览，但结论必须下钻到具体 section/metric。 |
| Tensor pipe utilization | Tensor Core 管线利用率 | Tensor Core pipeline 有多忙。 | GEMM 中如果低，通常说明数据搬运、同步或调度在饿住 MMA。 |
| DRAM throughput | 全局内存带宽使用率 | 实测 HBM traffic rate。 | 高 DRAM throughput 但低 tensor utilization，常指向 memory-bound。 |
| L2 hit rate / throughput | L2 cache 行为 | L2 复用率和吞吐。 | persistent kernel、split-K、MoE grouped workload 中重要。 |

## MoE / LLM Kernel 术语

| 术语 | 全称 / 别名 | 含义 | 使用 / 调优语境 |
|---|---|---|---|
| MoE | Mixture of Experts | 模型层把 token 路由到不同 expert FFN。 | 计算侧主要由 routing、permute/unpermute、grouped GEMM、量化组成。 |
| Expert | per-route FFN block | 被路由选择的一组权重/FFN。 | 每个 expert 接收的 token 数可能不同。 |
| Top-k routing | 每 token 选择 k 个 expert | router 把每个 token 发往一个或多个 expert。 | 造成 expert size 不规则和负载不均。 |
| Permute | 按 expert gather token | 重排 token，使同一个 expert 的 token 连续。 | 使 grouped GEMM 可以高效执行。 |
| Unpermute | scatter 回原 token 顺序 | expert 计算后把 token 恢复到原顺序。 | 优化实现里常和 reduction/scale 融合。 |
| Grouped GEMM | 多个独立 GEMM 合并执行 | 一次 kernel/library 调用中执行多个 expert GEMM。 | MoE 计算侧核心 primitive。 |
| Segmented GEMM | 可变大小 grouped GEMM | 每组 GEMM 的 M size/token count 不同。 | 比 uniform batched GEMM 更贴合 MoE。 |
| `moe_align_block_size` | expert token block 对齐 | 把 expert token 数 padding/reorder 到 tile/block 倍数。 | 降低 grouped GEMM 的尾块低效。 |
| Split-K | 沿 K 维拆分规约 | 多个 CTA 计算不同 K 区间的 partial result，最后规约。 | 对大 K 或负载均衡有用，但会引入规约开销。 |
| Stream-K | persistent tile 调度策略 | work tile 在 SM 间动态分配，提高利用率。 | 高级 GEMM scheduler 和 CUTLASS 讨论里常见。 |

## 来源指针

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`mma.sync`、`wgmma.mma_async`、`tcgen05`、`mbarrier`、`cp.async`、`cp.async.bulk.tensor`。
- [CUDA Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)：PDL 语义。
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)：Blackwell CLC 和 work stealing。
- [CUDA Programming Guide: Advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：cluster、同步和高级执行概念。
- [CUTLASS CuTe Layout docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)：CuTe `Layout`、shape、stride、坐标映射。
- [CUTLASS CuTe quickstart](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)：CuTe `Tensor` 和层级化 thread/data 抽象。
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：profiler section、metrics、诊断工作流。
- 本地课程文件：`reference/glossary.html`、`repos/LeetCUDA/kernels/swizzle/`、`repos/how-to-optim-algorithm-in-cuda/cutlass/`。
