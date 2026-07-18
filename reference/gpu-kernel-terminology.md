# GPU Kernel Terminology

Last updated: 2026-07-09

This glossary is tuned for the current learning mission: Tensor Core instructions, CUTLASS/CuTe, Hopper/Blackwell GEMM, MoE compute kernels, and `ncu`-driven diagnosis.

## Quick Review Of The Original Table

Your table is directionally correct. The important refinements are:

- `WMMA` is not `WGMMA`: `WMMA` is the CUDA C++ warp-level API; `WGMMA` is Hopper warpgroup-level PTX.
- `UMMA` is not the safest standalone term. Treat it as a community/library shorthand and pair it with the official PTX name `tcgen05.mma` or CUTLASS SM100 MMA names.
- `TMA` starts on Hopper but remains relevant on Blackwell. Do not read it as a Hopper-only idea.
- `TMEM` is Blackwell tensor memory for Tensor Core accumulators. Do not confuse it with SMEM, TMA, or ordinary shared memory.
- `Stall` should be read through profiler metrics, not as a single root cause. `memory dependency`, `barrier`, `not selected`, and `pipe throttle` point to different fixes.

## Threading And Execution

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| Thread | CUDA thread | Smallest CUDA execution instance. | One lane of scalar CUDA execution; too low-level for Tensor Core reasoning by itself. |
| Lane | Warp lane | Thread index inside a warp, usually `lane_id = threadIdx.x % 32`. | Essential for `ldmatrix`, `mma.sync`, warp shuffle, and fragment mapping. |
| Warp | 32-thread execution group | SIMT group scheduled together on NVIDIA GPUs. | Basic unit for `mma.sync`, `ldmatrix`, warp-level reductions, and coalescing. |
| Warpgroup | 4 contiguous warps, 128 threads | Hopper WGMMA's basic collaboration group. | Hopper GEMM/attention mainloop often splits producer/consumer warpgroups. |
| CTA / Thread block | Cooperative Thread Array | CUDA thread block. Threads in one CTA can synchronize and share SMEM. | Kernel tiling usually starts from CTA tile sizing. |
| CTA cluster / Thread block cluster | Cluster of CTAs | Group of CTAs scheduled together with cluster-level features. | Important for Hopper/Blackwell cluster launch, TMA multicast, distributed shared memory, and persistent scheduling. |
| Persistent kernel | Persistent CTA/cluster scheduling | Kernel keeps CTAs or clusters resident and repeatedly pulls work from a scheduler/queue. | Useful for grouped GEMM, MoE, decode, and load-imbalanced workloads; watch occupancy, queue overhead, and fairness. |
| CLC | Cluster Launch Control | Blackwell-era dynamic work redistribution mechanism; a block/cluster can cancel not-yet-started work and take its task. | Relevant for load balancing persistent and irregular kernels. |
| PDL | Programmatic Dependent Launch | CUDA-level mechanism that lets a dependent kernel begin before the previous kernel fully completes, if the dependency is explicitly signaled. | Useful when kernel B has an independent prologue and only later needs kernel A's results. |
| GDC | Grid Dependency Control | Lower-level PTX/device-side mechanism related to controlling dependent grid execution. | Use the exact PTX/CUDA term in comments because PDL and GDC live at different abstraction levels. |

## Tensor Core Instruction Layer

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| Tensor Core | Tensor matrix engine | Specialized SM hardware for small matrix multiply-accumulate. | The hardware unit behind `mma.sync`, `wgmma.mma_async`, and `tcgen05.mma`. |
| MMA | Matrix Multiply-Accumulate | Generic name for matrix multiply-add instruction families. | Common in PTX/SASS, CUTLASS/CuTe, Triton lowering, and architecture docs. |
| `mma.sync` | Warp-level MMA PTX | Warp-synchronous PTX instruction, e.g. `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. | Ada/Ampere-style Tensor Core ground truth; learn fragment layout here first. |
| WMMA | Warp Matrix Multiply Accumulate | CUDA C++ API in `nvcuda::wmma` that wraps Tensor Core fragments and `mma_sync`. | Good teaching API; hides fragment layout and usually gives less control than inline PTX/CuTe. |
| WGMMA | Warpgroup Matrix Multiply-Accumulate | Hopper warpgroup-level async MMA PTX, e.g. `wgmma.mma_async`. | Core Hopper/H20 GEMM path; pairs with TMA, mbarrier, and warp specialization. |
| `tcgen05.mma` | Tensor Core generation 5 MMA | Blackwell SM100 PTX instruction family for fifth-generation Tensor Cores. | Core B200 / B300 (SM100) GEMM path; tied to TMEM and newer FP8/FP4/block-scaled flows. |
| TMMA | Tensor Memory MMA / informal Blackwell MMA shorthand | Informal shorthand people may use for Blackwell Tensor Core + TMEM MMA flows. | Prefer writing `tcgen05.mma` next to it to avoid ambiguity. |
| UMMA | Unified MMA / informal library shorthand | Community/library shorthand often used around Blackwell's unified MMA path. | Do not treat as a standalone PTX opcode; pair with `tcgen05.mma` or CUTLASS SM100 MMA type names. |
| HMMA | Half-precision MMA SASS family | SASS-level Tensor Core instruction naming often seen on Volta/Ampere/Ada disassembly. | Shows up in `nvdisasm`/`cuobjdump`; useful for confirming Tensor Core lowering. |
| GMMA | Generic / warpgroup MMA SASS-family naming | SASS-level naming around Hopper WGMMA paths. | Useful when matching PTX `wgmma.mma_async` to generated machine code. |
| Fragment | Per-thread register slice | The part of A/B/C/D held by each lane for a collective MMA instruction. | Explains why `mma.sync` operands are register lists, not matrix pointers. |
| Accumulator | C/D fragment | Registers, or Blackwell TMEM locations, holding partial sums. | Controls precision, register pressure, epilogue complexity, and spill risk. |
| Instruction tile | MMA atom shape | The tile shape consumed by one hardware instruction, e.g. `m16n8k16`. | Base unit for building warp tiles, CTA tiles, and CUTLASS `TiledMMA`. |
| `row.col` | A layout / B layout PTX modifier | In `mma.sync`, tells PTX how A and B fragments are interpreted. | Drives whether B must be loaded transposed, pre-transposed, or with special `ldmatrix` use. |

## Memory And Data Movement

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| GMEM / HBM | Global memory | Device DRAM. Highest capacity, high latency. | Main bandwidth bottleneck for low arithmetic-intensity kernels. |
| L2 | Level-2 cache | Chip-wide cache shared by SMs. | Important for reuse across CTAs, persistent kernels, and cache residency tuning. |
| L1 / SMEM | L1 cache / Shared memory | On-chip memory near each SM; SMEM is explicitly managed. | Most Tensor Core GEMMs stage A/B tiles through SMEM. |
| Register file | Per-thread register storage | Fastest ordinary programmable storage, but limited. | Register pressure affects occupancy and spilling. |
| Local memory | Per-thread memory backed by global memory | Used for spills and large per-thread arrays that cannot stay in registers. | If `ncu` shows local memory traffic, inspect register pressure and indexing. |
| TMEM | Tensor Memory | Blackwell on-chip memory used by Tensor Core/tcgen05 accumulator flows. | B200 / B300-specific mental model; separate from registers and SMEM. |
| DSMEM | Distributed Shared Memory | Cluster-level shared-memory addressing across CTAs in a cluster. | Relevant for cluster-level algorithms and some Hopper/Blackwell cooperative kernels. |
| `cp.async` | Async copy global to shared | Ampere+ PTX copy path for staging GMEM to SMEM without a normal register round trip. | Foundation of multistage software pipelines before Hopper TMA. |
| TMA | Tensor Memory Accelerator | Hardware tensor copy path using descriptors for multidimensional tile movement, usually GMEM <-> SMEM. | Hopper/Blackwell GEMM and attention often combine TMA, mbarrier, and WGMMA. |
| `cp.async.bulk.tensor` | TMA PTX copy instruction family | PTX-level tensor copy instruction family behind many TMA operations. | Look for this when checking whether code truly uses TMA. |
| Tensor map / `CUtensorMap` | TMA tensor descriptor | Encodes multidimensional layout, strides, box shape, swizzle, and related TMA metadata. | Needed when reading CUTLASS/CuTe TMA code or writing raw TMA examples. |
| `ldmatrix` | Load matrix | Warp-level PTX instruction that loads SMEM matrix data into MMA fragment registers. | Key bridge in Ampere/Ada path: `SMEM -> registers -> mma.sync`. |
| Swizzle | Address permutation | Reorders SMEM layout, often XOR-based, to reduce bank conflicts and match Tensor Core access patterns. | Critical for `ldmatrix`, CUTLASS shared layouts, and MoE/GEMM tile staging. |
| Bank conflict | Shared-memory bank serialization | Multiple lanes hit conflicting SMEM banks. | Often diagnosed around `ldmatrix`, transposes, and poorly swizzled shared layouts. |
| Coalescing | Memory access combining | Warp memory accesses align into efficient memory transactions. | First check for GMEM load/store efficiency before deeper Tensor Core tuning. |
| Vectorized load/store | Packed memory operation | Load/store multiple elements per instruction, e.g. `int4`, `float4`, 128-bit copies. | Reduces instruction count and improves bandwidth utilization when aligned. |

## Synchronization And Pipelines

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| `mbarrier` | Memory / async barrier | Shared-memory barrier object used to coordinate async operations such as TMA and WGMMA pipelines. | Producer/consumer synchronization in Hopper/Blackwell mainloops. |
| `bar.sync` / `__syncthreads()` | CTA barrier | Synchronizes all threads in a CTA. | Simpler than mbarrier but too coarse for high-performance async pipelines. |
| Pipeline stage | Buffer stage | One SMEM buffer slot in a multistage mainloop. | Stage count trades SMEM usage for latency hiding. |
| Double buffering | Two-stage pipeline | Alternates current compute tile and next load tile. | Minimal useful overlap pattern. |
| Multistage pipeline | 3+ stage pipeline | Several prefetched tiles are in flight. | Common in high-performance GEMM; limited by SMEM and register pressure. |
| Warp specialization | Producer/consumer warp split | Some warps/warpgroups move data, others compute. | Standard Hopper/Blackwell high-performance GEMM/attention pattern. |
| `wgmma.fence` | WGMMA ordering primitive | Ensures WGMMA sees operand/register state correctly before async MMA issue. | Required in raw WGMMA flows; hidden by CUTLASS/CuTe wrappers. |
| `wgmma.commit_group` | WGMMA async group commit | Commits issued WGMMA operations as an async group. | Similar mental model to `cp.async.commit_group`, but for WGMMA. |
| `wgmma.wait_group` | WGMMA async group wait | Waits until a bounded number of WGMMA groups remain pending. | Controls when accumulators can be safely consumed. |

## CUTLASS / CuTe Abstraction Layer

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | NVIDIA template library for GEMM/conv and related kernels. | Major reference for production Tensor Core kernels. |
| CuTe | CUTLASS Tensor Expressions | CUTLASS 3.x core layout/tensor metaprogramming layer. | Necessary for reading modern CUTLASS kernels and generated Agent code. |
| Layout | `Shape + Stride` mapping | Maps logical coordinates to linear offsets. | CuTe foundation; explains swizzles, tiling, and thread-value layouts. |
| Tensor | Pointer + Layout | CuTe object representing data plus its coordinate mapping. | Lets copies/MMA operate on layouts rather than raw indexing. |
| Shape | Extent tuple | Logical dimensions of a tensor/layout. | Used in tile shape, atom shape, CTA shape, and instruction shape. |
| Stride | Offset step tuple | Linear offset movement for each logical coordinate. | Determines row-major, col-major, and custom layouts. |
| MMA Atom | Hardware MMA operation wrapper | CuTe/CUTLASS wrapper for one MMA instruction family. | Example: an atom wrapping `mma.sync`, `wgmma`, or SM100 MMA. |
| Copy Atom | Hardware copy operation wrapper | CuTe wrapper for a copy instruction such as `ldmatrix`, `cp.async`, or TMA. | Helps connect global-to-shared and shared-to-register data movement. |
| TiledMMA | Tiled composition of MMA atoms | Maps a hardware MMA atom across threads/warps/warpgroups to cover a larger tile. | The main abstraction for reading CUTLASS Tensor Core compute. |
| TiledCopy | Tiled composition of copy atoms | Maps hardware copy atoms across threads to move a tile. | Used for GMEM->SMEM, SMEM->register, TMA, and epilogues. |
| CTA tile | Thread-block output tile | Matrix tile computed by one CTA. | Tuning axis for occupancy, reuse, and scheduler granularity. |
| Warp tile | Per-warp tile | Portion of CTA tile computed by one warp. | Important in Ampere/Ada `mma.sync` kernels. |
| Warpgroup tile | Per-warpgroup tile | Portion of CTA tile computed by one warpgroup. | Important in Hopper WGMMA kernels. |
| Epilogue | Output post-processing | Converts accumulator fragments to final output: store, bias, activation, scale, quantize, etc. | Often nontrivial in LLM kernels because of quantization and fused operations. |

## Precision And Quantization

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| FP16 / `.f16` | IEEE half precision | 16-bit float with 5 exponent bits and 10 mantissa bits. | Common A/B input type for `mma.sync`; packed as `.b32` pairs in many fragments. |
| BF16 / `.bf16` | bfloat16 | 16-bit float with FP32-like exponent width and fewer mantissa bits. | Common training/inference precision; not the same as `.f16`. |
| TF32 / `.tf32` | TensorFloat-32 | 19-bit-ish Tensor Core format stored/operated through 32-bit paths. | Used for FP32-like GEMM acceleration on Ampere+. |
| FP8 E4M3 / E5M2 | 8-bit floating point formats | FP8 variants with different exponent/mantissa tradeoffs. | Hopper/Blackwell inference and quantized GEMM. |
| FP4 / MXFP4 | 4-bit floating point / microscaling FP4 | Blackwell low-precision path, often with block scaling. | Important for B200 / B300 MoE and next-gen quantized GEMM. |
| Block scaling | Per-block scale factors | Small groups of values share scale metadata. | Used to recover numeric range for FP8/FP4-style low precision. |
| W4A16 / W8A8 | Weight/activation precision shorthand | Quantization shorthand, e.g. 4-bit weights with 16-bit activations. | Common in LLM inference GEMM discussions. |
| Dequantization | Convert quantized values to compute format | Applies scale/zero-point or related metadata before/during compute. | Can live in mainloop, epilogue, or separate kernel. |

## Performance Diagnosis

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| SASS | NVIDIA native machine code | Final GPU machine instructions after compilation. | Inspect with `cuobjdump`, `nvdisasm`, or Nsight Compute source/SASS views. |
| PTX | Parallel Thread Execution | NVIDIA virtual ISA emitted before final machine code. | Good for instruction semantics; not always one-to-one with SASS. |
| Stall | Scheduler reason no instruction issued | Profiler category for why a warp could not issue. | Diagnose by subtype, not by the word "stall" alone. |
| Memory dependency | Stall subtype | Warp waits on a previous memory operation. | Check latency hiding, cache behavior, memory-level parallelism, and pipeline depth. |
| Barrier stall | Stall subtype | Warp waits at a synchronization point. | Check `__syncthreads`, mbarrier, WGMMA waits, and producer/consumer imbalance. |
| Not selected | Scheduler did not pick an eligible warp | Warp was eligible but another warp issued. | Often not a bug; interpret with eligible warp count and issue rate. |
| Pipe throttle | Execution pipe pressure | Instruction pipe is saturated or constrained. | Tensor, math, memory, or LSU pipe may be the limiting resource. |
| Spill | Register spill to local memory | Compiler places register values in local memory due to register pressure. | Look for local load/store traffic and high register count; may require lowering tile size/stages or simplifying epilogue. |
| Occupancy | Active warps/CTAs relative to hardware limit | Capacity measure affected by registers, SMEM, and block size. | High occupancy is not always best for Tensor Core kernels; enough latency hiding is the goal. |
| Register pressure | Register demand per thread | High register use can reduce occupancy or cause spills. | Major tuning axis in MMA mainloops and epilogues. |
| Arithmetic intensity | FLOPs per byte | Compute-to-memory ratio. | Used with roofline to decide memory-bound vs compute-bound. |
| Roofline | Performance bound model | Compares arithmetic intensity against memory bandwidth and compute peak. | First-pass diagnosis before micro-optimizing instructions. |
| SOL | Speed of Light | Nsight Compute's high-level utilization comparison to hardware peak. | Useful summary; always drill into section metrics before concluding. |
| Tensor pipe utilization | Tensor Core pipeline usage | How busy Tensor Core pipelines are. | Low value in GEMM usually means data movement, synchronization, or scheduling is starving MMA. |
| DRAM throughput | Global memory bandwidth usage | Measured HBM traffic rate. | High value with low tensor utilization suggests memory-bound behavior. |
| L2 hit rate / throughput | L2 cache behavior | Reuse and traffic at L2. | Important for persistent kernels, split-K, and MoE grouped workloads. |

## MoE / LLM Kernel Terms

| Term | Full name / Alias | Meaning | Usage / Tuning Context |
|---|---|---|---|
| MoE | Mixture of Experts | Model layer routes tokens to expert FFNs. | Compute side is dominated by routing, permute/unpermute, grouped GEMM, and quantization. |
| Expert | Per-route FFN block | One of many weight matrices selected by routing. | Each expert may receive a variable token count. |
| Top-k routing | Select k experts per token | Router sends each token to one or more experts. | Causes irregular expert sizes and load imbalance. |
| Permute | Token gather by expert | Reorders tokens so tokens for the same expert are contiguous. | Enables efficient grouped GEMM. |
| Unpermute | Token scatter back | Restores token order after expert computation. | Often fused with reduction/scale in optimized implementations. |
| Grouped GEMM | Batched GEMM with many independent problems | Executes many expert GEMMs in one kernel/library call. | Core MoE compute primitive. |
| Segmented GEMM | Variable-size grouped GEMM | GEMM groups have different M sizes/token counts. | Matches MoE better than uniform batched GEMM. |
| `moe_align_block_size` | Expert token block alignment | Pads/reorders expert token counts to tile/block multiples. | Reduces tail inefficiency in grouped GEMM. |
| Split-K | Split reduction over K dimension | Multiple CTAs compute partial K ranges and reduce. | Useful for large K or load balancing, but adds reduction overhead. |
| Stream-K | Persistent tile scheduling strategy | Work tiles are dynamically assigned across SMs to improve utilization. | Common concept in advanced GEMM schedulers and CUTLASS discussions. |

## Source Pointers

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): `mma.sync`, `wgmma.mma_async`, `tcgen05`, `mbarrier`, `cp.async`, `cp.async.bulk.tensor`.
- [CUDA Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html): PDL semantics.
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html): Blackwell CLC and work stealing.
- [CUDA Programming Guide: Advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html): clusters, synchronization, and advanced execution concepts.
- [CUTLASS CuTe Layout docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html): CuTe `Layout`, shape, stride, and coordinate mapping.
- [CUTLASS CuTe quickstart](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html): CuTe `Tensor` and hierarchical thread/data abstractions.
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): profiler sections, metrics, and diagnosis workflow.
- Local course files: `reference/glossary.html`, `repos/LeetCUDA/kernels/swizzle/`, `repos/how-to-optim-algorithm-in-cuda/cutlass/`.
