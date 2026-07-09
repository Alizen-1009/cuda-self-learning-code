#include <cuda_fp16.h>
  #include <stdint.h>

  __device__ __forceinline__
  uint32_t smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  }

  __device__ __forceinline__
  void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr)
    );
  }

  __device__ __forceinline__
  void ldmatrix_x2_trans(uint32_t r[2], uint32_t addr) {
    asm volatile(
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
      "{%0, %1}, [%2];\n"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(addr)
    );
  }

  __device__ __forceinline__
  void mma_m16n8k16_f32(float d[4], uint32_t a[4], uint32_t b[2], float c[4]) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
  }

  然后在 kernel 里，核心流程就是：

  __global__ void one_mma_tile(const half* A, const half* B, float* C) {
    __shared__ half sA[16][16]; // A tile: m16k16
    __shared__ half sB[16][8];  // B tile: k16n8

    int tid = threadIdx.x;
    int lane = tid & 31;

    // 1. Global -> Shared
    // 这里为了教学用普通 load/store 表示。
    // 高性能版本会把这一步换成 cp.async。
    if (tid < 256) {
      int i = tid / 16;
      int j = tid % 16;
      sA[i][j] = A[i * 16 + j];
    }

    if (tid < 128) {
      int i = tid / 8;
      int j = tid % 8;
      sB[i][j] = B[i * 8 + j];
    }

    __syncthreads();

    // 2. Shared -> Register fragment
    uint32_t RA[4]; // 每线程 A fragment: 4 个 b32 = 8 个 fp16
    uint32_t RB[2]; // 每线程 B fragment: 2 个 b32 = 4 个 fp16

    // 每个 lane 给 ldmatrix 一个 shared memory 行地址。
    // ldmatrix 会 warp 协作地读取 shared memory，并把数据摆成 mma 要的 fragment。
    uint32_t addrA = smem_addr(&sA[lane % 16][(lane / 16) * 8]);
    uint32_t addrB = smem_addr(&sB[lane % 8][(lane / 8) * 8]);

    ldmatrix_x4(RA, addrA);
    ldmatrix_x2_trans(RB, addrB);

    // 3. Register fragment -> Tensor Core
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float out[4];

    mma_m16n8k16_f32(out, RA, RB, acc);

    // 4. out[0..3] 是当前线程持有的 C fragment
    // 真实 GEMM 还要按 PTX fragment 映射把它写回 C。
  }

  这里你要抓住三个对象：

  sA / sB

  是 shared memory 里的普通二维 tile。

  RA / RB

  是 ldmatrix 之后得到的寄存器 fragment，已经不是普通二维矩阵视角了。

  out / acc

  是 mma.sync 的累加器 fragment。每个线程只拿 C tile 里的一小部分。

  所以这句：

  Shared ──ldmatrix──▶ Register ──mma.sync──▶ 累加

  具体就是：

  ldmatrix_x4(RA, addrA);      // sA -> RA
  ldmatrix_x2_trans(RB, addrB);// sB -> RB
  mma_m16n8k16_f32(out, RA, RB, acc); // RA/RB -> tensor core -> out

  前面的：

  Global ──cp.async──▶ Shared

  在教学骨架里我用普通 load/store 代替了。真实高性能代码会写成类似：

  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16;\n"
    :
    : "r"(smem_addr(&sA[row][col])),
      "l"(&A[gmem_offset])
  );
