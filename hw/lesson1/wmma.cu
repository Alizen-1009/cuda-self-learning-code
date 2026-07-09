#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
// Support A and B matrix with row-major inorder to compare with the kernels
// using CUDA Cores in hgemm.cu and hgemm_async.cu.

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template <const int WMMA_M, const int WMMA_N, const int WMMA_K, typename scalar_t>
__global__ void wmma_gemm_m16n16k16_naive_kernel(const scalar_t *A, const scalar_t *B,
                                 float *C, int M, int N, int K) {
  // Leading dimensions.
  int A_start = blockIdx.y * WMMA_M * K;
  int B_start = blockIdx.x * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, scalar_t, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  
  int BK = div_ceil(K, WMMA_K);
  for(int i = 0; i < BK; i++) {
    int A_offset = A_start + i * WMMA_K;
    int B_offset = B_start + i * WMMA_K * N;
    wmma::load_matrix_sync(a_frag, A + A_offset, K);
    wmma::load_matrix_sync(b_frag, B + B_offset, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }
  int C_offset = blockIdx.y * WMMA_M * N + blockIdx.x * WMMA_N;
  wmma::store_matrix_sync(C + C_offset, c_frag, N, wmma::mem_row_major);

}

#define CHECK_CUDA(T) TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(T)                                                    \
  TORCH_CHECK((T).is_contiguous(), #T " must be contiguous")
#define CHECK_2D(T) TORCH_CHECK((T).dim() == 2, #T " must be a 2D tensor")
#define CHECK_INPUT(T)                                                         \
  CHECK_CUDA(T);                                                               \
  CHECK_CONTIGUOUS(T);                                                         \
  CHECK_2D(T)

torch::Tensor wmma_gemm_m16n16k16_naive(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "A and B must have the same dtype");
  TORCH_CHECK(A.scalar_type() == torch::kHalf ||
                  A.scalar_type() == torch::kBFloat16,
              "only float16 and bfloat16 are supported");

  const int M = A.size(0);
  const int K = A.size(1);
  TORCH_CHECK(B.size(0) == K, "B.size(0) must equal A.size(1)");
  const int N = B.size(1);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  TORCH_CHECK(M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0,
              "this naive WMMA kernel requires M, N, K to be multiples of 16");

  c10::cuda::CUDAGuard device_guard(A.device());
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

  dim3 block(WARP_SIZE);
  dim3 grid(N / WMMA_N, M / WMMA_M);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (A.scalar_type() == torch::kHalf) {
    wmma_gemm_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K, half>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
            C.data_ptr<float>(), M, N, K);
  } else {
    wmma_gemm_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
            C.data_ptr<float>(), M, N, K);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wmma_gemm_m16n16k16_naive", &wmma_gemm_m16n16k16_naive,
        "Naive 16x16x16 WMMA GEMM with FP32 accumulator");
}
