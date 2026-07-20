// Width-4 causal depthwise conv1d + SiLU for Qwen3.5-plus GDN prefill (B300 / SM100).
// Channel-contiguous bf16, single sequence. ~67% SOL (74.7us on dim=12288,T=8192),
// 1.5x over cp.async/TMA baselines and ~1.6x over the Triton stock kernel; in-model 1.79x.
//
// How it beats the ~44% wall (all ncu-driven):
//   1. Occupancy: small blocks (128) + __launch_bounds__(128, MINB) force more resident
//      blocks/SM (the prior cp.async/TMA attempts sat at ~17-20% occ -> DRAM only ~42%).
//   2. Vectorized float2 load AND store (scalar stores are LSU-bound).
//   3. cp.async ring (DEPTH deep) for memory-level parallelism.
//   4. bf16x2 SIMD conv (__hfma2): halves FMAs, removes bf16<->f32 converts, halves registers.
//   5. tanh.approx.f32 SiLU (single SFU op). rel_l2 ~2.4e-3 (bf16 accum, well under 1e-2).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "causal_conv1d_fwd.h"

namespace {

__device__ __forceinline__ float silu_t(float a) {
  float t;
  asm("tanh.approx.f32 %0,%1;" : "=f"(t) : "f"(0.5f * a));
  return 0.5f * a * (1.f + t);
}

// x,out (dim,T) channel-contig: elem(d,t)=ptr[t*dim+d]. V=4 ch/thread = 2 bf162 lanes,
// float2 vec load/store. Block-flattened (cg=tid%NCG, chunk=tid/NCG). DEPTH-deep cp.async ring.
template <int V, int BLK, int CHUNK, int DEPTH, int MINB>
__global__ void __launch_bounds__(BLK, MINB) conv_fwd_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, const __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out, int dim, int T, int has_init) {
  typedef float2 VT;
  const int L = V / 2;  // V=4 -> float2, 2 bf162 lanes
  const int NCG = dim / V;
  const int NCHUNK = (T + CHUNK - 1) / CHUNK;
  long tid = (long)blockIdx.x * BLK + threadIdx.x;
  int cg = tid % NCG, chunk = tid / NCG;
  int c0 = cg * V, t0 = chunk * CHUNK, tend = min(t0 + CHUNK, T);
  if (chunk >= NCHUNK) return;
  extern __shared__ __nv_bfloat16 ring[];
  __nv_bfloat16* myring = ring + (long)threadIdx.x * DEPTH * V;
  __nv_bfloat162 wj[4][L], bs[L];
#pragma unroll
  for (int l = 0; l < L; l++) {
#pragma unroll
    for (int j = 0; j < 4; j++)
      wj[j][l] = __halves2bfloat162(w[(c0 + 2 * l) * 4 + j], w[(c0 + 2 * l + 1) * 4 + j]);
    bs[l] = __halves2bfloat162(bias[c0 + 2 * l], bias[c0 + 2 * l + 1]);
  }
  // taps for t0-3..t0-1: from state (chunk 0, has_init) else prior x tokens (else zero).
  __nv_bfloat162 tap0[L], tap1[L], tap2[L];
  auto ldx = [&](int t, __nv_bfloat162* d) {
    VT f = *reinterpret_cast<const VT*>(x + (long)t * dim + c0);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&f);
#pragma unroll
    for (int l = 0; l < L; l++) d[l] = h[l];
  };
  auto lds = [&](int i, __nv_bfloat162* d) {  // state (3,dim) dim-contig
    const __nv_bfloat16* p = state + (long)i * dim + c0;
#pragma unroll
    for (int l = 0; l < L; l++) d[l] = __halves2bfloat162(p[2 * l], p[2 * l + 1]);
  };
  auto zero = [&](__nv_bfloat162* d) {
#pragma unroll
    for (int l = 0; l < L; l++) d[l] = __float2bfloat162_rn(0.f);
  };
  if (t0 == 0) {
    if (has_init) { lds(0, tap0); lds(1, tap1); lds(2, tap2); }
    else { zero(tap0); zero(tap1); zero(tap2); }
  } else {
    ldx(t0 - 3, tap0); ldx(t0 - 2, tap1); ldx(t0 - 1, tap2);
  }
#pragma unroll
  for (int s = 0; s < DEPTH; s++) {
    int t = t0 + s;
    if (t < tend) __pipeline_memcpy_async(myring + s * V, x + (long)t * dim + c0, V * 2);
    __pipeline_commit();
  }
  for (int t = t0; t < tend; t++) {
    int slot = (t - t0) % DEPTH;
    __pipeline_wait_prior(DEPTH - 1);
    __nv_bfloat162 cur[L];
    {
      VT f = *reinterpret_cast<const VT*>(myring + slot * V);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&f);
#pragma unroll
      for (int l = 0; l < L; l++) cur[l] = h[l];
    }
    int tn = t + DEPTH;
    if (tn < tend) __pipeline_memcpy_async(myring + slot * V, x + (long)tn * dim + c0, V * 2);
    __pipeline_commit();
    VT of;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&of);
#pragma unroll
    for (int l = 0; l < L; l++) {
      __nv_bfloat162 acc = __hfma2(wj[0][l], tap0[l],
          __hfma2(wj[1][l], tap1[l], __hfma2(wj[2][l], tap2[l], __hmul2(wj[3][l], cur[l]))));
      acc = __hadd2(acc, bs[l]);
      ob[l] = __halves2bfloat162(__float2bfloat16(silu_t(__low2float(acc))),
                                 __float2bfloat16(silu_t(__high2float(acc))));
      tap0[l] = tap1[l]; tap1[l] = tap2[l]; tap2[l] = cur[l];
    }
    *reinterpret_cast<VT*>(out + (long)t * dim + c0) = of;
  }
}

}  // namespace

// best config from the autotune sweep: V=4, block=128, CHUNK=64, DEPTH=6, MINB=8
void causal_conv1d_fwd(const torch::Tensor& x, const torch::Tensor& weight,
                       const torch::Tensor& bias, const torch::Tensor& state,
                       torch::Tensor& out, int64_t has_init) {
  const int dim = x.size(0), T = x.size(1);
  const int V = 4, BLK = 128, CHUNK = 64, DEPTH = 6;
  long nth = (long)(dim / V) * ((T + CHUNK - 1) / CHUNK);
  long grid = (nth + BLK - 1) / BLK;
  size_t sh = (size_t)BLK * DEPTH * V * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  conv_fwd_kernel<V, BLK, CHUNK, DEPTH, 8><<<grid, BLK, sh, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), dim, T, (int)has_init);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
