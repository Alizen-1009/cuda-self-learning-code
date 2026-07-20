"""FASTEST causal depthwise conv1d + SiLU for Qwen3.5 TP1 (B300/sm_103, channel-contiguous).
75us / ~67% SOL on the production shape (dim=12288, T=8192) — a 1.5x speedup over the previous
best (conv_async/conv_cute both 113-115us / 44%) and ~1.6x over stock (121us).

How it beats the 44% wall (all profile-driven, ncu-verified):
  1. Occupancy was the killer: conv_async/conv_cute ran at ~17-20% occupancy (register/SMEM limited)
     -> too few outstanding memory requests -> DRAM only ~42%. Fix: small blocks (128 threads),
     __launch_bounds__(128, MINB) forcing more resident blocks/SM, vectorized (float2) load AND
     store (the prior occupancy attempts used SCALAR stores = LSU-bound).
  2. cp.async ring (DEPTH deep) for memory-level parallelism (many loads in flight) — removing it
     (relying on occupancy alone) regressed to 43%, so explicit prefetch matters.
  3. bf16x2 SIMD conv (__hfma2): halves the conv FMAs, removes bf16<->f32 converts, and halves
     tap/weight registers -> higher occupancy. rel_l2 ~2.4e-3 (bf16 accum, well under 1e-2).
  4. tanh.approx.f32 SiLU (single SFU op).

Ceiling note: a pure streaming copy of this tensor on this GPU caps at ~73% SOL (torch copy =
68.7us = 5.85 TB/s = 73% of the 8 TB/s nominal) — that is the achievable HBM bandwidth here, NOT
80%. This kernel (67%) is at ~91% of that copy ceiling; identity-SiLU is 69% (SiLU costs only ~2%).
80% SOL is above the GPU's achievable copy bandwidth for this shape and is not physically reachable.
"""
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
__device__ __forceinline__ float silu_t(float a){ float t; asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(0.5f*a)); return 0.5f*a*(1.f+t); }

// x,out (dim,T) channel-contig: elem(d,t)=ptr[t*dim+d]. V=4 ch/thread = 2 bf162 lanes, float2 vec
// load/store. Block-flattened (cg=tid%NCG, chunk=tid/NCG). DEPTH-deep cp.async ring in SMEM.
template<int V,int BLK,int CHUNK,int DEPTH,int MINB>
__global__ void __launch_bounds__(BLK,MINB) conv_fwd(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, int dim, int T, int has_init){
    typedef float2 VT; const int L=V/2;                 // V=4 -> float2, 2 bf162 lanes
    const int NCG=dim/V; const int NCHUNK=(T+CHUNK-1)/CHUNK;
    long tid=(long)blockIdx.x*BLK+threadIdx.x; int cg=tid%NCG, chunk=tid/NCG;
    int c0=cg*V, t0=chunk*CHUNK, tend=min(t0+CHUNK,T);
    if(chunk>=NCHUNK) return;
    extern __shared__ __nv_bfloat16 ring[];
    __nv_bfloat16* myring = ring + (long)threadIdx.x*DEPTH*V;
    __nv_bfloat162 wj[4][L], bs[L];
    #pragma unroll
    for(int l=0;l<L;l++){
        #pragma unroll
        for(int j=0;j<4;j++) wj[j][l]=__halves2bfloat162(w[(c0+2*l)*4+j], w[(c0+2*l+1)*4+j]);
        bs[l]=__halves2bfloat162(bias[c0+2*l], bias[c0+2*l+1]);
    }
    // taps for t0-3..t0-1: from state (chunk 0, has_init) else prior x tokens (else zero).
    __nv_bfloat162 tap0[L],tap1[L],tap2[L];
    auto ldx=[&](int t,__nv_bfloat162* d){ VT f=*reinterpret_cast<const VT*>(x+(long)t*dim+c0);
        const __nv_bfloat162* h=reinterpret_cast<const __nv_bfloat162*>(&f);
        #pragma unroll
        for(int l=0;l<L;l++)d[l]=h[l]; };
    auto lds=[&](int i,__nv_bfloat162* d){ const __nv_bfloat16* p=state+(long)i*dim+c0; // state (3,dim) dim-contig
        #pragma unroll
        for(int l=0;l<L;l++)d[l]=__halves2bfloat162(p[2*l],p[2*l+1]); };
    auto zero=[&](__nv_bfloat162* d){
        #pragma unroll
        for(int l=0;l<L;l++)d[l]=__float2bfloat162_rn(0.f); };
    if(t0==0){ if(has_init){ lds(0,tap0); lds(1,tap1); lds(2,tap2);} else { zero(tap0); zero(tap1); zero(tap2);} }
    else { ldx(t0-3,tap0); ldx(t0-2,tap1); ldx(t0-1,tap2); }
    #pragma unroll
    for(int s=0;s<DEPTH;s++){ int t=t0+s; if(t<tend) __pipeline_memcpy_async(myring+s*V, x+(long)t*dim+c0, V*2); __pipeline_commit(); }
    for(int t=t0;t<tend;t++){
        int slot=(t-t0)%DEPTH; __pipeline_wait_prior(DEPTH-1);
        __nv_bfloat162 cur[L]; { VT f=*reinterpret_cast<const VT*>(myring+slot*V); const __nv_bfloat162* h=reinterpret_cast<const __nv_bfloat162*>(&f);
            #pragma unroll
            for(int l=0;l<L;l++)cur[l]=h[l]; }
        int tn=t+DEPTH; if(tn<tend){ __pipeline_memcpy_async(myring+slot*V, x+(long)tn*dim+c0, V*2);} __pipeline_commit();
        VT of; __nv_bfloat162* ob=reinterpret_cast<__nv_bfloat162*>(&of);
        #pragma unroll
        for(int l=0;l<L;l++){
            __nv_bfloat162 acc=__hfma2(wj[0][l],tap0[l],__hfma2(wj[1][l],tap1[l],__hfma2(wj[2][l],tap2[l],__hmul2(wj[3][l],cur[l]))));
            acc=__hadd2(acc,bs[l]);
            ob[l]=__halves2bfloat162(__float2bfloat16(silu_t(__low2float(acc))), __float2bfloat16(silu_t(__high2float(acc))));
            tap0[l]=tap1[l]; tap1[l]=tap2[l]; tap2[l]=cur[l];
        }
        *reinterpret_cast<VT*>(out+(long)t*dim+c0)=of;
    }
}
// best config from the autotune sweep: V=4, block=128, CHUNK=64, DEPTH=6, MINB=8
torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor state, int has_init){
    int dim=x.size(0), T=x.size(1); auto out=torch::empty_like(x);
    const int V=4, BLK=128, CHUNK=64, DEPTH=6;
    long nth=(long)(dim/V)*((T+CHUNK-1)/CHUNK); long grid=(nth+BLK-1)/BLK;
    size_t sh=(size_t)BLK*DEPTH*V*2;
    conv_fwd<V,BLK,CHUNK,DEPTH,8><<<grid,BLK,sh>>>((const __nv_bfloat16*)x.data_ptr(),(const __nv_bfloat16*)w.data_ptr(),
        (const __nv_bfloat16*)bias.data_ptr(),(const __nv_bfloat16*)state.data_ptr(),(__nv_bfloat16*)out.data_ptr(),dim,T,has_init);
    return out;
}
'''
_m = load_inline(name="conv_fast_bf162", cpp_sources="torch::Tensor run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);",
                 cuda_sources=_SRC, functions=["run"], extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100a"], verbose=False)

def causal_conv1d_run(x, weight, bias, initial_state=None, activation="silu"):
    # production layout is channel-contiguous (dim stride 1). The atrex-bench harness input.py makes
    # token-contiguous x (stride(1)==1) — convert so this stays a correct drop-in either way. (On the
    # real channel-contiguous production input this branch is a no-op; see __main__ for the real bench.)
    if x.stride(0) != 1:
        x = x.t().contiguous().t()      # (dim,T) with dim stride 1
    if initial_state is not None:
        s = initial_state.squeeze(0) if initial_state.dim() == 3 else initial_state  # (dim, w-1) or (w-1, dim)
        # want state as (3, dim) dim-contiguous: elem(i,d)=state[i*dim+d]
        if s.shape[0] == x.shape[0]:      # (dim, w-1) -> transpose to (w-1, dim)
            s = s.transpose(0, 1).contiguous()
        else:
            s = s.contiguous()
        return _m.run(x, weight, bias, s, 1)
    dummy = torch.empty(1, device=x.device, dtype=x.dtype)
    return _m.run(x, weight, bias, dummy, 0)

def run(x, weight, bias):
    dummy = torch.empty(1, device=x.device, dtype=x.dtype)
    return _m.run(x, weight, bias, dummy, 0)

if __name__ == "__main__":
    import torch.nn.functional as F, sys, triton
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 12288
    wd = 4; torch.manual_seed(0)
    x = (torch.randn(T, dim, device="cuda", dtype=torch.bfloat16) * 0.1).transpose(0, 1)
    weight = torch.randn(dim, wd, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1
    seq = F.pad(x.unsqueeze(0).float(), (wd - 1, 0))
    ref = F.silu(F.conv1d(seq, weight.unsqueeze(1).float(), bias.float(), groups=dim)[:, :, -T:]).squeeze(0)
    act = run(x, weight, bias); torch.cuda.synchronize()
    rel = (act.float() - ref).norm().item() / ref.norm().item()
    us = triton.testing.do_bench(lambda: run(x, weight, bias), warmup=25, rep=100, return_mode="median") * 1e3
    sol = 2 * dim * T * 2 / 8e12 * 1e6
    print(f"CONV-FAST dim={dim} T={T} rel_l2={rel:.2e} us={us:.2f} SOL={sol:.1f}us pct_SOL={100*sol/us:.1f}%")
