"""conv_fast: env-gated fast prefill drop-in for vLLM causal_conv1d_fn (Qwen3.5 GDN).

Proven kernel from _conv1d_opt/handoff_result/conv_fast.py (74.7us / 67.4% SOL on the
production shape dim=12288,T=8192, rel_l2 2.4e-3). This module wraps it behind a NARROW
guard so it only fires on the exact production prefill regime and otherwise returns None
so the caller falls back to the stock Triton kernel.

Guard (all must hold, else -> None -> stock):
  * bf16, channel-last x (stride(0)==1), weight width==4, activation silu/swish
  * single sequence (query_start_loc == [0, T]), fresh prefill (has_initial_state all False)
  * conv_states + cache_indices present, dims match

On the fast path we also write the trailing (width-1) input tokens back into
conv_states[cache_indices[0]] to match stock cache semantics (verified in test_integration.py).
"""
import torch

_M = None


def _ext():
    global _M
    if _M is None:
        from torch.utils.cpp_extension import load_inline
        _M = load_inline(
            name="conv_fast_bf162",
            cpp_sources="torch::Tensor run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);",
            cuda_sources=_SRC, functions=["run"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a"], verbose=False)
    return _M


_SRC = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
__device__ __forceinline__ float silu_t(float a){ float t; asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(0.5f*a)); return 0.5f*a*(1.f+t); }

template<int V,int BLK,int CHUNK,int DEPTH,int MINB>
__global__ void __launch_bounds__(BLK,MINB) conv_fwd(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, int dim, int T, int has_init){
    typedef float2 VT; const int L=V/2;
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
    __nv_bfloat162 tap0[L],tap1[L],tap2[L];
    auto ldx=[&](int t,__nv_bfloat162* d){ VT f=*reinterpret_cast<const VT*>(x+(long)t*dim+c0);
        const __nv_bfloat162* h=reinterpret_cast<const __nv_bfloat162*>(&f);
        #pragma unroll
        for(int l=0;l<L;l++)d[l]=h[l]; };
    auto lds=[&](int i,__nv_bfloat162* d){ const __nv_bfloat16* p=state+(long)i*dim+c0;
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


def try_fast_prefill(x, weight, bias, conv_states, query_start_loc,
                     cache_indices, has_initial_state, activation):
    """Return out (dim, cu_seqlen) if eligible for the fast prefill path, else None."""
    try:
        if activation not in ("silu", "swish", True):
            return None
        if x.dtype != torch.bfloat16 or x.dim() != 2:
            return None
        if x.stride(0) != 1:                       # need channel-last (dim contiguous)
            return None
        if weight is None or weight.shape[1] != 4:  # width == 4
            return None
        if conv_states is None or cache_indices is None or query_start_loc is None:
            return None
        if query_start_loc.numel() != 2:           # single sequence (batch == 1)
            return None
        if has_initial_state is not None and bool(has_initial_state.to(torch.bool).any()):
            return None                            # fresh prefill only
        dim, cu = x.shape
        if cu < 4 or conv_states.dim() != 3 or conv_states.shape[1] != dim:
            return None
        s0 = int(query_start_loc[0].item()); e0 = int(query_start_loc[1].item())
        if s0 != 0 or e0 != cu:                    # whole x is the one sequence
            return None

        w = weight if weight.is_contiguous() else weight.contiguous()
        b = bias if bias is not None else torch.zeros(dim, device=x.device, dtype=x.dtype)
        dummy = torch.empty(1, device=x.device, dtype=x.dtype)
        out = _ext().run(x, w, b, dummy, 0)        # (dim, cu) bf16

        # conv_states cache writeback: trailing (width-1) input tokens, most-recent last.
        sl = conv_states.shape[2]                  # width - 1
        line = int(cache_indices[0].item())
        if 0 <= line < conv_states.shape[0]:
            conv_states[line, :, :].copy_(x[:, cu - sl:])
        return out
    except Exception:
        return None
