"""conv1d(width-4)+SiLU + split(q/k/v rearrange), NO l2norm — matches the mentor's 72us
`fused_gdn_kernel_cutlass`. SGLang does l2norm INSIDE the GDN core (use_qk_l2norm_in_kernel=True),
so preprocessing is just conv+split = memory-bound conv_fast work over all 12288 channels,
writing directly to q/k/v [1,T,H,d] layout. Floor ~69us.
"""
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
typedef float2 VT;
__device__ __forceinline__ float silu_t(float a){ float t; asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(0.5f*a)); return 0.5f*a*(1.f+t); }

// conv_fast over ALL channels; route each group's write to q/k/v by channel range. No l2norm.
template<int V,int BLK,int CHUNK,int DEPTH,int MINB>
__global__ void __launch_bounds__(BLK,MINB) conv_split(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ qo, __nv_bfloat16* __restrict__ ko, __nv_bfloat16* __restrict__ vo,
    int DIM, int T, int KEYDIM, int VDIM){
  const int L=V/2; const int NCG=DIM/V; const int NCHUNK=(T+CHUNK-1)/CHUNK;
  long tid=(long)blockIdx.x*BLK+threadIdx.x; int cg=tid%NCG, chunk=tid/NCG;
  int c0=cg*V, t0=chunk*CHUNK, tend=min(t0+CHUNK,T);
  if(chunk>=NCHUNK) return;
  // route: which output buffer + base index for this group (groups are 4-aligned; KEYDIM,2*KEYDIM mult of 4)
  __nv_bfloat16* obuf; int obase; int ostride;
  if(c0<KEYDIM){ obuf=qo; obase=c0; ostride=KEYDIM; }
  else if(c0<2*KEYDIM){ obuf=ko; obase=c0-KEYDIM; ostride=KEYDIM; }
  else { obuf=vo; obase=c0-2*KEYDIM; ostride=VDIM; }
  extern __shared__ __nv_bfloat16 ring[];
  __nv_bfloat16* myr=ring+(long)threadIdx.x*DEPTH*V;
  __nv_bfloat162 wj[4][L],bs[L];
  #pragma unroll
  for(int l=0;l<L;l++){
    #pragma unroll
    for(int j=0;j<4;j++) wj[j][l]=__halves2bfloat162(w[(c0+2*l)*4+j],w[(c0+2*l+1)*4+j]);
    bs[l]=__halves2bfloat162(bias[c0+2*l],bias[c0+2*l+1]);
  }
  __nv_bfloat162 t0v[L],t1v[L],t2v[L];
  auto ldx=[&](int t,__nv_bfloat162*d){ VT f=*reinterpret_cast<const VT*>(x+(long)t*DIM+c0); const __nv_bfloat162* hh=reinterpret_cast<const __nv_bfloat162*>(&f);
    #pragma unroll
    for(int l=0;l<L;l++)d[l]=hh[l]; };
  auto zero=[&](__nv_bfloat162*d){
    #pragma unroll
    for(int l=0;l<L;l++)d[l]=__float2bfloat162_rn(0.f); };
  if(t0==0){ zero(t0v);zero(t1v);zero(t2v);} else { ldx(t0-3,t0v);ldx(t0-2,t1v);ldx(t0-1,t2v);}
  #pragma unroll
  for(int s=0;s<DEPTH;s++){ int t=t0+s; if(t<tend) __pipeline_memcpy_async(myr+s*V,x+(long)t*DIM+c0,V*2); __pipeline_commit(); }
  for(int t=t0;t<tend;t++){
    int slot=(t-t0)%DEPTH; __pipeline_wait_prior(DEPTH-1);
    __nv_bfloat162 cur[L]; { VT f=*reinterpret_cast<const VT*>(myr+slot*V); const __nv_bfloat162* hh=reinterpret_cast<const __nv_bfloat162*>(&f);
      #pragma unroll
      for(int l=0;l<L;l++)cur[l]=hh[l]; }
    int tn=t+DEPTH; if(tn<tend) __pipeline_memcpy_async(myr+slot*V,x+(long)tn*DIM+c0,V*2); __pipeline_commit();
    VT of; __nv_bfloat162* ob=reinterpret_cast<__nv_bfloat162*>(&of);
    #pragma unroll
    for(int l=0;l<L;l++){
      __nv_bfloat162 acc=__hadd2(__hfma2(wj[0][l],t0v[l],__hfma2(wj[1][l],t1v[l],__hfma2(wj[2][l],t2v[l],__hmul2(wj[3][l],cur[l])))),bs[l]);
      ob[l]=__halves2bfloat162(__float2bfloat16(silu_t(__low2float(acc))),__float2bfloat16(silu_t(__high2float(acc))));
      t0v[l]=t1v[l];t1v[l]=t2v[l];t2v[l]=cur[l];
    }
    *reinterpret_cast<VT*>(obuf+(long)t*ostride+obase)=of;
  }
}
template<int CHUNK,int DEPTH,int MINB>
static void launch(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor qo,torch::Tensor ko,torch::Tensor vo,int DIM,int T,int KEYDIM,int VDIM){
  const int V=4,BLK=128;
  long nth=(long)(DIM/V)*((T+CHUNK-1)/CHUNK); long grid=(nth+BLK-1)/BLK; size_t sh=(size_t)BLK*DEPTH*V*2;
  conv_split<V,BLK,CHUNK,DEPTH,MINB><<<grid,BLK,sh>>>((const __nv_bfloat16*)x.data_ptr(),(const __nv_bfloat16*)w.data_ptr(),
      (const __nv_bfloat16*)bias.data_ptr(),(__nv_bfloat16*)qo.data_ptr(),(__nv_bfloat16*)ko.data_ptr(),(__nv_bfloat16*)vo.data_ptr(),DIM,T,KEYDIM,VDIM);
}
std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor bias, int NHQK, int NHV, int HD, int chunk, int depth){
  int DIM=x.size(0),T=x.size(1); int KEYDIM=NHQK*HD, VDIM=NHV*HD;
  auto opt=torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());
  auto qo=torch::empty({1,T,NHQK,HD},opt), ko=torch::empty({1,T,NHQK,HD},opt), vo=torch::empty({1,T,NHV,HD},opt);
  if(chunk==64&&depth==6) launch<64,6,8>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else if(chunk==96&&depth==8) launch<96,8,6>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else if(chunk==128&&depth==8) launch<128,8,4>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else if(chunk==128&&depth==12) launch<128,12,4>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else if(chunk==256&&depth==8) launch<256,8,4>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else if(chunk==64&&depth==8) launch<64,8,8>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  else launch<64,6,8>(x,w,bias,qo,ko,vo,DIM,T,KEYDIM,VDIM);
  return {qo,ko,vo};
}
'''
_m = load_inline(name="conv_split_ks", cpp_sources="std::vector<torch::Tensor> run(torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,int,int);",
                 cuda_sources=_SRC, functions=["run"], extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100a"], verbose=False)

def conv_split(x, w, b, NHQK=16, NHV=64, HD=128, chunk=64, depth=6):
    return _m.run(x, w, b, NHQK, NHV, HD, chunk, depth)

if __name__ == "__main__":
    import os, sys, triton, torch.nn.functional as F
    NHQK,NHV,HD,WD=16,64,128,4; DIM=2*NHQK*HD+NHV*HD; KEYDIM=NHQK*HD
    T=int(sys.argv[1]) if len(sys.argv)>1 else 8192
    torch.manual_seed(0)
    x=(torch.randn(T,DIM,device="cuda",dtype=torch.bfloat16)*0.05).transpose(0,1)
    w=(torch.randn(DIM,WD,device="cuda",dtype=torch.bfloat16)*0.1).contiguous()
    b=torch.randn(DIM,device="cuda",dtype=torch.bfloat16)*0.1
    # reference: silu(conv) split to q/k/v (NO l2norm)
    seq=F.pad(x.unsqueeze(0).float(),(WD-1,0))
    mixed=F.silu(F.conv1d(seq,w.unsqueeze(1).float(),b.float(),groups=DIM)[:,:,-T:]).squeeze(0).transpose(0,1)  # (T,dim)
    rq=mixed[:,:KEYDIM].reshape(T,NHQK,HD); rk=mixed[:,KEYDIM:2*KEYDIM].reshape(T,NHQK,HD); rv=mixed[:,2*KEYDIM:].reshape(T,NHV,HD)
    q,k,v=conv_split(x,w,b); torch.cuda.synchronize()
    rel=lambda a,r:(a.squeeze(0).float()-r.float()).norm().item()/r.float().norm().item()
    print(f"rel q={rel(q,rq):.2e} k={rel(k,rk):.2e} v={rel(v,rv):.2e}")
    sol=2*DIM*T*2/8e12*1e6
    for ch,dp in [(64,6),(64,8),(96,8),(128,8),(128,12),(256,8)]:
        us=triton.testing.do_bench(lambda: conv_split(x,w,b,chunk=ch,depth=dp),warmup=25,rep=100,return_mode="median")*1e3
        print(f"  CONV+SPLIT chunk={ch} depth={dp}: {us:.2f}us pct_SOL={100*sol/us:.1f}%")
