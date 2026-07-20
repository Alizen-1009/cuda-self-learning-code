#pragma once

#include <torch/all.h>

// Width-4 causal depthwise conv1d + SiLU for Qwen3.5 GDN, channel-contiguous bf16.
//   x, out : (dim, T), dim-contiguous (stride(0)==1) -> elem(d,t) = ptr[t*dim + d]
//   weight : (dim, 4) row-major
//   bias   : (dim,)
//   state  : (3, dim) dim-contiguous initial taps (used only when has_init != 0)
//   out    : (dim, T) pre-allocated, written in place
void causal_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& state,
    torch::Tensor& out,
    int64_t has_init);
