"""Width-4 causal depthwise conv1d + SiLU for Qwen3.5-plus GDN prefill (B300 / SM100).

Channel-contiguous bf16, single sequence. ~67% SOL on the production shape
(dim=12288, T=8192): 74.7us standalone (1.6x over the Triton stock kernel), 1.79x in-model.
This is at ~91% of the card's achievable HBM copy ceiling (~73% SOL); it is memory-bound,
so there is no further standalone headroom (see the op_test bench).
"""
from typing import Optional

import torch
from torch import Tensor

from ..core.compile_cu import cuda_kernel


@cuda_kernel()
def _causal_conv1d_fwd_kernel(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    state: Tensor,
    out: Tensor,
    has_init: int,
) -> None: ...


def can_use_causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation="silu",
) -> bool:
    """Narrow guard: only the exact regime this kernel is built for.

    SM100/Blackwell, bf16, width==4, channel-contiguous x (dim stride 1). Callers should
    fall back to their stock kernel when this returns False.
    """
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability(x.device)[0] != 10:  # SM100
            return False
        if activation not in ("silu", "swish", True):
            return False
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(0) != 1:
            return False
        if weight is None or weight.dim() != 2 or weight.shape[1] != 4:
            return False
        return True
    except Exception:
        return False


def causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
) -> Tensor:
    """Causal depthwise conv1d (width 4) + SiLU on channel-contiguous bf16 input.

    Args:
        x:             (dim, T) bf16, channel-contiguous (``x.stride(0) == 1``).
        weight:        (dim, 4) bf16 conv weight.
        bias:          (dim,) bf16, or None.
        initial_state: optional (dim, 3) or (3, dim) or (1, dim, 3) initial taps; when
                       None the kernel assumes a fresh sequence (zero-padded left).

    Returns:
        out: (dim, T) bf16, silu(conv1d(x)).
    """
    assert x.dim() == 2 and x.stride(0) == 1, "x must be (dim, T) channel-contiguous (dim stride 1)"
    assert weight.dim() == 2 and weight.shape[1] == 4, "weight must be (dim, 4)"
    dim, _ = x.shape
    w = weight if weight.is_contiguous() else weight.contiguous()
    b = bias if bias is not None else torch.zeros(dim, device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)

    if initial_state is not None:
        s = initial_state
        if s.dim() == 3:
            s = s.squeeze(0)
        # kernel wants state as (3, dim) dim-contiguous: elem(i, d) = state[i*dim + d]
        s = s.transpose(0, 1).contiguous() if s.shape[0] == dim else s.contiguous()
        _causal_conv1d_fwd_kernel(x, w, b, s, out, 1)
    else:
        dummy = torch.empty(1, device=x.device, dtype=x.dtype)
        _causal_conv1d_fwd_kernel(x, w, b, dummy, out, 0)
    return out
