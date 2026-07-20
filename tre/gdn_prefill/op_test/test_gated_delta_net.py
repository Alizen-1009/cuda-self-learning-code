import atrex
import triton
import triton.language as tl
import pytest
import torch
import torch.nn.functional as F


@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold,
                          (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](g,
                                  A_log,
                                  a,
                                  dt_bias,
                                  seq_len,
                                  num_heads,
                                  beta,
                                  threshold,
                                  8,
                                  num_warps=1)
    return g


def fix_query_key_value_ordering(
    mixed_qkvz,
    mixed_ba,
    tp_size
):
    """
    Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
    """
    num_v_heads = 32
    num_k_heads = 16
    head_v_dim = 128
    head_k_dim = 128
    # mixed_qkvz shape: [T, 6144]->[T, 8, 768]
    # mixed_qkvz -> q, k, v, z = [T, 8, 128], [T, 8, 128], [T, 8, 256], [T, 8, 256]
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        num_k_heads // tp_size,
        (head_k_dim + head_k_dim +
            (head_v_dim + head_v_dim) * num_v_heads //
            num_k_heads),
    )
    # mixed_ba shape: [T, 32]->[T, 8, 4]
    # mixed_ba -> b, a = [T, 8, 2], [T, 8, 2]
    new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
        num_k_heads // tp_size,
        2 * num_v_heads // num_k_heads,
    )

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        (num_v_heads // num_k_heads * head_v_dim),
        (num_v_heads // num_k_heads * head_v_dim),
    ]
    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads
    ]
    # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
    # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
    #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
    (query, key, value, z) = torch.split(mixed_qkvz,
                                         split_arg_list_qkvz,
                                         dim=2)
    (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

    # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
    value = value.reshape(value.size(0), -1, head_v_dim)
    z = z.reshape(z.size(0), -1, head_v_dim)
    b = b.reshape(b.size(0), num_v_heads // tp_size)
    a = a.reshape(a.size(0), num_v_heads // tp_size)
    torch_q = query.reshape((query.shape[0], -1))
    torch_k = key.reshape((key.shape[0], -1))
    torch_v = value.reshape((value.shape[0], -1))
    qkv = torch.cat((torch_q, torch_k, torch_v), dim=-1)
    return qkv, z, b, a


@pytest.mark.parametrize("m", [1, 4, 6, 8, 12, 16, 120, 512])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("n", [[6144, 32], [3072, 16], [1536, 8]])
@pytest.mark.parametrize("backend", ["gluon", "flydsl"])
def test_gdn_pre(m, k, n, backend):
    M = m
    K = k
    N1, N2 = n
    tp_size = 64 // N2
    num_heads = N1 // 768
    num_vz_heads = num_heads * 2
    qkv_dim: tl.constexpr = 512  # q(128) + k(128) + v(128 * 2)
    vz_dim: tl.constexpr = 128
    b_dim: tl.constexpr = 2
    a_dim: tl.constexpr = 2

    Nqkv = num_heads * qkv_dim
    Nz = num_vz_heads * vz_dim
    Nb = num_heads * b_dim
    Na = num_heads * a_dim
    hidden_states = torch.rand((M, K), dtype=torch.bfloat16).cuda() - 0.5
    in_qkvz_proj = torch.rand((N1, K), dtype=torch.bfloat16).cuda() - 0.5
    in_ba_proj = torch.rand((N2, K), dtype=torch.bfloat16).cuda() - 0.5
    A_log = torch.rand((N2 // 2,), dtype=torch.float32).cuda() - 0.5
    dt_bias = torch.rand((N2 // 2,), dtype=torch.bfloat16).cuda() - 0.5

    if backend == "gluon":
        qkv, z, b, g = atrex.gated_delta_net_pre(hidden_states, in_qkvz_proj, in_ba_proj, A_log, dt_bias)
    else:
        ctx = atrex.gated_delta_net_pre_flydsl_build(M, K, num_heads, in_qkvz_proj, in_ba_proj, A_log, dt_bias)
        qkv, z, b, g = atrex.gated_delta_net_pre_flydsl(ctx, hidden_states)

    z = z.reshape(z.size(0), -1, vz_dim)
    mixed_qkvz = torch.matmul(hidden_states, in_qkvz_proj.T)
    mixed_ba = torch.matmul(hidden_states, in_ba_proj.T)
    torch_qkv, torch_z, torch_beta, torch_alpha = fix_query_key_value_ordering(mixed_qkvz, mixed_ba, tp_size)
    torch_g = fused_gdn_gating(A_log, torch_alpha, dt_bias)
    torch_beta = torch_beta.sigmoid()

    assert torch.allclose(qkv, torch_qkv, atol=0, rtol=0)
    assert torch.allclose(z, torch_z, atol=0, rtol=0)
    assert torch.allclose(b, torch_beta, atol=0, rtol=0)
    assert torch.allclose(g, torch_g, atol=1e-3, rtol=1e-3)


def test_gdn_time():
    K = 2048
    quantiles = [0.5, 0.2, 0.8]
    for M in [1, 4, 6, 8, 12, 16, 120, 512]:
        for N1,N2 in[[6144, 32], [3072, 16], [1536, 8]]:
            print(f"M,K,N1,N2 = {M},{K},{N1},{N2}")

            hidden_states = torch.rand((M, K), dtype=torch.bfloat16).cuda() - 0.5
            in_qkvz_proj = torch.rand((N1, K), dtype=torch.bfloat16).cuda() - 0.5
            in_ba_proj = torch.rand((N2, K), dtype=torch.bfloat16).cuda() - 0.5
            A_log = torch.rand((N2 // 2,), dtype=torch.bfloat16).cuda() - 0.5
            dt_bias = torch.rand((N2 // 2,), dtype=torch.bfloat16).cuda() - 0.5
            triton_ms, _, _ = triton.testing.do_bench(lambda: atrex.gated_delta_net_pre(hidden_states, in_qkvz_proj, in_ba_proj, A_log, dt_bias), quantiles=quantiles)
            print(f"triton_ms: {triton_ms}")
            num_heads = N1 // 768
            ctx = atrex.gated_delta_net_pre_flydsl_build(M, K, num_heads, in_qkvz_proj, in_ba_proj, A_log, dt_bias)
            triton_ms, _, _ = triton.testing.do_bench(lambda: atrex.gated_delta_net_pre_flydsl(ctx, hidden_states), quantiles=quantiles)
            print(f"flydsl_ms: {triton_ms}")

if __name__ == "__main__":
    test_gdn_time()

'''
MI308X perf
| M | K | N1 | N2 | triton_ms | flydsl_ms | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 6144 | 32 | 0.04376000165939331 | 0.03643999993801117 | 1.201 |
| 1 | 2048 | 3072 | 16 | 0.03311999887228012 | 0.023399999365210533 | 1.415 |
| 1 | 2048 | 1536 | 8 | 0.030319999903440475 | 0.01899999938905239 | 1.596 |
| 4 | 2048 | 6144 | 32 | 0.0408799983561039 | 0.03675999864935875 | 1.112 |
| 4 | 2048 | 3072 | 16 | 0.033160001039505005 | 0.025520000606775284 | 1.299 |
| 4 | 2048 | 1536 | 8 | 0.03136000037193298 | 0.020600000396370888 | 1.522 |
| 6 | 2048 | 6144 | 32 | 0.039319999516010284 | 0.041439998894929886 | 0.949 |
| 6 | 2048 | 3072 | 16 | 0.03415999934077263 | 0.02419999986886978 | 1.412 |
| 6 | 2048 | 1536 | 8 | 0.03203999996185303 | 0.020600000396370888 | 1.555 |
| 8 | 2048 | 6144 | 32 | 0.04252000153064728 | 0.04472000151872635 | 0.951 |
| 8 | 2048 | 3072 | 16 | 0.031599998474121094 | 0.02576100081205368 | 1.227 |
| 8 | 2048 | 1536 | 8 | 0.03215999901294708 | 0.019200000911951065 | 1.675 |
| 12 | 2048 | 6144 | 32 | 0.04323999956250191 | 0.04659999907016754 | 0.928 |
| 12 | 2048 | 3072 | 16 | 0.03003999963402748 | 0.025961000472307205 | 1.157 |
| 12 | 2048 | 1536 | 8 | 0.03223999962210655 | 0.01940000057220459 | 1.662 |
| 16 | 2048 | 6144 | 32 | 0.04284000024199486 | 0.046720001846551895 | 0.917 |
| 16 | 2048 | 3072 | 16 | 0.03452000021934509 | 0.02696000039577484 | 1.280 |
| 16 | 2048 | 1536 | 8 | 0.03200000151991844 | 0.01955999992787838 | 1.636 |
| 120 | 2048 | 6144 | 32 | 0.1783200055360794 | 0.10075999796390533 | 1.770 |
| 120 | 2048 | 3072 | 16 | 0.09564000368118286 | 0.05979999899864197 | 1.599 |
| 120 | 2048 | 1536 | 8 | 0.06347999721765518 | 0.04820000007748604 | 1.317 |
| 512 | 2048 | 6144 | 32 | 0.6577609777450562 | 0.24328100681304932 | 2.704 |
| 512 | 2048 | 3072 | 16 | 0.3361999988555908 | 0.1409199982881546 | 2.386 |
| 512 | 2048 | 1536 | 8 | 0.17815999686717987 | 0.09324000030755997 | 1.911 |
'''