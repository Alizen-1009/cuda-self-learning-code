"""Vendored SM100 (Blackwell) Chunk-Gated-Delta-Rule prefill CuTe kernel.

Based on flashinfer 0.6.9 `flashinfer/gdn_kernels/blackwell/` (pure cutlass.*,
Apache-2.0), with documented atrex performance and host-configuration changes;
see PROVENANCE.md. Guarded import follows the atrex
`gdn_delta_rule_varlen_sm120` convention.

Exposes the in-place SM100 adapter `chunk_gated_delta_rule_sm100(q, k, v, gate,
beta, output, cu_seqlens, initial_state, output_state, scale, ...)` (gate is
LINEAR space; the kernel takes log internally).
"""
try:
    from .gdn_prefill import chunk_gated_delta_rule_sm100

    _has_sm100_prefill_dsl = True
except (ImportError, RuntimeError):
    chunk_gated_delta_rule_sm100 = None  # type: ignore
    _has_sm100_prefill_dsl = False


__all__ = [
    "chunk_gated_delta_rule_sm100",
    "_has_sm100_prefill_dsl",
]
