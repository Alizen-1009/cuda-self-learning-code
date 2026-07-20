# Provenance — gdn_prefill_sm100 (vendored SM100 GDN prefill CuTe kernel)

## Source
- **Package:** flashinfer **0.6.9** (tag `v0.6.9`)
- **Path:** `flashinfer/gdn_kernels/blackwell/`
- **License:** Apache-2.0 (© 2025 FlashInfer team)
- **Note:** based on 0.6.9 after an A/B speed test against the 0.6.14
  kernel (in the atrex bench, fi 0.6.9 beat the 0.6.14 kernel on B300; we run
  0.6.9 under the SAME launcher to isolate kernel-version vs wrapper effects).
  0.6.9 already carries the cutlass-dsl >=4.4.2 `TmaInfo` compat shim, so it
  compiles on the 4.6.0.dev0 backend. It LACKS later fixes (#3581 SM100 hang,
  #3536 zero-len seq, #3715 fp8/fp16 state, #3742 ~20-25% perf) — see history.

## Vendored files
- `gated_delta_net_chunked.py`        ← `blackwell/gated_delta_net_chunked.py` (locally optimized; see below)
- `gated_delta_net_tile_scheduler.py` ← `blackwell/gated_delta_net_tile_scheduler.py` (byte-verbatim)
- `gdn_prefill.py`                    ← `blackwell/gdn_prefill.py` (local host-configuration cleanup)

They import only `torch`, `cutlass.*`, and `cuda.bindings.driver` plus
intra-package relative imports (`.gated_delta_net_chunked`,
`.gated_delta_net_tile_scheduler`) — **zero `flashinfer` imports**. The 0.6.9
launcher obtains active-cluster information from `cutlass.utils.HardwareInfo()`.

## Local changes
- `gated_delta_net_chunked.py` — V3 coalesced recurrent-state GMEM transfers and
  the v21 state-input R2T hoist (`915753cf`), measured at 112.8 TFLOPS / 2018
  GB/s on B300. Fixed SM100-only accumulator/tile/persistent configuration is
  kept inside the class instead of being threaded through its constructor.
- `gdn_prefill.py` — removes redundant fixed configuration and uses the active
  cluster count directly for persistent-CTA workspace sizing.
- `__init__.py` — re-exports `chunk_gated_delta_rule_sm100` with an atrex-style
  guarded import (`(ImportError, RuntimeError)` → `_has_sm100_prefill_dsl =
  False`). Not a kernel change.

## Gate convention
The vendored kernel consumes the forget gate in **LINEAR** space (it computes
`cumsumlog = sum log(gate)` internally). The launcher passes linear gate
directly — no `log()` wrapper (unlike the previous 0.6.7 kernel, which consumed
log-space gates).

## Version selection history
The 0.6.7 `blackwell_prefill` kernel was vendored because the then-current
0.6.12 GDN kernel failed to `cute.compile` on the cutlass-dsl 4.5.x backend
(MLIR legalization ICE at `tcgen05.make_tmem_copy`). The 0.6.14 kernel declares
`nvidia-cutlass-dsl>=4.5.0` and carries explicit `>=4.4.2` compatibility shims
(e.g. the `TmaInfo` replacement), is ~2.3× faster than 0.6.7, and adds native
checkpoint + fp8/fp16-state support. atrex already depends on
`nvidia-cutlass-dsl`, so this adds **no new dependency**.

The kernel was subsequently pinned back to the faster 0.6.9 base under the
same atrex launcher (`c3b0edf1`), then received the local optimizations listed
above. It must therefore be treated as a maintained fork, not a byte-verbatim
upstream snapshot.

## Consumer
`python/atrex/api/chunk_gdn_prefill_sm100.py` (the chunk-parallel launcher)
imports `chunk_gated_delta_rule_sm100` from here and runs the chunk-parallel
single-pass scan + fused Triton correction on top of it.
