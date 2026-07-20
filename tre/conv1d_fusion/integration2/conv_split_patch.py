"""Apply/revert env-gated conv_split fusion into vLLM qwen3_next.py (CONV_SPLIT_FUSE=1).

Point A (conv call): when gated+eligible, SKIP causal_conv1d_fn (keep mixed_qkv_non_spec pre-conv,
do conv_states writeback), set _cs_fuse.
Point B (split): when _cs_fuse, run conv_split (conv+SiLU+split+rearrange) + torch l2norm(q,k)
instead of fused_conv_split_l2norm_rearrange. Op order/gating/l2norm-as-separate preserved.

  python conv_split_patch.py apply | revert | status
"""
import os, shutil, sys, importlib.util

OPS = os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin),
                   "model_executor/layers/mamba/ops")
QN = os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin),
                  "model_executor/models/qwen3_next.py")
BAK = QN + ".convsplitorig"
EXT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_split_ext.py")
EXT_DST = os.path.join(OPS, "conv_split_ext.py")

A_ANCHOR = '''            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)'''

A_NEW = '''            import os as _os
            _cs_fuse = False
            if _os.environ.get("CONV_SPLIT_FUSE") == "1":
                try:
                    from vllm.model_executor.layers.mamba.ops import conv_split_ext as _cse
                    if _cse.gate(conv_weights, non_spec_query_start_loc, has_initial_state, mixed_qkv_non_spec_T):
                        _cse.writeback(conv_state, non_spec_state_indices_tensor, mixed_qkv_non_spec_T)
                        _cs_fuse = True  # keep mixed_qkv_non_spec PRE-conv; conv done at split point
                except Exception:
                    _cs_fuse = False
            if not _cs_fuse:
                mixed_qkv_non_spec = causal_conv1d_fn(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)'''

B_ANCHOR = '''            query_non_spec, key_non_spec, value_non_spec = (
                fused_conv_split_l2norm_rearrange(
                    mixed_qkv_non_spec,
                    self.num_k_heads // self.tp_size,
                    self.num_v_heads // self.tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
            non_spec_l2norm_done = True'''

B_NEW = '''            if _cs_fuse:
                from vllm.model_executor.layers.mamba.ops import conv_split_ext as _cse
                query_non_spec, key_non_spec, value_non_spec = _cse.fused_conv_split(
                    mixed_qkv_non_spec.transpose(0, 1), conv_weights, self.conv1d.bias,
                    self.num_k_heads // self.tp_size, self.num_v_heads // self.tp_size,
                    self.head_k_dim, do_l2norm=False)  # l2norm applied AFTER gating
            else:
                query_non_spec, key_non_spec, value_non_spec = (
                    fused_conv_split_l2norm_rearrange(
                        mixed_qkv_non_spec,
                        self.num_k_heads // self.tp_size,
                        self.num_v_heads // self.tp_size,
                        self.head_k_dim,
                        self.head_v_dim,
                    )
                )
            non_spec_l2norm_done = True'''


C_ANCHOR = '''        g, beta = fused_gdn_gating(self.A_log, a.contiguous(), b, self.dt_bias)'''

C_NEW = '''        g, beta = fused_gdn_gating(self.A_log, a.contiguous(), b, self.dt_bias)
        if locals().get("_cs_fuse", False) and query_non_spec is not None:
            from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd as _l2fwd
            query_non_spec = _l2fwd(query_non_spec)
            key_non_spec = _l2fwd(key_non_spec)'''


def apply():
    if not os.path.exists(BAK):
        shutil.copy2(QN, BAK); print("backed up ->", BAK)
    src = open(BAK).read()
    for name, anc in (("A", A_ANCHOR), ("B", B_ANCHOR), ("C", C_ANCHOR)):
        if src.count(anc) != 1:
            print(f"!! anchor {name} count={src.count(anc)} (need 1); abort"); sys.exit(1)
    src = src.replace(A_ANCHOR, A_NEW, 1).replace(B_ANCHOR, B_NEW, 1).replace(C_ANCHOR, C_NEW, 1)
    open(QN, "w").write(src)
    shutil.copy2(EXT_SRC, EXT_DST)
    print("installed ext ->", EXT_DST); print("patched ->", QN)


def revert():
    if os.path.exists(BAK):
        shutil.copy2(BAK, QN); print("restored", QN)
    if os.path.exists(EXT_DST):
        os.remove(EXT_DST); print("removed", EXT_DST)


def status():
    s = open(QN).read()
    print("bak:", os.path.exists(BAK), "| ext:", os.path.exists(EXT_DST),
          "| gated:", "_cs_fuse" in s)


if __name__ == "__main__":
    {"apply": apply, "revert": revert, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
