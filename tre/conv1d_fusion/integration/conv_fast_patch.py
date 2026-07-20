"""Apply / revert the env-gated conv_fast fast-path into vLLM causal_conv1d.py.

  python conv_fast_patch.py apply     # backup -> .convfastorig, install ext, insert gate
  python conv_fast_patch.py revert    # restore .convfastorig
  python conv_fast_patch.py status

The gate fires only when env CONV1D_FAST=1 AND the narrow guard in conv_fast_ext passes;
otherwise stock runs. Idempotent.
"""
import os, shutil, sys, importlib.util

OPS = os.path.join(
    os.path.dirname(importlib.util.find_spec("vllm").origin),
    "model_executor/layers/mamba/ops")
CC = os.path.join(OPS, "causal_conv1d.py")
BAK = CC + ".convfastorig"
EXT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_fast_ext.py")
EXT_DST = os.path.join(OPS, "conv_fast_ext.py")

MARK = "# --- conv_fast gate (CONV1D_FAST) ---"
ANCHOR = '    if isinstance(activation, bool) and activation:\n        activation = "silu"\n'
GATE = (ANCHOR +
        f"\n    {MARK}\n"
        "    import os as _os\n"
        '    if _os.environ.get("CONV1D_FAST") == "1":\n'
        "        try:\n"
        "            from vllm.model_executor.layers.mamba.ops import conv_fast_ext as _cfe\n"
        "            _r = _cfe.try_fast_prefill(x, weight, bias, conv_states,\n"
        "                                       query_start_loc, cache_indices,\n"
        "                                       has_initial_state, activation)\n"
        "            if _r is not None:\n"
        "                return _r.to(x.dtype)\n"
        "        except Exception:\n"
        "            pass\n")


def apply():
    if not os.path.exists(BAK):
        shutil.copy2(CC, BAK)
        print(f"backed up -> {BAK}")
    src = open(BAK).read()          # always patch from the pristine backup
    if ANCHOR not in src:
        print("!! anchor not found; abort"); sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(f"!! anchor not unique ({src.count(ANCHOR)}); abort"); sys.exit(1)
    patched = src.replace(ANCHOR, GATE, 1)
    open(CC, "w").write(patched)
    shutil.copy2(EXT_SRC, EXT_DST)
    print(f"installed ext -> {EXT_DST}")
    print(f"patched gate  -> {CC}")


def revert():
    if os.path.exists(BAK):
        shutil.copy2(BAK, CC)
        print(f"restored {CC} from backup")
    else:
        print("no backup; nothing to revert")
    if os.path.exists(EXT_DST):
        os.remove(EXT_DST); print(f"removed {EXT_DST}")


def status():
    print("CC   :", CC)
    print("bak  :", os.path.exists(BAK))
    print("ext  :", os.path.exists(EXT_DST))
    print("gated:", MARK in open(CC).read())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"apply": apply, "revert": revert, "status": status}[cmd]()
