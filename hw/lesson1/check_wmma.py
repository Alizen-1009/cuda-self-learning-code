#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

torch = None
load = None


DEFAULT_CASES = [
    (16, 16, 16),
    (32, 32, 16),
    (64, 32, 48),
    (128, 128, 64),
    (256, 128, 256),
]


def parse_case(text):
    fields = text.lower().replace(",", "x").split("x")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("case must look like MxNxK, e.g. 128x128x64")
    try:
        m, n, k = (int(x) for x in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case dimensions must be integers") from exc
    if m <= 0 or n <= 0 or k <= 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    if m % 16 or n % 16 or k % 16:
        raise argparse.ArgumentTypeError("M, N, and K must all be multiples of 16")
    return m, n, k


def normalize_arch(arch):
    if arch == "auto":
        return None
    arch = arch.lower().removeprefix("sm_").removeprefix("compute_")
    if arch.isdigit() and len(arch) == 2:
        return f"{arch[0]}.{arch[1]}"
    return arch


def extension_name(arch):
    tag = "auto" if arch is None else arch.replace(".", "")
    return f"lesson1_wmma_sm{tag}"


def build_extension(cu_path, arch, verbose):
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch

    build_dir = cu_path.parent / ".torch_extensions" / extension_name(arch)
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=extension_name(arch),
        sources=[str(cu_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )


def make_inputs(m, n, k, dtype, device, seed, scale):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn((m, k), device=device, dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device=device, dtype=torch.float32, generator=generator)
    return (a * scale).to(dtype).contiguous(), (b * scale).to(dtype).contiguous()


def compare(actual, expected, atol, rtol):
    diff = (actual - expected).abs()
    max_abs = diff.max()
    max_rel = (diff / (expected.abs() + 1e-6)).max()
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)

    flat_idx = diff.argmax().item()
    row = flat_idx // actual.size(1)
    col = flat_idx % actual.size(1)
    return {
        "ok": bool(ok),
        "max_abs": float(max_abs.item()),
        "max_rel": float(max_rel.item()),
        "row": row,
        "col": col,
        "actual": float(actual[row, col].item()),
        "expected": float(expected[row, col].item()),
    }


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(ext, m, n, k, dtype, device, seed, scale, atol, rtol, bench):
    a, b = make_inputs(m, n, k, dtype, device, seed, scale)

    actual = ext.wmma_gemm_m16n16k16_naive(a, b)
    expected = a.float() @ b.float()
    result = compare(actual, expected, atol=atol, rtol=rtol)

    status = "PASS" if result["ok"] else "FAIL"
    print(
        f"[{status}] dtype={str(dtype).replace('torch.', ''):<8} "
        f"shape=({m},{n},{k}) "
        f"max_abs={result['max_abs']:.6g} max_rel={result['max_rel']:.6g}"
    )
    if not result["ok"]:
        print(
            "       worst "
            f"C[{result['row']},{result['col']}] "
            f"actual={result['actual']:.9g} expected={result['expected']:.9g}"
        )

    if bench:
        kernel_ms = time_cuda(lambda: ext.wmma_gemm_m16n16k16_naive(a, b), 10, 50)
        torch_ms = time_cuda(lambda: a.float() @ b.float(), 10, 50)
        print(f"       time kernel={kernel_ms:.4f} ms torch_ref={torch_ms:.4f} ms")

    return result["ok"]


def main():
    global torch, load

    parser = argparse.ArgumentParser(description="Correctness check for lesson1 WMMA GEMM")
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "all"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arch", default="auto", help="auto, 89, 90, 8.9, 9.0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch as torch_module
    from torch.utils.cpp_extension import load as load_function

    torch = torch_module
    load = load_function

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("--device must be a CUDA device, e.g. cuda or cuda:0")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    arch = normalize_arch(args.arch)
    cu_path = Path(__file__).with_name("wmma.cu")
    ext = build_extension(cu_path, arch=arch, verbose=args.verbose)

    prop = torch.cuda.get_device_properties(device)
    print(f"device={prop.name} capability=sm_{prop.major}{prop.minor}")

    dtypes = []
    if args.dtype in ("fp16", "all"):
        dtypes.append(torch.float16)
    if args.dtype in ("bf16", "all"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Current CUDA device does not support bfloat16")
        dtypes.append(torch.bfloat16)

    cases = args.cases or DEFAULT_CASES
    ok = True
    for dtype in dtypes:
        if dtype == torch.float16:
            atol = 2e-2 if args.atol is None else args.atol
            rtol = 2e-2 if args.rtol is None else args.rtol
        else:
            atol = 2e-1 if args.atol is None else args.atol
            rtol = 5e-2 if args.rtol is None else args.rtol

        for i, (m, n, k) in enumerate(cases):
            case_seed = args.seed + i
            ok &= run_case(
                ext,
                m,
                n,
                k,
                dtype=dtype,
                device=device,
                seed=case_seed,
                scale=args.scale,
                atol=atol,
                rtol=rtol,
                bench=args.bench,
            )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
