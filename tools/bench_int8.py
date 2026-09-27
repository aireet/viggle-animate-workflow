"""Benchmark comfy-kitchen's INT8 ConvRot linear against the shapes this project produces.

    python tools/bench_int8.py --device cuda:0 [--shapes blocks] [--batch 1,8,64,512]

Reports, per (M, N, K): which backend actually served the call, milliseconds, TFLOP/s and the
effective weight bandwidth. Run it on every architecture you care about (sm_89 vs sm_120) -- the
point of the kernel pack is that the *same* checkpoint performs well on both, and this is the
measurement that decides what still needs a kernel.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: (name, N, K) of the quantized linears in MiniMax-H3, one per distinct shape.
MODEL_SHAPES = {
    "qkv": (21504, 5376),
    "out_proj": (5376, 7168),
    "fc1": (28672, 5376),
    "fc2": (5376, 14336),
}


def backend_report() -> str:
    try:
        import comfy_kitchen as ck
    except ImportError:
        return "comfy_kitchen not importable -- run inside a ComfyUI environment"
    lines = []
    for name, status in sorted(ck.list_backends().items()):
        lines.append(f"  {name}: {status}")
    try:
        lines.append(f"  device capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
        lines.append(f"  torch: {torch.__version__} cuda {torch.version.cuda}")
    except Exception as err:  # noqa: BLE001
        lines.append(f"  (no cuda device: {err})")
    return "\n".join(lines)


def bench_one(m: int, n: int, k: int, iters: int, warmup: int, convrot_groupsize: int = 256) -> dict:
    import comfy_kitchen as ck

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    qweight, wscales = ck.quantize_int8_convrot_weight(weight, convrot_groupsize)

    def call():
        return ck.int8_linear(
            x, qweight, wscales, convrot=True, convrot_groupsize=convrot_groupsize, out_dtype=torch.bfloat16
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iters

    flops = 2.0 * m * n * k
    weight_bytes = n * k + n * 4  # int8 weights + fp32 row scales
    return {
        "ms": seconds * 1e3,
        "tflops": flops / seconds / 1e12,
        "weight_gib_s": weight_bytes / seconds / 2**30,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="1,8,64,512", help="comma-separated M values")
    ap.add_argument("--shapes", default="all", choices=["all", *MODEL_SHAPES])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    print("backends:")
    print(backend_report())

    names = list(MODEL_SHAPES) if args.shapes == "all" else [args.shapes]
    batches = [int(v) for v in args.batch.split(",") if v.strip()]
    print(f"{'shape':10s} {'M':>6s} {'ms':>9s} {'TFLOP/s':>9s} {'wGiB/s':>8s}")
    for name in names:
        n, k = MODEL_SHAPES[name]
        for m in batches:
            try:
                stats = bench_one(m, n, k, args.iters, args.warmup)
                print(f"{name:10s} {m:6d} {stats['ms']:9.3f} {stats['tflops']:9.2f} {stats['weight_gib_s']:8.1f}")
            except Exception as err:  # noqa: BLE001 - report and continue across shapes
                print(f"{name:10s} {m:6d}   FAILED: {type(err).__name__}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
