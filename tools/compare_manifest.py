"""Compare two safetensors checkpoints by header only (keys, dtypes, shapes, sizes).

    python3 tools/compare_manifest.py ours.safetensors reference.safetensors

Reports keys that are missing on either side, dtype/shape mismatches, and the byte totals, which
is how this project validates that a converted checkpoint is layout-compatible with the
reference one before spending GPU time on renders.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}


def read_header(path: Path) -> dict:
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return header


def tensor_bytes(entry: dict) -> int:
    size = DTYPE_SIZE[entry["dtype"]]
    count = 1
    for dim in entry["shape"]:
        count *= dim
    return size * count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("--limit", type=int, default=15, help="how many mismatches to print")
    ap.add_argument(
        "--intersect",
        action="store_true",
        help="only demand parity for tensors present in both (partial/trial conversions)",
    )
    args = ap.parse_args(argv)

    ours = read_header(Path(args.ours))
    ref = read_header(Path(args.reference))

    only_ours = sorted(set(ours) - set(ref))
    only_ref = sorted(set(ref) - set(ours))
    mismatches = []
    for key in sorted(set(ours) & set(ref)):
        a, b = ours[key], ref[key]
        if a["dtype"] != b["dtype"] or list(a["shape"]) != list(b["shape"]):
            mismatches.append(f"{key}: ours {a['dtype']}{a['shape']} vs ref {b['dtype']}{b['shape']}")

    print(f"ours     : {len(ours):5d} tensors, {sum(tensor_bytes(e) for e in ours.values()) / 2**30:.2f} GiB -> {args.ours}")
    print(f"reference: {len(ref):5d} tensors, {sum(tensor_bytes(e) for e in ref.values()) / 2**30:.2f} GiB -> {args.reference}")
    print(f"only in ours ({len(only_ours)}): {only_ours[: args.limit]}")
    print(f"only in reference ({len(only_ref)}): {only_ref[: args.limit]}")
    print(f"dtype/shape mismatches ({len(mismatches)}):")
    for line in mismatches[: args.limit]:
        print("   ", line)

    ok = not mismatches and (args.intersect or (not only_ours and not only_ref))
    print("PARITY OK" if ok else "PARITY DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
