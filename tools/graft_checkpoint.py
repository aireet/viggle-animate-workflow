"""Byte-level tensor grafting between two checkpoints (isolation experiments).

Copies the raw bytes of selected tensors from ``--donor`` into a copy of ``--base``. Requires
identical name/dtype/shape for the grafted tensors, which lets us swap one subsystem (e.g. the
whole adaln curve) between two checkpoints and re-render, instead of guessing which part is at
fault. Works in place on the file's existing offsets -- no full rewrite.

    python tools/graft_checkpoint.py --base a.safetensors --donor b.safetensors \\
        --pattern 'adaln_t_table|adaln_proj' --out hybrid.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--pattern", required=True, help="regex over tensor names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base_path, donor_path, out_path = Path(args.base), Path(args.donor), Path(args.out)
    base_header, base_data = read_header(base_path)
    donor_header, donor_data = read_header(donor_path)

    matcher = re.compile(args.pattern)
    selected = [name for name in base_header if matcher.search(name)]
    if not selected:
        raise SystemExit("pattern matched nothing in the base checkpoint")

    if out_path.resolve() != base_path.resolve():
        print(f"copying {base_path.name} -> {out_path.name}", flush=True)
        shutil.copyfile(base_path, out_path)

    grafted = skipped = 0
    with out_path.open("r+b") as out_fh, donor_path.open("rb") as donor_fh:
        for name in selected:
            a, b = base_header.get(name), donor_header.get(name)
            if a is None or b is None or a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                skipped += 1
                continue
            start, end = a["data_offsets"]
            if end - start != b["data_offsets"][1] - b["data_offsets"][0]:
                skipped += 1
                continue
            donor_fh.seek(donor_data + b["data_offsets"][0])
            blob = donor_fh.read(end - start)
            out_fh.seek(base_data + start)
            out_fh.write(blob)
            grafted += 1

    print(f"grafted {grafted} tensors from {donor_path.name} into {out_path.name} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
