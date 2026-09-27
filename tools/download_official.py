"""Download the official Viggle-Animate transformer shards (about 39 GB)."""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download


def main() -> int:
    dest = sys.argv[1] if len(sys.argv) > 1 else "/root/work/data/viggle-animate"
    path = snapshot_download(
        repo_id="Viggle/Viggle-Animate",
        allow_patterns=["transformer/*"],
        local_dir=dest,
        max_workers=8,
    )
    print("downloaded to", path, flush=True)
    print("READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
