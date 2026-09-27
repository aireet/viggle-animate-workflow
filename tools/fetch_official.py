"""Fetch the official Viggle-Animate transformer shards with the standard library only.

Runs anywhere a plain ``python3`` exists (e.g. the build host). One streaming connection per
shard, several shards in flight at once; each shard resumes from the contiguous prefix already
on disk and is verified against its safetensors header before being declared complete.

    python3 tools/fetch_official.py --out <data-dir>/models/viggle-official --workers 6
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = "Viggle/Viggle-Animate"
SUBDIR = "transformer"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{SUBDIR}"
INDEX = f"https://huggingface.co/{REPO}/raw/main/{SUBDIR}/diffusion_pytorch_model.safetensors.index.json"

CHUNK = 8 << 20
PRINT_LOCK = threading.Lock()


def _open(url: str, start: int | None = None, end: int | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": "slimdit-fetch"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    return urllib.request.urlopen(req, timeout=300)


def shard_size(url: str) -> int:
    with _open(url, 0, 0) as resp:
        content_range = resp.headers.get("Content-Range")
        if content_range:
            return int(content_range.split("/")[-1])
    raise RuntimeError(f"server did not report a size for {url}")


def verify_header(path: Path) -> None:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"{path.name}: truncated header length")
        (length,) = struct.unpack("<Q", raw)
        header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    size = max(v["data_offsets"][1] for v in header.values())
    if path.stat().st_size != 8 + length + size:
        raise RuntimeError(f"{path.name}: header/data size mismatch")


def download_shard(url: str, dest: Path, retries: int = 5) -> None:
    """Stream a shard to ``dest.part``, resuming from whatever is already there."""
    total = shard_size(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    done = part.stat().st_size if part.exists() else 0
    if done > total:
        part.unlink()
        done = 0
    if done == total:
        part.replace(dest)
        verify_header(dest)
        return

    for attempt in range(retries):
        try:
            with _open(url, done) as resp, part.open("r+b" if part.exists() else "wb") as fh:
                fh.seek(done)
                while True:
                    blob = resp.read(CHUNK)
                    if not blob:
                        break
                    fh.write(blob)
                    done += len(blob)
                    if attempt == 0 or done % (64 << 20) < CHUNK:
                        with PRINT_LOCK:
                            print(f"  {dest.name}: {done * 100 / total:5.1f}%", flush=True)
        except Exception as err:  # noqa: BLE001 - any transport hiccup is retryable
            with PRINT_LOCK:
                print(f"  {dest.name}: retry {attempt + 1}/{retries} at {done} bytes ({err})", flush=True)
            time.sleep(2 + 3 * attempt)
            continue
        if done == total:
            break

    if done != total:
        raise RuntimeError(f"{dest.name}: incomplete after {retries} attempts ({done}/{total})")
    part.replace(dest)
    verify_header(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination directory (shards land inside)")
    ap.add_argument("--workers", type=int, default=6, help="shards downloaded concurrently")
    ap.add_argument("--shards", type=int, default=0, help="only fetch the first N shards (0 = all)")
    args = ap.parse_args(argv)

    with _open(INDEX) as resp:
        index = json.loads(resp.read())
    shards = sorted(set(index["weight_map"].values()))
    if args.shards:
        shards = shards[: args.shards]
    dest_dir = Path(args.out)

    pending: list[str] = []
    for i, shard in enumerate(shards, 1):
        dest = dest_dir / shard
        if dest.exists():
            try:
                verify_header(dest)
                print(f"[{i}/{len(shards)}] {shard} already complete", flush=True)
                continue
            except Exception as err:  # noqa: BLE001 - re-fetch a corrupt leftover
                print(f"[{i}/{len(shards)}] {shard}: {err}; re-fetching", flush=True)
                dest.unlink()
        pending.append(shard)

    queue = list(pending)
    error: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            with lock:
                if not queue or error:
                    return
                shard = queue.pop(0)
            try:
                print(f"start {shard}", flush=True)
                download_shard(f"{BASE}/{shard}", dest_dir / shard)
                print(f"done  {shard}", flush=True)
            except BaseException as err:  # noqa: BLE001 - surface any failure
                with lock:
                    error.append(err)

    threads = [threading.Thread(target=worker) for _ in range(min(args.workers, max(1, len(queue))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if error:
        print(f"FAILED: {error[0]}", flush=True)
        return 1
    print("READY", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
