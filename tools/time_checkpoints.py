"""Time the same workflow across several checkpoints, twice each (cold + warm), via the ComfyUI API.

    python3 time_checkpoints.py --host 127.0.0.1 --port 8188 --workflow smoke-prompt.json \\
        --checkpoints name1.safetensors name2.safetensors

Prints, per run: wall time (submit -> finished) and the server's own "Prompt executed" figure is
captured separately from the container log. The first run of each checkpoint includes loading it
from disk; the second is the steady state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def run_once(base: str, graph: dict, checkpoint: str, prefix: str, timeout: float, seed: int | None = None) -> dict:
    graph["13"]["inputs"]["unet_name"] = checkpoint
    graph["21"]["inputs"]["filename_prefix"] = prefix
    for node in graph.values():
        if node.get("class_type") == "UNETLoader":
            node["inputs"]["unet_name"] = checkpoint
        if node.get("class_type") == "VHS_VideoCombine":
            node["inputs"]["filename_prefix"] = prefix
        if node.get("class_type") == "RandomNoise" and seed is not None:
            # ComfyUI caches per-node results: an unchanged seed makes the whole sampler chain a
            # cache hit (~0.7 s), so a *real* warm re-run needs a new noise seed.
            node["inputs"]["noise_seed"] = seed

    t0 = time.time()
    started = post(f"{base}/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    first_seen = None
    while time.time() - t0 < timeout:
        history = get(f"{base}/history/{started}")
        entry = history.get(started)
        if entry and entry.get("outputs"):
            status = entry.get("status", {})
            if status.get("status_str") != "success":
                return {"ok": False, "error": json.dumps(status)[:200], "wall": time.time() - t0}
            exec_ms = None
            for name, payload in status.get("messages", []):
                if name == "execution_start":
                    first_seen = payload.get("timestamp")
                if name == "execution_success":
                    exec_ms = payload.get("timestamp", 0) - (first_seen or 0)
            return {"ok": True, "wall": time.time() - t0, "exec": (exec_ms or 0) / 1000.0}
        time.sleep(1)
    return {"ok": False, "error": "timeout", "wall": time.time() - t0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=2347709, help="seed for run 1; run N uses base+N-1")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    graph = json.loads(Path(args.workflow).read_text())
    graph = graph.get("prompt", graph)

    print(f"{'checkpoint':<48s} {'run':>3s} {'seed':>9s} {'wall s':>8s} {'service s':>10s}")
    for ckpt in args.checkpoints:
        for i in range(1, args.repeats + 1):
            prefix = f"timing/{Path(ckpt).stem[:28]}_{i}"
            seed = args.seed_base + i - 1
            result = run_once(base, json.loads(json.dumps(graph)), ckpt, prefix, args.timeout, seed=seed)
            if result["ok"]:
                print(f"{ckpt:<48s} {i:>3d} {seed:>9d} {result['wall']:>8.1f} {result['exec']:>10.1f}")
            else:
                print(f"{ckpt:<48s} {i:>3d} {seed:>9d} FAILED: {result['error']}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
