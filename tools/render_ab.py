"""Submit a ComfyUI workflow (API format) to a running instance and collect its outputs.

    python tools/render_ab.py --host <comfy-host> --port 8188 \\
        --workflow smoke-prompt.json --set 13.unet_name=my.safetensors --set 10.noise_seed=7 \\
        --outdir /tmp/ab

``--set <node_id>.<input>=<value>`` patches the graph; values are JSON-decoded when possible so
numbers, lists and the existing ``["13", 0]`` style links keep working. Prints the produced files
and can wait for the job to finish (``--wait``), which is what the A/B renders use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


def _json_get(url: str) -> dict:
    return json.loads(_get(url))


def patch(graph: dict, assignments: list[str]) -> None:
    for item in assignments:
        target, _, raw = item.partition("=")
        node_id, _, input_name = target.partition(".")
        if node_id not in graph:
            raise SystemExit(f"node {node_id} not in workflow")
        if input_name not in graph[node_id]["inputs"]:
            raise SystemExit(f"node {node_id} has no input {input_name!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        graph[node_id]["inputs"][input_name] = value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--set", dest="assignments", action="append", default=[])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    body = json.loads(Path(args.workflow).read_text())
    graph = body.get("prompt", body)
    patch(graph, args.assignments)

    client_id = str(uuid.uuid4())
    result = _post(f"{base}/prompt", {"prompt": graph, "client_id": client_id})
    prompt_id = result["prompt_id"]
    print(f"queued {prompt_id}", flush=True)
    if not args.wait:
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        history = _json_get(f"{base}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(f"job failed: {json.dumps(status)[:400]}")
            if entry.get("outputs"):
                break
        time.sleep(5)
    else:
        raise SystemExit("timed out waiting for the prompt")

    saved: list[str] = []
    for node_outputs in entry["outputs"].values():
        for kind in ("gifs", "videos", "images", "audio"):
            for item in node_outputs.get(kind, []):
                query = urllib.parse.urlencode(
                    {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                )
                blob = _get(f"{base}/view?{query}")
                target = outdir / item["filename"]
                target.write_bytes(blob)
                saved.append(f"{target} ({len(blob) / 2**20:.1f} MiB)")
    print("outputs:\n  " + "\n  ".join(saved) if saved else "no outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
