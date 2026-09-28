"""Poll a ComfyUI prompt until it finishes, then print status, timing and output files."""

import json
import os
import sys
import time
import urllib.request

PID = sys.argv[1]
BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")  # 8188 is ComfyUI's default port


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


t0 = time.time()
last = None
while time.time() - t0 < 1080:
    queue = get("/queue")
    running, pending = len(queue["queue_running"]), len(queue["queue_pending"])
    if (running, pending) != last:
        print(f"[{time.time() - t0:4.0f}s] running={running} pending={pending}", flush=True)
        last = (running, pending)
    if running == 0 and pending == 0 and get("/history/" + PID):
        break
    time.sleep(10)

history = get("/history/" + PID)
if not history:
    print("timeout: prompt never finished")
    raise SystemExit(1)

entry = history[PID]
messages = entry["status"]["messages"]
stamps = [m[1]["timestamp"] / 1000 for m in messages if isinstance(m[1], dict) and "timestamp" in m[1]]
if stamps:
    print(f"wall clock: {(time.time() - t0) / 60:.1f} min")
print("status:", entry["status"]["status_str"])
for name, payload in messages:
    if name in ("execution_error", "execution_interrupted"):
        print("error:", json.dumps(payload, ensure_ascii=False)[:500])

files = []
for _node, out in (entry.get("outputs") or {}).items():
    for kind, items in out.items():
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("filename"):
                files.append((kind, item["filename"], round(item.get("size", 0) / 1e6, 1)))
print("outputs:", files)
