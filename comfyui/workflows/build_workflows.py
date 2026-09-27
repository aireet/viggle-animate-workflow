"""Generate the shipped ComfyUI workflow: four nodes, two of them file pickers.

    [Load Video] ┐
    [Load Image] ┼→ [Viggle Animate] → [Save Video]
                 └──────── audio ────────────┘

Starts from the vendor/user workflow (``base/viggle-animate-5090.json``) and keeps its video
loader, image loader and save node (widget values included, so the defaults stay the evaluated
ones); everything between them collapses into the one node. Slots are looked up by name from the
base, never by index.

    python3 build_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "viggle-animate-5090.json"
OUT = HERE / "viggle-animate-slimdit.json"

NODE_ID = 100
NOTE_ID = 101
NOTE = (
    "三步:\n"
    "  1. 左边放驱动视频 + 参考图(参考图最好是同一镜头里改绘的一帧)\n"
    "  2. 点 Run\n"
    "  3. 结果在右边,已带驱动视频的原声\n\n"
    "生成节点里的默认值就是官方评测用的配置:3 步、124 帧、shift 3/3。\n"
    "想换权重/LoRA 就在那个节点的下拉里选;跑第二次请改 seed(否则直接命中缓存)。"
)


def main() -> int:
    base = json.loads(BASE.read_text())
    by_id = {n["id"]: n for n in base["nodes"]}
    slots = {l[0]: (l[1], l[2]) for l in base["links"]}  # link id -> (origin node, origin slot)

    def keep(node_id: int, pos, title: str) -> dict:
        node = by_id[node_id]
        node["pos"] = pos
        node["title"] = title
        node["flags"] = {}
        node["size"] = [node.get("size", [270, 100])[0], node.get("size", [270, 100])[1]]
        return node

    video = keep(3, [40, 40], "① 驱动视频(动作来源)")
    image = keep(5, [40, 340], "① 参考图")
    save = keep(21, [1000, 40], "③ 保存 mp4(带原声)")

    def out_slot(node, name: str) -> int:
        return next(i for i, s in enumerate(node["outputs"]) if s["name"] == name)

    def in_slot(node, name: str) -> int:
        return next(i for i, s in enumerate(node["inputs"]) if s["name"] == name)

    generator = {
        "id": NODE_ID,
        "type": "ViggleAnimateSlimDiT",
        "pos": [540, 40],
        "size": [400, 330],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "video", "type": "IMAGE", "link": None},
            {"name": "reference_image", "type": "IMAGE", "link": None},
        ],
        "outputs": [{"name": "frames", "type": "IMAGE", "links": [], "slot_index": 0}],
        "properties": {"Node name for S&R": "ViggleAnimateSlimDiT"},
        "widgets_values": [
            "3",  # steps
            124,  # length
            0,  # seed
            "randomize",  # seed's control_after_generate -- the widget takes two values
            "minimax_h3_ref2va_slimdit_int8_convrot.safetensors",
            "viggle_animate_dmd_lora.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "fixed_embed_fwd_anyframe.safetensors",
            3.0,
            3.0,
        ],
        "title": "② 生成(默认值即官方配置)",
        "color": "#432",
        "bgcolor": "#653",
    }

    note = {
        "id": NOTE_ID,
        "type": "Note",
        "pos": [40, 620],
        "size": [620, 230],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": [NOTE],
        "title": "说明",
        "color": "#432",
        "bgcolor": "#653",
    }

    nodes = [video, image, generator, save, note]
    for node in nodes:
        for slot in node.get("outputs", []):
            slot["links"] = []

    links: list[list] = []

    def connect(source, source_slot: int, target, target_name: str, type_: str) -> None:
        slot = in_slot(target, target_name)
        link_id = len(links) + 1
        target["inputs"][slot]["link"] = link_id
        source["outputs"][source_slot]["links"].append(link_id)
        links.append([link_id, source["id"], source_slot, target["id"], slot, type_])

    connect(video, out_slot(video, "IMAGE"), generator, "video", "IMAGE")
    connect(video, out_slot(video, "audio"), save, "audio", "AUDIO")
    connect(image, out_slot(image, "IMAGE"), generator, "reference_image", "IMAGE")
    connect(generator, 0, save, "images", "IMAGE")

    for index, node in enumerate(nodes):
        node["order"] = index

    out = {
        "id": "viggle-animate-slimdit",
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": base.get("config", {}),
        "extra": base.get("extra", {}),
        "version": base.get("version", 0.4),
    }

    # self-check: every link resolves and every generator input is wired
    for source_id, source_slot, target_id, target_slot, _type in [(l[1], l[2], l[3], l[4], l[5]) for l in links]:
        nodes_by_id = {n["id"]: n for n in nodes}
        assert source_id in nodes_by_id, f"link source {source_id} missing"
        assert target_id in nodes_by_id, f"link target {target_id} missing"
        assert source_slot < len(nodes_by_id[source_id]["outputs"]), "output slot out of range"
        assert target_slot < len(nodes_by_id[target_id]["inputs"]), "input slot out of range"
    for slot in generator["inputs"]:
        assert slot["link"] is not None, f"generator input {slot['name']} not wired"
    for slot in save["inputs"]:
        if slot["name"] in ("images", "audio"):
            assert slot["link"] is not None, f"save input {slot['name']} not wired"

    OUT.write_text(json.dumps(out, indent=1))
    print(f"{OUT.name}: {len(nodes)} nodes, {len(links)} links, {sum(1 for n in nodes if n['type'] == 'Note')} note")
    print("nodes:", ", ".join(n["type"] for n in nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
