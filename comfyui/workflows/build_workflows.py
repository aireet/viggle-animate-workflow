"""Generate the shipped ComfyUI workflow: a five-node one-click graph with visible links.

    viggle-animate-slimdit.json   5 nodes  -- one-click: video + still -> generator -> save

Slots are looked up
by node type and slot *name*, never by index, and every generated file is self-checked.

    python3 build_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "viggle-animate-5090.json"

CHECKPOINT = "minimax_h3_ref2va_slimdit_int8_convrot.safetensors"
LORA = "viggle_animate_dmd_lora.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
TEXT_COND = "fixed_embed_fwd_anyframe.safetensors"

NOTE_ONECLICK = (
    "一键版:两步就能出片。\n"
    "  1. 左边放驱动视频 + 参考图(参考图最好是同一镜头里改绘的一帧)\n"
    "  2. 点 Run,结果在右边,已带驱动视频的原声\n\n"
    "生成节点里的默认值 = 官方评测配置(3 步 / 124 帧 / shift 3/3)。\n"
    "想看每一步在做什么、想改中间的数据流,用另一份「全流程版」。"
)


class Graph:
    def __init__(self, base: dict):
        self.base = base
        self.by_type: dict[str, dict] = {}
        for node in base["nodes"]:
            self.by_type.setdefault(node["type"], node)
        self.slots = {l[0]: (l[1], l[2]) for l in base["links"]}
        self.nodes: list[dict] = []
        self.links: list[list] = []

    def take(self, type_: str, pos: list[int], title: str | None = None) -> dict:
        node = json.loads(json.dumps(self.by_type[type_]))  # deep copy: the base stays untouched
        node["pos"] = pos
        node.pop("title", None)
        if title:
            node["title"] = title
        node["flags"] = {}
        node["color"] = node.pop("color", None)
        node["bgcolor"] = node.pop("bgcolor", None)
        if node["color"] is None:
            node.pop("color")
            node.pop("bgcolor")
        for slot in node.get("outputs", []):
            slot["links"] = []
        self.nodes.append(node)
        return node

    def add(self, node: dict) -> dict:
        self.nodes.append(node)
        return node

    def connect(self, source: dict, out_name: str, target: dict, in_name: str) -> None:
        out_slot = next(i for i, s in enumerate(source["outputs"]) if s["name"] == out_name)
        in_slot = next(i for i, s in enumerate(target["inputs"]) if s["name"] == in_name)
        link_id = len(self.links) + 1
        target["inputs"][in_slot]["link"] = link_id
        source["outputs"][out_slot]["links"].append(link_id)
        self.links.append([link_id, source["id"], out_slot, target["id"], in_slot, source["outputs"][out_slot]["type"]])

    def save(self, name: str, note: str | None, ds: dict | None = None, note_pos: list[int] | None = None) -> None:
        if note is not None:
            note_node = {
                "id": max(n["id"] for n in self.nodes) + 1,
                "type": "Note",
                "pos": note_pos or [40, max(n["pos"][1] + n["size"][1] for n in self.nodes) + 60],
                "size": [620, 160],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [note],
                "color": "#432",
                "bgcolor": "#653",
                "title": "说明",
            }
            self.nodes.append(note_node)

        for index, node in enumerate(self.nodes):
            node["order"] = index

        doc = {
            "id": name.replace(".json", ""),
            "revision": 0,
            "last_node_id": max(n["id"] for n in self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": self.base.get("config", {}),
            "extra": {**self.base.get("extra", {}), **({"ds": ds} if ds else {})},
            "version": self.base.get("version", 0.4),
        }
        by_id = {n["id"]: n for n in self.nodes}
        for _id, origin, origin_slot, target, target_slot, type_ in self.links:
            assert origin in by_id and target in by_id, f"{name}: dangling link endpoint"
            assert origin_slot < len(by_id[origin]["outputs"]), f"{name}: bad output slot"
            assert target_slot < len(by_id[target]["inputs"]), f"{name}: bad input slot"
            assert by_id[origin]["outputs"][origin_slot]["type"] == type_, f"{name}: link type mismatch"
            assert by_id[target]["inputs"][target_slot]["type"] == type_, f"{name}: {target} input type mismatch"
        (HERE / name).write_text(json.dumps(doc, indent=1))
        print(f"{name}: {len(self.nodes)} nodes, {len(self.links)} links, {sum(1 for n in self.nodes if n['type'] == 'Note')} note")


def loader_node(node_id: int, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "SlimDiTLoader",
        "pos": pos,
        "size": [380, 290],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [], "slot_index": 0},
            {"name": "vae", "type": "VAE", "links": [], "slot_index": 1},
            {"name": "sigmas", "type": "SIGMAS", "links": [], "slot_index": 2},
            {"name": "sampler", "type": "SAMPLER", "links": [], "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "SlimDiTLoader"},
        "widgets_values": [CHECKPOINT, LORA, VAE, "3", 3.0, 3.0],
        "title": "SlimDiT Loader (权重 + LoRA + VAE + 步数)",
    }


def one_click(base: dict) -> None:
    g = Graph(base)
    # Tight block so nothing hides behind the workflow panel; the saved view below then shows all
    # five nodes and all four links the moment the workflow opens.
    video = g.take("VHS_LoadVideo", [0, 0])
    video["size"] = [340, 280]
    image = g.take("LoadImage", [0, 320])
    image["size"] = [340, 330]
    save = g.take("VHS_VideoCombine", [840, 0])
    save["size"] = [400, 330]
    gen = g.add({
        "id": 100,
        "type": "ViggleAnimateSlimDiT",
        "pos": [400, 60],
        "size": [380, 300],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "video", "type": "IMAGE", "link": None},
            {"name": "reference_image", "type": "IMAGE", "link": None},
        ],
        "outputs": [{"name": "frames", "type": "IMAGE", "links": [], "slot_index": 0}],
        "properties": {"Node name for S&R": "ViggleAnimateSlimDiT"},
        "widgets_values": ["3", 124, 0, "randomize", CHECKPOINT, LORA, VAE, TEXT_COND, 3.0, 3.0],
        "title": "② 生成(默认值即官方配置)",
        "color": "#432",
        "bgcolor": "#653",
    })
    g.connect(video, "IMAGE", gen, "video")
    g.connect(image, "IMAGE", gen, "reference_image")
    g.connect(video, "audio", save, "audio")
    g.connect(gen, "frames", save, "images")
    g.save("viggle-animate-slimdit.json", NOTE_ONECLICK, ds={"scale": 0.6, "offset": [470, 80]}, note_pos=[400, 420])



def main() -> int:
    base = json.loads(BASE.read_text())
    one_click(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
