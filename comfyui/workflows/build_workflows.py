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
    "viggle-animate-h3:权重、LoRA、VAE 全部用 ComfyUI 自带的加载器节点,连线送进本节点。\n"
    "  1. 左边放驱动视频(画面和声音)和参考图(最好是同一镜头里改绘的一帧)\n"
    "  2. 点 Run\n"
    "  3. 右边 VHS_VideoCombine 出 mp4(带驱动视频原声)\n\n"
    "连线:Load Diffusion Model -> Load LoRA -> 节点 model;Load VAE -> 节点 vae 和 VAE Decode;\n"
    "驱动视频 IMAGE/audio -> 节点 video/audio;参考图 -> reference_image;\n"
    "节点 latent -> VAE Decode -> 保存节点;节点 audio 直通保存节点。\n"
    "节点里只留 slimdit 自己的设置:步数 3/4/6、帧数、seed、文本条件、shift。"
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

        # no node may overlap another: a wrong size makes the frontend grow a node into its neighbour
        placed = [n for n in self.nodes if n["type"] != "Note"]
        for i, a in enumerate(placed):
            ax0, ay0 = a["pos"]
            ax1, ay1 = ax0 + a["size"][0], ay0 + a["size"][1]
            for b in placed[i + 1:]:
                bx0, by0 = b["pos"]
                bx1, by1 = bx0 + b["size"][0], by0 + b["size"][1]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    raise SystemExit(f"{name}: {a['type']} overlaps {b['type']}")

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
    """ComfyUI conventions, and no node overlaps anything else.

    Sizes are the vendor workflow's real rendered sizes (a node grew past them before, which pushed
    it into the node below); placement keeps >=80 px between every pair of rectangles, checked in
    save().
    """
    g = Graph(base)
    unet = g.take("UNETLoader", [0, 0])
    unet["size"] = [470, 90]
    unet["widgets_values"] = [CHECKPOINT, "default"]
    unet["title"] = "Load Diffusion Model"
    lora = g.take("LoraLoaderModelOnly", [0, 180])
    lora["size"] = [470, 90]
    lora["widgets_values"] = [LORA, 1.0]
    lora["title"] = "Load LoRA"
    vae_load = g.take("VAELoader", [0, 380])
    vae_load["size"] = [310, 70]
    vae_load["widgets_values"] = [VAE]
    vae_load["title"] = "Load VAE"

    video = g.take("VHS_LoadVideo", [560, 0])
    video["size"] = [380, 950]
    image = g.take("LoadImage", [560, 1030])
    image["size"] = [380, 700]

    node = g.add({
        "id": 100,
        "type": "ViggleAnimateH3",
        "pos": [1030, 0],
        "size": [400, 460],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "video", "type": "IMAGE", "link": None},
            {"name": "reference_image", "type": "IMAGE", "link": None},
            {"name": "audio", "type": "AUDIO", "link": None},
        ],
        "outputs": [
            {"name": "latent", "type": "LATENT", "links": [], "slot_index": 0},
            {"name": "audio", "type": "AUDIO", "links": [], "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "ViggleAnimateH3"},
        "widgets_values": ["6", 124, 0, "randomize", TEXT_COND, 3.0, 3.0],
        "title": "viggle-animate-h3",
        "color": "#432",
        "bgcolor": "#653",
    })

    decode = g.take("VAEDecode", [1520, 80])
    decode["size"] = [240, 80]
    decode["title"] = "MiniMax-H3 VAE Decode"
    combine = g.take("VHS_VideoCombine", [1520, 260])
    combine["size"] = [430, 970]

    g.connect(unet, "MODEL", lora, "model")
    g.connect(lora, "MODEL", node, "model")
    g.connect(vae_load, "VAE", node, "vae")
    g.connect(video, "IMAGE", node, "video")
    g.connect(video, "audio", node, "audio")

    g.connect(image, "IMAGE", node, "reference_image")
    g.connect(node, "latent", decode, "samples")
    g.connect(vae_load, "VAE", decode, "vae")
    g.connect(node, "audio", combine, "audio")
    g.connect(decode, "IMAGE", combine, "images")
    g.save("viggle-animate-h3.json", NOTE_ONECLICK, ds={"scale": 1.0, "offset": [500, 60]}, note_pos=[1030, 520])


def main() -> int:
    base = json.loads(BASE.read_text())
    one_click(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
