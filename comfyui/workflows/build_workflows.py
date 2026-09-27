"""Generate the shipped ComfyUI workflow: one loader node instead of six.

Starts from the vendor/user workflow (``base/viggle-animate-5090.json``), keeps the nodes a user
actually touches, and folds UNETLoader + LoraLoaderModelOnly + VAELoader + MiniMaxH3SigmaShift +
KSamplerSelect + the sigma list into a single ``SlimDiTLoader``. Links are re-pointed by *source
node type*, so the graph stays valid whatever the original link numbering was.

    python3 build_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "viggle-animate-5090.json"
OUT = HERE / "viggle-animate-slimdit-int8.json"

#: Nodes folded into SlimDiTLoader: type -> (loader output slot, loader output type)
FOLDED = {
    "UNETLoader": (0, "MODEL"),
    "LoraLoaderModelOnly": (0, "MODEL"),
    "MiniMaxH3SigmaShift": (0, "MODEL"),
    "VAELoader": (1, "VAE"),
    "KSamplerSelect": (3, "SAMPLER"),
    "ManualSigmas": (2, "SIGMAS"),
}

LOADER_ID = 100
LOADER_TITLE = "加载器:权重 + LoRA + VAE + 步数"
NOTE = (
    "用法(三步):\n"
    "① 左边放:驱动视频 + 角色参考图(参考图最好是同一镜头的改绘帧)\n"
    "② 中间只有三处可动:步数(3/4/6)、length=帧数(跟驱动视频一致)、seed(改它=重新生成)\n"
    "③ 右边出 mp4(带驱动视频的原声)\n\n"
    "加载器一个节点就带模型/LoRA/VAE/步数;灰色条是折叠的辅助节点,不用碰。\n"
    "注意力路由由 SlimDiT 节点包在后台生效(启动时 SLIMDIT_ATTN=auto)。"
)

TITLES = {
    "SlimDiTLoader": LOADER_TITLE,
    "VHS_LoadVideo": "① 驱动视频(动作来源)",
    "LoadImage": "① 角色参考图",
    "ViggleTextCondLoader": "文本条件(预加载)",
    "ImageScaleToTotalPixels": "缩放 0.4MP",
    "ViggleAnimateConditioning": "② 组装条件(length=帧数)",
    "RandomNoise": "② 噪声种子(改它=重新生成)",
    "BasicGuider": "引导(每步调模型)",
    "SamplerCustomAdvanced": "② 采样(步数由加载器决定)",
    "VAEDecode": "VAE 解码",
    "VHS_VideoCombine": "③ 保存 mp4 (24fps)",
    "Note": "说明",
}

#: node type -> (column, row, collapsed)
LAYOUT = {
    "VHS_LoadVideo": (0, 0, False),
    "LoadImage": (0, 1, False),
    "ViggleTextCondLoader": (0, 2, True),
    "Note": (0, 3, False),
    "ImageScaleToTotalPixels": (1, 2, True),
    "ViggleAnimateConditioning": (1, 0, False),
    "SlimDiTLoader": (2, 0, False),
    "RandomNoise": (2, 1, False),
    "BasicGuider": (2, 2, True),
    "SamplerCustomAdvanced": (3, 0, False),
    "VAEDecode": (3, 2, True),
    "VHS_VideoCombine": (4, 0, False),
}

COL_X, ROW_H = 470, 150
STAGE_TITLES = {
    0: "① 准备 · 输入素材",
    1: "② 准备 · 条件(Viggle)",
    2: "② 模型 / 采样设置",
    3: "② 采样",
    4: "③ 合成 · 出片",
}
STAGE_COLORS = {0: "#335", 1: "#353", 2: "#533", 3: "#533", 4: "#235"}


def loader_node() -> dict:
    def slot(name, type_, slot_index):
        return {"name": name, "type": type_, "links": [], "slot_index": slot_index}

    return {
        "id": LOADER_ID,
        "type": "SlimDiTLoader",
        "pos": [COL_X * 2 + 40, ROW_H * 1],
        "size": [420, 260],
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
        "widgets_values": [
            "minimax_h3_ref2va_slimdit_int8_convrot.safetensors",
            "viggle_animate_dmd_lora.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "3",
            3.0,
            3.0,
        ],
        "title": LOADER_TITLE,
        "color": "#432",
        "bgcolor": "#653",
    }


def main() -> int:
    base = json.loads(BASE.read_text())
    kept = [n for n in base["nodes"] if n["type"] not in FOLDED]
    by_id = {n["id"]: n for n in base["nodes"]}

    # original link id -> (origin node id, origin slot), so inputs can be re-pointed by source type
    origin = {l[0]: (l[1], l[2]) for l in base["links"]}

    loader = loader_node()
    nodes = kept + [loader]
    links: list[list] = []
    next_link = 1

    # Output slots still carry the base workflow's link ids; rebuild the live set from scratch.
    for node in nodes:
        for slot in node.get("outputs", []):
            slot["links"] = []

    for node in nodes:
        node["pos"] = [COL_X * LAYOUT.get(node["type"], (3, 0, True))[0] + 40,
                       ROW_H * LAYOUT.get(node["type"], (3, 0, True))[1]]
        node["title"] = TITLES.get(node["type"], node["type"])
        node["flags"] = {"collapsed": bool(LAYOUT.get(node["type"], (3, 0, True))[2])}
        if node["type"] == "Note":
            node["widgets_values"] = [NOTE]
            node["size"] = [380, 300]
        if not node["flags"]["collapsed"]:
            node["color"], node["bgcolor"] = "#432" if node["type"] == "SlimDiTLoader" else "#233", "#653" if node["type"] == "SlimDiTLoader" else "#355"

        if node["type"] == "SlimDiTLoader":
            continue

        for index, slot in enumerate(node.get("inputs", [])):
            old_link = slot.get("link")
            if old_link is None or old_link not in origin:
                continue
            source_id, source_slot = origin[old_link]
            source_type = by_id.get(source_id, {}).get("type")
            if source_type in FOLDED:
                out_slot, out_type = FOLDED[source_type]
                slot["link"] = next_link
                links.append([next_link, LOADER_ID, out_slot, node["id"], index, out_type])
                loader["outputs"][out_slot]["links"].append(next_link)
                next_link += 1
            else:
                slot["link"] = next_link
                src_node = next(n for n in nodes if n["id"] == source_id)
                out_type = src_node["outputs"][source_slot]["type"]
                links.append([next_link, source_id, source_slot, node["id"], index, out_type])
                src_node["outputs"][source_slot]["links"].append(next_link)
                next_link += 1

    groups, bbox = [], {}
    for node in nodes:
        col = LAYOUT.get(node["type"], (3, 0, True))[0]
        lo, hi = bbox.get(col, (10**9, -10**9))
        height = 300 if node["flags"]["collapsed"] else node.get("size", [210, 120])[1]
        bbox[col] = (min(lo, node["pos"][1] - 40), max(hi, node["pos"][1] + height + 120))
    for col, (lo, hi) in sorted(bbox.items()):
        groups.append({
            "title": STAGE_TITLES[col],
            "bounding": [COL_X * col - 20, lo, 440, hi - lo],
            "color": STAGE_COLORS[col],
            "font_size": 24,
        })

    def order(node):
        return LAYOUT.get(node["type"], (5, 0, True))[0] * 10 + LAYOUT.get(node["type"], (5, 0, True))[1]

    for index, node in enumerate(sorted(nodes, key=order)):
        node["order"] = index

    out = {
        "id": "slimdit-int8",
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": next_link - 1,
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "config": base.get("config", {}),
        "extra": base.get("extra", {}),
        "version": base.get("version", 0.4),
    }
    OUT.write_text(json.dumps(out, indent=1))
    collapsed = sum(1 for n in nodes if n["flags"]["collapsed"])
    print(f"{OUT.name}: {len(nodes)} nodes (was {len(base['nodes'])}), {len(links)} links, "
          f"{len(groups)} groups, {collapsed} collapsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
