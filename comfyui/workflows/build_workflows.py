"""Rebuild the shipped workflows as a three-stage canvas (Prepare / Render / Finish).

ComfyUI cannot hide the sampling machinery behind one node, but it can group it and collapse it:
`flags.collapsed` leaves only a titled chip, and the group boxes make the stages readable at a
glance -- the same three-step story the official Gradio workflow tells. Only the nodes a user
actually touches stay expanded.

    python3 build_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: node type -> (column, row, collapsed)
LAYOUT = {
    # stage 1: inputs + conditioning
    "VHS_LoadVideo": (0, 0, False),
    "LoadImage": (0, 1, False),
    "ViggleTextCondLoader": (0, 2, True),
    "ImageScaleToTotalPixels": (1, 2, True),
    "ViggleAnimateConditioning": (1, 0, False),
    "VAELoader": (1, 3, True),
    # stage 2: the sampling pipeline
    "UNETLoader": (2, 0, True),
    "LoraLoaderModelOnly": (2, 1, True),
    "MiniMaxH3SigmaShift": (2, 2, True),
    "RandomNoise": (2, 3, False),
    "KSamplerSelect": (2, 4, True),
    "ManualSigmas": (2, 5, True),
    "BasicGuider": (3, 0, True),
    "SamplerCustomAdvanced": (3, 1, False),
    # stage 3: output
    "VAEDecode": (4, 0, True),
    "VHS_VideoCombine": (4, 1, False),
    "Note": (0, 4, False),
}

TITLES = {
    "VHS_LoadVideo": "① 驱动视频(动作来源)",
    "LoadImage": "① 角色参考图",
    "ViggleTextCondLoader": "文本条件(预计算,一般不用动)",
    "ImageScaleToTotalPixels": "缩放 0.4MP",
    "ViggleAnimateConditioning": "② 组装条件(length=帧数)",
    "VAELoader": "视频 VAE",
    "UNETLoader": "SlimDiT int8 权重",
    "LoraLoaderModelOnly": "DMD LoRA",
    "MiniMaxH3SigmaShift": "shift 3/3",
    "RandomNoise": "② 噪声种子(改它=重新生成)",
    "KSamplerSelect": "euler",
    "ManualSigmas": "4 步 sigmas",
    "BasicGuider": "guider",
    "SamplerCustomAdvanced": "② 采样(4 步)",
    "VAEDecode": "VAE 解码",
    "VHS_VideoCombine": "③ 保存 mp4 (24fps)",
}

COL_X, COL_Y = 470, 400
ROW_H = 110

NOTE = (
    "用法(三步):\n"
    "① 左边放:驱动视频 + 角色参考图(参考图最好是同一镜头的改绘帧)\n"
    "② 中间只有两个可动:length = 帧数(跟驱动视频一致),RandomNoise 的 seed(改它就重新生成)\n"
    "③ 右边出 mp4(带驱动视频的原声)\n\n"
    "灰掉的方框是加载模型/VAE/采样参数,折叠起来了,一般不用碰。\n"
    "注意力路由/加速由 SlimDiT 节点包在后台生效(启动时 SLIMDIT_ATTN=auto)。"
)


def build(src: Path, dst: Path, kind: str) -> None:
    wf = json.loads(src.read_text())
    groups = []
    bbox = {}

    for node in wf["nodes"]:
        spec = LAYOUT.get(node["type"])
        if spec is None:
            continue
        col, row, collapsed = spec
        node["pos"] = [COL_X * col + 40, COL_Y // 2 + ROW_H * row]
        node["title"] = TITLES.get(node["type"], node["type"])
        flags = node.setdefault("flags", {})
        flags["collapsed"] = bool(collapsed)
        # keep the essential nodes visually prominent
        if not collapsed:
            node["color"] = "#432" if col == 2 else "#233"
            node["bgcolor"] = "#653" if col == 2 else "#355"
        if node["type"] == "Note":
            node["widgets_values"] = [NOTE]
            node["size"] = [380, 260]
        lo, hi = bbox.get(col, (10**9, -10**9))
        w, h = node.get("size", [210, 100])
        bbox[col] = (min(lo, node["pos"][1] - 40), max(hi, node["pos"][1] + (260 if collapsed else h) + 130))

    stage_titles = {
        0: "① 准备 · 输入素材",
        1: "② 准备 · 条件(Viggle)",
        2: "② 渲染 · 采样 4 步",
        3: "② 渲染 · 采样循环",
        4: "③ 合成 · 出片",
    }
    stage_colors = {0: "#335", 1: "#353", 2: "#533", 3: "#533", 4: "#235"}
    for col, (lo, hi) in sorted(bbox.items()):
        groups.append({
            "title": stage_titles[col],
            "bounding": [COL_X * col - 20, lo, 430, hi - lo],
            "color": stage_colors[col],
            "font_size": 24,
        })
    wf["groups"] = groups
    wf["id"] = f"slimdit-{kind}"
    dst.write_text(json.dumps(wf, indent=1))
    print(f"{dst.name}: {len(wf['nodes'])} nodes, {len(groups)} groups, "
          f"{sum(1 for n in wf['nodes'] if n.get('flags', {}).get('collapsed'))} collapsed")


def main() -> int:
    for name, kind in (("viggle-animate-slimdit-int8.json", "int8"), ("viggle-animate-slimdit-nvfp4.json", "nvfp4")):
        src = HERE / name
        if src.exists():
            build(src, src, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
