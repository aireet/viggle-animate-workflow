# ComfyUI-SlimDiT

Two nodes for turning an official diffusion checkpoint into a slim one, from inside ComfyUI.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> slimdit-tmp && cp -r slimdit-tmp/comfyui/ComfyUI-SlimDiT .
# or symlink: ln -s "$(pwd)/slimdit-tmp/comfyui/ComfyUI-SlimDiT" ComfyUI-SlimDiT
pip install -r ComfyUI-SlimDiT/requirements.txt
```

The nodes import the `slimdit` package from the repository root, so keep the checkout around
(the clone above is expected next to `ComfyUI-SlimDiT`).

## Nodes

| node | what it does |
|---|---|
| **SlimDiT Convert (INT8 + curves)** | Downloads the official transformer shards from HuggingFace (cached in `~/.cache/slimdit/official`), rebuilds the adaln path as a shared rank-`r` curve table, quantizes every block linear to INT8 ConvRot and writes `models/diffusion_models/<output_name>`. Then load it with the stock **Load Diffusion Model** node. |
| **SlimDiT Inspect Checkpoint** | Prints size, tensor count, quantized-linears, curve rank/grid of any checkpoint header — cheap way to confirm what a file actually is. |

Conversion is CPU-only but takes ~20 minutes for MiniMax-H3 (66 GB in, 21 GB out); ComfyUI will
show the console log while it runs.
