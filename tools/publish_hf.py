"""Publish the project and/or a converted checkpoint to a HuggingFace repo.

    HF_TOKEN=... python tools/publish_hf.py --repo-id <user>/slimdit --card
    HF_TOKEN=... python tools/publish_hf.py --repo-id <user>/MiniMax-H3-SlimDiT \\
        --checkpoints /path/to/checkpoints/*.safetensors

The card is generated from the values measured in this repository, so it stays honest: update
the numbers here if a future run changes them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CARD = """---
library_name: comfyui
base_model: Viggle/Viggle-Animate
tags: [comfyui, minimax-h3, viggle, int8, quantization, diffusion-transformer]
---

# MiniMax-H3 (Viggle-Animate) SlimDiT

INT8 ConvRot checkpoints for the MiniMax-H3 ref2va finetune, with the per-block adaln path
replaced by a shared rank-8 curve basis. Built analytically from the released BF16 weights by
[slimdit](https://huggingface.co/{repo}) — no distillation or fine-tuning.

| | |
|---|---|
| source | `Viggle/Viggle-Animate` (BF16, 66 GB) |
| this repo | INT8 ConvRot, curve rank 8 |
| curve error | 0.019 % relative on the exact modulation family |
| int8 error | ~0.9-1.2 % per row |
| load in ComfyUI | stock **Load Diffusion Model** |

## Files

{files}

## Usage

Place the checkpoint in `ComfyUI/models/diffusion_models/` and load it with **Load Diffusion
Model**. The curve layout (`adaln_t_table` + `[out_features, rank]` adaln linears) is detected
from the tensor shapes, so no extra nodes are needed to run it.

```bash
python3 tools/fetch_official.py --out /models/viggle-official          # 66 GB source
python -m slimdit.convert --src /models/viggle-official --out out.safetensors
```

Apache-2.0.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="target repo, e.g. <user>/slimdit")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="checkpoint files to upload")
    ap.add_argument("--source-repo", default="WarpEngine-github/slimdit", help="repo credited in the card")
    ap.add_argument("--card", action="store_true", help="also upload the generated model card")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--code", action="store_true", help="upload the slimdit package + comfyui node")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    uploads: list[str] = []

    root = Path(__file__).resolve().parent.parent
    if args.code:
        for pattern in ("slimdit/*.py", "comfyui/**/*", "tools/*.py", "README.md", "pyproject.toml"):
            for path in root.glob(pattern):
                if path.is_file():
                    target = str(path.relative_to(root))
                    api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=args.repo_id)
                    uploads.append(target)
        print(f"uploaded {len(uploads)} code files", flush=True)

    files = []
    for pattern in args.checkpoints:
        for path in sorted(Path().glob(pattern)) if "*" in pattern else [Path(pattern)]:
            if not path.is_file():
                print(f"skip missing {path}", flush=True)
                continue
            files.append(path.name)
            print(f"uploading {path.name} ({path.stat().st_size / 2**30:.1f} GiB)", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.repo_id)

    if args.card:
        listing = "\n".join(f"- `{name}`" for name in files) or "- (code only)"
        card = CARD.format(repo=args.source_repo, files=listing)
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo_id)
        print("uploaded model card", flush=True)

    print(f"done: https://huggingface.co/{args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
