"""The node pack must surface one inference node, with the developer tools behind an opt-in.

The pack is loaded by path here (ComfyUI does the same), so this also covers the import guard that
keeps ``folder_paths`` optional.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parent.parent / "comfyui" / "ComfyUI-SlimDiT" / "nodes.py"


def load_nodes(monkeypatch, *, dev: bool):
    monkeypatch.setenv("SLIMDIT_DEV_NODES", "1" if dev else "0")
    spec = importlib.util.spec_from_file_location("slimdit_nodes_under_test", PACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_library_ships_two_nodes(monkeypatch):
    library = set(load_nodes(monkeypatch, dev=False).NODE_CLASS_MAPPINGS)
    # the two nodes this pack is about; the vendored Viggle-Animate pair only appears inside ComfyUI
    assert {"SlimDiTLoader", "ViggleAnimateH3"} <= library
    assert library <= {"SlimDiTLoader", "ViggleAnimateH3", "ViggleTextCondLoader", "ViggleAnimateConditioning"}


def test_dev_nodes_need_the_env_var(monkeypatch):
    assert set(load_nodes(monkeypatch, dev=True).NODE_CLASS_MAPPINGS) == {
        "SlimDiTLoader",
        "ViggleAnimateH3",
        "SlimDiTConvert",
        "SlimDiTInspect",
        "SlimDiTSolAttnStats",
        "SlimDiTSigmas",
    }


def test_loader_carries_the_weight_choices(monkeypatch):
    required = load_nodes(monkeypatch, dev=False).SlimDiTLoader.INPUT_TYPES()["required"]
    assert list(required) == ["checkpoint", "lora", "vae", "steps", "shift_video", "shift_audio"]
    assert required["steps"][0] == ["3", "4", "6"]


def test_h3_node_takes_three_inputs_and_outputs_the_latent(monkeypatch):
    """The graph depends on this shape: video + still (+ audio) in, latent + vae + audio out."""
    node = load_nodes(monkeypatch, dev=False).ViggleAnimateH3
    required = node.INPUT_TYPES()["required"]
    assert list(required)[:5] == ["model", "vae", "video", "reference_image", "audio"]
    assert required["model"][0] == "MODEL"
    assert required["vae"][0] == "VAE"
    assert required["audio"][0] == "AUDIO"
    assert required["steps"][0] == ["3", "4", "6"]
    assert required["length"][1]["default"] == 124
    assert node.RETURN_NAMES == ("latent", "audio")
