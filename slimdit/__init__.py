"""slimdit -- analytic low-rank + INT8/NVFP4 slimming for ComfyUI diffusion models.

First target: MiniMax-H3 (Viggle-Animate) in ComfyUI, producing checkpoints that plain
``Load Diffusion Model`` consumes with no custom nodes.
"""

from . import curve, hadamard, int8, mapping

__all__ = ["curve", "hadamard", "int8", "mapping"]
__version__ = "0.1.0"
