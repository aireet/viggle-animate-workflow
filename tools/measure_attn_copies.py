"""What do the three contiguous q/k/v copies inside the sol-attn override actually cost?

The override hands comfy-kitchen three contiguous (1, S, H, D) tensors, materialised from the
model's packed (S, 3*H*D) bf16 buffer (row stride 3*H*D, per-head stride D). This measures, at the
real production shape, the time of those copies and the peak VRAM they add -- i.e. whether the
"zero-copy attention" item from exp-viggle-ops/RESULTS.md is a speed item or a headroom item.

    python3 measure_attn_copies.py
"""

from __future__ import annotations

import torch


def main() -> int:
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    heads, dim, seq = 56, 128, 30026  # packed sequence at 124 frames, 480x832
    row = heads * dim

    packed = torch.empty(seq, 3 * row, dtype=torch.bfloat16, device=device)
    packed.normal_(0, 0.02)
    q = packed[:, :row].view(1, seq, heads, dim).transpose(1, 2)
    k = packed[:, row:2 * row].view(1, seq, heads, dim).transpose(1, 2)
    v = packed[:, 2 * row:].view(1, seq, heads, dim).transpose(1, 2)
    print(f"q view: shape {tuple(q.shape)} strides {tuple(q.stride())} contiguous={q.is_contiguous()}")
    print(f"packed buffer: {packed.numel() * 2 / 2**30:.2f} GiB")

    def copies():
        return [t.transpose(1, 2).contiguous() for t in (q, k, v)]

    for _ in range(3):
        out = copies()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    iters = 20
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        out = copies()
    end.record()
    torch.cuda.synchronize()
    per_call_ms = start.elapsed_time(end) / iters
    peak_delta = (torch.cuda.max_memory_allocated() - base) / 2**30
    bytes_copied = 3 * seq * row * 2

    print(f"three copies: {per_call_ms:.3f} ms/call, {bytes_copied / 2**20:.0f} MiB moved, peak +{peak_delta:.2f} GiB")
    print(f"at 300 attention calls per render: {per_call_ms * 300 / 1000:.2f} s total")
    print(f"card total {torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB, "
          f"currently allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    del out
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
