"""Can comfy-kitchen's sol_attn consume the model's strided q/k/v directly?

The override currently materialises three contiguous (1, S, H, D) copies per call (~1.2 GiB peak,
0.5 s per render at 124 frames). If the kernel accepts the model's strided views, that memory and
those copies disappear -- a two-line change instead of a new kernel.

    python3 probe_sol_attn_strided.py
"""

from __future__ import annotations

import time

import torch


def bench(fn, iters=5):
    for _ in range(1):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    import comfy_kitchen as ck

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    heads, dim, seq = 56, 128, 30026
    row = heads * dim
    sink = [0, 4096]

    packed = torch.empty(seq, 3 * row, dtype=torch.bfloat16, device=device).normal_(0, 0.02)
    q = packed[:, :row].view(1, seq, heads, dim).transpose(1, 2)
    k = packed[:, row:2 * row].view(1, seq, heads, dim).transpose(1, 2)
    v = packed[:, 2 * row:].view(1, seq, heads, dim).transpose(1, 2)
    print("strided q:", tuple(q.shape), "contiguous:", q.is_contiguous())

    def contig_call():
        return ck.sol_attn(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                           v.transpose(1, 2).contiguous(), tau=1.0, scale=None,
                           sink_blocks=sink, sink_q=None, topk_ratio=0.0, tail=True)

    def strided_call():
        return ck.sol_attn(q, k, v, tau=1.0, scale=None, sink_blocks=sink, sink_q=None,
                           topk_ratio=0.0, tail=True)

    reference = contig_call()
    print(f"contiguous path: {bench(contig_call):.2f} ms")

    try:
        candidate = strided_call()
        torch.cuda.synchronize()
        same = torch.allclose(reference, candidate, atol=1e-2, rtol=1e-2)
        diff = (reference.float() - candidate.float()).abs().max().item()
        print(f"strided path:    {bench(strided_call):.2f} ms   matches contiguous: {same} (max diff {diff:.4f})")
        print("VERDICT: zero-copy is a two-line change" if same else "VERDICT: strided runs but numerics differ")
    except Exception as exc:  # noqa: BLE001
        print(f"strided path FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        print("VERDICT: needs a kernel that reads strided q/k/v in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
