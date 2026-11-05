"""Measure strict FP32 inference for the short combat attention shapes on CUDA.

The primitive benchmark uses deterministic synthetic activations and the length
groups seen in the 8192-state combat benchmark. It does not measure a rollout.
"""

import argparse
import json
import math
import statistics

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from rl.training.resource_monitor import ResourceMonitor


@triton.jit
def _short_attention(
    query, key, value, present, output,
    HEADS: tl.constexpr, TOKENS: tl.constexpr, QUERIES: tl.constexpr,
    WIDTH: tl.constexpr, KEY_TILE: tl.constexpr,
    q_batch: tl.constexpr, q_head: tl.constexpr, q_token: tl.constexpr, q_dim: tl.constexpr,
    k_batch: tl.constexpr, k_head: tl.constexpr, k_token: tl.constexpr, k_dim: tl.constexpr,
    v_batch: tl.constexpr, v_head: tl.constexpr, v_token: tl.constexpr, v_dim: tl.constexpr,
    m_batch: tl.constexpr, m_token: tl.constexpr, SCALE: tl.constexpr,
):
    pair = tl.program_id(0)
    batch, head = pair // HEADS, pair % HEADS
    rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, KEY_TILE)
    dims = tl.arange(0, WIDTH)
    q = tl.load(
        query + batch * q_batch + head * q_head + rows[:, None] * q_token + dims[None, :] * q_dim,
        rows[:, None] < QUERIES, 0.0,
    )
    k = tl.load(
        key + batch * k_batch + head * k_head + cols[None, :] * k_token + dims[:, None] * k_dim,
        cols[None, :] < TOKENS, 0.0,
    )
    valid = (cols < TOKENS) & (tl.load(present + batch * m_batch + cols * m_token, cols < TOKENS, False) != 0)
    scores = tl.where(valid[None, :], tl.dot(q, k, input_precision="ieee") * SCALE, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, 1)[:, None])
    weights /= tl.sum(weights, 1)[:, None]
    v = tl.load(
        value + batch * v_batch + head * v_head + cols[:, None] * v_token + dims[None, :] * v_dim,
        cols[:, None] < TOKENS, 0.0,
    )
    y = tl.dot(weights, v, input_precision="ieee")
    # Store in the order consumed by the next linear layer.
    tl.store(output + ((batch * QUERIES + rows[:, None]) * HEADS + head) * WIDTH + dims[None, :],
             y, rows[:, None] < QUERIES)


def short_attention(q, k, v, present):
    batch, heads, queries, width = q.shape
    tokens = k.shape[2]
    out = torch.empty((batch, queries, heads, width), device=q.device, dtype=q.dtype)
    _short_attention[(batch * heads, triton.cdiv(queries, 16))](
        q, k, v, present, out, heads, tokens, queries, width,
        max(16, triton.next_power_of_2(tokens)), *q.stride(), *k.stride(), *v.stride(),
        *present.stride(), 1 / math.sqrt(width), num_warps=4,
    )
    return out.transpose(1, 2)


def torch_attention(q, k, v, present):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=present[:, None, None, :])


def make_inputs(batch, tokens, queries):
    weights = torch.randn((batch, tokens, 3, 4, 16), device="cuda")
    q, k, v = weights.permute(2, 0, 3, 1, 4)
    if queries != tokens:
        indices = torch.randint(tokens, (batch, queries), device="cuda")
        q = weights[:, :, 0][torch.arange(batch, device="cuda")[:, None], indices].transpose(1, 2)
    mask = torch.rand((batch, tokens), device="cuda") > 0.15
    mask[:, 0] = True
    return q, k, v, mask


def measure(fn, packets, reference, replays):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False)

    def forward():
        return [(compiled if len(q) >= 256 else fn)(q, k, v, m) for q, k, v, m in packets]

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = forward()
    graph.replay()
    torch.cuda.synchronize()
    for result, expected in zip(outputs, reference, strict=True):
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    times = []
    for _ in range(4):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / replays)
    return {"milliseconds": statistics.median(times[1:]), "rounds_ms": times}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20267105)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be positive")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    packets = [make_inputs(b, t, q) for b, t, last_q in ((4032, 21, 10), (4032, 28, 14), (128, 41, 20))
               for q in (t, last_q)]
    reference = [torch_attention(*packet) for packet in packets]
    results = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "triton_version": triton.__version__, "seed": args.seed,
               "input_shapes": [(list(q.shape), list(k.shape)) for q, k, _, _ in packets],
               "replays": args.replays, "precision": "fp32", "tf32": False}
    methods = (("torch", torch_attention), ("triton", short_attention), ("torch_repeat", torch_attention))
    with ResourceMonitor(interval=1, gpu=True, torch_device=torch.device("cuda:0")) as monitor:
        for name, fn in methods:
            results[name] = measure(fn, packets, reference, args.replays)
    results["resources"] = monitor.summary
    print(json.dumps(results))


if __name__ == "__main__":
    main()
