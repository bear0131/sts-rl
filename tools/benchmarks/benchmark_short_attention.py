"""Measure strict FP32 inference for the short combat attention shapes on CUDA.

The primitive benchmark uses deterministic synthetic activations and the length
groups seen in the 8192-state combat benchmark or fully packed variable-length
sequences. Both modes check the CUDA output against PyTorch attention.
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


@triton.jit
def _ragged_attention(
    qkv, present, token_offsets, query_offsets, query_token, output,
    HEADS: tl.constexpr, WIDTH: tl.constexpr, ALL_QUERIES: tl.constexpr,
    QUERY_TILE: tl.constexpr, KEY_TILE: tl.constexpr, SCALE: tl.constexpr,
):
    pair = tl.program_id(0)
    row, head = pair // HEADS, pair % HEADS
    begin, end = tl.load(token_offsets + row), tl.load(token_offsets + row + 1)
    first, limit = tl.load(query_offsets + row), tl.load(query_offsets + row + 1)
    first += tl.program_id(1) * QUERY_TILE
    if first < limit:
        queries = first + tl.arange(0, QUERY_TILE)
        dims = tl.arange(0, WIDTH)
        columns = tl.arange(0, KEY_TILE)
        if ALL_QUERIES:
            source = queries
        else:
            source = tl.load(query_token + queries, queries < limit, 0)
        stride: tl.constexpr = 3 * HEADS * WIDTH
        base = qkv + head * WIDTH
        q = tl.load(base + source[:, None] * stride + dims[None, :], queries[:, None] < limit, 0.0)
        if end - begin <= KEY_TILE:
            keys = begin + columns
            valid = (keys < end) & tl.load(present + keys, keys < end, False)
            k = tl.load(base + HEADS * WIDTH + keys[None, :] * stride + dims[:, None], keys[None, :] < end, 0.0)
            scores = tl.where(valid[None, :], tl.dot(q, k, input_precision="ieee") * SCALE, -float("inf"))
            weights = tl.where(valid[None, :], tl.exp(scores - tl.max(scores, 1)[:, None]), 0.0)
            weights /= tl.maximum(tl.sum(weights, 1), 1.0)[:, None]
            v = tl.load(base + 2 * HEADS * WIDTH + keys[:, None] * stride + dims[None, :], keys[:, None] < end, 0.0)
            y = tl.dot(weights, v, input_precision="ieee")
        else:
            maximum = tl.full((QUERY_TILE,), -float("inf"), tl.float32)
            total = tl.full((QUERY_TILE,), 0.0, tl.float32)
            acc = tl.full((QUERY_TILE, WIDTH), 0.0, tl.float32)
            for tile in range(0, tl.cdiv(end - begin, KEY_TILE)):
                keys = begin + tile * KEY_TILE + columns
                valid = (keys < end) & tl.load(present + keys, keys < end, False)
                k = tl.load(base + HEADS * WIDTH + keys[None, :] * stride + dims[:, None], keys[None, :] < end, 0.0)
                scores = tl.where(valid[None, :], tl.dot(q, k, input_precision="ieee") * SCALE, -float("inf"))
                updated = tl.maximum(maximum, tl.max(scores, 1))
                scale = tl.where(total > 0, tl.exp(maximum - updated), 0.0)
                weights = tl.where(valid[None, :], tl.exp(scores - updated[:, None]), 0.0)
                v = tl.load(base + 2 * HEADS * WIDTH + keys[:, None] * stride + dims[None, :], keys[:, None] < end, 0.0)
                acc = acc * scale[:, None] + tl.dot(weights, v, input_precision="ieee")
                total = total * scale + tl.sum(weights, 1)
                maximum = updated
            y = acc / tl.maximum(total, 1.0)[:, None]
        tl.store(output + (queries[:, None] * HEADS + head) * WIDTH + dims[None, :], y, queries[:, None] < limit)


def ragged_attention(qkv, present, token_offsets, query_offsets, query_token, max_queries, all_queries):
    """Score packed QKV [token, 3, head, dim] with offsets of length batch + 1.

    query_token contains global token indices within each state's own interval.
    For all_queries, query_offsets equals token_offsets and every token is a query.
    """
    heads, width = qkv.shape[-2:]
    count = len(qkv) if all_queries else len(query_token)
    result = qkv.new_empty((count, heads, width))
    tile = 32 if max_queries > 16 else 16
    _ragged_attention[((len(token_offsets) - 1) * heads, triton.cdiv(max_queries, tile))](
        qkv, present, token_offsets, query_offsets, query_token, result,
        heads, width, all_queries, tile, 32, 1 / math.sqrt(width), num_warps=2,
    )
    return result


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


def make_ragged_inputs(batch, tokens):
    # Variable lengths span both the single-tile and the online softmax paths.
    rows = torch.arange(batch, device="cuda")
    lengths = torch.randint(1, tokens + 1, (batch,), device="cuda")
    lengths[:4] = lengths.new_tensor([0, tokens, tokens, 1])
    position = torch.arange(tokens, device="cuda")
    real = position[None, :] < lengths[:, None]
    present = (torch.rand((batch, tokens), device="cuda") > 0.15) & real
    present[1] = False
    present[2, :32] = False
    present[2, -1] = True
    values = torch.randn((batch, tokens, 3, 4, 16), device="cuda")
    q, k, v = values.permute(2, 0, 3, 1, 4)
    expected = torch_attention(q, k, v, present).transpose(1, 2)
    packed = values[real].contiguous()
    offset = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).int()
    # Repeated and non-monotone queries must preserve the caller's action order.
    local = torch.randint(tokens, (batch, 14), device="cuda") % lengths.clamp_min(1)[:, None]
    qlengths = torch.minimum(lengths, torch.randint(1, 15, (batch,), device="cuda"))
    active = torch.arange(14, device="cuda")[None, :] < qlengths[:, None]
    qoffset = torch.cat((lengths.new_zeros(1), qlengths.cumsum(0))).int()
    indices = (offset[:-1, None] + local)[active].int()
    live = present[real]
    packets = [(packed, live, offset, offset, indices, tokens, True),
               (packed, live, offset, qoffset, indices, 14, False)]
    reference = [expected[real], expected[rows[:, None], local][active]]
    return packets, reference


def measure(fn, packets, reference, replays, *, compile_large=True):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False) if compile_large else fn

    def forward():
        return [(compiled if len(packet[0]) >= 256 else fn)(*packet) for packet in packets]

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
    parser.add_argument("--layout", choices=("fixed", "ragged"), default="fixed")
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20267105)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be positive")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {"device": torch.cuda.get_device_name(), "torch_version": torch.__version__, "cuda": torch.version.cuda,
               "triton_version": triton.__version__, "seed": args.seed, "layout": args.layout,
               "replays": args.replays, "precision": "fp32", "tf32": False}
    if args.layout == "ragged":
        packets, reference = make_ragged_inputs(4096, 48)
        results["states"] = 4096
        results["packed_tokens"] = len(packets[0][0])
        results["selected_queries"] = len(packets[1][4])
        methods = (("triton", ragged_attention),)
    else:
        packets = [make_inputs(b, t, q) for b, t, last_q in ((4032, 21, 10), (4032, 28, 14), (128, 41, 20))
                   for q in (t, last_q)]
        reference = [torch_attention(*packet) for packet in packets]
        results["input_shapes"] = [(list(q.shape), list(k.shape)) for q, k, _, _ in packets]
        methods = (("torch", torch_attention), ("triton", short_attention), ("torch_repeat", torch_attention))
    with ResourceMonitor(interval=1, gpu=True, torch_device=torch.device("cuda:0")) as monitor:
        for name, fn in methods:
            results[name] = measure(fn, packets, reference, args.replays, compile_large=args.layout == "fixed")
    results["resources"] = monitor.summary
    print(json.dumps(results))


if __name__ == "__main__":
    main()
