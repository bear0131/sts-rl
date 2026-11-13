"""Frozen 64-wide search blocks using the accepted GPU search kernel layout.

FP32 tensors/accumulators, tf32x3 projections and IEEE attention match the old
ragged search arithmetic. This is inference-only; learner autograd is untouched.
The caller's CUDA graph owns all temporary tensors and parameter lifetimes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(x, weight, bias, eps: tl.constexpr):
    d = tl.arange(0, 64)
    centered = x - tl.sum(x, 1)[:, None] / 64.
    normalized = centered * tl.rsqrt(tl.sum(centered * centered, 1) / 64. + eps)[:, None]
    return normalized * tl.load(weight + d)[None, :] + tl.load(bias + d)[None, :]


@triton.jit
def _qkv(X, NW, NB, W, B, Y, N: tl.constexpr, EPS: tl.constexpr):
    d = tl.arange(0, 64)
    columns = tl.program_id(1) * 64 + d
    weight = tl.load(W + columns[None, :] * 64 + d[:, None])
    for block in range(tl.program_id(0), tl.cdiv(N, 32), tl.num_programs(0)):
        rows = block * 32 + tl.arange(0, 32)
        x = tl.load(X + rows[:, None] * 64 + d[None, :], rows[:, None] < N, 0.)
        z = _norm(x, NW, NB, EPS)
        y = tl.dot(z, weight, input_precision='tf32x3') + tl.load(B + columns)[None, :]
        tl.store(Y + rows[:, None] * 192 + columns[None, :], y, rows[:, None] < N)


@triton.jit
def _attention(QKV, PRESENT, OUT, T: tl.constexpr, KT: tl.constexpr):
    state, head = tl.program_id(0) // 4, tl.program_id(0) % 4
    qi = tl.program_id(1) * 16 + tl.arange(0, 16)
    ki, di = tl.arange(0, KT), tl.arange(0, 16)
    q = tl.load(QKV + (state*T+qi[:, None])*192 + head*16 + di[None, :], qi[:, None] < T, 0.)
    k = tl.load(QKV + (state*T+ki[None, :])*192 + 64 + head*16 + di[:, None], ki[None, :] < T, 0.)
    live = tl.load(PRESENT + state*T + ki, ki < T, False)
    scores = tl.where(live[None, :], tl.dot(q, k, input_precision='ieee') * .25, -float('inf'))
    probability = tl.exp(scores - tl.max(scores, 1)[:, None])
    probability = probability / tl.sum(probability, 1)[:, None]
    v = tl.load(QKV + (state*T+ki[:, None])*192 + 128 + head*16 + di[None, :], ki[:, None] < T, 0.)
    output = tl.dot(probability, v, input_precision='ieee')
    tl.store(OUT + (state*T+qi[:, None])*64 + head*16 + di[None, :], output, qi[:, None] < T)


@triton.jit
def _mix_feed(A, X, MW, MB, NW, NB, F1, B1, F2, B2, OUT, N: tl.constexpr, EPS: tl.constexpr):
    d, h = tl.arange(0, 64), tl.arange(0, 128)
    for block in tl.range(tl.program_id(0), tl.cdiv(N, 32), tl.num_programs(0), disable_licm=True):
        rows = block * 32 + tl.arange(0, 32)
        a = tl.load(A + rows[:, None]*64 + d[None, :], rows[:, None] < N, 0.)
        residual = tl.load(X + rows[:, None]*64 + d[None, :], rows[:, None] < N, 0.)
        wm = tl.load(MW + d[None, :]*64 + d[:, None])
        residual += tl.dot(a, wm, input_precision='tf32x3') + tl.load(MB + d)[None, :]
        normalized = _norm(residual, NW, NB, EPS)
        w1 = tl.load(F1 + h[None, :]*64 + d[:, None])
        hidden = tl.maximum(tl.dot(normalized, w1, input_precision='tf32x3') + tl.load(B1 + h)[None, :], 0.)
        w2 = tl.load(F2 + d[None, :]*128 + h[:, None])
        output = residual + (tl.dot(hidden, w2, input_precision='tf32x3') + tl.load(B2 + d)[None, :])
        tl.store(OUT + rows[:, None]*64 + d[None, :], output, rows[:, None] < N)


def frozen_block(block, x, present):
    if (torch.is_grad_enabled() or block.training or x.dtype != torch.float32
            or not x.is_cuda or x.shape[-1] != 64 or block.heads != 4
            or block.ff1.weight.shape != (128, 64)):
        raise ValueError('resident kernels require frozen CUDA FP32 64x4/128 blocks')
    x, present = x.contiguous(), present.contiguous()
    batch, length, _ = x.shape
    qkv = torch.empty((batch*length, 192), device=x.device, dtype=x.dtype)
    attention, output = torch.empty_like(x), torch.empty_like(x)
    grid = min(triton.cdiv(batch*length, 32), torch.cuda.get_device_properties(x.device).multi_processor_count * 8)
    _qkv[(grid, 3)](x, block.norm1.weight, block.norm1.bias, block.qkv.weight,
        block.qkv.bias, qkv, batch*length, block.norm1.eps, num_warps=4, num_stages=1)
    _attention[(batch*4, triton.cdiv(length, 16))](qkv, present, attention,
        length, triton.next_power_of_2(length), num_warps=1, num_stages=1)
    _mix_feed[(grid,)](attention, x, block.mix.weight, block.mix.bias,
        block.norm2.weight, block.norm2.bias, block.ff1.weight, block.ff1.bias,
        block.ff2.weight, block.ff2.bias, output, batch*length, block.norm2.eps,
        num_warps=4, num_stages=1)
    return output
