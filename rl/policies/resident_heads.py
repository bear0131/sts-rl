"""Current combat probabilities, evaluated only for live search consumers.

The archived scorer's device counts and legal-action dispatch are reused here;
the current pointer, conditional-target and prefix factors remain unchanged.
Illegal logits are private scratch, never an input to the search sampler.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def shared_queries(STATE, PW, PB, GW, EW, EB, COUNT, STATUS, QUERY, END,
                   N: tl.constexpr, SS: tl.constexpr):
    r = tl.program_id(0)*32+tl.arange(0, 32)
    d = tl.arange(0, 64)
    active = (r < N) & (r < tl.load(COUNT)) & (tl.load(STATUS+7) == 0)
    q = tl.zeros((32, 64), tl.float32)
    g = tl.zeros((32, 64), tl.float32)
    end = tl.zeros((32,), tl.float32)
    if tl.sum(active.to(tl.int32), 0) > 0:
        x = tl.load(STATE+r[:, None]*SS+d[None, :], active[:, None], 0.)
        q = tl.dot(x, tl.load(PW+d[:, None]+64*d[None, :]), input_precision='tf32x3')
        q += tl.load(PB+d)[None, :]
        g = tl.dot(x, tl.load(GW+d[:, None]+64*d[None, :]), input_precision='tf32x3')
        end = tl.sum(x*tl.load(EW+d)[None, :], 1)+tl.load(EB)
    tl.store(QUERY+r[:, None]*128+d[None, :], q, r[:, None] < N)
    tl.store(QUERY+r[:, None]*128+64+d[None, :], g, r[:, None] < N)
    tl.store(END+r, end, r < N)


@triton.jit
def source_targets(HAND, POTION, MONSTER, QUERY, END, KEY_W, KEY_B, SOURCE_W,
                   TARGET_W, LEGAL, COUNT, STATUS, OUT, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, 16)
    t = tl.arange(0, 16)
    d = tl.arange(0, 64)
    live = (row < tl.load(COUNT)) & (tl.load(STATUS+7) == 0)
    legal = tl.load(LEGAL+row*198+1+s[:, None]*6+t[None, :],
                    live & (s[:, None] < 15) & (t[None, :] < 6), False)
    number = tl.sum(legal.to(tl.int32), 1)
    output = tl.full((16, 16), -1.e9, tl.float32)
    if tl.sum(number, 0) > 0:
        x = tl.load(HAND+row*STRIDE+s[:, None]*64+d[None, :],
                     ((s < 10) & (number > 0))[:, None], 0.)
        x += tl.load(POTION+row*STRIDE+(s[:, None]-10)*64+d[None, :],
                      ((s >= 10) & (s < 15) & (number > 0))[:, None], 0.)
        key = tl.dot(x, tl.load(KEY_W+d[:, None]+64*d[None, :]), input_precision='tf32x3')
        key += tl.load(KEY_B+d)[None, :]
        pointer = tl.sum(key*tl.load(QUERY+row*128+d)[None, :], 1)*.125
        conditional = tl.zeros((16, 16), tl.float32)
        if tl.sum((number > 1).to(tl.int32), 0) > 0:
            q = tl.dot(x, tl.load(SOURCE_W+d[:, None]+64*d[None, :]), input_precision='tf32x3')
            q += tl.load(QUERY+row*128+64+d)[None, :]
            monsters = tl.load(MONSTER+row*STRIDE+t[:, None]*64+d[None, :], t[:, None] < 5, 0.)
            target = tl.dot(monsters, tl.load(TARGET_W+d[:, None]+64*d[None, :]), input_precision='tf32x3')
            score = tl.dot(q, tl.trans(target), input_precision='tf32x3')*.125
            score = tl.where(t[None, :] == 5, 0., score)
            score = tl.where(legal, score, -1.e9)
            maximum = tl.max(score, 1)
            # Match scores - logsumexp(scores), including FP32 rounding.
            normalizer = maximum+tl.log(tl.sum(tl.exp(score-maximum[:, None]), 1))
            conditional = tl.where((number > 1)[:, None], score-normalizer[:, None], 0.)
        output = tl.where(legal, pointer[:, None]+conditional, -1.e9)
    tl.store(OUT+row*198+1+s[:, None]*6+t[None, :], output,
              (s[:, None] < 15) & (t[None, :] < 6))
    end_legal = tl.load(LEGAL+row*198, live, False)
    tl.store(OUT+row*198, tl.where(end_legal, tl.load(END+row), -1.e9))


@triton.jit
def selection_scores(X, MAP, QUERY, KW, KB, NW, NB, LEGAL, COUNT, STATUS, OUT,
                      EPS: tl.constexpr, C: tl.constexpr, W: tl.constexpr, N: tl.constexpr):
    r = tl.program_id(0)*32+tl.arange(0, 32)
    d = tl.arange(0, 64)
    row, slot = r//96, r%96
    valid = (row < N) & (row < tl.load(COUNT)) & (tl.load(STATUS+7) == 0)
    legal = tl.load(LEGAL+row*198+91+slot, valid, False) & (slot < W)
    score = tl.full((32,), -1.e9, tl.float32)
    if tl.sum(legal.to(tl.int32), 0) > 0:
        index = tl.load(MAP+row*C+C-W+slot, legal, -1)
        x = tl.load(X+index[:, None]*64+d[None, :], (legal & (index >= 0))[:, None], 0.)
        u = x-tl.sum(x, 1)[:, None]/64.
        x = u*tl.rsqrt(tl.sum(u*u, 1)/64.+EPS)[:, None]
        x = x*tl.load(NW+d)[None, :]+tl.load(NB+d)[None, :]
        x = tl.where((index >= 0)[:, None], x, 0.)
        key = tl.dot(x, tl.load(KW+d[:, None]+64*d[None, :]), input_precision='tf32x3')
        key += tl.load(KB+d)[None, :]
        query = tl.load(QUERY+row[:, None]*128+d[None, :], legal[:, None], 0.)
        score = tl.where(legal, tl.sum(query*key, 1)*.125, -1.e9)
    tl.store(OUT+row*198+91+slot, score, row < N)


@triton.jit
def subset_scores(STATE, HAND, SELECTED, REMAINING, LEGAL, COUNT, STATUS,
                   HW, HB, SW, SB, PW, KW, EW, EB, COMPLETIONS, OUT,
                   SS: tl.constexpr, HS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 16)
    d = tl.arange(0, 64)
    h = tl.arange(0, 32)
    live = (row < tl.load(COUNT)) & (tl.load(STATUS+7) == 0)
    legal = tl.load(LEGAL+row*198+187+i, live & (i < 11), False)
    scores = tl.full((16,), -1.e9, tl.float32)
    if tl.sum(legal.to(tl.int32), 0) > 0:
        raw = tl.load(HAND+row*HS+i[:, None]*64+d[None, :], i[:, None] < 10, 0.)
        hand = tl.dot(raw, tl.load(HW+d[:, None]+64*h[None, :]), input_precision='tf32x3')
        hand += tl.load(HB+h)[None, :]
        chosen = tl.load(SELECTED+row*10+i, i < 10, 0.)
        prefix = tl.sum(hand*chosen[:, None], 0)
        state = tl.load(STATE+row*SS+d)
        state = tl.sum(state[:, None]*tl.load(SW+d[:, None]+64*h[None, :]), 0)+tl.load(SB+h)
        projection = tl.sum(prefix[:, None]*tl.load(PW+h[:, None]+32*h[None, :]), 0)
        query = libdevice.tanh(state+projection)
        key = tl.dot(hand, tl.load(KW+h[:, None]+32*h[None, :]), input_precision='tf32x3')
        scores = tl.sum(key*query[None, :], 1)*(32.**-.5)
        candidates = legal & (i < 10)
        later = tl.sum(candidates.to(tl.int32), 0)-tl.cumsum(candidates.to(tl.int32), 0)
        capacity = tl.minimum(tl.maximum(tl.load(REMAINING+row).to(tl.int64)-1, 0), 10)
        counts = tl.load(COMPLETIONS+later*11+capacity)
        scores += tl.log(counts)
        stop = tl.sum(query*tl.load(EW+h), 0)+tl.load(EB)
        scores = tl.where(i == 10, stop, scores)
        scores = tl.where(legal, scores, -1.e9)
        maximum = tl.max(scores, 0)
        normalizer = maximum+tl.log(tl.sum(tl.exp(scores-maximum), 0))
        scores = tl.where(legal, scores-normalizer, -1.e9)
    tl.store(OUT+row*198+187+i, scores, i < 11)


@torch.compiler.disable
def frozen_combat_logits(net, state, blocks, legal, selected, remaining, row_count):
    """Private legal logits for resident CUDA FP32; eager/training stay canonical."""
    if (torch.is_grad_enabled() or net.training or not state.is_cuda
            or state.dtype != torch.float32 or state.shape[1] != 64
            or not net.autoregressive_actions or legal.shape[1] != 198):
        raise ValueError('resident heads require frozen CUDA FP32 autoregressive64 combat')
    attention, mapping, status, consumers, width = blocks['_resident_selection']
    head = net.complete_head
    subset = head.subset
    if subset.hand.out_features != 32:
        raise ValueError('resident prefix decoder must have width32')
    n = len(state)
    query = torch.empty((n, 128), device=state.device)
    end = torch.empty(n, device=state.device)
    out = torch.empty((n, 198), device=state.device)
    legal, selected, remaining = legal.contiguous(), selected.contiguous(), remaining.contiguous()
    shared_queries[(triton.cdiv(n, 32),)](state, net.pointer_query.weight, net.pointer_query.bias,
        head.global_query.weight, net.end.weight, net.end.bias, row_count, status, query, end,
        n, state.stride(0), num_warps=4, enable_fp_fusion=False)
    hand, potion, monster = (blocks[k] for k in ('hand', 'potion', 'monster'))
    assert hand.stride(0) == potion.stride(0) == monster.stride(0)
    source_targets[(n,)](hand, potion, monster, query, end, net.pointer_key.weight,
        net.pointer_key.bias, head.source_query.weight, head.target_key.weight, legal, row_count,
        status, out, hand.stride(0), num_warps=4, enable_fp_fusion=False)
    selection_scores[(triton.cdiv(n*96, 32),)](attention, mapping, query, net.pointer_key.weight,
        net.pointer_key.bias, net.norm.weight, net.norm.bias, legal, row_count, status, out,
        net.norm.eps, consumers, width, n, num_warps=4, enable_fp_fusion=False)
    subset_scores[(n,)](state, hand, selected, remaining, legal, row_count, status,
        subset.hand.weight, subset.hand.bias, subset.state.weight, subset.state.bias,
        subset.prefix.weight, subset.choice_key.weight, subset.stop.weight, subset.stop.bias,
        subset.completion_counts, out, state.stride(0), hand.stride(0),
        num_warps=4, enable_fp_fusion=False)
    return out
