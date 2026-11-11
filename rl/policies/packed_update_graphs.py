"""Bounded CUDA forward/backward graphs for the two common deck-latent PPO heads."""
from collections import Counter
from copy import deepcopy

import torch
from torch import nn

from env.core import vocab


class _FieldsRead(dict):
    def __init__(self, fields):
        super().__init__(fields)
        self.used = set()

    def __contains__(self, key):
        present = super().__contains__(key)
        if present:
            self.used.add(key)
        return present

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


class _RewardHead(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.overworld = core.overworld
        self.tokenizer = core.upper_tokenizer

    def forward(self, obs, widths):
        return self.overworld(obs, self.tokenizer, vocab.PHASE_VOCAB_INDEX['combat_reward'],
                              upper_active_widths=widths)


class _CapturedHead:
    def __init__(self, module, observations, context, rows, widths):
        self.shape = rows, widths
        self.weights = tuple(p for p in module.parameters() if p.requires_grad)
        self.busy = False
        reward = isinstance(module, _RewardHead)

        def fields_for(obs, z):
            fields = dict(obs, _run_context=z)
            if not reward:
                fields['_choice_width'] = widths
            return fields

        probe = _FieldsRead(fields_for({k: t[:1] for k, t in observations.items()}, context[:1]))
        with torch.no_grad():
            if reward:
                module(probe, widths)
            else:
                module(probe)
        self.keys = tuple(sorted(probe.used.intersection(observations)))
        self.inputs = []
        for tensor in (context, *(observations[key] for key in self.keys)):
            static = tensor.detach()[:1].expand(rows, *tensor.shape[1:]).clone()
            static[:len(tensor)].copy_(tensor.detach())
            self.inputs.append(static)
        self.inputs[0].requires_grad_(True)

        # Capture with stream-local autograd leaves. Keep one leaf per tied
        # parameter and share its storage with the optimizer's original tensor.
        # Reparameterizing the live module would also mutate sibling heads that
        # refer to the same card embedding module.
        leaves = tuple(nn.Parameter(p.detach()) for p in self.weights)
        memo = {id(p): leaf for p, leaf in zip(self.weights, leaves)}
        memo.update({id(buffer): buffer for buffer in module.buffers()})
        self.graph_module = deepcopy(module, memo)

        def compute():
            fields = fields_for(dict(zip(self.keys, self.inputs[1:])), self.inputs[0])
            args = (fields, widths) if reward else (fields,)
            logits, values, *_ = self.graph_module(*args)
            return torch.cat((logits, values.reshape(rows, 1)), dim=1)

        inputs = (self.inputs[0], *leaves)
        with torch.enable_grad():
            warm = compute()
            torch.autograd.grad(warm, inputs, torch.ones_like(warm), allow_unused=True)
            del warm
            self.forward_graph = torch.cuda.CUDAGraph()
            self.backward_graph = torch.cuda.CUDAGraph()
            options = dict(pool=torch.cuda.graph_pool_handle(), stream=torch.cuda.current_stream(),
                           capture_error_mode='thread_local')
            with torch.cuda.graph(self.forward_graph, **options):
                result = compute()
            self.output = result.detach()
            self.upstream = torch.empty_like(result)
            with torch.cuda.graph(self.backward_graph, **options):
                grads = torch.autograd.grad(result, inputs, self.upstream, allow_unused=True)
        self.used = tuple(i for i, grad in enumerate(grads) if grad is not None)
        self.gradients = tuple(grads[i] for i in self.used)

    def replay(self, obs, context):
        packed = _Replay.apply(self, context, *(obs[key] for key in self.keys), *self.weights)
        return packed[:, :-1], packed[:, -1:]


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, entry, context, *args):
        if entry.busy:
            raise RuntimeError('overlapping uses of a packed training graph')
        entry.busy = True
        ctx.entry, ctx.rows = entry, len(context)
        torch._foreach_copy_([t[:ctx.rows] for t in entry.inputs], [context, *args[:len(entry.keys)]])
        entry.forward_graph.replay()
        return entry.output[:ctx.rows].clone()

    @staticmethod
    def backward(ctx, gradient):
        entry, n = ctx.entry, ctx.rows
        entry.upstream[:n].copy_(gradient)
        entry.upstream[n:].zero_()
        entry.backward_graph.replay()
        # Owned copies must survive later replays and deferred gradient additions.
        src = [g[:n] if i == 0 else g for i, g in zip(entry.used, entry.gradients)]
        copies = [torch.empty_like(g) for g in src]
        torch._foreach_copy_(copies, src)
        grads = dict(zip(entry.used, copies))
        entry.busy = False
        return (None, grads.get(0), *([None] * len(entry.keys)),
                *(grads.get(i + 1) for i in range(len(entry.weights))))


class PackedHeadGraphs:
    def __init__(self, core):
        self.core = core
        self.reward = _RewardHead(core)
        self.entries = {}
        self.retired = set()
        self.stats = Counter()
        self.capturing = None

    def _entry(self, name, module, obs, z, rows, widths):
        entry = self.entries.get(name)
        if entry is None and name not in self.retired:
            self.capturing = name
            entry = _CapturedHead(module, obs, z, rows, widths)
            self.capturing = None
            self.entries[name] = entry
            self.stats[name + '_captures'] += 1
        return entry if entry is not None and entry.shape == (rows, widths) else None

    def combat(self, obs, z, width):
        slots = obs['select_native_indices'].shape[1]
        fixed = 5 + obs['hand_ids'].shape[1] + obs['monster_id'].shape[1] + obs['potions'].shape[1]
        aligned_width = min(slots, (fixed + width + 7) // 8 * 8 - fixed)
        rows = 896
        if rows * .7 <= len(z) <= rows and aligned_width <= 23:
            entry = self._entry('combat', self.core.combat, obs, z, rows, min(23, slots))
            if entry is not None:
                self.stats['combat_replays'] += 1
                return entry.replay(obs, z)
        # This is also the original path used by the qualified shape-miss arm.
        return self.core.combat_with_context(obs, z)

    def upper(self, obs, z, phase, deck, relic, kept):
        if phase == vocab.PHASE_VOCAB_INDEX['combat_reward'] and len(z) >= 192:
            constant = 2 + obs['potions'].shape[1] + int('reward_keys' in obs)
            constant += sum(obs[k].shape[1] for k in
                            ('offer_card_ids', 'reward_gold', 'reward_relic_ids', 'reward_potion_ids'))
            widths = (min(obs['deck_ids'].shape[1], (deck + 31) // 32 * 32),
                      min(obs['relics'].shape[1], (relic + 15) // 16 * 16),
                      (kept + constant + 15) // 16 * 16)
            entry = self._entry('upper', self.reward, obs, z, (len(z) + 63) // 64 * 64, widths)
            if entry is not None:
                self.stats['upper_replays'] += 1
                return entry.replay(obs, z)
        return self.core.overworld(dict(obs, _run_context=z), self.core.upper_tokenizer, phase)

    def recover_oom(self):
        for entry in self.entries.values():
            entry.busy = False
        if self.capturing is not None:
            # The constructor failed before the graph could enter the cache.
            # Retry this geometry eagerly before shrinking any physical block.
            self.retired.add(self.capturing)
            self.stats[self.capturing + '_retired'] += 1
            self.capturing = None
            return True
        for name in ('upper', 'combat'):
            if name in self.entries:
                del self.entries[name]
                self.retired.add(name)
                self.stats[name + '_retired'] += 1
                return True
        return False
