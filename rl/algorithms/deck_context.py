"""Exact chain-rule accumulation for repeated deck contexts in one Adam batch.

Keep only unique latent values and their adjoints between physical PPO chunks.
Recompute the large encoder with those adjoints before gradient synchronization;
shared card parameters also retain their ordinary actor/critic gradients.
"""
from contextlib import contextmanager

import torch


CONTEXT_INDEX = '_deck_context_index'


class ContextGradientBatch:
    def __init__(self, core, observations, config):
        if not core.training or not torch.is_grad_enabled():
            raise ValueError('context gradient replay requires training with gradients')
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in core.context.modules()):
            raise ValueError('context gradient replay requires a deterministic encoder')
        self.core = core
        self.parameters = list(core.context.parameters())
        self.versions = tuple((id(p), p._version) for p in self.parameters)
        keys = core.context.key_tensor(observations)
        unique, self.inverse = torch.unique(keys, dim=0, return_inverse=True)
        first = torch.full((len(unique),), len(keys), device=keys.device, dtype=torch.long)
        first.scatter_reduce_(0, self.inverse, torch.arange(len(keys), device=keys.device), reduce='amin')
        self.observations = {k: observations[k][first] for k in (*core.context.keys, 'phase')}
        # Bound the actual padded encoder sequence, including public map tokens.
        obs = self.observations
        deck = int((obs['deck_ids'] != 0).any(0).nonzero()[-1].item()) + 1 if obs['deck_ids'].any() else 1
        relic = int((obs['relics'] != 0).any(0).nonzero()[-1].item()) + 1 if obs['relics'].any() else 1
        self.cap = min(config.max_rows, max(1, config.token_budget // (1 + deck + relic + obs['map_full_rooms'].shape[1])))
        self.oom_retries = 0
        values = []
        start = 0
        while start < len(unique):
            count = min(self.cap, len(unique) - start)
            values.extend(self._forward(start, count))
            start += count
        self.latents = torch.cat(values).requires_grad_()
        self.input_rows, self.unique_rows = len(keys), len(unique)

    def _slice(self, start, count):
        return {k: v[start:start + count] for k, v in self.observations.items()}

    def _split(self, start, count, operation):
        if count <= 1:
            raise torch.OutOfMemoryError('one deck context does not fit device memory')
        self.oom_retries += 1
        self.cap = min(self.cap, max(1, count // 2))
        if self.latent_device.type == 'cuda':
            torch.cuda.empty_cache()
        return operation(start, count // 2) + operation(start + count // 2, count - count // 2)

    @property
    def latent_device(self):
        return self.observations['phase'].device

    @torch.no_grad()
    def _forward(self, start, count):
        # Leave training=True: the recomputation must use the same attention
        # path. Inference fast paths can have different floating point rounding.
        failed = False
        try:
            return [self.core.context(self._slice(start, count)).detach()]
        except torch.OutOfMemoryError:
            failed = True
        if failed:
            return self._split(start, count, self._forward)

    def reset_gradient(self):
        self.latents.grad = None

    def backward(self):
        if self.versions != tuple((id(p), p._version) for p in self.parameters):
            raise RuntimeError('context weights changed before gradient replay')
        if self.latents.grad is None:
            raise RuntimeError('context replay received no latent gradients')
        start = 0
        while start < self.unique_rows:
            count = min(self.cap, self.unique_rows - start)
            self._backward(start, count)
            start += count

    def _backward(self, start, count):
        # A partially failed backward may have touched shared card gradients.
        # Isolate all encoder grads, then commit only a successful whole chunk.
        previous = [p.grad for p in self.parameters]
        for p in self.parameters:
            p.grad = None
        value, failed = None, False
        try:
            value = self.core.context(self._slice(start, count))
            value.backward(self.latents.grad[start:start + count])
        except torch.OutOfMemoryError:
            failed = True
        except BaseException:
            for p, old in zip(self.parameters, previous, strict=True):
                p.grad = old
            raise
        value = None
        if failed:
            for p, old in zip(self.parameters, previous, strict=True):
                p.grad = old
            return self._split(start, count, self._backward)
        old_grads, additions = [], []
        for p, old in zip(self.parameters, previous, strict=True):
            if old is not None:
                if p.grad is not None:
                    old_grads.append(old); additions.append(p.grad)
                p.grad = old
        if additions:
            torch._foreach_add_(old_grads, additions)
        return []


@contextmanager
def context_gradient_batch(model, batch, config):
    if not config.context_gradient_replay:
        if config.context_grouping:
            core = model.policy.core
            # These integer identities are valid only inside this original
            # optimizer batch, and travel with rows through every permutation.
            keys = core.context.key_tensor(batch.observations)
            _, inverse = torch.unique(keys, dim=0, return_inverse=True)
            observations = dict(batch.observations, **{CONTEXT_INDEX: inverse})
            data = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
            yield type(batch)(**dict(data, observations=observations)), None
        else:
            yield batch, None
        return
    core = model.policy.core
    if getattr(core, '_context_gradient_batch', None) is not None:
        raise RuntimeError('nested deck context gradient batches are not supported')
    replay = ContextGradientBatch(core, batch.observations, config)
    core._context_gradient_batch = replay
    observations = dict(batch.observations, **{CONTEXT_INDEX: replay.inverse})
    data = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
    try:
        yield type(batch)(**dict(data, observations=observations)), replay
    finally:
        core._context_gradient_batch = None
        replay.latents.grad = None
