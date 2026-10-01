"""Controlled search workload with identical replies and optional real inference.

Both modes return zero policy logits and value 0.5 to the search. Shadow mode
also runs the real network and discards its output. This isolates inference
cost without changing the tree workload; it does not measure policy strength.
The collector, observation encoding, transfers and native search still run.
"""
import time

import numpy as np
import torch

from rl.search.native import forward_leaves


class VirtualInference:
    def __init__(self, model, *, shadow=False):
        self.model = model
        self.shadow = shadow
        self.backend = getattr(model, '_native_leaf_inference', None)
        self.root_forward = model.policy.core.actor_critic
        self.value_forward = model.policy.core.value
        self.root_calls = self.leaf_calls = self.leaf_rows = 0
        self.value_calls = 0
        self.model_seconds = 0.0
        self.max_rows = 0
        self.logits = self.values = None
        self._measuring_value = False

    def install(self):
        self.model.policy.core.actor_critic = self.root
        self.model.policy.core.value = self.value
        self.model._native_leaf_inference = self
        return self

    def synchronize(self):
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)

    def root(self, observations, **kwargs):
        if self._measuring_value:
            return self.root_forward(observations, **kwargs)
        self.root_calls += 1
        if self.shadow:
            self.synchronize()
            started = time.perf_counter()
            self.root_forward(observations, **kwargs)
            self.synchronize()
            self.model_seconds += time.perf_counter() - started
        count = len(next(iter(observations.values())))
        return (torch.zeros((count, self.model.action_space.n), device=self.model.device),
                torch.full((count, 1), .5, device=self.model.device))

    def __call__(self, observations, width):
        count = len(next(iter(observations.values())))
        self.leaf_calls += 1
        self.leaf_rows += count
        if self.shadow:
            self.synchronize()
            started = time.perf_counter()
            forward_leaves(self.model, observations, autocast=self.model.autocast_context,
                           leaf_backend=self.backend)
            self.synchronize()
            self.model_seconds += time.perf_counter() - started
        if count > self.max_rows:
            self.logits = np.zeros((count, self.model.action_space.n), dtype=np.float32)
            self.values = np.full(count, .5, dtype=np.float32)
            self.max_rows = count
        return self.logits[:count], self.values[:count]

    def value(self, observations):
        self.value_calls += 1
        if self.shadow:
            self.synchronize()
            started = time.perf_counter()
            self._measuring_value = True
            try:
                self.value_forward(observations)
            finally:
                self._measuring_value = False
            self.synchronize()
            self.model_seconds += time.perf_counter() - started
        count = len(next(iter(observations.values())))
        return torch.full((count, 1), .5, device=self.model.device)

    def statistics(self):
        return dict(mode='shadow' if self.shadow else 'virtual', root_calls=self.root_calls,
                    value_calls=self.value_calls,
                    leaf_calls=self.leaf_calls, leaf_rows=self.leaf_rows,
                    model_seconds=self.model_seconds,
                    graphs=None if self.backend is None else self.backend.statistics())

    def clear(self):
        if self.backend is not None:
            self.backend.clear()
