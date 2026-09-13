"""Native NOSL protocol shared by rollout, evaluation and interactive search.

The caller owns environments, root policy sampling and randomness. This module
owns batched leaf decoding/inference/backprop; it never resets a game or updates
model weights. Transport callbacks support local environments and VecEnv IPC.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import time

import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from env.core.run_env import unpack_nosl_observation_buffer
from rl.policies.model import observation_for_policy


@dataclass(frozen=True)
class NativeSearchConfig:
    simulations: int = 128
    depth: int = 0
    leaf_batch_size: int = 1
    c_puct: float = 1.5
    prior_temperature: float = 1.0
    end_turn_as_leaf: bool = False
    root_search_mode: int = 2
    gumbel_candidates: int = 0
    gumbel_scale: float = 1.0
    gumbel_value_scale: float = 0.1
    gumbel_max_visit_init: float = 50.0
    gumbel_rescale_q: bool = True
    gumbel_use_mixed_value: bool = True
    gumbel_interior_uniformity: float = 0.0
    gumbel_interior_min_visits: int = 1
    gumbel_interior_coverage_fraction: float = 0.0
    gumbel_interior_value_mode: int = 2
    gumbel_interior_max_mix_visits: float = 64.0
    paper_full_gumbel: bool = True

    def __post_init__(self):
        if self.simulations < 1 or self.depth < 0 or self.leaf_batch_size < 1:
            raise ValueError("search requires simulations >= 1, depth >= 0, leaf_batch_size >= 1")
        if self.root_search_mode not in (0, 1, 2):
            raise ValueError("root_search_mode must be 0 (PUCT), 1 (Gumbel) or 2 (Full Gumbel)")
        if self.paper_full_gumbel and self.root_search_mode != 2:
            raise ValueError("paper_full_gumbel requires root_search_mode=2")
        if self.prior_temperature <= 0 or self.c_puct < 0 or self.gumbel_candidates < 0:
            raise ValueError("invalid search temperature, PUCT coefficient or candidate count")
        if any(isinstance(value, float) and not np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("search parameters must be finite")

    @classmethod
    def from_model(cls, model, **overrides):
        values = {f.name: getattr(model, 'nosl_is_az_' + f.name, f.default) for f in fields(cls)}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    @classmethod
    def from_mapping(cls, values):
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})


def observation_as_tensor(observation, device):
    """Transfer fields in one contiguous buffer per dtype, retaining shapes."""
    groups = {}
    for key, value in observation.items():
        array = np.ascontiguousarray(value)
        groups.setdefault(array.dtype, []).append((key, array))
    if any(dtype not in (np.dtype(np.int64), np.dtype(np.float32)) for dtype in groups):
        return obs_as_tensor(observation, device)
    result = {}
    for fields_by_dtype in groups.values():
        flats = [array.reshape(-1) for _, array in fields_by_dtype]
        packed = np.concatenate(flats) if len(flats) > 1 else flats[0]
        tensor = th.as_tensor(packed, device=device)
        offset = 0
        for key, array in fields_by_dtype:
            result[key] = tensor[offset:offset + array.size].reshape(array.shape)
            offset += array.size
    return result


def supports_compact(model):
    core = getattr(model.policy, 'core', None)
    return hasattr(core, 'combat_actor_critic') and hasattr(core, 'combat_observation_keys')


def forward_root(model, observations, *, lock=None):
    tensor, _ = model.policy.obs_to_tensor(observation_for_policy(model, observations))
    with lock or nullcontext(), th.inference_mode():
        logits, values = model.policy.core.actor_critic(tensor)
    return logits.float().cpu().numpy(), values.float().reshape(-1).cpu().numpy()


def forward_leaves(model, observations, *, compact=True, lock=None, autocast=nullcontext):
    tensor = observation_as_tensor(observations, model.device)
    core = model.policy.core
    with lock or nullcontext(), th.inference_mode(), autocast():
        if compact:
            active = np.flatnonzero(np.any(observations['relics'] != 0, axis=0))
            width = int(active[-1]) + 1 if active.size else 1
            logits, values = core.combat_actor_critic(tensor, active_relic_width=width)
        else:
            logits, values = core.actor_critic(tensor)
    return logits.float().cpu().numpy(), values.float().reshape(-1).cpu().numpy()


def decode_leaves(responses):
    """Decode each response in stable worker/request order, including mixed transports."""
    records, batches = [], []
    for index in sorted(responses):
        response = responses[index]
        if isinstance(response, list):
            response = {'leaves': response}
        packed = unpack_nosl_observation_buffer(response)
        meta = response.get('leaf_meta') if packed is not None else response.get('leaves')
        if not meta:
            raise RuntimeError(f'native search {index} returned no leaves before completion')
        if packed is None:
            packed = {key: np.stack([leaf['observation'][key] for leaf in meta])
                      for key in meta[0]['observation']}
        if any(len(value) != len(meta) for value in packed.values()):
            raise ValueError('native leaf metadata/observation batch size mismatch')
        records.extend((index, leaf) for leaf in meta)
        batches.append(packed)
    keys = tuple(batches[0])
    if any(set(batch) != set(keys) for batch in batches):
        raise ValueError('native workers returned incompatible observation schemas')
    observations = {key: np.ascontiguousarray(np.concatenate([b[key] for b in batches], axis=0))
                    for key in keys}
    return records, observations


def complete_searches(model, call_workers, responses, *, leaf_batch_size,
                      compact=True, lock=None, autocast=nullcontext,
                      prior_mode='model', check=lambda: None):
    """Finish already-started trees with one forward across all active workers.

    Returns results and counters. The caller must abort owned sessions when an
    exception occurs; training workers are disposed by their enclosing runner.
    """
    if prior_mode not in ('model', 'uniform', 'none'):
        raise ValueError(f'unknown prior mode: {prior_mode}')
    results = {}
    stats = dict(leaf_batches=0, leaf_evaluations=0, cpp_leaf_decode_seconds=0.0,
                 cpp_leaf_model_seconds=0.0, cpp_backprop_rpc_seconds=0.0)
    by_worker = {index: {'leaf_evaluations': 0, 'leaf_batches': 0} for index in responses}
    while responses:
        check()
        active = {}
        for index, response in responses.items():
            if response.get('done', False):
                results[index] = response['result']
            else:
                active[index] = response
        if not active:
            break
        started = time.perf_counter()
        records, observations = decode_leaves(active)
        stats['cpp_leaf_decode_seconds'] += time.perf_counter() - started
        stats['leaf_batches'] += 1
        stats['leaf_evaluations'] += len(records)
        for index in active:
            by_worker[index]['leaf_batches'] += 1
        for index, _ in records:
            by_worker[index]['leaf_evaluations'] += 1
        started = time.perf_counter()
        logits, values = forward_leaves(model, observations, compact=compact, lock=lock, autocast=autocast)
        stats['cpp_leaf_model_seconds'] += time.perf_counter() - started
        updates = {}
        for position, (index, leaf) in enumerate(records):
            policy = logits[position] if prior_mode == 'model' else np.zeros_like(logits[position])
            if not leaf.get('needs_policy', True):
                policy = np.zeros(0, dtype=np.float64)
            updates.setdefault(index, []).append((int(leaf['request_id']), float(values[position]), policy))
        check()
        started = time.perf_counter()
        responses = call_workers('nosl_is_az_cpp_backprop_and_leaf_batch',
                                 {i: ((payload,), {'max_leaves': leaf_batch_size}) for i, payload in updates.items()})
        if set(responses) != set(active):
            raise RuntimeError('native worker response IDs do not match requests')
        stats['cpp_backprop_rpc_seconds'] += time.perf_counter() - started
    stats['by_worker'] = by_worker
    return results, stats


def search(model, env, observation, config, *, nonce, decision_index=0, stochastic=False,
           lock=None, root_outputs=None, prior_mode='model', check=lambda: None):
    """Single-tree adapter used by interactive and fixed-scenario evaluation."""
    started = time.monotonic()
    base = env.unwrapped
    compact = supports_compact(model)
    logits, value = root_outputs if root_outputs is not None else forward_root(model, observation, lock=lock)
    logits = np.asarray(logits).reshape(-1)
    parameters = asdict(config)
    if prior_mode != 'model':
        logits = np.zeros_like(logits)
    if prior_mode == 'none':
        parameters['c_puct'] = 0.0
    try:
        check()
        response = base.nosl_is_az_cpp_begin(
            logits, root_value=float(np.asarray(value).reshape(-1)[0]),
            state_nonce=int(nonce) & ((1 << 64) - 1), decision_index=decision_index,
            stochastic=stochastic, compact_observation=compact,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0, **parameters)
        results, stats = complete_searches(
            model, lambda method, calls: {i: getattr(base, method)(*args, **kw)
                                          for i, (args, kw) in calls.items()},
            {0: response}, leaf_batch_size=config.leaf_batch_size, compact=compact,
            lock=lock, prior_mode=prior_mode, check=check)
    except BaseException:
        base.nosl_is_az_cpp_abort()
        raise
    result = dict(results[0])
    result.update(stats, elapsed_seconds=time.monotonic() - started)
    return result
