"""Native NOSL protocol shared by rollout, evaluation and interactive search.

The caller owns environments, root policy sampling and randomness. This module
owns batched leaf decoding/inference/backprop; it never resets a game or updates
model weights. Transport callbacks support local environments and VecEnv IPC.
"""
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import time

import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from env.core.run_env import unpack_nosl_observation_buffer
from rl.policies.model import observation_for_policy


STRICT_SEARCH_DEFAULTS = {
    'leaf_batch_size': 1,
    'prior_temperature': 1.0, 'gumbel_candidates': 16, 'gumbel_scale': 1.0,
    'gumbel_value_scale': 0.1, 'gumbel_max_visit_init': 50.0,
    'gumbel_rescale_q': True, 'gumbel_use_mixed_value': True,
    'gumbel_interior_uniformity': 0.0, 'gumbel_interior_min_visits': 0,
    'gumbel_interior_coverage_fraction': 0.0,
    'gumbel_interior_value_mode': 0, 'gumbel_interior_max_mix_visits': 64.0,
}


def apply_strict_training_settings(model, *mirrors):
    """Keep new models, resumed models and their CLI metadata consistent."""
    if model.nosl_is_az_search_mode != 'full_gumbel':
        raise ValueError("paper Full-Gumbel mode requires search_mode='full_gumbel'")
    for name, value in dict(STRICT_SEARCH_DEFAULTS, dirichlet_alpha=0.0, dirichlet_epsilon=0.0).items():
        for target in (model, *mirrors):
            setattr(target, 'nosl_is_az_' + name, value)


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
    fixed_worldline: bool = False

    def __post_init__(self):
        if self.simulations < 1 or self.depth < 0 or self.leaf_batch_size < 1:
            raise ValueError("search requires simulations >= 1, depth >= 0, leaf_batch_size >= 1")
        if self.root_search_mode not in (0, 1, 2):
            raise ValueError("root_search_mode must be 0 (PUCT), 1 (Gumbel) or 2 (Full Gumbel)")
        if self.paper_full_gumbel and self.root_search_mode != 2:
            raise ValueError("paper_full_gumbel requires root_search_mode=2")
        if self.paper_full_gumbel:
            for name, value in STRICT_SEARCH_DEFAULTS.items():
                object.__setattr__(self, name, value)
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


@dataclass(frozen=True)
class LeafInferenceSchedule:
    """Cross-tree batching only; it never changes one tree's leaf concurrency."""

    target_batch_size: int = 32
    max_wait_seconds: float = 0.0005
    mode: str = 'fixed_wait'

    def __post_init__(self):
        if self.target_batch_size < 1:
            raise ValueError("leaf inference target batch size must be positive")
        if not np.isfinite(self.max_wait_seconds) or self.max_wait_seconds < 0.0:
            raise ValueError("leaf inference maximum wait must be finite and non-negative")
        if self.mode not in ('fixed_wait', 'adaptive'):
            raise ValueError("leaf inference schedule mode must be fixed_wait or adaptive")


class AdaptiveLeafBatchController:
    """Low-frequency backpressure controller over fixed CUDA-friendly buckets."""

    def __init__(self, max_batch_size, *, initial_batch_size=None,
                 window_batches=256, evidence_windows=3):
        if max_batch_size < 1 or window_batches < 1 or evidence_windows < 1:
            raise ValueError("adaptive leaf batching limits must be positive")
        self.max_batch_size = int(max_batch_size)
        self.window_batches = int(window_batches)
        self.evidence_windows = int(evidence_windows)
        self.buckets = self._make_buckets(self.max_batch_size)
        initial = (
            max(1, self.max_batch_size // 2)
            if initial_batch_size is None
            else int(initial_batch_size)
        )
        self.bucket_index = min(
            range(len(self.buckets)),
            key=lambda index: abs(self.buckets[index] - initial),
        )
        self.changes = 0
        self.increases = 0
        self.decreases = 0
        self.rollbacks = 0
        self.last_decision = 'hold'
        self._trial_origin_index = None
        self._trial_baseline_rate = None
        self._increase_evidence = 0
        self._decrease_evidence = 0
        self._worker_latency_ewma = None
        self._worker_latency_deviation = 0.0
        self._reset_window()

    @staticmethod
    def _make_buckets(maximum):
        quantum = 16 if maximum >= 64 else 1
        values = []
        for fraction in (0.25, 0.375, 0.5, 0.75, 1.0):
            value = int(round((maximum * fraction) / quantum)) * quantum
            values.append(min(maximum, max(1, value)))
        values.append(maximum)
        return tuple(sorted(set(values)))

    @property
    def current_batch_size(self):
        return self.buckets[self.bucket_index]

    def compatible_with(self, schedule):
        return self.max_batch_size == schedule.target_batch_size

    def target(self, active_capacity):
        return max(1, min(self.current_batch_size, int(active_capacity)))

    def observe_worker_latency(self, seconds):
        if seconds is None or not np.isfinite(seconds) or seconds < 0.0:
            return
        value = float(seconds)
        if self._worker_latency_ewma is None:
            self._worker_latency_ewma = value
            return
        alpha = 0.05
        difference = abs(value - self._worker_latency_ewma)
        self._worker_latency_ewma += alpha * (value - self._worker_latency_ewma)
        self._worker_latency_deviation += alpha * (
            difference - self._worker_latency_deviation
        )

    def watchdog_seconds(self, floor_seconds):
        floor = max(float(floor_seconds), 0.001)
        if self._worker_latency_ewma is None:
            return max(floor, 0.005)
        estimate = self._worker_latency_ewma + 4.0 * self._worker_latency_deviation
        return min(0.1, max(floor, estimate))

    def observe_batch(self, *, batch_size, target, active_capacity,
                      ready_backlog, queue_delay_seconds, model_seconds,
                      cycle_seconds):
        # Tail batches are capacity-limited and must not train the outer
        # controller to shrink an otherwise healthy steady-state batch.
        if active_capacity < self.current_batch_size:
            return
        self._window_count += 1
        self._window_fill += min(1.0, float(batch_size) / max(1, target))
        self._window_backlog += float(ready_backlog)
        self._window_queue_delay += float(queue_delay_seconds)
        self._window_model += float(model_seconds)
        self._window_leaves += int(batch_size)
        self._window_seconds += float(cycle_seconds)
        if self._window_count < self.window_batches:
            return

        count = self._window_count
        fill = self._window_fill / count
        backlog = self._window_backlog / count
        queue_delay = self._window_queue_delay / count
        model_seconds = self._window_model / count
        queue_to_model = queue_delay / max(model_seconds, 1e-9)
        low = fill < 0.80 and backlog < 0.10 * self.current_batch_size \
            and queue_to_model > 0.25
        high = fill > 0.95 and backlog > 0.50 * self.current_batch_size

        leaves_per_second = self._window_leaves / max(
            self._window_seconds, 1e-9
        )
        self.last_window = {
            'fill': fill,
            'backlog': backlog,
            'queue_delay_seconds': queue_delay,
            'model_seconds': model_seconds,
            'leaves_per_second': leaves_per_second,
        }
        self._reset_window()
        if self._trial_origin_index is not None:
            if leaves_per_second < 0.95 * self._trial_baseline_rate:
                self.bucket_index = self._trial_origin_index
                self.rollbacks += 1
                self.changes += 1
                self.last_decision = 'rollback'
            else:
                self.last_decision = 'accept'
            self._trial_origin_index = None
            self._trial_baseline_rate = None
            self._increase_evidence = 0
            self._decrease_evidence = 0
            return
        if low:
            self._decrease_evidence += 1
            self._increase_evidence = 0
        elif high:
            self._increase_evidence += 1
            self._decrease_evidence = 0
        else:
            self._increase_evidence = 0
            self._decrease_evidence = 0
            self.last_decision = 'hold'
            return

        if (
            self._decrease_evidence >= self.evidence_windows
            and self.bucket_index > 0
        ):
            origin = self.bucket_index
            self.bucket_index -= 1
            self.decreases += 1
            self.changes += 1
            self.last_decision = 'decrease'
            self._trial_origin_index = origin
            self._trial_baseline_rate = leaves_per_second
            self._decrease_evidence = 0
        elif (
            self._increase_evidence >= self.evidence_windows
            and self.bucket_index + 1 < len(self.buckets)
        ):
            origin = self.bucket_index
            self.bucket_index += 1
            self.increases += 1
            self.changes += 1
            self.last_decision = 'increase'
            self._trial_origin_index = origin
            self._trial_baseline_rate = leaves_per_second
            self._increase_evidence = 0

    def _reset_window(self):
        self._window_count = 0
        self._window_fill = 0.0
        self._window_backlog = 0.0
        self._window_queue_delay = 0.0
        self._window_model = 0.0
        self._window_leaves = 0
        self._window_seconds = 0.0


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


# Existing training/benchmark callers inherit the model-owned runtime backend.
# Evaluation strategies can explicitly supply None (eager) or their own cache.
MODEL_LEAF_BACKEND = object()


def forward_leaves(model, observations, *, compact=True, lock=None, autocast=nullcontext,
                   leaf_backend=MODEL_LEAF_BACKEND):
    backend = (getattr(model, '_native_leaf_inference', None)
               if leaf_backend is MODEL_LEAF_BACKEND else leaf_backend)
    if backend is not None and compact:
        active = np.flatnonzero(np.any(observations['relics'] != 0, axis=0))
        width = int(active[-1]) + 1 if active.size else 1
        with lock or nullcontext():
            return backend(observations, width)
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
    records, spans = [], []
    validated_sources = set()
    keys = None
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
        span = getattr(packed, '_nosl_observation_span', None)
        if span is None:
            source, start, end = packed, 0, len(meta)
            if any(len(value) != end for value in source.values()):
                raise ValueError('native leaf metadata/observation batch size mismatch')
        else:
            source, start, end = span
            if end - start != len(meta) or start < 0:
                raise ValueError('native leaf metadata/observation batch size mismatch')
        source_id = id(source)
        if source_id not in validated_sources:
            if span is not None:
                row_count = len(next(iter(source.values())))
                if any(len(value) != row_count for value in source.values()):
                    raise ValueError('native leaf metadata/observation batch size mismatch')
            if keys is None:
                keys = tuple(source)
            elif set(source) != set(keys):
                raise ValueError('native workers returned incompatible observation schemas')
            validated_sources.add(source_id)
        if span is not None and end > len(next(iter(source.values()))):
            raise ValueError('native leaf metadata/observation batch size mismatch')
        records.extend((index, leaf) for leaf in meta)
        if spans and spans[-1][0] is source and spans[-1][2] == start:
            spans[-1] = (source, spans[-1][1], end)
        else:
            spans.append((source, start, end))
    # Keep concatenate even for one span: callers retain their independent,
    # contiguous result rather than a mutable alias into a received packet.
    observations = {key: np.ascontiguousarray(np.concatenate([
                        source[key][start:end] for source, start, end in spans
                    ], axis=0))
                    for key in keys}
    return records, observations


def complete_searches(model, call_workers, responses, *, leaf_batch_size,
                      compact=True, lock=None, autocast=nullcontext,
                      prior_mode='model', check=lambda: None,
                      leaf_backend=MODEL_LEAF_BACKEND):
    """Finish already-started trees with one forward across all active workers.

    Returns results and counters. The caller must abort owned sessions when an
    exception occurs; training workers are disposed by their enclosing runner.
    """
    if prior_mode not in ('model', 'uniform', 'none'):
        raise ValueError(f'unknown prior mode: {prior_mode}')
    results = {}
    stats = dict(leaf_batches=0, leaf_evaluations=0, cpp_leaf_decode_seconds=0.0,
                 cpp_leaf_model_seconds=0.0, cpp_backprop_rpc_seconds=0.0,
                 inference_batch_size_max=0)
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
        stats['inference_batch_size_max'] = max(
            stats['inference_batch_size_max'], len(records)
        )
        for index in active:
            by_worker[index]['leaf_batches'] += 1
        for index, _ in records:
            by_worker[index]['leaf_evaluations'] += 1
        started = time.perf_counter()
        logits, values = forward_leaves(model, observations, compact=compact, lock=lock,
                                       autocast=autocast, leaf_backend=leaf_backend)
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
    stats['inference_batch_size_mean'] = (
        float(stats['leaf_evaluations'] / stats['leaf_batches'])
        if stats['leaf_batches']
        else 0.0
    )
    stats['by_worker'] = by_worker
    return results, stats


def _leaf_count(response):
    if isinstance(response, list):
        return len(response)
    packed = (response.get('observation_batch') is not None or
              (response.get('observation_buffer') is not None and
               response.get('observation_spec') is not None))
    meta = response.get('leaf_meta') if packed else response.get('leaves')
    if not meta:
        raise RuntimeError('native search returned no leaves before completion')
    return len(meta)


def complete_searches_streaming(
    model,
    stream,
    begin_calls,
    *,
    leaf_batch_size,
    schedule=LeafInferenceSchedule(),
    compact=True,
    lock=None,
    autocast=nullcontext,
    prior_mode='model',
    check=lambda: None,
    controller=None,
):
    """Finish independent trees without a cross-worker simulation barrier.

    Each tree still has exactly one outstanding leaf. Worker groups are the
    scheduling unit because one subprocess owns and serializes several envs.
    Dynamic GPU batches contain only leaves from distinct trees, so the C++
    visit order, virtual-visit behavior and NOSL nonce sequence are unchanged.
    """
    if leaf_batch_size != 1:
        raise ValueError(
            'streaming search requires leaf_batch_size=1 to preserve '
            'single-leaf tree semantics'
        )
    if prior_mode not in ('model', 'uniform', 'none'):
        raise ValueError(f'unknown prior mode: {prior_mode}')
    if not isinstance(schedule, LeafInferenceSchedule):
        schedule = LeafInferenceSchedule(**dict(schedule))
    if schedule.mode == 'adaptive':
        if controller is None:
            controller = getattr(model, '_nosl_leaf_batch_controller', None)
            if (
                controller is None
                or not controller.compatible_with(schedule)
            ):
                controller = AdaptiveLeafBatchController(
                    schedule.target_batch_size
                )
                try:
                    model._nosl_leaf_batch_controller = controller
                except AttributeError:
                    pass
    else:
        controller = None

    expected = set(begin_calls)
    results = {}
    stats = dict(
        leaf_batches=0,
        leaf_evaluations=0,
        cpp_leaf_decode_seconds=0.0,
        cpp_leaf_model_seconds=0.0,
        cpp_backprop_rpc_seconds=0.0,
        cpp_stream_wait_seconds=0.0,
        cpp_ready_empty_wait_seconds=0.0,
        inference_batch_size_max=0,
        inference_target_batch_sum=0,
        inference_target_batch_count=0,
        inference_ready_backlog_sum=0,
        inference_ready_backlog_count=0,
        inference_ready_backlog_max=0,
        inference_leaf_queue_wait_seconds=0.0,
        inference_leaf_worker_wait_seconds=0.0,
        inference_full_flushes=0,
        inference_drained_flushes=0,
        inference_watchdog_flushes=0,
    )
    initial_controller_counts = (
        (controller.changes, controller.increases,
         controller.decreases, controller.rollbacks)
        if controller is not None else (0, 0, 0, 0)
    )
    by_worker = {
        index: {'leaf_evaluations': 0, 'leaf_batches': 0}
        for index in begin_calls
    }
    # Keep a whole worker group atomic. Sending only part of a returned group
    # would leave that process idle until a later GPU batch supplies the rest.
    ready = deque()
    ready_leaf_count = 0

    def batch_target(active_capacity):
        if controller is not None:
            return controller.target(active_capacity)
        return max(1, min(schedule.target_batch_size, int(active_capacity)))

    def receive(timeout, *, ready_empty=False):
        started = time.perf_counter()
        packets = stream.receive(timeout=timeout)
        elapsed = time.perf_counter() - started
        stats['cpp_stream_wait_seconds'] += elapsed
        if ready_empty:
            stats['cpp_ready_empty_wait_seconds'] += elapsed
        if controller is not None:
            for packet in packets:
                controller.observe_worker_latency(
                    getattr(packet, 'elapsed_seconds', None)
                )
        return packets

    def ingest(packets):
        nonlocal ready_leaf_count
        received_at = time.perf_counter()
        for packet in packets:
            active = {}
            for index, response in packet.responses.items():
                if index not in expected:
                    raise RuntimeError(f'unexpected native search response {index}')
                if response.get('done', False):
                    results[index] = response['result']
                else:
                    active[index] = response
            if active:
                count = sum(_leaf_count(response) for response in active.values())
                ready.append((received_at, packet.group_index, active, count))
                ready_leaf_count += count

    started = time.perf_counter()
    stream.submit('nosl_is_az_cpp_begin', begin_calls)
    stats['cpp_begin_rpc_seconds'] = time.perf_counter() - started

    while len(results) < len(expected):
        check()
        if ready:
            batch_cycle_started = time.perf_counter()
            active_capacity = len(expected) - len(results)
            target = batch_target(active_capacity)
            flush_reason = None
            while ready_leaf_count < target and stream.pending:
                oldest_at = ready[0][0]
                wait_budget = (
                    controller.watchdog_seconds(schedule.max_wait_seconds)
                    if controller is not None
                    else schedule.max_wait_seconds
                )
                wait_left = wait_budget - (time.perf_counter() - oldest_at)
                if wait_left <= 0.0:
                    flush_reason = 'watchdog'
                    break
                packets = receive(wait_left)
                if not packets:
                    flush_reason = 'watchdog'
                    break
                ingest(packets)
                active_capacity = len(expected) - len(results)
                target = batch_target(active_capacity)

            # Drain packets already waiting in the OS pipe. This does not wait,
            # but exposes producer backlog to the controller before selection.
            if controller is not None and ready_leaf_count >= target:
                while stream.pending:
                    packets = receive(0.0)
                    if not packets:
                        break
                    ingest(packets)
                    active_capacity = len(expected) - len(results)
                    target = batch_target(active_capacity)

            if ready_leaf_count >= target:
                flush_reason = 'full'
            elif not stream.pending:
                flush_reason = 'drained'
            else:
                flush_reason = flush_reason or 'watchdog'

            selected = {}
            selected_count = 0
            selected_packets = []
            selection_started = time.perf_counter()
            while ready and selected_count < target:
                received_at, group_index, responses, count = ready.popleft()
                overlap = set(selected).intersection(responses)
                if overlap:
                    raise RuntimeError(f'duplicate queued native responses: {sorted(overlap)}')
                selected.update(responses)
                selected_count += count
                ready_leaf_count -= count
                selected_packets.append((received_at, group_index, count))

            queue_wait_seconds = sum(
                max(0.0, selection_started - received_at) * count
                for received_at, _group_index, count in selected_packets
            )
            stats['inference_target_batch_sum'] += target
            stats['inference_target_batch_count'] += 1
            stats['inference_ready_backlog_sum'] += ready_leaf_count
            stats['inference_ready_backlog_count'] += 1
            stats['inference_ready_backlog_max'] = max(
                stats['inference_ready_backlog_max'], ready_leaf_count
            )
            stats['inference_leaf_queue_wait_seconds'] += queue_wait_seconds
            stats[f'inference_{flush_reason}_flushes'] += 1

            decode_started = time.perf_counter()
            records, observations = decode_leaves(selected)
            decode_seconds = time.perf_counter() - decode_started
            stats['cpp_leaf_decode_seconds'] += decode_seconds
            stats['leaf_batches'] += 1
            stats['leaf_evaluations'] += len(records)
            stats['inference_batch_size_max'] = max(
                stats['inference_batch_size_max'], len(records)
            )
            for index in selected:
                by_worker[index]['leaf_batches'] += 1
            for index, _leaf in records:
                by_worker[index]['leaf_evaluations'] += 1

            model_started = time.perf_counter()
            logits, values = forward_leaves(
                model,
                observations,
                compact=compact,
                lock=lock,
                autocast=autocast,
            )
            model_seconds = time.perf_counter() - model_started
            stats['cpp_leaf_model_seconds'] += model_seconds
            updates = {}
            for position, (index, leaf) in enumerate(records):
                policy = (
                    logits[position]
                    if prior_mode == 'model'
                    else np.zeros_like(logits[position])
                )
                if not leaf.get('needs_policy', True):
                    policy = np.zeros(0, dtype=np.float64)
                updates.setdefault(index, []).append(
                    (int(leaf['request_id']), float(values[position]), policy)
                )
            check()
            dispatch_started = time.perf_counter()
            stream.submit(
                'nosl_is_az_cpp_backprop_and_leaf_batch',
                {
                    index: ((payload,), {'max_leaves': leaf_batch_size})
                    for index, payload in updates.items()
                },
            )
            dispatch_finished = time.perf_counter()
            dispatch_seconds = dispatch_finished - dispatch_started
            stats['cpp_backprop_rpc_seconds'] += dispatch_seconds
            worker_wait_seconds = sum(
                max(0.0, dispatch_finished - received_at) * count
                for received_at, _group_index, count in selected_packets
            )
            stats['inference_leaf_worker_wait_seconds'] += worker_wait_seconds
            if controller is not None:
                controller.observe_batch(
                    batch_size=selected_count,
                    target=target,
                    active_capacity=active_capacity,
                    ready_backlog=ready_leaf_count,
                    queue_delay_seconds=(
                        queue_wait_seconds / selected_count
                        if selected_count else 0.0
                    ),
                    model_seconds=model_seconds,
                    cycle_seconds=dispatch_finished - batch_cycle_started,
                )
            continue

        if not stream.pending:
            missing = sorted(expected.difference(results))
            raise RuntimeError(
                f'native search stream stopped before results for {missing}'
            )
        ingest(receive(None, ready_empty=True))

    if stream.pending or ready:
        raise RuntimeError('native search completed with outstanding streamed work')
    stats['inference_batch_size_mean'] = (
        float(stats['leaf_evaluations'] / stats['leaf_batches'])
        if stats['leaf_batches']
        else 0.0
    )
    stats['inference_batch_target_current'] = (
        controller.current_batch_size
        if controller is not None
        else schedule.target_batch_size
    )
    stats['inference_controller_changes'] = (
        controller.changes - initial_controller_counts[0]
        if controller is not None else 0
    )
    stats['inference_controller_increases'] = (
        controller.increases - initial_controller_counts[1]
        if controller is not None else 0
    )
    stats['inference_controller_decreases'] = (
        controller.decreases - initial_controller_counts[2]
        if controller is not None else 0
    )
    stats['inference_controller_rollbacks'] = (
        controller.rollbacks - initial_controller_counts[3]
        if controller is not None else 0
    )
    stats['inference_watchdog_seconds'] = (
        controller.watchdog_seconds(schedule.max_wait_seconds)
        if controller is not None else schedule.max_wait_seconds
    )
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
