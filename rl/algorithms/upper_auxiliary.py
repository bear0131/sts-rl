"""Frozen public lookahead targets alongside on-policy upper PPO.

Search does not select the executed action or change its old log probability.
Only the searched action subset receives a conditional supervised gradient.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch

from env.core import vocab as V
from env.search.upper_public import PHASES, UPPER_KEYS, SearchBudget, backup
from rl.policies.precision import full_precision
from rl.search.native import observation_as_tensor


@dataclass(frozen=True)
class UpperAuxiliaryConfig:
    roots_per_phase: int = 4
    collection_seconds: float = 6.
    depth: int = 3
    transitions: int = 64
    root_actions: int = 32
    seconds: float = 2.
    micro_batch_size: int = 128
    loss_coef: float = .02
    temperature: float = .02
    min_spread: float = .01

    def __post_init__(self):
        SearchBudget(**self.search_budget())
        if (any(not math.isfinite(x) for x in asdict(self).values())
                or type(self.roots_per_phase) is not int or self.roots_per_phase < 1
                or type(self.micro_batch_size) is not int or self.micro_batch_size < 1
                or min(self.collection_seconds, self.loss_coef, self.temperature) <= 0
                or self.min_spread < 0):
            raise ValueError('invalid upper auxiliary search configuration')

    def search_budget(self):
        return {k: getattr(self, k) for k in ('depth', 'transitions', 'root_actions', 'seconds')}


def config(model):
    value = getattr(model.policy, 'upper_auxiliary_config', None)
    return UpperAuxiliaryConfig(**value) if value is not None else None


def begin_collection(model, steps):
    model._upper_auxiliary = dict(steps=steps, windows=set(), trees=[], seconds=0.,
        counts={f'{phase}_{key}': 0. for phase in PHASES
                for key in ('attempts', 'trees', 'labels')}, timeouts=0., transitions=0.)


def collect(model, observations, masks, logits, step, call_workers, paused=()):
    cfg = config(model)
    if cfg is None:
        return
    state = model._upper_auxiliary
    window = min(cfg.roots_per_phase - 1, step * cfg.roots_per_phase // state['steps'])
    phases = observations['phase'].reshape(-1)
    for phase in PHASES:
        key = (phase, window)
        if key in state['windows'] or state['seconds'] >= cfg.collection_seconds:
            continue
        candidates = np.flatnonzero((phases == V.PHASE_VOCAB_INDEX[phase]) & (masks.sum(axis=1) > 1))
        candidates = [int(i) for i in candidates if i not in paused]
        if not candidates:
            continue
        # A deterministic rotation neither consumes nor reseeds training RNG.
        index = candidates[(model.num_timesteps + step) % len(candidates)]
        state['windows'].add(key)
        state['counts'][f'{phase}_attempts'] += 1
        budget = cfg.search_budget()
        budget['seconds'] = min(budget['seconds'], cfg.collection_seconds - state['seconds'])
        started = time.perf_counter()
        result = call_workers('upper_search_tree', {index: ((budget,), {})})[index]
        state['seconds'] += time.perf_counter() - started
        state['transitions'] += result['transitions']
        state['timeouts'] += result['reason'] == 'timeout'
        if result['reason'] != 'ok':
            continue
        state['counts'][f'{phase}_trees'] += 1
        state['trees'].append(dict(tree=result, logits=np.array(logits[index], copy=True),
            observation={k: np.array(v[index], copy=True)
            for k, v in observations.items() if k in UPPER_KEYS}))


def target_distribution(values, cfg):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite auxiliary Q values')
    if values.size < 2 or np.ptp(values) < cfg.min_spread:
        return None
    weights = np.exp((values - values.max()) / cfg.temperature)
    return (weights / weights.sum()).astype(np.float32)


def conditional_loss(logits, targets, masks):
    log_prob = torch.log_softmax(logits.float().masked_fill(~masks, -1e9), dim=1)
    return -(targets * log_prob).sum(dim=1).mean()


def prepare(model):
    """Evaluate all leaves before any upper Adam; discard trees after reduction."""
    cfg = config(model)
    if cfg is None:
        return [], {}
    state = model._upper_auxiliary
    started = time.perf_counter()
    leaves = [(row, i, node['observation']) for row, record in enumerate(state['trees'])
              for i, node in enumerate(record['tree']['nodes'])
              if not node['children'] and not node['terminal']]
    estimates = [{} for _ in state['trees']]
    training = model.policy.training
    model.policy.set_training_mode(False)
    try:
        with torch.no_grad(), full_precision(model.device):
            for start in range(0, len(leaves), cfg.micro_batch_size):
                batch = leaves[start:start + cfg.micro_batch_size]
                obs = observation_as_tensor({k: np.stack([x[2][k] for x in batch])
                                             for k in batch[0][2]}, model.device)
                values = model.policy.predict_values(obs).flatten().float().cpu().numpy()
                for (row, node, _), value in zip(batch, values):
                    estimates[row][node] = float(value)
    finally:
        model.policy.set_training_mode(training)
    prepared = []
    spread_sum = entropy_sum = disagreement_sum = 0.
    for row, record in enumerate(state['trees']):
        actions, values = backup(record['tree'], estimates[row], model.policy.core.upper_config.gamma)
        weights = target_distribution(values, cfg)
        if weights is None:
            continue
        mask = np.zeros(model.action_space.n, dtype=bool)
        target = np.zeros(model.action_space.n, dtype=np.float32)
        mask[actions], target[actions] = True, weights
        prepared.append(dict(observation=record['observation'], target=target, mask=mask))
        state['counts'][f'{record["tree"]["phase"]}_labels'] += 1
        spread_sum += float(np.ptp(values))
        entropy_sum += float(-(weights * np.log(np.maximum(weights, 1e-30))).sum())
        disagreement_sum += int(np.argmax(record['logits'][actions]) != np.argmax(values))
    stats = dict(aux_count=float(len(prepared)), aux_leaf_samples=float(len(leaves)),
        aux_transition_count=state['transitions'], aux_timeout_count=state['timeouts'],
        aux_search_seconds=state['seconds'], aux_teacher_seconds=time.perf_counter() - started,
        aux_spread_sum=spread_sum, aux_entropy_sum=entropy_sum, aux_disagreement_sum=disagreement_sum,
        **{'aux_' + k: v for k, v in state['counts'].items()})
    state['trees'] = []
    return prepared, stats


def backward(model, rows, mean_count):
    """Global mean conditional CE, accumulated into the existing upper Adam."""
    cfg = config(model)
    if cfg is None or not mean_count:
        return 0.
    total = 0.
    for start in range(0, len(rows), cfg.micro_batch_size):
        batch = rows[start:start + cfg.micro_batch_size]
        obs = observation_as_tensor({k: np.stack([r['observation'][k] for r in batch])
                                     for k in batch[0]['observation']}, model.device)
        with full_precision(model.device):
            logits, _ = model.policy.core.actor_critic(obs)
            targets = torch.as_tensor(np.stack([r['target'] for r in batch]), device=model.device)
            masks = torch.as_tensor(np.stack([r['mask'] for r in batch]), device=model.device)
            loss = conditional_loss(logits, targets, masks)
            (loss * (cfg.loss_coef * len(batch) / mean_count)).backward()
        total += float(loss.detach()) * len(batch)
    return total
