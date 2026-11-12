"""Model-free reference scheduler for paired, pre-reward anchored chains.

Work units are (root, atomic action, world). The scorer receives a frontier
batch, never a callback from inside a single simulated transition. Exact thin
action/prefix state remains owned by the existing environment. This is the
semantic reference for a native resident implementation, not a speed claim.
"""
from dataclasses import dataclass, field
import time

import numpy as np

from env.core.outer_return import strategic_reward


@dataclass
class ChainRequest:
    root: int
    observation: dict
    action_mask: np.ndarray
    kind: str  # combat, cutoff, anchor
    cache: dict = field(default_factory=dict)  # opaque, branch-owned scorer state


@dataclass
class _Lane:
    root: int
    action: int
    environment: object
    rng: object
    steps: int
    reward: float
    observation: dict
    done: bool
    cache: dict = field(default_factory=dict)


def _step(lane, action, *, defer_observation=False):
    env = lane.environment
    act, floor = env.thick.gc.act, env.thick.gc.floor_num
    previous = env._defer_combat_observation
    env._defer_combat_observation = defer_observation
    env._combat_observation_pending = False
    try:
        obs, _, terminated, truncated, info = env.step(int(action))
        if env._combat_observation_pending:
            obs = None
    finally:
        env._defer_combat_observation = previous
        env._combat_observation_pending = False
    if truncated or info.get('dead_screen'):
        raise RuntimeError('censored search branch cannot supply a complete teacher')
    lane.reward += strategic_reward(act, floor, env.thick.run_stats(),
                                    done=terminated, censored=False)
    lane.observation, lane.done = obs, bool(terminated)
    lane.steps += 1


def paired_anchor_search(roots, budgets, scorer, *, seed, max_depth=200,
                         max_active=256, timeout_seconds=120., observation_batch=None):
    """Return per-root action means using complete equal paired-world sets.

    ``scorer(requests)`` returns one (logits, scalar value) per request; anchor
    rows only consume the value. It owns the model, shared context and any
    inference caches. Shadow search must not change the real policy's RNG.
    An optional native root encoder fills deferred live combat observations at
    the next scored frontier. Anchors, terminals and thin prefixes retain the
    ordinary codec. Forced steps consume no observation and need no encoding.
    """
    if len(roots) != len(budgets) or min(max_depth, max_active) < 1 or timeout_seconds <= 0:
        raise ValueError('invalid anchored search batch limits')
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    results, jobs, root_metadata = [], [], []
    stats = dict(chains=0, atomic_steps=0, policy_rows=0, forced_rows=0, anchor_rows=0,
                 cutoff_rows=0, terminal_rows=0, max_active=0, budget_floor_overage=0)
    for root_index, (env, budget) in enumerate(zip(roots, budgets, strict=True)):
        if budget < 1 or env._phase() != 'combat' or env._at_combat_exit_anchor():
            raise ValueError('search roots must be live combat with positive budgets')
        legal = np.flatnonzero(env.action_masks())
        result = dict(actions=legal, values=np.zeros(len(legal)), visits=np.zeros(len(legal), np.int64))
        results.append(result)
        root_metadata.append((dict(env._joint_policy_actions()), int(env._obs()['select_zone'].item()) == 2))
        relics = tuple(r['id'] for r in env.thick.gc.relic_state())
        if len(legal) <= 1 or 'Runic Dome' in relics:
            result['skipped'] = 'forced' if len(legal) <= 1 else 'hidden_current_intent'
            continue
        worlds = max(1, budget // len(legal))
        stats['budget_floor_overage'] += max(0, worlds * len(legal) - budget)
        seeds = rng.integers(0, 2**63 - 1, worlds, dtype=np.int64)
        for world in seeds:
            for action_index in range(len(legal)):
                jobs.append((root_index, action_index, int(world)))
    next_job, lanes = 0, []
    while next_job < len(jobs) or lanes:
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError('anchored search budget not completed; no partial teacher accepted')
        while next_job < len(jobs) and len(lanes) < max_active:
            root, action_index, world = jobs[next_job]; next_job += 1
            source = roots[root]
            branch = source.fork_run(worldline=world)
            branch.thick.bc.nosl_redeterminize(world)
            branch._invalidate_backend_cache()
            action = int(results[root]['actions'][action_index])
            mapping, draw_selection = root_metadata[root]
            bits = int(mapping[action])
            # Draw-pile candidate handles are execution metadata, not identity.
            if bits >> 29 == 2 and draw_selection:
                bits = int(branch.thick.bc.remap_draw_selection_from(source.thick.bc, bits))
            inverse = {int(b): int(a) for a, b in branch._joint_policy_actions()}
            if bits not in inverse:
                raise RuntimeError('root atomic action lost its identity in a sampled world')
            lane = _Lane(root, action_index, branch, np.random.default_rng(world ^ 0x504F4C494359),
                         0, 0., {}, False)
            _step(lane, inverse[bits], defer_observation=observation_batch is not None)
            lanes.append(lane)
            stats['chains'] += 1
        stats['max_active'] = max(stats['max_active'], len(lanes))
        requests, requested_lanes = [], []
        for lane in lanes:
            if lane.done:
                continue
            env = lane.environment
            kind = ('anchor' if env._at_combat_exit_anchor() else
                    'cutoff' if lane.steps >= max_depth else 'combat')
            mask = env.action_masks()
            if kind == 'combat' and np.count_nonzero(mask) == 1:
                _step(lane, np.flatnonzero(mask)[0], defer_observation=observation_batch is not None)
                stats['forced_rows'] += 1
                continue
            requests.append(ChainRequest(lane.root, lane.observation, mask, kind, lane.cache))
            requested_lanes.append(lane)
        pending = [i for i, lane in enumerate(requested_lanes) if lane.observation is None]
        if pending:
            encoded = observation_batch.encode([requested_lanes[i].environment for i in pending])
            for row, i in enumerate(pending):
                observation = {key: value[row] for key, value in encoded.items()}
                requested_lanes[i].observation = requests[i].observation = observation
        scored = scorer(requests) if requests else []
        if len(scored) != len(requests):
            raise ValueError('scorer must return exactly one result per frontier row')
        # Only rows which issued requests consume outputs; forced steps remain
        # live and may now be terminal/anchor on the following frontier.
        request_ids = {id(lane): (request, output)
                       for lane, request, output in zip(requested_lanes, requests, scored, strict=True)}
        survivors = []
        for lane in lanes:
            pair = request_ids.get(id(lane))
            finished = lane.done
            if pair is not None:
                request, (logits, value) = pair
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite search leaf value')
                if request.kind != 'combat':
                    lane.reward += float(value); finished = True
                    stats[request.kind + '_rows'] += 1
                else:
                    legal = np.flatnonzero(request.action_mask)
                    scores = np.asarray(logits, np.float64)[legal]
                    if not len(legal) or not np.isfinite(scores).all():
                        raise FloatingPointError('empty/nonfinite search action distribution')
                    probability = np.exp(scores - scores.max()); probability /= probability.sum()
                    _step(lane, lane.rng.choice(legal, p=probability),
                          defer_observation=observation_batch is not None)
                    stats['policy_rows'] += 1
                    # Finalize on the next iteration, so every exit is classified.
                    survivors.append(lane)
                    continue
            if finished:
                stats['terminal_rows'] += int(lane.done)
                result = results[lane.root]
                result['values'][lane.action] += lane.reward
                result['visits'][lane.action] += 1
                stats['atomic_steps'] += lane.steps
                lane.environment.close()
            else:
                survivors.append(lane)
        lanes = survivors
    for result in results:
        counts = result['visits']
        if counts.any():
            if not (counts > 0).all() or not (counts == counts[0]).all():
                raise RuntimeError('paired world coverage must be complete for every legal root action')
            result['values'] /= counts
    stats['seconds'] = time.monotonic() - start
    return results, stats
