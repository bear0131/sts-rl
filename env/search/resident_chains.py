"""Paired world chains over owned CUDA battle states, without a model dependency.

The logical jobs and PCG64 draws match anchored_chains. Physical wave size is
an inference batching choice; it never changes budgets or per-world streams.
"""
import time
import numpy as np
import torch


def root_payload(env, bridge):
    env = env.unwrapped
    legal = np.flatnonzero(env.action_masks())
    observation = env._obs()
    result = dict(snapshot=bridge.snapshot(env.thick.bc), prefix=sum(1 << k for k in (env._multi or ())),
        actions=legal, bits=np.asarray([dict(env._joint_policy_actions())[a] for a in legal], np.uint32),
        draw_selection=int(observation['select_zone'].item()) == 2, offset=env._policy_offset,
        skipped=('forced' if len(legal) <= 1 else 'hidden_current_intent'
                 if 'Runic Dome' in [r['id'] for r in env.thick.gc.relic_state()] else None))
    return result


@torch.no_grad()
def resident_anchor_search(roots, budgets, scorer, arena, *, seed, max_depth=200,
                           wave_capacity=4096, timeout_seconds=120., groups=None):
    start = time.monotonic()
    stats = dict(chains=0, atomic_steps=0, policy_rows=0, forced_rows=0, anchor_rows=0,
        cutoff_rows=0, terminal_rows=0, max_active=0, budget_floor_overage=0)
    results, jobs = [], []
    generators = {}
    for i, (root, budget) in enumerate(zip(roots, budgets, strict=True)):
        group = 0 if groups is None else groups[i]
        if group not in generators:
            group_seed = seed if groups is None else int(np.random.SeedSequence([seed, group]).generate_state(1, dtype=np.uint64)[0])
            generators[group] = np.random.default_rng(group_seed)
        legal = root['actions']
        result = dict(actions=legal, values=np.zeros(len(legal)), visits=np.zeros(len(legal), np.int64))
        results.append(result)
        if root['skipped']:
            result['skipped'] = root['skipped']; continue
        worlds = max(1, budget // len(legal))
        stats['budget_floor_overage'] += max(0, worlds*len(legal)-budget)
        for world in generators[group].integers(0, 2**63-1, worlds, dtype=np.int64):
            for a in range(len(legal)):
                jobs.append((i, a, int(world)))
    if not jobs:
        return results, dict(stats, seconds=time.monotonic()-start)
    arena.load([r['snapshot'] for r in roots], [r['prefix'] for r in roots])
    nr = len(roots)
    for step in range(max((len(r.get('replay', ())) for r in roots), default=0)):
        arena.observe(nr)
        arena.apply([r['replay'][step] if step < len(r.get('replay', ())) else -1 for r in roots])
    for begin in range(0, len(jobs), wave_capacity):
        wave = jobs[begin:begin+wave_capacity]
        n = len(wave)
        source, action, worlds = (np.asarray(v) for v in zip(*wave))
        dest = np.arange(nr, nr+n)
        arena.fork(source, dest, worlds)
        arena.first(source, dest, [roots[r]['bits'][a] for r, a, _ in wave],
                    [roots[r]['draw_selection'] for r in source])
        # Pre-generate the same independent PCG64 uniforms. A lane advances its
        # cursor only for a policy choice, never for forced or prefix actions.
        uniforms = {w: np.random.default_rng(int(w) ^ 0x504F4C494359).random(max_depth) for w in set(worlds)}
        randoms = torch.as_tensor(np.stack([uniforms[w] for w in worlds]), device=arena.device)
        owners = torch.as_tensor(source, device=arena.device)
        live = np.ones(n, bool); depth = np.ones(n, np.int64); cursor = np.zeros(n, np.int64)
        final = np.zeros(n, np.float64)
        stats['chains'] += n; stats['max_active'] = max(stats['max_active'], n)
        while live.any():
            if time.monotonic()-start > timeout_seconds:
                raise TimeoutError('resident paired budget incomplete; no partial teacher accepted')
            raw = {k: v[nr:] for k, v in arena.observe(nr+n).items()}
            counts = raw['mask'].sum(1)
            meta = torch.stack((raw['outcome'], counts), 1).cpu().numpy()
            dead = live & (meta[:, 0] == 2)
            anchor = live & np.isin(meta[:, 0], [1, 3])
            cutoff = live & (meta[:, 0] == 0) & (depth >= max_depth)
            combat = live & ~(dead | anchor | cutoff)
            if np.any(combat & (meta[:, 1] < 1)):
                raise RuntimeError('resident combat has no legal action')
            forced = combat & (meta[:, 1] == 1)
            policy = combat & ~forced
            selected = torch.full((nr+n,), -1, dtype=torch.int32, device=arena.device)
            force_rows = np.flatnonzero(forced)
            selected[nr+force_rows] = raw['mask'][force_rows].argmax(1).int()
            for kind, subset in (('anchor', anchor), ('combat', policy | cutoff)):
                rows = np.flatnonzero(subset)
                if not len(rows): continue
                logits, values = scorer.score(raw, owners, rows, anchor=kind == 'anchor')
                if not torch.isfinite(values).all():
                    raise FloatingPointError('nonfinite resident leaf value')
                if kind == 'anchor':
                    final[rows] = values.cpu().numpy(); continue
                terminal = cutoff[rows]
                if terminal.any(): final[rows[terminal]] = values[terminal].cpu().numpy()
                choose = rows[~terminal]
                if len(choose):
                    legal = raw['mask'][choose].bool()
                    scores = logits[~terminal].double().masked_fill(~legal, -torch.inf)
                    probability = (scores-scores.max(1, keepdim=True).values).exp()
                    probability = probability/probability.sum(1, keepdim=True)
                    cdf = probability.cumsum(1); cdf = cdf/cdf[:, -1:]
                    u = randoms[choose, cursor[choose]]
                    sampled = (cdf <= u[:, None]).sum(1).int()
                    selected[nr+choose] = sampled
                    cursor[choose] += 1
            for kind, subset in (('terminal', dead), ('anchor', anchor), ('cutoff', cutoff),
                                 ('forced', forced), ('policy', policy)):
                stats[kind+'_rows'] += int(subset.sum())
            live &= ~(dead | anchor | cutoff)
            depth[live] += 1
            arena.apply(selected)
        stats['atomic_steps'] += int(depth.sum())
        for j, (root, a, _) in enumerate(wave):
            results[root]['values'][a] += final[j]
            results[root]['visits'][a] += 1
    for result in results:
        count = result['visits']
        if count.any():
            if not (count > 0).all() or not (count == count[0]).all():
                raise RuntimeError('incomplete paired resident worlds')
            result['values'] /= count
    stats['seconds'] = time.monotonic()-start
    return results, stats
