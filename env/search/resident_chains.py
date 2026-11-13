"""Paired world chains over owned CUDA battle states, without a model dependency.

The logical jobs and PCG64 draws match anchored_chains. Physical wave size is
an inference batching choice; it never changes budgets or per-world streams.
"""
import time
import numpy as np
import torch


def root_payload(env, bridge, *, observation=None, snapshot=None):
    env = env.unwrapped
    legal = np.flatnonzero(env.action_masks())
    # The native vector already owns the observation of this exact real root.
    # Its select-zone is all the exporter consumes; rebuilding the full map,
    # deck and combat observation for every search has no additional consumer.
    if observation is None:
        observation = env._obs()
    joint = dict(env._joint_policy_actions())
    result = dict(snapshot=bridge.snapshot(env.thick.bc) if snapshot is None else snapshot,
        prefix=sum(1 << k for k in (env._multi or ())),
        actions=legal, bits=np.asarray([joint[a] for a in legal], np.uint32),
        draw_selection=int(observation['select_zone'].item()) == 2, offset=env._policy_offset,
        skipped=('forced' if len(legal) <= 1 else 'hidden_current_intent'
                 if 'Runic Dome' in [r['id'] for r in env.thick.gc.relic_state()] else None))
    return result


@torch.no_grad()
def resident_anchor_search(roots, budgets, scorer, arena, *, seed, max_depth=200,
                           wave_capacity=4096, timeout_seconds=120., groups=None,
                           group_seeds=None):
    start = time.monotonic()
    stats = dict(chains=0, atomic_steps=0, policy_rows=0, forced_rows=0, anchor_rows=0,
        cutoff_rows=0, terminal_rows=0, max_active=0, budget_floor_overage=0)
    results, job_parts = [], []
    generators = {}
    for i, (root, budget) in enumerate(zip(roots, budgets, strict=True)):
        group = 0 if groups is None else groups[i]
        if group not in generators:
            group_seed = (group_seeds[group] if group_seeds is not None else seed if groups is None
                          else int(np.random.SeedSequence([seed, group]).generate_state(1, dtype=np.uint64)[0]))
            generators[group] = np.random.default_rng(group_seed)
        legal = root['actions']
        result = dict(actions=legal, values=np.zeros(len(legal)), visits=np.zeros(len(legal), np.int64))
        results.append(result)
        if root['skipped']:
            result['skipped'] = root['skipped']; continue
        worlds = max(1, budget // len(legal))
        stats['budget_floor_overage'] += max(0, worlds*len(legal)-budget)
        world_seeds = generators[group].integers(0, 2**63-1, worlds, dtype=np.int64)
        # Pack the old batched job boundary in the same world-major/action-major
        # order. It owns logical IDs separately from physical reusable lanes.
        job_parts.append(np.column_stack((np.full(worlds*len(legal), i, np.int64),
            np.tile(np.arange(len(legal), dtype=np.int64), worlds), np.repeat(world_seeds, len(legal)))))
    if not job_parts:
        return results, dict(stats, seconds=time.monotonic()-start)
    jobs = np.concatenate(job_parts)
    offsets = np.r_[0, np.cumsum([len(r['actions']) for r in results])]
    value_sums = np.zeros(offsets[-1], np.float64)
    visit_sums = np.zeros(offsets[-1], np.int64)
    bits = np.concatenate([r['bits'] for r in roots])
    draw_selection = np.asarray([r['draw_selection'] for r in roots])
    arena.load([r['snapshot'] for r in roots], [r['prefix'] for r in roots])
    nr = len(roots)
    for step in range(max((len(r.get('replay', ())) for r in roots), default=0)):
        arena.observe(nr)
        arena.apply([r['replay'][step] if step < len(r.get('replay', ())) else -1 for r in roots])
    for begin in range(0, len(jobs), wave_capacity):
        wave = jobs[begin:begin+wave_capacity]
        n = len(wave)
        source, action, worlds = wave.T
        dest = np.arange(nr, nr+n)
        arena.fork(source, dest, worlds)
        arena.first(source, dest, bits[offsets[source]+action], draw_selection[source])
        # Pre-generate the same independent PCG64 uniforms. A lane advances its
        # cursor only for a policy choice, never for forced or prefix actions.
        uniforms = {w: np.random.default_rng(int(w) ^ 0x504F4C494359).random(max_depth) for w in set(worlds)}
        randoms = torch.as_tensor(np.stack([uniforms[w] for w in worlds]), device=arena.device)
        owners = torch.as_tensor(source, device=arena.device)
        scorer.bind_wave(arena, owners, nr)
        live = np.ones(n, bool); depth = np.ones(n, np.int64); cursor = np.zeros(n, np.int64)
        final = np.zeros(n, np.float64)
        anchors = np.zeros(n, bool)
        stats['chains'] += n; stats['max_active'] = max(stats['max_active'], n)
        if 'observe_live' in arena.functions:
            final, anchors, counts, atomic = scorer.run_wave(arena, owners, randoms,
                max_depth, start+timeout_seconds)
            for key, value in counts.items():
                stats[key] += value
            stats['atomic_steps'] += atomic
            live[:] = False
        while live.any():
            if time.monotonic()-start > timeout_seconds:
                raise TimeoutError('resident paired budget incomplete; no partial teacher accepted')
            raw = {k: v[nr:] for k, v in arena.observe(nr+n).items()}
            meta = scorer.metadata(raw, owners)
            dead = live & (meta[:, 0] == 2)
            anchor = live & np.isin(meta[:, 0], [1, 3])
            anchors |= anchor
            cutoff = live & (meta[:, 0] == 0) & (depth >= max_depth)
            combat = live & ~(dead | anchor | cutoff)
            if np.any(combat & (meta[:, 1] < 1)):
                raise RuntimeError('resident combat has no legal action')
            forced = combat & (meta[:, 1] == 1)
            policy = combat & ~forced
            selected = torch.full((nr+n,), -1, dtype=torch.int32, device=arena.device)
            force_rows = np.flatnonzero(forced)
            selected[nr+force_rows] = raw['mask'][force_rows].argmax(1).int()
            rows = np.flatnonzero(policy | cutoff)
            if len(rows):
                # The frozen graph owns the FP64 CDF. Counterfactual cutoff
                # samples are ignored and do not advance any policy cursor.
                u = randoms[rows, np.minimum(cursor[rows], max_depth-1)]
                sampled, values = scorer.score(raw, owners, rows, metadata=meta, uniforms=u)
                if not torch.isfinite(values).all():
                    raise FloatingPointError('nonfinite resident leaf value')
                terminal = cutoff[rows]
                if terminal.any(): final[rows[terminal]] = values[terminal].cpu().numpy()
                choose = rows[~terminal]
                if len(choose):
                    selected[nr+choose] = sampled[~terminal]
                    cursor[choose] += 1
            for kind, subset in (('terminal', dead), ('anchor', anchor), ('cutoff', cutoff),
                                 ('forced', forced), ('policy', policy)):
                stats[kind+'_rows'] += int(subset.sum())
            live &= ~(dead | anchor | cutoff)
            depth[live] += 1
            arena.apply(selected)
        # Completed branches stay frozen in their owned arena slots. Batch all
        # pre-reward values once, after advancement, so terminal arrivals do not
        # trigger a host cache lookup and a full decoder launch at every depth.
        rows = np.flatnonzero(anchors)
        if len(rows):
            raw = {k: v[nr:] for k, v in arena.observe(nr+n).items()}
            _, values = scorer.score(raw, owners, rows, anchor=True)
            if not torch.isfinite(values).all():
                raise FloatingPointError('nonfinite resident anchor value')
            final[rows] = values.cpu().numpy()
        if 'observe_live' not in arena.functions:
            stats['atomic_steps'] += int(depth.sum())
        # Unbuffered add.at keeps the original paired-world addition order,
        # without dispatching two Python dictionary updates for every chain.
        slots = offsets[source] + action
        np.add.at(value_sums, slots, final)
        np.add.at(visit_sums, slots, 1)
    for i, result in enumerate(results):
        result['values'] = value_sums[offsets[i]:offsets[i+1]].copy()
        result['visits'] = visit_sums[offsets[i]:offsets[i+1]].copy()
        count = result['visits']
        if count.any():
            if not (count > 0).all() or not (count == count[0]).all():
                raise RuntimeError('incomplete paired resident worlds')
            result['values'] /= count
    stats['seconds'] = time.monotonic()-start
    return results, stats
