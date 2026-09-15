"""Fixed public-observation HP probes; synthetic interventions, not game oracles.

The corpus is independent of checkpoint weights and persisted without pickle.
Native roots/boundaries supply contexts; HP sweeps deliberately do not execute
mechanics. Inversions flag a review candidate, not necessarily a game-rule bug.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def maximum_hp(observation, axis, slot=0):
    if axis == 'player':
        maximum = float(observation['player_scalars'][1]) * 200
        if maximum >= 200:
            raise ValueError('capped player max HP requires a raw-state corpus')
        return int(round(maximum))
    scalars = observation['monster_scalars'][slot]
    if scalars[0] <= 0 or scalars[1] <= 0 or scalars[2] <= 0:
        raise ValueError('enemy must be alive and present')
    if scalars[1] >= 1:
        raise ValueError('capped enemy HP requires a raw-state corpus')
    # Column 0 is presence, NOT max HP. Recover max from current and ratio.
    return int(round(float(scalars[1]) * 200 / float(scalars[2])))


def intervene(observation, axis, hp, *, slot=0):
    """Copy and update all scalar aliases for one current-HP coordinate."""
    result = {k: np.asarray(v).copy() for k, v in observation.items()}
    if 'player_status_language' in result:
        raise ValueError('tokenized HP needs re-encoding; scalar-only probes unsupported')
    if axis == 'player':
        scalars = result['player_scalars']
        maximum = maximum_hp(observation, axis)
        if not 1 <= hp <= maximum:
            raise ValueError('player HP outside living range')
        scalars[0], scalars[5] = hp / 200, hp / maximum
        if 'run_hp' in result:
            result['run_hp'][0], result['run_hp'][2] = hp / 200, hp / maximum
    elif axis == 'enemy':
        if not 0 <= slot < len(result['monster_id']) or not result['monster_id'][slot]:
            raise ValueError('enemy slot is empty or invalid')
        scalars = result['monster_scalars'][slot]
        maximum = maximum_hp(observation, axis, slot)
        if not 1 <= hp <= maximum:
            raise ValueError('enemy HP outside living range')
        scalars[1], scalars[2] = hp / 200, hp / maximum
    else:
        raise ValueError('axis must be player or enemy')
    return result


def summarize(hps, values, axis, *, tolerance=1e-5):
    hps, values = np.asarray(hps), np.asarray(values, dtype=np.float64)
    if axis not in ('player', 'enemy') or tolerance < 0:
        raise ValueError('invalid sensitivity settings')
    if len(hps) < 2 or hps.shape != values.shape or len(set(hps)) != len(hps):
        raise ValueError('need at least two distinct HP points with paired values')
    order = np.argsort(hps)
    hps, values = hps[order], values[order]
    if not np.isfinite(values).all():
        return {'finite': False, 'points': len(values)}
    # Positive adverse difference = opposite to the usual expected direction.
    adverse = np.diff(values) * (-1 if axis == 'player' else 1)
    wrong = adverse > tolerance
    return dict(finite=True, points=len(values), value_min_hp=float(values[0]),
                value_max_hp=float(values[-1]), delta_max_minus_min=float(values[-1]-values[0]),
                adverse_steps=int(wrong.sum()), adjacent_steps=len(adverse),
                adverse_fraction=float(wrong.mean()),
                largest_adverse_step=float(max(0, adverse.max())),
                total_adverse_variation=float(np.maximum(adverse, 0).sum()),
                value_below_zero_fraction=float(np.mean(values < -tolerance)),
                tolerance=tolerance)


def corpus_digest(corpus):
    digest = hashlib.sha256()
    for meta, obs in corpus:
        digest.update(json.dumps(meta, sort_keys=True).encode())
        for key, array in sorted(obs.items()):
            array = np.asarray(array)
            digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def save_corpus(corpus, path):
    arrays = {f'{i}/{k}': np.asarray(v) for i, (_, obs) in enumerate(corpus) for k, v in obs.items()}
    arrays['metadata'] = np.asarray(json.dumps([meta for meta, _ in corpus]))
    arrays['schema_version'] = np.asarray(1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def load_corpus(path):
    with np.load(path, allow_pickle=False) as archive:
        if int(archive['schema_version']) != 1:
            raise ValueError('unsupported HP corpus schema')
        metadata = json.loads(str(archive['metadata']))
        return [(meta, {k.split('/', 1)[1]: archive[k].copy()
                        for k in archive.files if k.startswith(f'{i}/')})
                for i, meta in enumerate(metadata)]


def build_corpus(model, encounters, seeds):
    from env.core import vocab
    from rl.evaluation.nosl.scenarios import SCENARIOS, make_env
    import slaythespire as sts

    result = []
    for encounter in encounters:
        if encounter not in sts.encounter_names():
            raise ValueError(f'unknown encounter: {encounter}')
        for seed in seeds:
            env = make_env(model, SCENARIOS['bash_plus_strike'], False, seed)
            try:
                env.thick.gc.set_relics([])
                env.thick.bc = sts.create_battle(env.thick.gc, encounter)
                env._invalidate_backend_cache()
                root = env._obs()
                raw = env.thick.bc.preview_end_turn_boundary(4 << 29)
                states = [('root', root)]
                if raw is not None and raw['player']['hp'] > 0:
                    states.append(('end_turn_boundary', env._native_leaf_observation(raw)))
                for kind, obs in states:
                    ids = [int(x) for x in obs['monster_id'] if int(x)]
                    result.append((dict(encounter=encounter, seed=int(seed), kind=kind,
                                        monsters=[vocab.MONSTER_VOCAB[x] for x in ids],
                                        relics=[], fixture='bash_plus_strike'), obs))
            finally:
                env.close()
    return result


def evaluate_corpus(model, corpus, *, axes=('player', 'enemy'), batch_size=64, tolerance=1e-5):
    from rl.search.native import forward_root
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    model.policy.set_training_mode(False)
    curves = []
    for meta, base in corpus:
        for axis in axes:
            slots = [0] if axis == 'player' else [i for i, x in enumerate(base['monster_id'])
                if x and base['monster_scalars'][i, 0] > 0 and base['monster_scalars'][i, 1] > 0]
            for slot in slots:
                maximum = maximum_hp(base, axis, slot)
                if maximum < 2:
                    continue
                hps = list(range(1, maximum + 1))
                values = []
                for start in range(0, len(hps), batch_size):
                    observations = [intervene(base, axis, hp, slot=slot)
                                    for hp in hps[start:start+batch_size]]
                    batch = {k: np.stack([o[k] for o in observations]) for k in base}
                    _, predicted = forward_root(model, batch)
                    values.extend(float(x) for x in predicted)
                summary = summarize(hps, values, axis, tolerance=tolerance)
                if not summary['finite']:
                    raise FloatingPointError(f'nonfinite value in {meta}, {axis}, slot {slot}')
                current_hp = np.asarray(hps)/200 if axis == 'player' else float(base['player_scalars'][0])
                excess = np.asarray(values)-current_hp
                summary.update(above_current_hp_fraction=float(np.mean(excess > tolerance)),
                               max_above_current_hp=float(max(0, excess.max())))
                curves.append(dict(**meta, axis=axis, slot=slot if axis == 'enemy' else None,
                                   hps=hps, values=values, summary=summary))
    return dict(schema_version=2, corpus_sha256=corpus_digest(corpus),
                protocol='fixed encoded HP interventions; no mechanic transitions during sweep',
                curves=curves, summary=dict(curves=len(curves),
                    curves_with_inversions=sum(c['summary']['adverse_steps'] > 0 for c in curves),
                    finite=True))
