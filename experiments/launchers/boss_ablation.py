"""Explicit offline forks for the two COMBAT06 Boss ablations.

The source remains immutable. Model/Adam, environment journals and RNG are
preserved; only obsolete Boss replay/pending rows are removed in the fork.
The existing distributed trainer starts the resulting bundle via --fork-state.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from stable_baselines3.common.save_util import data_to_json, json_to_data

from rl.algorithms.combat_distribution import boss_ablation_identity
from rl.training.checkpoints import read_bundle


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _state_digest(value):
    """Content digest independent of pickle storage IDs and array ownership."""
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, dict):
            for key in sorted(v):
                visit(key); visit(v[key])
        elif isinstance(v, (tuple, list)):
            digest.update(type(v).__name__.encode()); visit(len(v))
            for item in v: visit(item)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            digest.update(str(a.dtype).encode()); digest.update(repr(a.shape).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        else:
            digest.update(repr((type(v).__name__, v)).encode())
    visit(value)
    return digest.hexdigest()


def remove_boss_replay(state):
    """Retain non-Boss rows in chronological order, including sparse policies."""
    state = deepcopy(state)
    count = state['size']
    if not count:
        return state, 0
    boss = state['observations']['combat_is_boss'][:count].reshape(-1)
    if not np.isin(boss, [0, 1]).all():
        raise ValueError('replay Boss flag is not binary')
    keep = np.flatnonzero(boss == 0)
    keep = keep[np.argsort(state['arrays']['sample_ids'][keep], kind='stable')]
    state['observations'] = {k: v[keep].copy() for k, v in state['observations'].items()}
    state['arrays'] = {k: None if v is None else v[keep].copy() for k, v in state['arrays'].items()}
    if state['joint'] is not None:
        old = state['joint']; offsets = old['offsets']
        parts = [slice(int(offsets[i]), int(offsets[i+1])) for i in keep]
        state['joint'] = dict(offsets=np.cumsum([0]+[p.stop-p.start for p in parts], dtype=np.int64),
            ids=np.concatenate([old['ids'][p] for p in parts]) if parts else old['ids'][:0].copy(),
            probabilities=np.concatenate([old['probabilities'][p] for p in parts]) if parts else old['probabilities'][:0].copy())
    state['size'] = len(keep)
    state['position'] = len(keep) % state['capacity']
    return state, count-len(keep)


def fork_bundle(source, destination, *, rank, world_size, mode):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('fork needs a separate new destination')
    manifest = read_bundle(source, rank=rank, world_size=world_size)
    original = manifest['outer_return_contract']
    if manifest['schema_version'] != 2 or original.get('boss_ablation') or original['boss_win_weight'] != .9:
        raise ValueError('fork must start at the original COMBAT06 complete state')
    identity = boss_ablation_identity(original, mode)
    state = torch.load(source/'learner.pt', map_location='cpu', weights_only=False)
    if state['outer_return_contract'] != original or state['replay']['objective_identity'] != original:
        raise ValueError('inconsistent source identities')
    retained_keys = set(state) - {'outer_return_contract', 'replay', 'nosl_is_az_pending_fights'}
    retained_digest = _state_digest({k: state[k] for k in retained_keys})
    state['replay'], removed = remove_boss_replay(state['replay'])
    state['outer_return_contract'] = deepcopy(identity)
    state['replay']['objective_identity'] = deepcopy(identity)
    pending_removed = 0
    for i, samples in enumerate(state['nosl_is_az_pending_fights']):
        flags = {bool(s['observation']['combat_is_boss'][0]) for s in samples}
        if len(flags) > 1:
            raise ValueError('pending fight mixes Boss identities')
        if flags == {True}:
            pending_removed += len(samples)
            state['nosl_is_az_pending_fights'][i] = []
    destination.mkdir(parents=True)
    with zipfile.ZipFile(source/'model.zip') as z:
        data = json_to_data(z.read('data').decode())
        if data['outer_return_contract'] != original:
            raise ValueError('model identity differs from source manifest')
        data['outer_return_contract'] = identity
        data['policy_kwargs']['boss_ablation'] = mode
        with zipfile.ZipFile(destination/'model.zip', 'w', compression=zipfile.ZIP_DEFLATED) as out:
            for info in z.infolist():
                out.writestr(deepcopy(info), data_to_json(data) if info.filename == 'data' else z.read(info.filename))
        with zipfile.ZipFile(destination/'model.zip') as out:
            assert all(z.read(n) == out.read(n) for n in z.namelist() if n != 'data')
    torch.save(state, destination/'learner.pt', pickle_protocol=5)
    check = torch.load(destination/'learner.pt', map_location='cpu', weights_only=False)
    assert _state_digest({k: check[k] for k in retained_keys}) == retained_digest
    assert _state_digest(check) == _state_digest(state)
    del check
    os.link(source/'environment.pt', destination/'environment.pt')
    manifest = deepcopy(manifest)
    manifest['outer_return_contract'] = identity
    manifest['sha256'] = {n: _sha(destination/n) for n in manifest['sha256']}
    assert manifest['sha256']['environment.pt'] == _sha(source/'environment.pt')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
    receipt = dict(schema='combat06-boss-ablation-fork-v1', source=str(source), mode=mode,
        source_files=read_bundle(source, rank=rank, world_size=world_size)['sha256'],
        files=manifest['sha256'], global_step=manifest['global_step'],
        removed_boss_replay_rows=removed, retained_nonboss_replay_rows=state['replay']['size'],
        removed_boss_pending_rows=pending_removed,
        model_and_adam_members_identical=True, environment_bytes_identical=True,
        learner_rng_preserved=True, ordinary_fight_targets_preserved=True)
    receipt['unchanged_learner_fields_sha256'] = retained_digest
    (destination/'boss-ablation-receipt.json').write_text(json.dumps(receipt, indent=2))
    for path in destination.iterdir():
        with path.open('rb') as f:
            os.fsync(f.fileno())
    if hasattr(os, 'O_DIRECTORY'):
        fd = os.open(destination, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    read_bundle(destination, rank=rank, world_size=world_size)
    return receipt
