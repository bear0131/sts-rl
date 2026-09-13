"""Atomic per-rank learner recovery bundles, separate from inference checkpoints.

Environment instances restart on recovery. Incomplete fights are discarded;
completed replay targets, policy/Adam, parent RNG and global step are restored.
This is learner recovery, not bitwise continuation of live simulator instances.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle(path, *, rank, world_size):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] != 1 or (manifest['rank'], manifest['world_size']) != (rank, world_size):
        raise ValueError('recovery bundle rank/world size mismatch')
    for name in ('model.zip', 'learner.pt'):
        if _sha(path / name) != manifest['sha256'][name]:
            raise ValueError('recovery bundle checksum mismatch: ' + name)
    return manifest


def save_bundle(model, root, *, rank, world_size, run_id, wandb_id, config):
    root = Path(root) / f'rank{rank}'
    root.mkdir(parents=True, exist_ok=True)
    step = int(model.num_timesteps)
    destination = root / f'step-{step:012d}'
    if destination.exists():
        if (destination / 'manifest.json').exists():
            read_bundle(destination, rank=rank, world_size=world_size)
            return destination
        # A killed writer may leave files without its final commit marker.
        # Preserve that evidence, then allow recovery from the previous step
        # to reach and publish this step again.
        abandoned = root / f'.incomplete-{destination.name}-{uuid.uuid4().hex}'
        if destination.resolve().parent != root.resolve() or abandoned.resolve().parent != root.resolve():
            raise ValueError('incomplete recovery generation escaped its directory')
        abandoned.mkdir()
        for partial in destination.iterdir():
            if not partial.is_file() or partial.resolve().parent != destination.resolve():
                raise ValueError('unexpected content in incomplete recovery generation')
            partial.replace(abandoned / partial.name)
    # A generation becomes visible to readers only when its manifest appears.
    # Publishing a file avoids Windows directory-rename races with file scanners.
    temporary = destination
    temporary.mkdir(exist_ok=True)
    try:
        model.save(temporary / 'model.zip')
        replay = getattr(model, '_nosl_is_az_replay', None)
        state = dict(schema_version=1, global_step=step,
            replay=None if replay is None else replay.state_dict(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None)
        torch.save(state, temporary / 'learner.pt')
        manifest = dict(schema_version=1, rank=rank, world_size=world_size,
            global_step=step, run_id=run_id, wandb_id=wandb_id,
            environment_resume='restart; unfinished fights discarded',
            config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()},
            sha256={name:_sha(temporary / name) for name in ('model.zip','learner.pt')})
        marker = temporary / '.manifest.tmp'
        marker.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        marker.replace(temporary / 'manifest.json')
    except BaseException:
        if temporary.resolve().parent == root.resolve():
            shutil.rmtree(temporary)
        raise
    latest = root / '.latest.tmp'
    latest.write_text(destination.name,encoding='utf-8')
    latest.replace(root / 'latest')
    return destination


def restore_learner(model, path, *, rank, world_size):
    manifest = read_bundle(path, rank=rank, world_size=world_size)
    if manifest['global_step'] != model.num_timesteps:
        raise ValueError('policy and learner global steps disagree')
    state = torch.load(Path(path) / 'learner.pt', map_location='cpu', weights_only=False)
    if state['schema_version'] != 1 or state['global_step'] != model.num_timesteps:
        raise ValueError('invalid learner recovery state')
    if state['replay'] is not None:
        model._ensure_nosl_is_az_replay().load_state_dict(state['replay'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if model.device.type != 'cuda':
            raise ValueError('CUDA learner RNG requires CUDA recovery')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return manifest


def prune_bundles(root, *, rank, keep):
    if keep < 2:
        raise ValueError('keep at least two recovery generations')
    directory = Path(root) / f'rank{rank}'
    bundles = sorted(p for p in directory.glob('step-*') if (p/'manifest.json').is_file())
    for path in bundles[:-keep]:
        # Only complete, owned generation directories under the explicit root.
        if path.resolve().parent != directory.resolve():
            raise ValueError('recovery generation escaped its directory')
        shutil.rmtree(path)
