"""Atomic per-rank learner and live-environment recovery bundles."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


_BUNDLE_SPACE_RESERVE_BYTES = 256 * 1024 * 1024


def _bundle_bytes(path):
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _is_step_directory_name(name):
    step = name.removeprefix('step-')
    return name.startswith('step-') and len(step) >= 12 and step.isdigit()


def _is_incomplete_directory_name(name):
    if not name.startswith('.incomplete-step-'):
        return False
    step, separator, suffix = name.removeprefix(
        '.incomplete-step-'
    ).partition('-')
    return (
        bool(separator)
        and len(step) >= 12
        and step.isdigit()
        and len(suffix) == 32
        and all(character in '0123456789abcdef' for character in suffix)
    )


def ensure_bundle_disk_space(
    root, *, rank, reserve_bytes=_BUNDLE_SPACE_RESERVE_BYTES,
):
    """Fail before an atomic save when another generation cannot fit."""
    directory = Path(root) / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    bundles = [
        path for path in directory.glob('step-*')
        if (path / 'manifest.json').is_file()
    ]
    previous_bytes = max((_bundle_bytes(path) for path in bundles), default=0)
    growth_reserve = max(reserve_bytes, previous_bytes // 10)
    required_bytes = previous_bytes + growth_reserve
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            'insufficient disk space for atomic recovery bundle: '
            f'rank={rank} free_bytes={free_bytes} '
            f'required_bytes={required_bytes} '
            f'previous_bundle_bytes={previous_bytes}'
        )
    return {
        'free_bytes': free_bytes,
        'required_bytes': required_bytes,
        'previous_bundle_bytes': previous_bytes,
    }


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_sha256(observation):
    digest = hashlib.sha256()
    values = observation if isinstance(observation, dict) else {"observation": observation}
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _capture_environment(model):
    env = model.get_env()
    if env is None:
        return None
    supported = env.has_attr("training_state_checkpointable")
    if not supported:
        return None
    states = env.env_method("export_training_state")
    if len(states) != env.num_envs:
        raise ValueError("environment export count mismatch")
    return {
        "schema_version": 1,
        "states": states,
        "episode_returns": np.asarray(
            getattr(env, "episode_returns", np.zeros(env.num_envs)),
            dtype=np.float32,
        ).copy(),
        "episode_lengths": np.asarray(
            getattr(env, "episode_lengths", np.zeros(env.num_envs)),
            dtype=np.int32,
        ).copy(),
        "episode_count": int(getattr(env, "episode_count", 0)),
        "last_observation_sha256": _observation_sha256(model._last_obs),
    }


def _restore_environment_states(env, states):
    try:
        restore_many = env.restore_training_states
    except AttributeError:
        restored = [
            env.env_method(
                "restore_training_state", state, indices=index
            )[0]
            for index, state in enumerate(states)
        ]
        observations = [item[0] for item in restored]
        if isinstance(observations[0], dict):
            return {
                key: np.stack([observation[key] for observation in observations])
                for key in observations[0]
            }
        return np.stack(observations)
    return restore_many(states)


def read_bundle(path, *, rank, world_size):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] not in (1, 2) or (manifest['rank'], manifest['world_size']) != (rank, world_size):
        raise ValueError('recovery bundle rank/world size mismatch')
    required = ('model.zip', 'learner.pt') + (
        ('environment.pt',) if manifest['schema_version'] >= 2 else ()
    )
    for name in required:
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
    ensure_bundle_disk_space(root.parent, rank=rank)
    # A generation becomes visible to readers only when its manifest appears.
    # Publishing a file avoids Windows directory-rename races with file scanners.
    temporary = destination
    temporary.mkdir(exist_ok=True)
    try:
        environment = _capture_environment(model)
        schema_version = 2 if environment is not None else 1
        model.save(temporary / 'model.zip')
        replay = getattr(model, '_nosl_is_az_replay', None)
        state = dict(schema_version=schema_version, global_step=step,
            replay=None if replay is None else replay.state_dict(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None)
        if schema_version >= 2:
            state.update(
                last_episode_starts=np.asarray(
                    model._last_episode_starts, dtype=bool
                ).copy(),
                nosl_is_az_pending_fights=copy.deepcopy(getattr(
                    model, '_nosl_is_az_pending_fights', None,
                )),
                ep_info_buffer=copy.deepcopy(list(model.ep_info_buffer or ())),
                ep_success_buffer=copy.deepcopy(list(model.ep_success_buffer or ())),
                episode_num=int(getattr(model, '_episode_num', 0)),
            )
        torch.save(state, temporary / 'learner.pt')
        files = ['model.zip', 'learner.pt']
        if environment is not None:
            torch.save(environment, temporary / 'environment.pt')
            files.append('environment.pt')
        manifest = dict(schema_version=schema_version, rank=rank, world_size=world_size,
            global_step=step, run_id=run_id, wandb_id=wandb_id,
            environment_resume=(
                'action-journal replay; live episodes restored'
                if schema_version >= 2
                else 'restart; unfinished fights discarded'
            ),
            config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()},
            sha256={name:_sha(temporary / name) for name in files})
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


def restore_learner(
    model, path, *, rank, world_size, restore_environment: bool = True,
):
    """Restore learner state, optionally replacing the live environment.

    ``restore_environment=False`` is an explicit fork boundary for runs that
    intentionally change the number of actors.  Policy/Adam/replay/RNG state
    still comes from the source bundle, while all environment-local progress is
    reset to the freshly constructed VecEnv.  Silent state-count mismatches are
    therefore impossible.
    """
    manifest = read_bundle(path, rank=rank, world_size=world_size)
    if manifest['global_step'] != model.num_timesteps:
        raise ValueError('policy and learner global steps disagree')
    state = torch.load(Path(path) / 'learner.pt', map_location='cpu', weights_only=False)
    if state['schema_version'] != manifest['schema_version'] or state['global_step'] != model.num_timesteps:
        raise ValueError('invalid learner recovery state')
    if state['replay'] is not None:
        model._ensure_nosl_is_az_replay().load_state_dict(state['replay'])
    def restore_rng() -> None:
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            if model.device.type != 'cuda':
                raise ValueError('CUDA learner RNG requires CUDA recovery')
            torch.cuda.set_rng_state_all(state['cuda_rng'])

    restore_rng()
    model._recovery_environment_restored = False
    if manifest['schema_version'] >= 2 and restore_environment:
        environment = torch.load(
            Path(path) / 'environment.pt', map_location='cpu', weights_only=False
        )
        env = model.get_env()
        if env is not None:
            if environment.get('schema_version') != 1:
                raise ValueError('invalid environment recovery state')
            observations = _restore_environment_states(
                env, environment['states']
            )
            if _observation_sha256(observations) != environment[
                'last_observation_sha256'
            ]:
                raise ValueError('learner/environment observation mismatch')
            model._last_obs = observations
            model._last_episode_starts = np.asarray(
                state['last_episode_starts'], dtype=bool
            ).copy()
            if hasattr(env, 'episode_returns'):
                env.episode_returns[:] = environment['episode_returns']
            if hasattr(env, 'episode_lengths'):
                env.episode_lengths[:] = environment['episode_lengths']
            if hasattr(env, 'episode_count'):
                env.episode_count = int(environment['episode_count'])
            model._nosl_is_az_pending_fights = copy.deepcopy(
                state['nosl_is_az_pending_fights']
            )
            if model.ep_info_buffer is not None:
                model.ep_info_buffer.clear()
                model.ep_info_buffer.extend(copy.deepcopy(state['ep_info_buffer']))
            if model.ep_success_buffer is not None:
                model.ep_success_buffer.clear()
                model.ep_success_buffer.extend(copy.deepcopy(
                    state['ep_success_buffer']
                ))
            model._episode_num = int(state['episode_num'])
            model._recovery_environment_restored = True
    elif manifest['schema_version'] >= 2:
        # A geometry-changing fork cannot consume the source VecEnv's state
        # array.  Reset every environment and discard source-local pending
        # fights/episode windows while retaining learner state above.
        env = model.get_env()
        if env is None:
            raise ValueError('environment reset fork requires a live VecEnv')
        model._last_obs = env.reset()
        model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
        model._nosl_is_az_pending_fights = [
            [] for _ in range(env.num_envs)
        ]
        if hasattr(env, 'episode_returns'):
            env.episode_returns[...] = 0.0
        if hasattr(env, 'episode_lengths'):
            env.episode_lengths[...] = 0
        if hasattr(env, 'episode_count'):
            env.episode_count = 0
        if model.ep_info_buffer is not None:
            model.ep_info_buffer.clear()
        if model.ep_success_buffer is not None:
            model.ep_success_buffer.clear()
        model._episode_num = 0
        # VecEnv.reset() may execute wrappers in the parent process.  Restore
        # the learner streams after that reset so a geometry fork cannot shift
        # replay sampling or optimizer randomness merely by constructing new
        # actors.
        restore_rng()
    return manifest


def prune_bundles(root, *, rank, keep):
    if keep < 1:
        raise ValueError('keep at least one recovery generation')
    directory = Path(root) / f'rank{rank}'
    bundles = sorted(p for p in directory.glob('step-*') if (p/'manifest.json').is_file())
    for path in bundles[:-keep]:
        # Only complete, owned generation directories under the explicit root.
        if path.resolve().parent != directory.resolve():
            raise ValueError('recovery generation escaped its directory')
        shutil.rmtree(path)
    # Interrupted generations are unusable without a manifest. Retain only
    # the newest one as diagnostic evidence so repeated restarts stay bounded.
    incomplete = [
        path for path in directory.iterdir()
        if path.is_dir() and (
            _is_incomplete_directory_name(path.name)
            or (
                _is_step_directory_name(path.name)
                and not (path / 'manifest.json').is_file()
            )
        )
    ] if directory.exists() else []
    incomplete.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    for path in incomplete[:-1]:
        if path.resolve().parent != directory.resolve():
            raise ValueError('incomplete recovery generation escaped its directory')
        shutil.rmtree(path)
