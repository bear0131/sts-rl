"""Atomic per-rank learner and live-environment recovery bundles."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import uuid
import zipfile

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
    reclaim_timeout_seconds=5.0,
):
    """Allow quota accounting to catch up after pruning, then enforce space."""
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
    # Unlink can return before XFS project quotas release a large replay file.
    # Keep the full safety margin; only recheck the observed free space briefly.
    deadline = time.monotonic() + reclaim_timeout_seconds
    while free_bytes < required_bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))
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
    if len(states) != env.num_envs:
        raise ValueError('environment recovery count mismatch; explicit complete-cohort growth required')
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


def _grow_environment_slots(model, manifest, state, environment, *, rank, world_size):
    """Append seeded slots while retaining every source slot and learner field.

    Complete-game collectors require a natural-game boundary; continuous PPO
    retains its live action journals. Per-slot arrays must match the source.
    Caller restores learner RNG after replay.
    """
    env = model.get_env()
    growth = getattr(model, 'episode_rollout_growth', None)
    old_count = len(environment['states'])
    pure_ppo = getattr(model, 'nosl_supervision_contract', None) == 'shared-wholerun-ppo-room-v1'
    if pure_ppo:
        if (env is None or env.num_envs <= old_count
                or manifest['config'].get('n_envs') != old_count
                or model.nosl_is_az or model.hierarchical_combat_hp):
            raise ValueError('continuous PPO environment growth geometry mismatch')
        growth = dict(mode='continuous-ppo', source_games_per_rank=old_count,
            games_per_rank=env.num_envs, source_round=getattr(model, 'ppo_rollouts_completed', 0),
            next_global_seed=int(manifest['config']['seed']) + manifest['global_step'] * world_size)
    elif (env is None or not growth or old_count != growth['source_games_per_rank']
            or env.num_envs != growth['games_per_rank'] or env.num_envs <= old_count
            or manifest.get('rollout_round') != growth['source_round']
            or manifest['config'].get('episode_rollout_config', {}).get('games_per_rank') != old_count
            or getattr(model, 'episode_rollout_world_size', None) != world_size
            or getattr(model, 'episode_rollout_rank', None) != rank):
        raise ValueError('environment growth source/geometry/boundary mismatch')
    if (getattr(model, '_outer_return', False) or state['replay'] is not None
            or any(state.get('upper_pending') or ())
            or any(state.get('nosl_is_az_pending_fights') or ())):
        raise ValueError('environment growth source/geometry/boundary mismatch')
    extra = env.num_envs - old_count
    # Copy only per-slot data; source files and loaded optimizer state are never edited.
    state = dict(state)
    environment = dict(environment)
    for name, fill, dtype, owner in (
            ('last_episode_starts', True, bool, state),
            ('episode_returns', 0., np.float32, environment),
            ('episode_lengths', 0, np.int32, environment)):
        values = np.asarray(owner[name], dtype=dtype)
        if values.shape != (old_count,):
            raise ValueError('environment growth source array mismatch: ' + name)
        owner[name] = np.concatenate((values, np.full(extra, fill, dtype=dtype)))
    for name, fill in (('upper_pending', None), ('nosl_is_az_pending_fights', []),
                       ('upper_episode_ids', growth['source_round'])):
        values = state.get(name)
        if values is not None:
            if pure_ppo and len(values) == 0:
                continue  # Inactive NOSL/joint-PPO queues have no per-slot state.
            if len(values) != old_count:
                raise ValueError('environment growth source slot mismatch: ' + name)
            state[name] = copy.deepcopy(list(values)) + [copy.deepcopy(fill) for _ in range(extra)]
    if state.get('outer_pending_rewards') is not None:
        values = np.asarray(state['outer_pending_rewards'])
        if values.shape != (old_count,) or np.any(values):
            raise ValueError('environment growth cannot discard pending outer rewards')
        state['outer_pending_rewards'] = np.concatenate((values, np.zeros(extra, dtype=values.dtype)))
    first_seed = growth['next_global_seed'] + rank * env.num_envs + old_count
    for index in range(old_count, env.num_envs):
        env.env_method('reset', seed=first_seed + index - old_count, indices=index)
    appended = env.env_method('export_training_state', indices=list(range(old_count, env.num_envs)))
    if len(appended) != extra:
        raise ValueError('environment growth export count mismatch')
    environment['states'] = environment['states'] + appended
    model._recovery_environment_growth = dict(
        **growth, source_global_step=manifest['global_step'], restored_environments=old_count,
        appended_environments=extra, appended_first_seed=first_seed,
        appended_last_seed=first_seed + extra - 1,
        source_observation_sha256=environment['last_observation_sha256'])
    return state, environment, old_count


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
    identity = getattr(model, 'outer_return_contract', None)
    if identity is not None:
        from rl.algorithms import outer_return
        outer_return.setup(model)
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
            outer_return_contract=identity,
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
                outer_pending_rewards=copy.deepcopy(getattr(model, '_outer_pending_rewards', None)),
                upper_pending=copy.deepcopy(getattr(model, '_upper_pending', None)),
                upper_episode_ids=copy.deepcopy(getattr(model, '_upper_episode_ids', None)),
                ep_info_buffer=copy.deepcopy(list(model.ep_info_buffer or ())),
                ep_success_buffer=copy.deepcopy(list(model.ep_success_buffer or ())),
                episode_num=int(getattr(model, '_episode_num', 0)),
            )
        # Protocol 5 writes NumPy buffers directly instead of the protocol-2
        # bytes-to-Unicode round trip. Payload/schema and old-bundle loading
        # stay unchanged; supported Python runtimes are all >= 3.11.
        torch.save(state, temporary / 'learner.pt', pickle_protocol=5)
        files = ['model.zip', 'learner.pt']
        if environment is not None:
            torch.save(environment, temporary / 'environment.pt', pickle_protocol=5)
            files.append('environment.pt')
        from env.core.encode import OBS_VERSION
        manifest = dict(schema_version=schema_version, observation_version=OBS_VERSION,
            outer_return_contract=identity,
            rank=rank, world_size=world_size,
            global_step=step, run_id=run_id, wandb_id=wandb_id,
            rollout_round=getattr(model, 'episode_rollouts_completed',
                                  getattr(model, 'ppo_rollouts_completed', None)),
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
    model, path, *, rank, world_size, restore_environment: bool = True, grow_environments: bool = False,
):
    """Restore learner state, optionally replacing the live environment.

    ``restore_environment=False`` is an explicit recovery boundary for an
    incompatible simulator journal or a fork with changed actors. Policy,
    Adam, replay and learner RNG still come from the source bundle; unfinished
    environment episodes and their pending fight labels are reset.
    """
    manifest = read_bundle(path, rank=rank, world_size=world_size)
    if grow_environments and (not restore_environment or manifest['schema_version'] < 2):
        raise ValueError('environment growth requires complete environment restoration')
    identity = getattr(model, 'outer_return_contract', None)
    if manifest.get('outer_return_contract') != identity:
        raise ValueError('recovery objective/teacher/context/native contract mismatch')
    if identity is not None:
        from rl.algorithms import outer_return
        outer_return.setup(model)
    if manifest['global_step'] != model.num_timesteps:
        raise ValueError('policy and learner global steps disagree')
    state = torch.load(Path(path) / 'learner.pt', map_location='cpu', weights_only=False)
    if state.get('outer_return_contract') != identity:
        raise ValueError('learner objective/teacher/context/native contract mismatch')
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
        if grow_environments and env is None:
            raise ValueError('environment growth requires a live VecEnv')
        if env is not None:
            if environment.get('schema_version') != 1:
                raise ValueError('invalid environment recovery state')
            source_count = len(environment['states'])
            if grow_environments:
                state, environment, source_count = _grow_environment_slots(
                    model, manifest, state, environment, rank=rank, world_size=world_size)
            observations = _restore_environment_states(
                env, environment['states']
            )
            source_observations = ({key: value[:source_count] for key, value in observations.items()}
                                   if isinstance(observations, dict) else observations[:source_count])
            if _observation_sha256(source_observations) != environment[
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
            model._outer_pending_rewards = copy.deepcopy(state.get('outer_pending_rewards'))
            if getattr(model, '_outer_return', False):
                outer_return.validate_pending(model, env.num_envs)
            model._upper_pending = copy.deepcopy(state.get('upper_pending'))
            model._upper_episode_ids = copy.deepcopy(state.get('upper_episode_ids'))
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
            if grow_environments:
                restore_rng()
    elif manifest['schema_version'] >= 2:
        # Incompatible journals cannot be replayed under changed simulator
        # rules. Reset every environment and discard source-local pending
        # fights/episode windows while retaining learner state above.
        env = model.get_env()
        if env is None:
            raise ValueError('environment reset fork requires a live VecEnv')
        model._last_obs = env.reset()
        model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
        model._nosl_is_az_pending_fights = [
            [] for _ in range(env.num_envs)
        ]
        model._outer_pending_rewards = np.zeros(env.num_envs, dtype=np.float64)
        model._upper_pending = None
        model._upper_episode_ids = None
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


def _bundle_rollout_round(path):
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    round_value = manifest.get('rollout_round')
    if round_value is None:
        # Old complete-cohort bundles stored the explicit round in SB3 data.
        # Do not infer it from variable-length environment action counts.
        with zipfile.ZipFile(path / 'model.zip') as archive:
            round_value = json.loads(archive.read('data')).get('episode_rollouts_completed')
    if type(round_value) is not int or round_value < 1:
        raise ValueError('checkpoint retention requires an explicit positive rollout round')
    return round_value


def prune_bundles(root, *, rank, keep, round_multiple=0):
    """Keep newest generations, or latest round-multiple anchor plus current.

    Round-based retention must only run AFTER all ranks publish a new bundle;
    old anchor/current bundles remain available during the atomic write.
    """
    if keep < 1:
        raise ValueError('keep at least one recovery generation')
    if type(round_multiple) is not int or round_multiple < 0:
        raise ValueError('round_multiple must be a nonnegative integer')
    directory = Path(root) / f'rank{rank}'
    bundles = sorted(p for p in directory.glob('step-*')
                     if _is_step_directory_name(p.name) and (p/'manifest.json').is_file())
    retained = set(bundles[-keep:])
    if round_multiple and bundles:
        rounds = [_bundle_rollout_round(p) for p in bundles]
        if any(a >= b for a, b in zip(rounds, rounds[1:])):
            raise ValueError('checkpoint rollout rounds must increase with global steps')
        anchors = [p for p, r in zip(bundles, rounds) if r % round_multiple == 0]
        retained = {bundles[-1], *anchors[-1:]}
    for path in bundles:
        if path in retained:
            continue
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
