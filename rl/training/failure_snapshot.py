"""Bounded diagnostic scenes, separate from resumable learner checkpoints."""
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch


def _json_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return str(value)


def _resident(env):
    while hasattr(env, 'venv'):
        env = env.venv
    return env


def save_failure_scene(model, root, *, rank, error, context=None):
    """Best effort: never replace the original error, reset, step or infer."""
    if root is None:
        return None
    try:
        env = _resident(model.get_env())
        context = dict(context or getattr(env, '_step_failure', {}) or {})
        active = getattr(env, 'episode_rollout_active', None)
        indices = list(range(env.num_envs)) if active is None else np.flatnonzero(active).tolist()
        slot = context.get('slot')
        selected = [int(slot)] if slot is not None else indices[:4]
        directory = Path(root) / 'failure-scenes' / f'rank{rank}' / f'{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        directory.mkdir(parents=True, exist_ok=False)
        report = dict(schema_version=1, kind='diagnostic-only-not-learner-checkpoint',
                      rank=rank, global_step=int(model.num_timesteps),
                      error_type=type(error).__name__, error=str(error), context=context,
                      active_slots=indices, selected_slots=selected, slots=[])
        for index in selected:
            item = dict(slot=index)
            report['slots'].append(item)
            wrapped = env.envs[index]
            base = wrapped.unwrapped
            # Keep scene and journal independent: either can survive a broken
            # encoder/exporter. No extra policy call or environment transition.
            try:
                thick = base.thick
                scene = dict(canonical=thick.obs_canonical(),
                             combat=None if thick.bc is None else thick.bc.to_dict())
                name = f'slot{index}-scene.json'
                (directory / name).write_text(json.dumps(scene, ensure_ascii=False, indent=2,
                                                        default=_json_value), encoding='utf8')
                item['scene'] = name
            except Exception as capture_error:
                item['scene_error'] = repr(capture_error)
            try:
                state = wrapped.export_training_state()
                name = f'slot{index}-environment.pt'
                torch.save(state, directory / name)
                item['journal'] = name
                item['journal_actions'] = len(state.get('base', state).get('actions', []))
            except Exception as capture_error:
                item['journal_error'] = repr(capture_error)
        temporary = directory / 'manifest.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                        default=_json_value), encoding='utf8')
        temporary.replace(directory / 'manifest.json')
        print(f'FAILURE_SCENE {directory}', flush=True)
        return directory
    except Exception as capture_error:
        print(f'FAILURE_SCENE_UNAVAILABLE {type(capture_error).__name__}: {capture_error}', flush=True)
        return None


def install_truncation_recorder(model, root, *, rank):
    """Record up to four nonfatal truncations before VecEnv auto-reset."""
    if root is None:
        return
    env = _resident(model.get_env())
    captured = 0

    def record(context):
        nonlocal captured
        if captured < 4:
            captured += 1
            save_failure_scene(model, root, rank=rank,
                               error=RuntimeError('environment limit: '+str(context.get('reason','truncated'))),
                               context=context)

    env.failure_scene_recorder = record
