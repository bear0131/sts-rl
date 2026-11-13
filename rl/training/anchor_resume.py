"""Explicit, additive migration of whole-run PPO recovery bundles to exit anchors."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from gymnasium import spaces

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.shared_wholerun import SharedWholeRunPolicy
from rl.training.checkpoints import read_bundle, _sha, _observation_sha256
from rl.training.environments import observation_flags


def enable_anchor_policy(model):
    """Preserve all existing weights/Adam entries; insert two zero input columns."""
    old = model.policy
    if (type(old) is not SharedWholeRunPolicy or old.room_lambda != 1
            or old.combat_value_target != 'whole-run' or old.reward_flow != 'sequential-v1'
            or model.device.type != 'cpu' or model.nosl_is_az
            or model.hierarchical_combat_hp or model.gamma != 1 or model.gae_lambda != 1):
        raise ValueError('anchor migration requires CPU shared whole-run lambda=1 sequential PPO')
    space = deepcopy(model.observation_space)
    if 'combat_exit_anchor' in space.spaces:
        raise ValueError('source already has anchor observations')
    space.spaces['combat_exit_anchor'] = spaces.Box(0, 1, (2,), np.float32)
    kwargs = dict(model.policy_kwargs, combat_value_target='pre-reward-anchor')
    with torch.random.fork_rng():
        new = SharedWholeRunPolicy(space, model.action_space, model.lr_schedule, **kwargs)
    if list(dict(old.named_parameters())) != list(dict(new.named_parameters())):
        raise ValueError('anchor migration changed parameter ordering')
    name = 'core.combat.player_projection.weight'
    split = sum(int(np.prod(model.observation_space[k].shape)) for k in old.core.combat.scalar_keys)
    def expand(value):
        if value.ndim != 2 or value.shape[1] != split + 6:
            raise ValueError('unexpected scalar projection layout')
        return torch.cat((value[:, :split], value.new_zeros((len(value), 2)), value[:, split:]), 1)
    weights = old.state_dict()
    weights[name] = expand(weights[name])
    new.load_state_dict(weights, strict=True)
    adam = deepcopy(old.optimizer.state_dict())
    named = {id(p): n for n, p in old.named_parameters()}
    for live_group, saved_group in zip(old.optimizer.param_groups, adam['param_groups'], strict=True):
        for p, ident in zip(live_group['params'], saved_group['params'], strict=True):
            if named[id(p)] == name:
                for key, value in adam['state'].get(ident, {}).items():
                    if torch.is_tensor(value) and value.ndim:
                        adam['state'][ident][key] = expand(value)
    new.optimizer.load_state_dict(adam)
    model.policy, model.observation_space, model.policy_kwargs = new, space, kwargs
    for field in ('_last_obs', '_last_original_obs'):
        obs = getattr(model, field, None)
        if obs is not None:
            obs['combat_exit_anchor'] = np.zeros((len(next(iter(obs.values()))), 2), np.float32)
    return old


def migrate_anchor_bundle(source, destination):
    """Write a new bundle, preserving journals, wrappers, RNG and progress.

    Historical actions replay with anchors disabled until the migration boundary.
    Thus a past reward action cannot accidentally be interpreted as anchor continue.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    raw = json.loads((source / 'manifest.json').read_text())
    manifest = read_bundle(source, rank=raw['rank'], world_size=raw['world_size'])
    if manifest['schema_version'] != 2:
        raise ValueError('anchor continuation requires live environment recovery')
    model = AccumulatingMaskablePPO.load(source / 'model.zip', device='cpu')
    environment = torch.load(source / 'environment.pt', map_location='cpu', weights_only=False)
    old_observations, observations, masks = [], [], []
    for state in environment['states']:
        env = SlayTheSpireEnv(**observation_flags(model),
            ascension=manifest['config']['ascension'], combat_action_contract='autoregressive-v2',
            reward_flow='sequential-v1')
        try:
            obs, _ = env.restore_training_state(state['base'])
            if env._array_mapping_digest(obs) != state['visible_observation_sha256']:
                raise ValueError('anchor migration does not support transformed observations')
            obs = deepcopy(obs)
            old_observations.append(obs)
            masks.append(env.action_masks())
            new_obs = env.enable_combat_exit_anchor()
            for key in obs:
                np.testing.assert_array_equal(obs[key], new_obs[key], err_msg=key)
            np.testing.assert_array_equal(masks[-1], env.action_masks())
            state['base'] = env.export_training_state()
            state['visible_observation_sha256'] = env._array_mapping_digest(new_obs)
            observations.append(new_obs)
        finally:
            env.close()
    def stack(rows):
        return {k: np.stack([o[k] for o in rows]) for k in rows[0]}
    old_obs, new_obs = stack(old_observations), stack(observations)
    if _observation_sha256(old_obs) != environment['last_observation_sha256']:
        raise ValueError('source learner/environment observation mismatch')
    old_policy = enable_anchor_policy(model)
    old_policy.set_training_mode(False)
    model.policy.set_training_mode(False)
    # Same CPU FP32 implementation; the two inserted zero columns can change
    # GEMM reduction ordering. Check actual legal distributions and values.
    errors = {'value_max_abs': 0., 'probability_max_abs': 0.}
    with torch.no_grad():
        for start in range(0, len(observations), 16):
            a = {k: torch.as_tensor(v[start:start+16]) for k, v in old_obs.items()}
            b = {k: torch.as_tensor(v[start:start+16]) for k, v in new_obs.items()}
            mask = np.stack(masks[start:start+16])
            av, bv = old_policy.predict_values(a), model.policy.predict_values(b)
            ap = old_policy.get_distribution(a, action_masks=mask).distribution.probs
            bp = model.policy.get_distribution(b, action_masks=mask).distribution.probs
            torch.testing.assert_close(av, bv, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(ap, bp, atol=2e-5, rtol=2e-5)
            errors['value_max_abs'] = max(errors['value_max_abs'], (av-bv).abs().max().item())
            errors['probability_max_abs'] = max(errors['probability_max_abs'], (ap-bp).abs().max().item())
    environment['last_observation_sha256'] = _observation_sha256(new_obs)
    model._last_obs = new_obs
    destination.mkdir(parents=True)
    model.save(destination / 'model.zip')
    shutil.copyfile(source / 'learner.pt', destination / 'learner.pt')
    torch.save(environment, destination / 'environment.pt', pickle_protocol=5)
    manifest['config']['combat_value_target'] = 'pre-reward-anchor'
    manifest['migration'] = dict(kind='pre-reward-anchor-v1', source=str(source),
        source_sha256=manifest['sha256'], environments=len(observations),
        old_weights_and_adam_preserved=True, extra_zero_parameters=2 * old_policy._core_cfg['d_model'],
        learner_rng_byte_identical=True, **errors)
    manifest['sha256'] = {name: _sha(destination / name) for name in manifest['sha256']}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    read_bundle(destination, rank=manifest['rank'], world_size=manifest['world_size'])
    return manifest
