"""Executed segment labels, real learner updates and exact pending recovery."""
from types import SimpleNamespace
import copy
import json
import sys

import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecMonitor

from env.core import encode, vocab as V
from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms import outer_return as outer
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.outer_return import OuterReturnPolicy
from rl.training import distributed_sb3
from rl.training.environments import TrainingStateWrapper
from rl.training.native_vec_env import NativeSearchVecEnv
from rl.training.checkpoints import restore_learner

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(240)]


def test_executed_prefix_continuation_terminal_and_censor():
    class Teacher:
        def assert_integrity(self):
            pass
        def __call__(self, obs):
            return None, torch.full((len(obs['phase']), 1), 2.7)
    from contextlib import nullcontext
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(teacher=Teacher(),
        combat_actor_critic=lambda obs: (None, torch.full((len(obs['phase']), 1), -.4)))),
        device='cpu', autocast_context=nullcontext, _nosl_is_az_samples=[],
        _nosl_is_az_pending_fights=[[] for _ in range(4)])
    pending = model._nosl_is_az_pending_fights
    outer.begin_collection(model, 4)
    before = dict(phase=np.ones((4, 1), np.int32), act=np.full((4, 1), 3/encode.ACT_SCALE),
                  floor=np.full((4, 1), 50/encode.FLOOR_SCALE))
    for row in pending:
        row.append({})
    outer.stamp_roots(model, pending)
    # A non-boundary observed progress increment must survive a collection.
    progress = dict(run_stats=dict(act=3, floor=51, outcome='UNDECIDED'))
    assert outer.finish_step(model, before, before, [False] * 4, [progress] * 4, pending) == (0, 0, 0)
    outer.begin_collection(model, 4)
    for row in pending:
        row.append({})
    outer.stamp_roots(model, pending)
    before['floor'][:] = 51/encode.FLOOR_SCALE
    after = copy.deepcopy(before)
    after['phase'][0] = V.PHASE_VOCAB_INDEX['map']
    infos = [progress, dict(progress, outer_combat_exit=True),
             dict(run_stats=dict(act=3, floor=51, outcome='PLAYER_VICTORY')),
             dict(progress, **{'TimeLimit.truncated': True})]
    assert outer.finish_step(model, before, after, [False, False, True, True], infos, pending) == (6, 2, 3)
    assert [s['value_target'] for s in model._nosl_is_az_samples] == pytest.approx(
        [2.702, 2.7, -.398, -.4, 1.002, 1.])
    assert not any(pending) and not model._outer_pending_rewards.any()
    # Auto-reset observations cannot bootstrap a natural death.
    pending[0].append({})
    outer.stamp_roots(model, pending)
    death = dict(run_stats=dict(act=3, floor=51, outcome='PLAYER_LOSS'))
    outer.finish_step(model, before, after, [True]*4, [death]*4, pending)
    assert model._nosl_is_az_samples[-1]['value_target'] == 0


class QuickBackend(SimGameEnv):
    """Replayable controlled battles, through actual run callbacks."""
    def __init__(self):
        super().__init__(global_map_view=True)

    def _settle(self):
        super()._settle()
        if self.bc is not None:
            self.bc.set_player_strength(999)


def test_real_outer_two_rounds_payload_and_resume(outer_material, tmp_path, monkeypatch):
    def make_env(*_):
        def factory():
            return TrainingStateWrapper(TimeLimit(SlayTheSpireEnv(backend_factory=QuickBackend,
                global_map_view=True, combat_action_contract='autoregressive-v2',
                outer_return_context=True), max_episode_steps=500))
        env = VecMonitor(NativeSearchVecEnv([factory for _ in range(8)], threads=2))
        env.seed(42101)
        return env
    env = make_env()
    model = AccumulatingMaskablePPO(OuterReturnPolicy, env, policy_kwargs=outer_material,
        n_steps=128, batch_size=32, n_epochs=1, gamma=1., device='cpu', seed=42101,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_fixed_worldline=True, nosl_is_az_upper_strategy='model',
        nosl_is_az_simulations=4, nosl_is_az_updates=2,
        nosl_is_az_leaf_batch_size=1,
        nosl_is_az_batch_size=64, nosl_is_az_micro_batch_size=32,
        nosl_is_az_replay_capacity=2048)
    initial = tmp_path / 'initial.zip'
    assert model.policy.core.lower_initialization == 'random'
    assert all(p.requires_grad for p in model.policy.core.combat.parameters())
    lower_before = {key: value.clone() for key, value in model.policy.core.combat.state_dict().items()}
    model.save(initial)
    teacher = model.policy.core.teacher.assert_integrity()
    env.close()
    payloads = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=lambda **kw: SimpleNamespace(finish=lambda **kw: None),
        config=SimpleNamespace(update=lambda *a, **kw: None), define_metric=lambda *a, **kw: None,
        log=lambda payload: payloads.append(json.loads(json.dumps(payload, allow_nan=False)))))
    monkeypatch.setattr(distributed_sb3, 'make_env', make_env)
    output = tmp_path / 'trained.zip'
    argv = ['distributed_sb3', '--rank', '0', '--world-size', '1', '--checkpoint', str(initial),
        '--output', str(output), '--updates', '2', '--n-envs', '8', '--n-steps', '128',
        '--batch-size', '32', '--n-epochs', '1', '--device', 'cpu', '--amp', 'off', '--gamma', '1',
        '--vec', 'native', '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning',
        '--remote-root', str(tmp_path/'transport'), '--ssh-control-dir', str(tmp_path/'ssh'),
        '--run-id', 'ci-outer', '--wandb', '--wandb-id', 'ci-outer',
        '--training-state-dir', str(tmp_path/'state'), '--save-every-updates', '1', '--max-checkpoints', '2',
        '--rollout-resource-interval', '0.1']
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    assert len(payloads) == 2
    from rl.training.checkpoints import read_bundle
    generations = sorted((tmp_path/'state'/'rank0').glob('step-*'))
    assert [read_bundle(path, rank=0, world_size=1)['global_step'] for path in generations] == [1024, 2048]
    assert all(p['outer_return/labelled_roots'] > 0 for p in payloads)
    assert all(p['outer_return/upper_updates'] == 0 for p in payloads)
    assert all(p['nosl_is_az/updates'] == 2 for p in payloads)
    assert all(p['nosl_is_az/lower_cache_requests'] > 0 for p in payloads)
    assert all(p['nosl_is_az/upper_cache_requests'] > 0 for p in payloads)
    assert all(p['nosl_is_az/outer_settlement_seconds'] > 0 for p in payloads)
    restored_env = make_env()
    restored = AccumulatingMaskablePPO.load(output, env=restored_env, device='cpu')
    assert restored.policy.core.teacher.assert_integrity() == teacher
    for key in ('transformer.layers.0.linear1.weight', 'scalar.2.weight', 'end.weight'):
        assert not torch.equal(lower_before[key], restored.policy.core.combat.state_dict()[key]), key
    bundle = tmp_path/'state'/'rank0'/'step-000000002048'
    restore_learner(restored, bundle, rank=0, world_size=1)
    assert restored._recovery_environment_restored
    assert not hasattr(restored, '_rollout_inference')
    replay = restored._ensure_nosl_is_az_replay()
    assert replay.size > 0 and np.isfinite(replay.value_targets[:replay.size]).all()
    state = replay.state_dict()
    state['objective_identity'] = dict(state['objective_identity'], gamma=.99)
    with pytest.raises(ValueError, match='objective identity'):
        replay.load_state_dict(state)
    restored_env.close()
    argv[argv.index('--checkpoint') + 1] = str(bundle/'model.zip')
    argv[argv.index('--updates') + 1] = '1'
    argv += ['--resume-state', str(bundle)]
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    assert len(payloads) == 3 and payloads[-1]['global_step'] == 3072
    generations = sorted((tmp_path/'state'/'rank0').glob('step-*'))
    assert [read_bundle(path, rank=0, world_size=1)['global_step'] for path in generations] == [2048, 3072]
    final = AccumulatingMaskablePPO.load(output, device='cpu')
    assert final.policy.core.teacher.assert_integrity() == teacher
    assert final.policy.optimizer.state


@pytest.mark.parametrize('abort', [False, True])
def test_rollout_cache_clears_on_callback_abort_and_exception(outer_material, monkeypatch, abort):
    from rl.search.rollout_inference import RolloutInference
    env = VecMonitor(NativeSearchVecEnv([lambda: SlayTheSpireEnv(
        global_map_view=True, combat_action_contract='autoregressive-v2',
        outer_return_context=True)], threads=1))
    model = AccumulatingMaskablePPO(OuterReturnPolicy, env, policy_kwargs=outer_material,
        n_steps=2, batch_size=2, device='cpu', seed=42101, gamma=1.,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_fixed_worldline=True, nosl_is_az_upper_strategy='model',
        nosl_is_az_leaf_batch_size=1)
    _, callback = model._setup_learn(2)
    runtimes = []
    def collect(*args):
        runtime = model._rollout_inference
        runtimes.append(runtime)
        runtime.lower.forward(dict(x=np.ones((1, 1), np.float32)), lambda x: (x['x'], x['x']))
        if abort:
            return False
        raise RuntimeError('collector failed')
    monkeypatch.setattr(model, '_collect_nosl_is_az_rollouts', collect)
    try:
        if abort:
            assert not model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        else:
            with pytest.raises(RuntimeError, match='collector failed'):
                model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        assert not hasattr(model, '_rollout_inference')
        assert not runtimes[0].lower.entries
    finally:
        env.close()
