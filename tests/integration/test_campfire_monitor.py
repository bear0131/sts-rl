"""Native choices, their follow-up screens, and replay across a checkpoint."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.training import campfire_monitor as M

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('choice', ['rest', 'smith', 'recall', 'lift', 'toke', 'dig'])
def test_campfire_native_followup_and_resume_are_counted_once(campfire_env_type, choice):
    env = campfire_env_type(global_map_view=True, upper_public_context=True, event_option_observation=True,
                            combat_action_contract='autoregressive-v2')
    restored = campfire_env_type(global_map_view=True, upper_public_context=True, event_option_observation=True,
                                 combat_action_contract='autoregressive-v2')
    model = SimpleNamespace()

    def step(action):
        obs = {k: v[None] for k, v in env._obs().items()}
        mask = env.action_masks()[None]
        probs = torch.as_tensor(mask, dtype=torch.float32)
        probs /= probs.sum(-1, keepdim=True)
        policy = M.policy_probabilities(SimpleNamespace(
            distribution=SimpleNamespace(probs=probs)), obs)
        result = env.step(action)
        M.record_step(model, [action], mask, policy, [result[-1]])
        return result

    try:
        env.reset(seed=42)
        assert env.action_masks()[M.OPTIONS[choice]]
        step(M.OPTIONS[choice])
        assert M.counts(model)['campfire_decisions'] == 1
        snapshot = env.export_training_state()
        restored.reset(seed=42)
        restored.restore_training_state(snapshot)
        for key, value in env._obs().items():
            np.testing.assert_array_equal(value, restored._obs()[key])
        if choice in ('smith', 'toke'):
            assert env._phase() == 'card_select'
            step(int(np.flatnonzero(env.action_masks())[0]))
        elif choice == 'dig':
            assert env._phase() == 'combat_reward'
            step(M.OPTIONS['skip'])
        assert M.counts(model)['campfire_decisions'] == 1
        assert M.counts(model)[f'campfire_{choice}_selected'] == 1
        M.begin_collection(model)
        assert not any(M.counts(model).values())
    finally:
        env.close()
        restored.close()


def test_native_skip_only_when_no_main_option_remains(upper_search_env_factory):
    env = upper_search_env_factory('rest', relics=('Coffee Dripper', 'Fusion Hammer'))
    gc = env.thick.gc
    x, y = gc.cur_map_node_x, gc.cur_map_node_y
    env.step(M.OPTIONS['recall'])
    # Enter a controlled second rest with the ruby key already acquired.
    gc.cur_map_node_y = y - 1
    gc.cur_map_node_x = next(px for px in range(7) if x in env.thick.rmap.reachable(px, y - 1))
    gc.transition_to_map_node(x)
    env._invalidate_backend_cache()
    mask = env.action_masks()[None]
    assert [name for name, action in M.OPTIONS.items() if mask[0, action]] == ['skip']
    probs = torch.as_tensor(mask, dtype=torch.float32)
    probs /= probs.sum(-1, keepdim=True)
    policy = M.policy_probabilities(SimpleNamespace(distribution=SimpleNamespace(probs=probs)),
                                    {k: v[None] for k, v in env._obs().items()})
    result = env.step(M.OPTIONS['skip'])
    model = SimpleNamespace()
    M.record_step(model, [M.OPTIONS['skip']], mask, policy, [result[-1]])
    out = M.metrics(M.counts(model))
    assert out['campfire_skip_choice_rate'] == out['campfire_skip_policy_probability'] == 1
    assert 'campfire_rest_choice_rate' not in out
