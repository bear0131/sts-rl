import pytest
import torch
import json
import numpy as np

from rl.algorithms.ppo import PPOLossComponents
from tools.analysis.probe_upper_learning import json_scalar, loss_arm, should_stop, subset_learner


def test_subset_keeps_pending_credit_but_remaps_live_actor_and_does_not_mutate_source():
    source = dict(last_episode_starts=[False, True, False], upper_episode_ids=[7, 8, 9],
        upper_pending=[dict(env=0, episode=7, reward=.3, value=.2), None, dict(env=2, episode=9, reward=.4, value=.1)],
        nosl_is_az_pending_fights=[[], [], []])
    selected = subset_learner(source, [2, 0])
    assert selected['upper_episode_ids'] == [9, 7]
    assert selected['upper_pending'][0] == dict(env=0, episode=9, reward=.4, value=.1)
    assert selected['upper_pending'][1] == dict(env=1, episode=7, reward=.3, value=.2)
    assert source['upper_pending'][2]['env'] == 2
    with pytest.raises(ValueError):
        subset_learner(source, [2, 2])


def test_loss_arms_remove_only_requested_current_gradients_and_keep_zero_grad_graph():
    p, v, e = (torch.tensor(2., requires_grad=True) for _ in range(3))
    components = PPOLossComponents(p**2, v**2, e**2, p.detach()*0, p.detach()*0, 1, 0, 1, None, v**2)
    value_only = loss_arm(components, 'value')
    (value_only.policy_loss + .5*value_only.value_loss + .02*value_only.entropy_loss).backward()
    assert p.grad.item() == 0 and v.grad.item() == 2 and e.grad.item() == 0
    assert components.policy_loss.item() == 4


def test_stop_uses_fixed_panel_absolute_drop_and_handles_missing_groups():
    baseline = dict(open_probability=.71, take_probability=.54)
    assert should_stop(baseline, dict(open_probability=.60, take_probability=.55), .1) == 'open_probability_drop'
    assert should_stop(baseline, dict(open_probability=.70, take_probability=.55), .1) is None
    assert should_stop(baseline, dict(open_probability=None, take_probability=.43), .1) == 'take_probability_drop'


def test_real_episode_scalar_types_survive_diagnostic_publication():
    result = json.loads(json.dumps(dict(episode=dict(r=np.float32(.5), l=np.int32(42)), won=np.bool_(False)), default=json_scalar, allow_nan=False))
    assert result == dict(episode=dict(r=.5, l=42), won=False)
