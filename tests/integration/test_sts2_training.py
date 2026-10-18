import copy

import numpy as np
import pytest
import torch

from env.sts2.game import EnvConfig
from rl.algorithms.sts2_ppo import TrainConfig, Trainer

pytestmark = [pytest.mark.sts2, pytest.mark.training]


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('task', ['combat', 'run'])
def test_real_ppo_two_iterations_and_exact_cpu_resume(tmp_path, task):
    config = TrainConfig(num_envs=2, rollout_steps=16, minibatch=16, hidden=32)
    env_config = EnvConfig(task=task, max_steps=25)
    trainer = Trainer(config, env_config)
    first = trainer.iteration_step()
    checkpoint = tmp_path / 'iteration-1.pt'
    trainer.save(checkpoint)
    second = trainer.iteration_step()
    expected = copy.deepcopy(trainer.policy.state_dict())
    optimizer = copy.deepcopy(trainer.policy.optimizer.state_dict())
    fingerprints = [env.fingerprint() for env in trainer.envs]
    for row in (first, second):
        assert np.isfinite(row['loss'])
        assert row['parameter_delta_l2'] > 0
        assert row['samples'] == 32
    assert second['transitions'] == 64
    assert second['optimizer_updates'] == 8
    trainer.close()
    restored = Trainer(config, env_config)
    restored.restore(checkpoint)
    replayed = restored.iteration_step()
    equal_nested(expected, restored.policy.state_dict())
    equal_nested(optimizer, restored.policy.optimizer.state_dict())
    assert fingerprints == [env.fingerprint() for env in restored.envs]
    for key in ('loss', 'approx_kl', 'parameter_delta_l2', 'outcomes', 'phases'):
        assert second[key] == replayed[key]
    restored.close()
