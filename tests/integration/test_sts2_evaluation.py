import pytest

from env.sts2.game import EnvConfig, Sts2Env
from rl.algorithms.sts2_ppo import TrainConfig, Trainer
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.sts2 import Sts2Strategy, outcome

pytestmark = pytest.mark.sts2


def test_checkpoint_strategy_uses_shared_runner(tmp_path):
    config = EnvConfig(max_steps=3)
    trainer = Trainer(TrainConfig(num_envs=1, rollout_steps=4, minibatch=4, hidden=16), config)
    trainer.iteration_step()
    checkpoint = tmp_path / 'policy.pt'
    trainer.save(checkpoint)
    trainer.close()
    strategy = Sts2Strategy(checkpoint)
    rows = list(evaluate(strategy, Scenario('sts2', lambda: Sts2Env(config), outcome),
                         EvaluationConfig(episodes=2, num_envs=2, max_steps=3, timeout=30)))
    assert len(rows) == 2
    assert all(row['result'] == 'truncated' and row['steps'] == 3 for row in rows)
    assert all(row['strategy'] == 'sts2-policy' for row in rows)
