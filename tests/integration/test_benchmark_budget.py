"""The benchmark supervisor must actually enforce its advertised wall budget."""
import json
import subprocess
import sys
import pytest


def test_benchmark_timeout_exits_with_machine_readable_status(tmp_path):
    output = tmp_path / "timeout.json"
    result = subprocess.run(
        [sys.executable, "-m", "rl.benchmarks.train", "--output", str(output),
         "--timeout", "0.001"], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 124, result.stderr
    assert json.loads(output.read_text())["status"] == "timeout"


@pytest.mark.native
def test_leaf_benchmark_inherits_checkpoint_action_contract(tmp_path, combat_factory):
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.policies.entity_combat import EntityCombatPolicy
    from rl.benchmarks.train import parser, run_worker
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cpu', seed=0,
        n_steps=8, batch_size=4, policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)),
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules')
    checkpoint = tmp_path/'entity.zip'
    model.save(checkpoint)
    result = run_worker(parser().parse_args(['--workload','leaves','--checkpoint',str(checkpoint),
        '--output',str(tmp_path/'leaves.json'),'--n-envs','2','--leaf-repeats','2','--seed','0']))
    assert result['status'] == 'ok'
    assert result['timing']['leaves'] == 4
