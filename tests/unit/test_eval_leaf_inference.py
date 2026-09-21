"""Explicit graph mode preserves defaults and upper/combat ownership."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from rl.evaluation import eval as cli
from rl.evaluation import strategies as S
from rl.search.native import forward_leaves


def args(mode='eager'):
    return cli.build_parser().parse_args(['upper.zip', '--solver', 'native',
        '--search-mode', 'strict', '--simulations', '128', '--depth', '0',
        '--leaf-batch-size', '1', '--leaf-inference', mode])


def model(device='cpu', dtype=torch.float32):
    policy = SimpleNamespace(training=False, parameters=lambda: iter([torch.zeros(1, dtype=dtype)]),
                             core=SimpleNamespace(combat_actor_critic=object(), combat_observation_keys=()))
    return SimpleNamespace(device=torch.device(device), policy=policy)


def test_default_backend_leaves_runtime_unmodified():
    configured = cli.build_parser().parse_args(['upper.zip', '--solver', 'native'])
    assert configured.leaf_inference == 'eager'
    lower = model()
    S.native_strategy(lower, configured)
    assert not hasattr(lower, '_native_leaf_inference')


def test_fixed_worldline_cli_override_reaches_native_config():
    enabled = cli.build_parser().parse_args([
        'upper.zip', '--solver', 'native', '--fixed-worldline'])
    disabled = cli.build_parser().parse_args([
        'upper.zip', '--solver', 'native', '--no-fixed-worldline'])
    inherited = cli.build_parser().parse_args(['upper.zip', '--solver', 'native'])

    assert S.native_strategy(model(), enabled).routes['combat'].config.fixed_worldline is True
    assert S.native_strategy(model(), disabled).routes['combat'].config.fixed_worldline is False
    assert inherited.fixed_worldline is None


def test_cuda_graph_refuses_cpu():
    with pytest.raises(ValueError, match='CUDA combat model'):
        S.native_strategy(model(), args('cuda-graph'))


def test_cuda_graph_refuses_non_fp32():
    with pytest.raises(ValueError, match='eval-mode FP32'):
        S.native_strategy(model('cuda', torch.float16), args('cuda-graph'))


@pytest.mark.parametrize('solver', ['policy', 'rules', 'value'])
def test_cli_rejects_unsupported_solver_before_loading(monkeypatch, solver):
    monkeypatch.setattr(sys, 'argv', ['eval', 'missing.zip', '--solver', solver,
                                    '--leaf-inference', 'cuda-graph'])
    with pytest.raises(ValueError, match='requires --solver native'):
        cli.main()


def test_graph_cache_belongs_only_to_separate_combat_model(monkeypatch):
    class Cache:
        def __init__(self, core, device, **kwargs):
            self.core, self.device = core, device
            self.__dict__.update(kwargs)
        def statistics(self):
            return {'captures': 7}
    monkeypatch.setattr(S, 'CombatLeafGraphs', Cache)
    upper, lower = model('cuda'), model('cuda')
    strategy = S.compose_strategy(upper, args('cuda-graph'), lower)
    assert not hasattr(upper, '_native_leaf_inference')
    assert not hasattr(lower, '_native_leaf_inference')
    native = strategy.routes['combat'].routes['combat']
    assert native.leaf_backend.core is lower.policy.core
    manifest = S.strategy_manifest(native)
    assert manifest['config']['simulations'] == 128
    assert manifest['config']['leaf_batch_size'] == 1
    assert manifest['leaf_inference'] == dict(backend='cuda-graph', amp='off',
        capacity=32, batch_multiple=0, statistics={'captures': 7})
    eager = S.native_strategy(lower, args()).routes['combat']
    second = S.native_strategy(lower, args('cuda-graph')).routes['combat']
    assert eager.leaf_backend is None
    assert native.leaf_backend is not second.leaf_backend
    assert S.strategy_manifest(eager)['leaf_inference'] == {'backend': 'eager'}
    assert native.leaf_backend.core is second.leaf_backend.core


def test_explicit_backend_does_not_change_legacy_model_backend():
    calls = []
    def backend(observations, width):
        calls.append(width)
        return np.full((2, 1), 7.0), np.full(2, 8.0)
    core = SimpleNamespace(combat_actor_critic=lambda observations, active_relic_width:
                           (torch.full((2, 1), 3.0), torch.full((2,), 4.0)))
    model = SimpleNamespace(device='cpu', policy=SimpleNamespace(core=core),
                            _native_leaf_inference=backend)
    batch = {'relics': np.zeros((2, 3), dtype=np.int64)}
    inherited = forward_leaves(model, batch)
    eager = forward_leaves(model, batch, leaf_backend=None)
    inherited_again = forward_leaves(model, batch)
    np.testing.assert_array_equal(inherited[0], 7.0)
    np.testing.assert_array_equal(eager[0], 3.0)
    np.testing.assert_array_equal(inherited_again[0], 7.0)
    assert model._native_leaf_inference is backend
    assert calls == [1, 1]


@pytest.mark.parametrize('interrupted', [False, True])
def test_cli_reports_final_candidate_cache_statistics(tmp_path, monkeypatch, interrupted):
    class Cache:
        def __init__(self, core, device, **kwargs):
            self.__dict__.update(kwargs)
            self.captures = 0
        def statistics(self):
            return {'captures': self.captures}

    monkeypatch.setattr(S, 'CombatLeafGraphs', Cache)
    frozen = model('cuda')
    frozen.set_random_seed = lambda seed: None
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: frozen)
    monkeypatch.setattr(cli, 'make_scenario', lambda *a: None)
    output = tmp_path / 'eval.json'
    monkeypatch.setattr(sys, 'argv', ['eval', 'unused.zip', '--solver', 'native',
        '--leaf-inference', 'cuda-graph', '--compare-combats', '1', '--episodes', '1',
        '--output', str(output)])

    def finish(audit, *a, **kw):
        reference = audit.reference.routes['combat'].leaf_backend
        candidate = audit.candidates['native'].routes['combat'].leaf_backend
        assert reference is not candidate
        reference.captures = 2
        candidate.captures = 7
        yield dict(episode=0, seed=10000, result='win', floor=55, act=4, steps=3, raw_return=1.)
        if interrupted:
            raise TimeoutError('fixture budget')

    monkeypatch.setattr(cli, 'evaluate', finish)
    if interrupted:
        with pytest.raises(TimeoutError, match='fixture budget'):
            cli.main()
    else:
        cli.main()
    payload = json.loads(output.read_text())
    assert payload['state'] == ('error' if interrupted else 'complete')
    candidate = payload['combat_candidates']['native']['phases']['combat']['leaf_inference']
    reference = payload['effective_strategy']['phases']['combat']['leaf_inference']
    assert candidate['statistics'] == {'captures': 7}
    assert reference['statistics'] == {'captures': 2}
