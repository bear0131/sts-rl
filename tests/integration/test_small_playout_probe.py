"""The lower diagnostic must change real Adam updates, never its heldout data."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import env.core
import slaythespire as sts
from rl.algorithms import small_playout_probe as probe
from rl.algorithms.small_playout import update_online
from rl.algorithms.upper_probe import state_digest
from rl.policies.small_playout import OnlinePlayoutConfig, export_online_search_pair
from rl.policies.token_regret import OnlineTokenRegretPolicy, GUIDE_WEIGHT
from rl.search.paired_token_rollout import PairedTokenBatch

pytestmark = pytest.mark.native


@pytest.fixture
def probe_model(combat_factory, request):
    device = request.param
    if device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA qualification requires an actual GPU'
    torch.manual_seed(91)
    np.random.seed(91)
    net = OnlineTokenRegretPolicy(sts.small_playout_key_count()).to(device).eval()
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3,
        seed=190, combat_action_contract='autoregressive-v2')
    pool = sts.NeuralSearchBatch(1)
    try:
        from env.search.small_playout import PAIRED_CONTRACT
        pair = export_online_search_pair(net, collect_samples=True, guide_only=True)
        pair.update(search_contract=PAIRED_CONTRACT, sha256=state_digest(net.state_dict()))
        batch = PairedTokenBatch([env], pool, pair, net)
        batch.decisions({0: ((), {'state_nonce': 9})})
        sample, = batch.samples
    finally:
        pool.clear()
    samples = [dict(copy.deepcopy(sample), env=i // 2) for i in range(8)]
    policy = torch.nn.Module()
    policy.small_playout = net
    policy.core = torch.nn.Linear(1, 1).to(device)
    policy.core.joint_upper = True
    policy.set_training_mode = policy.train
    policy.small_playout_config = asdict(OnlinePlayoutConfig(context_mode='token-root-64x2',
        batch_size=4, micro_batch_size=2, epochs=2, learning_rate=3e-4))
    policy.small_playout_optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    policy.upper_optimizer = torch.optim.AdamW(policy.core.parameters(), lr=1e-4)
    model = SimpleNamespace(policy=policy, device=torch.device(device), _small_playout_samples=samples,
        _upper_completed=[dict(env=i, episode=1, terminal=True, reward=0.) for i in range(4)],
        policy_kwargs={'small_playout_config': copy.deepcopy(policy.small_playout_config)},
        _sync_device=lambda: torch.cuda.synchronize() if device == 'cuda' else None,
        max_grad_norm=1., logger=SimpleNamespace(record=lambda *args: None))
    update_online(model)
    assert policy.small_playout_optimizer.state_dict()['state']
    model._small_playout_samples = samples
    return model


@pytest.mark.parametrize('probe_model', ['cpu', 'cuda'], indirect=True)
@pytest.mark.parametrize('with_censor', [False, True])
def test_real_optimizer_rate_epochs_holdout_and_full_restore(probe_model, monkeypatch, tmp_path, with_censor):
    model = probe_model
    if with_censor:
        model._small_playout_samples = [r for r in model._small_playout_samples if r['env'] != 1]
        model._upper_completed = [r for r in model._upper_completed if r['env'] != 1]
        model.independent_rollout_config = dict(max_combat_turns=50)
        model.episode_rollout_stats = dict(games=4, completed_games=3, censored_games=1, unfinished_games=0, first_seed=20)
        model.episode_rollout_censored_slots = [dict(slot=1, seed=21, reason='combat-turn-limit:51>50')]
    before = state_digest((model.policy.state_dict(), model.policy.small_playout_optimizer.state_dict()))
    steps = model.small_playout_optimizer_steps
    source_rows = model._small_playout_samples
    calls = []
    original = probe.update_online
    def observed(m, **kwargs):
        calls.append((sorted({r['env'] for r in m._small_playout_samples}),
            [g['lr'] for g in m.policy.small_playout_optimizer.param_groups]))
        return original(m, **kwargs)
    monkeypatch.setattr(probe, 'update_online', observed)
    protocol = dict(seed=37, diagnostic=dict(world_size=1, games_per_rank=4, holdout_games_per_rank=1,
        allow_combat_turn_censoring=with_censor),
        lower_arms=[dict(id='control', changes={}),
            dict(id='half', changes={'learning_rate': 1.5e-4}),
            dict(id='one_epoch', changes={'epochs': 1}),
            dict(id='control_repeat', changes={})])
    upper_control = tmp_path / 'control-holdout.npz'
    upper_control.write_bytes(b'upper probe owns this artifact')
    result = probe.run_lower_probe(model, protocol, output_dir=tmp_path)
    assert upper_control.read_bytes() == b'upper probe owns this artifact'
    with np.load(tmp_path / 'lower-control-holdout.npz') as held:
        assert set(held['env']) == set(result['games_per_split'][1])
    assert result['completed'] and result['control_repeat_exact']
    assert result['source_state_restored'] and result['rng_restored'] and result['upper_optimizer_unchanged']
    assert [a['stats']['updates'] for a in result['arms']] == ([2, 2, 1, 2] if with_censor else [4, 4, 2, 4])
    assert [c[1] for c in calls] == [[3e-4], [1.5e-4], [3e-4], [3e-4]]
    assert all(c[0] == result['games_per_split'][0] for c in calls)
    assert result['arms'][0]['final_digest'] != result['arms'][1]['final_digest']
    assert model._small_playout_samples is source_rows and model.small_playout_optimizer_steps == steps
    assert state_digest((model.policy.state_dict(), model.policy.small_playout_optimizer.state_dict())) == before
    prepared = model.policy.small_playout.prepare_samples(source_rows, 1.)
    ids = np.arange(len(source_rows))
    model.policy.set_training_mode(False)
    measured = probe.evaluate_lower(model.policy.small_playout, prepared, ids,
        device=model.device, micro_batch_size=2)
    with torch.no_grad():
        inputs, target, _ = prepared.batch(ids, model.device)
        loss, _, _, forward, guide = model.policy.small_playout.update_terms(inputs, target)
    assert measured['reverse_kl'].sum() + GUIDE_WEIGHT * measured['guide_current_root_kl'].sum() == pytest.approx(float(loss), abs=2e-5)
    assert measured['forward_kl'].sum() == pytest.approx(float(forward), abs=2e-5)
    assert measured['guide_current_root_kl'].sum() == pytest.approx(float(guide), abs=2e-5)


@pytest.mark.parametrize('probe_model', ['cpu'], indirect=True)
def test_exception_restores_and_game_aliasing_is_rejected(probe_model, monkeypatch):
    model = probe_model
    before = state_digest((model.policy.state_dict(), model.policy.small_playout_optimizer.state_dict()))
    rows = model._small_playout_samples
    def fail(m, **kwargs):
        update_online(m, **kwargs)
        raise RuntimeError('injected failure after actual Adam steps')
    monkeypatch.setattr(probe, 'update_online', fail)
    protocol = dict(seed=37, diagnostic=dict(world_size=1, games_per_rank=4, holdout_games_per_rank=1),
        lower_arms=[dict(id='control', changes={})])
    with pytest.raises(RuntimeError, match='injected failure'):
        probe.run_lower_probe(model, protocol)
    assert model._small_playout_samples is rows
    assert state_digest((model.policy.state_dict(), model.policy.small_playout_optimizer.state_dict())) == before
    with pytest.raises(ValueError, match='one complete episode'):
        probe.split_lower_games(rows, model._upper_completed + [dict(env=0, episode=2)], 1, 37)
    with pytest.raises(ValueError, match='unknown game'):
        probe.split_lower_games(rows + [dict(env=99)], model._upper_completed, 1, 37)
