"""Exact COMBAT06 initialization and actual-boundary observation parity."""
import numpy as np
import pytest
import torch

from env.core.finite_curves import MATCHED_OBJECTIVE
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.combat_distribution import MEAN_CONTRACT
from rl.policies.outer_return import OuterCombatNetwork

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_all_weights_heads_and_nonboss_gradients_match(outer_return_core, frozen_upper_source, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA gate must also run on the production Linux GPU')
    _, source, env = outer_return_core
    networks = []
    for objective in (MEAN_CONTRACT, MATCHED_OBJECTIVE):
        with torch.random.fork_rng():
            torch.manual_seed(9001000)
            networks.append(OuterCombatNetwork(source.combat, frozen_upper_source[1],
                env.observation_space, initialization='random', objective=objective).to(device))
    old, new = networks
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(5, old.config.width, device=device)
    obs = dict(combat_is_boss=torch.zeros(5, 1, device=device))
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    with torch.autocast(device_type=device, dtype=dtype, enabled=device == 'cuda'):
        a, ao = old._value(x, obs)
        b, bo = new._value(x, obs)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ao:
        torch.testing.assert_close(ao[key], bo[key], rtol=0, atol=0)
    (ao['win_logits'].square().mean()+ao['hp_mean'].square().mean()).backward()
    (bo['win_logits'].square().mean()+bo['hp_mean'].square().mean()).backward()
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_scripted_actual_end_turn_and_terminal_observations_identical():
    import slaythespire as sts
    envs = [SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
        outer_return_context=True, combat_value_objective=objective)
        for objective in (MEAN_CONTRACT, MATCHED_OBJECTIVE)]
    try:
        for env in envs:
            env.reset(seed=42101)
            for _ in range(64):
                if env.thick.bc is not None:
                    break
                env.step(int(np.flatnonzero(env.action_masks())[0]))
            env.thick.bc = sts.create_battle(env.thick.gc, 'Cultist')
            env.thick.bc.apply_custom_state(dict(player=dict(hp=50,block=999),
                hand=['DEFEND_RED'], monsters=[dict(hp=30)]))
            env._invalidate_backend_cache()
        for terminal in (False, True):
            if terminal:
                for env in envs:
                    env.thick.bc.apply_custom_state(dict(hand=['CLEAVE'],player=dict(energy=3,strength=999)))
                    env.thick.bc.set_player_strength(999)
                    env._invalidate_backend_cache()
            bits = 0 if terminal else 4 << 29
            infos = []
            for env in envs:
                thin = env._native_action_to_thin(bits)
                assert thin is not None
                env._nosl_cpp_pending = (thin, bits)
                infos.append(env.step(thin)[-1])
            key = 'nosl_value_terminal_observation' if terminal else 'nosl_value_boundary_observation'
            assert key in infos[0] and key in infos[1]
            assert infos[0][key].keys() == infos[1][key].keys()
            for field in infos[0][key]:
                np.testing.assert_array_equal(infos[0][key][field], infos[1][key][field], err_msg=field)
            if terminal:
                assert infos[0]['nosl_value_terminal_outcome'] == infos[1]['nosl_value_terminal_outcome'] == 'win'
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize('outlet', ['win', 'escape', 'loss'])
@pytest.mark.parametrize('boss', [False, True])
def test_typed_terminal_transport_preserves_search_when_payoff_is_held_fixed(run_combat_factory, outlet, boss):
    """Change terminal transport alone, with identical old HP/Boss payoffs."""
    import slaythespire as sts
    from env.core import native_actions as C
    from env.core.combat_value import combat_terminal_target, combat_terminal_outcome
    root = run_combat_factory(potions=['Smoke Bomb'] if outlet == 'escape' else [])
    root.gc.cur_room = sts.Room.BOSS if boss else sts.Room.MONSTER
    root.bc = sts.create_battle(root.gc, 'Jaw Worm')
    battle = root.bc
    if outlet == 'win':
        battle.set_player_strength(999)
        action = C.AUTOREGRESSIVE_PLAY
    elif outlet == 'escape':
        action = C.AUTOREGRESSIVE_POTION + 5
    else:
        battle.apply_custom_state(dict(player=dict(hp=1)))
        action = 0
    logits = np.full(198, -1000., dtype=np.float64)
    logits[action] = 1000.
    original = battle.to_dict()
    results = []
    for typed in (False, True):
        session = battle.nosl_neural_search_begin(logits, simulations=128, depth=0,
            nonce=42101, root_value=.4, fixed_worldline=True, outer_return=typed,
            boss_win_weight=.9 if boss and not typed else 0.,
            root_search_mode=2, paper_full_gumbel=True,
            gumbel_candidates=16, gumbel_scale=0.)
        for _ in range(130):
            batch = session.next_batch_packed(1)
            if batch['done']:
                break
            exits = batch.get('combat_exits', {})
            for row, request in enumerate(batch['request_ids']):
                value, policy = .4, np.zeros(198)
                if row in exits:
                    leaf = exits[row]
                    value = combat_terminal_target(leaf)
                    if boss and combat_terminal_outcome(leaf) == 'win':
                        value = .9 + .1 * value
                    policy = []
                session.backprop(request, value, policy)
        else:
            pytest.fail('bounded search did not finish')
        results.append(session.result())
    for key in ('action_bits', 'root_visits', 'total_root_visits'):
        assert results[0][key] == results[1][key], key
    assert results[0]['root_value'] == pytest.approx(results[1]['root_value'], abs=1e-7)
    assert battle.to_dict() == original
