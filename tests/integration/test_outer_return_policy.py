"""Warm-start logits, new FP32 unbounded value and full native exit routing."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from env.core import native_actions as C
from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig, complete_searches_native, observation_as_tensor

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


@pytest.mark.parametrize('card,eligible', [('SECRET_TECHNIQUE', ['DEFEND_RED', 'SHRUG_IT_OFF']),
                                         ('SECRET_WEAPON', ['STRIKE_RED', 'BASH'])])
@pytest.mark.parametrize('nonce', [1, 42, 42101])
def test_draw_selection_root_preserves_public_handles_and_priors(outer_return_core, card, eligible, nonce):
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    excluded = 'STRIKE_RED' if card == 'SECRET_TECHNIQUE' else 'DEFEND_RED'
    env.thick.bc.apply_custom_state({'hand': [card], 'draw': [excluded]*12 + eligible})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    candidates = env.thick.bc.select_info()['candidates']
    public_index = next(i for i, candidate in enumerate(candidates)
                        if candidate['card'][0] == eligible[1])
    expected = C.AUTOREGRESSIVE_PICK + public_index
    assert env.action_masks()[expected]
    logits = np.full(env.action_space.n, -100., np.float32)
    logits[expected] = 100.
    pool = NativeSearchBatch([env], 1)
    try:
        config = NativeSearchConfig(simulations=4, depth=1, fixed_worldline=True,
                                    outer_return=True, gumbel_scale=0.)
        pool.begin({0: ((logits,), dict(**asdict(config), root_value=0.,
                                      state_nonce=nonce, compact_observation=True))})
        while not pool.done:
            pool.advance(np.zeros((pool.rows, env.action_space.n), np.float32), np.zeros(pool.rows, np.float32))
        result = pool.results()[0]
        assert result['action'] == expected
        assert result['native_action_bits'] == (2 << 29 | 13)
        assert env.thick.bc.to_dict() == before
        env.step(expected)
        assert env.thick.bc.to_dict()['hand'][-1][0] == eligible[1]
    finally:
        pool.close()



def _enter_combat(env):
    for _ in range(64):
        if env.thick.bc is not None:
            return
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    raise AssertionError('no combat within fixture budget')


def test_grid_leaf_choices_override_root_context_and_reach_lower(outer_return_core):
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_relics(['SacredBark'])
    env.thick.gc.set_potion_state(['Liquid Memories'])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.apply_custom_state({'discard': [('BASH', 1), ('DEFEND_RED', 0), ('ANGER', 0)]})
    env.thick.bc.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    env._invalidate_backend_cache()
    assert not env._obs()['selected_card_ids'].any()
    batch = NativeSearchBatch([env], 1)
    config = NativeSearchConfig(simulations=4, depth=1, outer_return=True, fixed_worldline=True, gumbel_scale=0.)
    try:
        batch.begin({0: ((np.zeros(198, np.float32),), dict(**asdict(config), root_value=0.,
                         state_nonce=42101, compact_observation=True))})
        observed = batch.observations()
        assert np.count_nonzero(observed['selected_card_ids']) == 1
        tensor = observation_as_tensor(observed, 'cpu')
        core.eval()
        with torch.inference_mode():
            chosen_context = core.combat.context_features(tensor, torch.float32)
            without_choice = dict(tensor, selected_card_ids=torch.zeros_like(tensor['selected_card_ids']))
            assert not torch.equal(chosen_context, core.combat.context_features(without_choice, torch.float32))
        assert not env._obs()['selected_card_ids'].any()
        core.teacher.assert_integrity()
    finally:
        batch.close()


def test_warm_actor_scalar_updates_and_frozen_teacher(outer_return_core):
    core, source, env = outer_return_core
    _enter_combat(env)
    obs = observation_as_tensor({k: v[None] for k, v in env._obs().items()}, 'cpu')
    core.eval()
    with torch.no_grad():
        expected = source.combat_actor_critic(obs)[0]
        actual = core.combat_actor_critic(obs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not hasattr(core.combat, 'distribution') and not hasattr(core.combat, 'transition')
    assert core.teacher.upper_tokenizer.card.vectors.data_ptr() != core.combat.card.vectors.data_ptr()
    frozen = core.teacher.assert_integrity()
    core.train()
    optimizer = torch.optim.Adam(core.combat.parameters(), lr=5e-5)
    for target in (2.7, -.4):
        optimizer.zero_grad()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, value = core.combat_actor_critic(obs)
            assert value.dtype == torch.float32
            loss = (value - target).square().mean() + .01 * logits[:, 0].square().mean()
        loss.backward()
        assert all(p.grad is None for p in core.teacher.parameters())
        optimizer.step()
        assert core.teacher.assert_integrity() == frozen
    with torch.no_grad():
        core.combat.scalar[-1].weight.zero_()
        core.combat.scalar[-1].bias.fill_(2.7)
        assert core.combat_actor_critic(obs)[1].item() == pytest.approx(2.7)


def test_native_exit_uses_teacher_on_settled_reward_page(outer_return_core):
    import slaythespire as sts
    core, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    core.eval()
    model = SimpleNamespace(policy=SimpleNamespace(core=core), device=torch.device('cpu'),
                            action_space=env.action_space)
    config = NativeSearchConfig(simulations=8, fixed_worldline=True, outer_return=True,
                               gumbel_scale=0., gumbel_value_scale=1., gumbel_rescale_q=False)
    logits = np.zeros(198, np.float32)
    logits[C.AUTOREGRESSIVE_PLAY:C.AUTOREGRESSIVE_POTION] = 20
    calls = {0: ((logits,), dict(**asdict(config), root_value=2.7,
                               state_nonce=42101, compact_observation=True))}
    batch = NativeSearchBatch([env], 2)
    seen = []
    original = batch.combat_exits
    def capture():
        exits = original()
        seen.extend(exits.values())
        return exits
    batch.combat_exits = capture
    try:
        result, stats = complete_searches_native(model, batch, calls)
        assert stats['leaf_evaluations'] > 0 and seen
        assert any(exit.kind == 'upper' and int(obs['phase'][0]) != 1 for exit, obs in seen)
        assert np.isfinite(result[0]['root_value'])
        assert env.thick.bc.to_dict() == before
        core.teacher.assert_integrity()
    finally:
        batch.close()


def test_material_manifest_guard(tmp_path):
    from rl.policies.outer_return import load_initialization_material
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest SHA256'):
        load_initialization_material(tmp_path, manifest_sha256='0' * 64, source_sha256='1' * 64)


def test_permanent_growth_context_matches_real_settlement(outer_return_core):
    import slaythespire as sts
    from env.core.outer_return import permanent_misc
    _, _, env = outer_return_core
    _enter_combat(env)
    env.thick.gc.set_deck_state([dict(id='RITUAL_DAGGER', upgrades=0, misc=15, bottle='NONE')])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Looter')
    env.thick.bc.set_player_strength(999)
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert env.thick.bc.is_battle_over
    assert env.thick.gc.deck_state()[0]['misc'] == 15
    assert permanent_misc(env.thick.gc, env.thick.bc)[0] == pytest.approx(.18)
    branch, _ = env.thick.settle_combat_leaf(env.thick.bc)
    assert branch.gc.deck_state()[0]['misc'] == 18
    assert permanent_misc(branch.gc)[0] == pytest.approx(.18)
