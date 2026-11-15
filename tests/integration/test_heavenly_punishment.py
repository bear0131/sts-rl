"""User-defined combat rule, independent of policy rewards and truncation."""
import numpy as np
import pytest

from tests.heavenly_punishment_fixtures import punishment_battle

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def play(bc):
    import slaythespire as sts
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))


@pytest.mark.parametrize('scenario', ['cards', 'mitigation', 'draw_hooks'])
def test_full480_punishment_card_threshold_and_true_damage(combat_factory, scenario):
    bc = punishment_battle(combat_factory(['DEFEND_RED'] * 10), scenario)
    for hp in (80, 60, 40):
        play(bc)
        assert bc.to_dict()['player']['hp'] == hp
    assert bc.to_dict()['player']['combat_counters']['cards_played'] == 102
    if scenario == 'mitigation':
        powers = dict(bc.to_dict()['player']['powers'])
        assert powers['Buffer'] == 3 and powers['Intangible'] == 3


def test_full480_punishment_turn_threshold_repeats_and_counter_resets(combat_factory):
    import slaythespire as sts
    bc = punishment_battle(combat_factory(['DEFEND_RED'] * 10), 'turns')
    assert bc.turn + 1 == 20  # Native turns are zero based; displayed turns are one based.
    bc.apply_custom_state({'turn': 18})
    for turn, hp in ((20, 80), (21, 60), (22, 40), (23, 20), (24, 0)):
        bc.execute(sts.Action(sts.ActionType.END_TURN))
        assert bc.turn + 1 == turn
        assert bc.to_dict()['player']['hp'] == hp
        if hp: assert bc.to_dict()['player']['combat_counters']['cards_played'] == 0
    assert bc.is_battle_over and bc.outcome == 2


@pytest.mark.parametrize('scenario,hp', [('fatal', 0), ('revive', 40), ('fairy', 24)])
def test_full480_punishment_normal_death_and_revive(combat_factory, scenario, hp):
    bc = punishment_battle(combat_factory(['DEFEND_RED'] * 10), scenario)
    play(bc)  # 100 is safe.
    play(bc)
    assert bc.to_dict()['player']['hp'] == hp
    assert bc.is_battle_over == (hp == 0)


def test_full480_punishment_autoplay_counts_and_disabled_rule_unchanged(combat_factory):
    bc = punishment_battle(combat_factory(['DEFEND_RED'] * 10), 'autoplay')
    control = bc.clone()
    control.heavenly_punishment = False
    play(bc)
    play(control)
    assert bc.to_dict()['player']['combat_counters']['cards_played'] == 101
    assert bc.to_dict()['player']['hp'] == 60
    assert control.to_dict()['player']['hp'] == 80


@pytest.mark.parametrize('scenario', ['cards', 'turns', 'mitigation', 'draw_hooks', 'autoplay', 'fatal', 'revive', 'fairy'])
def test_full480_punishment_native_roundtrip(combat_factory, scenario):
    import slaythespire as sts
    from rl.search.full480.run_device import cpu_bridge
    bc = punishment_battle(combat_factory(['DEFEND_RED'] * 10), scenario)
    cpu = cpu_bridge()
    restored = cpu.State(cpu.State(bc).snapshot()['words'])
    for _ in range(6):
        before = cpu.State(bc).snapshot()
        assert before['portable']
        np.testing.assert_array_equal(restored.snapshot()['words'], before['words'])
        if bc.is_battle_over: break
        bits = next((b for b in restored.actions() if sts.Action(b).action_type == sts.ActionType.CARD),
                    restored.actions()[0])
        restored.step(bits)
        bc.execute(sts.Action(bits))
    np.testing.assert_array_equal(restored.snapshot()['words'], cpu.State(bc).snapshot()['words'])


def test_full480_punishment_journal_activation_and_resume(latent_env, latent_training_factory):
    for _ in range(128):
        if latent_env.thick.bc is not None: break
        latent_env.step(int(np.flatnonzero(latent_env.action_masks())[0]))
    assert latent_env.thick.bc is not None
    old = latent_env.export_training_state()
    # Legacy saves carry neither field.
    old.pop('combat_rules'); old.pop('combat_rules_start_action')
    clone = latent_training_factory().unwrapped
    try:
        clone.restore_training_state(old)
        before = clone.thick.bc.to_dict()['player']['hp']
        clone.enable_heavenly_punishment()
        assert clone.thick.bc.to_dict()['player']['hp'] == before
        assert clone._combat_rules_start_action == len(old['actions'])
        saved = clone.export_training_state()
        clone.restore_training_state(saved)
        assert clone.thick.bc.heavenly_punishment
        for _ in range(24):
            _, _, done, truncated, _ = clone.step(int(np.flatnonzero(clone.action_masks())[0]))
            if done or truncated: break
        saved = clone.export_training_state()
        clone.restore_training_state(saved)
        assert clone.export_training_state()['observation_sha256'] == saved['observation_sha256']
    finally:
        clone.close()
