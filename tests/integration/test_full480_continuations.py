"""Full audited selection-source matrix against original native execution."""
import numpy as np
import pytest

from tests.full480_continuation_fixtures import AUDITED_CASES, pending_continuation, large_card_pile

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


@pytest.mark.parametrize('zone', ['draw', 'discard', 'exhaust'])
@pytest.mark.parametrize('count', [128, 129, 168, 257])
def test_full480_large_pile_roundtrip(combat_factory, zone, count):
    import slaythespire as sts
    from rl.search.full480.run_device import cpu_bridge
    env = combat_factory(['STRIKE_RED'] * 10, seed=20261111,
                         combat_action_contract='autoregressive-v2')
    bc = large_card_pile(env, zone, count)
    assert len(bc.card_state()[zone]) == count
    cpu = cpu_bridge()
    snap = cpu.State(bc).snapshot()
    assert snap['max_vector'] == count and snap['queue_size'] == 0
    assert snap['portable'], '128 is inline storage, not the GPU vector limit'
    restored = cpu.State(snap['words'])
    for _ in range(3):
        np.testing.assert_array_equal(restored.snapshot()['words'], cpu.State(bc).snapshot()['words'])
        bits = restored.actions()[0]
        restored.step(bits)
        bc.execute(sts.Action(bits))
    np.testing.assert_array_equal(restored.snapshot()['words'], cpu.State(bc).snapshot()['words'])


@pytest.mark.parametrize('case', AUDITED_CASES)
def test_full480_audited_continuation_roundtrip(combat_factory, case):
    import slaythespire as sts
    from rl.search.full480.run_device import cpu_bridge
    env = combat_factory(['STRIKE_RED'] * 10, seed=20261111,
                         combat_action_contract='autoregressive-v2')
    bc = pending_continuation(env, case)
    cpu = cpu_bridge()
    initial = cpu.State(bc).snapshot()
    # Armaments+ upgrades all; unupgraded True Grit exhausts at random.
    if case not in ('card:ARMAMENTS:1', 'card:TRUE_GRIT:0'):
        assert initial['input_state'] == 2, (case, bc.select_info())
    assert initial['portable'], (case, initial['queue_size'])
    restored = cpu.State(initial['words'])
    for _ in range(32):
        before = cpu.State(bc).snapshot()
        assert before['portable'], (case, before['queue_size'])
        np.testing.assert_array_equal(restored.snapshot()['words'], before['words'])
        if before['input_state'] != 2:
            break
        bits = restored.actions()[0]
        restored.step(bits)
        bc.execute(sts.Action(bits))
    else:
        pytest.fail(f'{case}: selection did not settle in 32 choices')
    np.testing.assert_array_equal(restored.snapshot()['words'], cpu.State(bc).snapshot()['words'])
    state = bc.to_dict()
    if case == 'flight_last':
        assert state['monsters'][0]['move'] == 'BYRD_STUNNED'
    if case == 'sharp_hide':
        assert state['player']['hp'] == 37
    if case in ('attack_powers', 'skill_powers'):
        assert state['player']['combat_counters']['panache_counter'] == 5
    if case == 'death_relics':
        assert dict(state['monsters'][1]['powers'])['Poison'] == 4
        assert state['player']['energy'] == 10
    if case == 'envenom_sadistic':
        assert dict(state['monsters'][0]['powers'])['Poison'] == 2
    if case in ('attack_relics', 'skill_relics'):
        assert not {'Weak', 'Frail'}.intersection(dict(state['player']['powers']))
    if case.startswith('potion:'):
        assert state['player']['hp'] == 45, 'Toy Ornithopter heals exactly once after all picks'
    if case.startswith('stasis'):
        cards = bc.card_state()
        assert cards['stasis'][0] is None
        returned = [card for zone in ('hand', 'draw', 'discard', 'exhaust')
                    for card in cards[zone] if card['uid'] == 30]
        assert len(returned) == 1
        assert returned[0]['special_data'] == 23 and returned[0]['upgrades'] == 1
        if case in ('stasis_full', 'stasis_discard'): assert returned[0] in cards['discard']
