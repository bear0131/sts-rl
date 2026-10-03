"""AUDIT01 simulator rule regressions, including the TRAIN32 fixes."""
import pytest
import slaythespire as sts
import numpy as np

from env.core import native_actions as C

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _battle(combat_factory, hand):
    env = combat_factory(['STRIKE_RED'] * 10)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': hand, 'draw': ['DEFEND_RED'] * 10,
        'discard': [], 'exhaust': [], 'player': {'energy': 20},
        'monsters': [{'max_hp': 500}]})
    battle.apply_custom_state({'monsters': [{'hp': 500}]})
    return battle


@pytest.mark.parametrize('upgraded,amount', [(0, 9), (1, 15)])
def test_dark_shackles_reduces_strength_for_one_turn(combat_factory, upgraded, amount):
    battle = _battle(combat_factory, [('DARK_SHACKLES', upgraded)])
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    powers = dict(battle.to_dict()['monsters'][0]['powers'])
    assert powers.get('Strength', 0) == -amount
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert dict(battle.to_dict()['monsters'][0]['powers']).get('Strength', 0) == 0


def test_panache_does_not_trigger_on_first_following_card(combat_factory):
    battle = _battle(combat_factory, ['PANACHE', 'ANGER'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    before = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert before - battle.to_dict()['monsters'][0]['hp'] == 6


def test_panache_triggers_only_on_each_fifth_card(combat_factory):
    battle = _battle(combat_factory, ['PANACHE'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({'hand': ['ANGER'] * 10})
    damage = []
    for _ in range(10):
        before = battle.to_dict()['monsters'][0]['hp']
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        damage.append(before - battle.to_dict()['monsters'][0]['hp'])
    assert damage == [6, 6, 6, 6, 16, 6, 6, 6, 6, 16]


def test_upgraded_forethought_allows_multiple_cards(combat_factory):
    battle = _battle(combat_factory, [('FORETHOUGHT', 1), 'STRIKE_RED', 'DEFEND_RED'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    select = battle.to_dict()['select']
    assert select['context'] == 'FORETHOUGHT'
    assert select['can_pick_any'] and select['multi']


@pytest.mark.parametrize('artifact', [1, 2])
@pytest.mark.parametrize('upgraded', [0, 1])
def test_shackles_artifact_blocks_loss_and_restoration(combat_factory, artifact, upgraded):
    battle = _battle(combat_factory, [('DARK_SHACKLES', upgraded)])
    battle.set_monster_status(0, 'ARTIFACT', artifact)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    powers = dict(battle.to_dict()['monsters'][0]['powers'])
    assert powers.get('Strength', 0) == 0
    assert powers.get('Artifact', 0) == artifact - 1
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert dict(battle.to_dict()['monsters'][0]['powers']).get('Strength', 0) == 0


@pytest.mark.parametrize('upgraded,damage', [(0, 10), (1, 14)])
def test_panache_skill_power_and_stacking_share_countdown(combat_factory, upgraded, damage):
    battle = _battle(combat_factory, [('PANACHE', upgraded), 'DEFEND_RED',
        'INFLAME', ('PANACHE', upgraded), 'DEFEND_RED', 'DEFEND_RED'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    hp = battle.to_dict()['monsters'][0]['hp']
    for _ in range(4):
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
        assert battle.to_dict()['monsters'][0]['hp'] == hp
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == hp - 2 * damage


@pytest.mark.parametrize('count', [1, 3, 9])
def test_forethought_plus_all_subsets_and_zero_are_legal(combat_factory, count):
    battle = _battle(combat_factory, [('FORETHOUGHT', 1)] + ['STRIKE_RED'] * count)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    actions = battle.rule_actions()
    assert {int(a.source_idx) for a in actions} == set(range(1 << count))
    assert all(int(a.action_type) == 3 for a in actions)
    before = battle.to_dict()
    battle.execute(sts.Action(sts.ActionType.MULTI_CARD_SELECT, 0))
    assert battle.to_dict()['hand'] == before['hand']
    assert battle.to_dict()['draw'] == before['draw']


@pytest.mark.parametrize('compact', [False, True])
def test_forethought_plus_prefix_search_and_execution(combat_factory, compact):
    env = combat_factory(['STRIKE_RED'] * 10, combat_action_contract='autoregressive-v2',
                         card_language=True)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': [('FORETHOUGHT', 1), 'BASH', 'ANGER', 'DEFEND_RED'],
        'draw': ['STRIKE_RED'], 'discard': [], 'player': {'energy': 10}})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    before = battle.to_dict()
    logits = np.zeros(198)
    response = env.nosl_is_az_cpp_begin(logits, state_nonce=42, simulations=8,
        depth=1, leaf_batch_size=1, compact_observation=compact)
    for _ in range(16):
        if response['done']:
            break
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, logits) for leaf in response['leaf_meta']], max_leaves=1)
    assert response['done']
    assert battle.to_dict() == before
    direct = battle.clone()
    direct.execute(sts.Action(sts.ActionType.MULTI_CARD_SELECT, 5))
    env.step(C.AUTOREGRESSIVE_MULTI)
    env.step(C.AUTOREGRESSIVE_MULTI + 2)
    assert battle.to_dict() == before
    env.step(C.AUTOREGRESSIVE_STOP)
    assert battle.to_dict() == direct.to_dict()
    assert [c[0] for c in battle.to_dict()['hand']] == ['ANGER']
    assert [c[0] for c in battle.to_dict()['draw']] == ['DEFEND_RED', 'BASH', 'STRIKE_RED']
    # Drawing the parked cards preserves their one-shot free flag.
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    costs = {c[0]: c[2] for c in battle.to_dict()['hand']}
    assert costs['BASH'] == costs['DEFEND_RED'] == 0


def test_unupgraded_forethought_still_requires_one_card(combat_factory):
    battle = _battle(combat_factory, ['FORETHOUGHT', 'BASH', 'ANGER'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    select = battle.to_dict()['select']
    assert not select['multi'] and not select['can_pick_zero']
    assert len(battle.rule_actions()) == 2
    battle.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, 0))
    assert [c[0] for c in battle.to_dict()['hand']] == ['ANGER']
    assert battle.to_dict()['draw'][0][0] == 'BASH'


def test_panache_partial_count_resets_on_new_turn(combat_factory):
    battle = _battle(combat_factory, ['PANACHE', 'DEFEND_RED', 'DEFEND_RED'])
    for _ in range(3):
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({'hand': ['DEFEND_RED'] * 5, 'player': {'energy': 20}})
    hp = battle.to_dict()['monsters'][0]['hp']
    for _ in range(4):
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
        assert battle.to_dict()['monsters'][0]['hp'] == hp
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == hp - 10


@pytest.mark.parametrize('card', [
    'ANGER', 'BASH', 'BODY_SLAM', 'CARNAGE', 'CLOTHESLINE', 'HEAVY_BLADE',
    'IRON_WAVE', 'RAMPAGE', 'THUNDERCLAP', 'TWIN_STRIKE', 'UPPERCUT',
    'DEFEND_RED', 'BLOODLETTING', 'DISARM', 'ENTRENCH', 'FLAME_BARRIER',
    'FLEX', 'GHOSTLY_ARMOR', 'IMPERVIOUS', 'METALLICIZE', 'INFLAME',
    'BARRICADE', 'CORRUPTION', 'DEMON_FORM',
])
@pytest.mark.parametrize('upgraded', [0, 1])
def test_equivalent_deterministic_plays_reach_same_public_future(combat_factory, card, upgraded):
    battle = _battle(combat_factory, [(card, upgraded), 'DEFEND_RED', (card, upgraded)])
    assert battle.equivalent_play_slots()[2] == 0
    left, right = battle.clone(), battle.clone()
    left.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    right.execute(sts.Action(sts.ActionType.CARD, 2, 0))
    assert left.nosl_future_key() == right.nosl_future_key()
    assert left.to_dict()['player'] == right.to_dict()['player']
    assert left.to_dict()['monsters'] == right.to_dict()['monsters']
