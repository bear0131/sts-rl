"""BRIDGE30: replay initialization consumes persistent real-card records."""
from copy import deepcopy

import pytest
import slaythespire as sts

from bridge.replay import _sync_gc
from tests.bridge_fixtures import real_card

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _frame(deck):
    return dict(floor=1, current_hp=70, max_hp=80, gold=99, deck=deck, relics=[])


@pytest.mark.parametrize('card_id', ['RitualDagger', 'Ritual Dagger'])
def test_replay_restores_current_ritual_growth_and_bottle_before_battle(card_id):
    frame = _frame([real_card(card_id, misc=39, damage=39, exhaust=True, bottle='ATTACK')])
    before = deepcopy(frame)
    gc = _sync_gc(42, 0, frame)
    assert gc.deck_state() == [dict(id='RITUAL_DAGGER', upgrades=0, misc=39, bottle='ATTACK')]
    battle = sts.create_battle(gc, 'Cultist')
    hp = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == hp - 39
    assert frame == before


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_replay_preserves_physical_bottles_among_duplicate_cards(seed):
    records = [real_card('Bash', index=i, base_cost=2) for i in range(20)]
    records += [real_card('Bash', index=20, base_cost=2, bottle='ATTACK'),
                real_card('Battle Trance', index=21, base_cost=0, bottle='SKILL'),
                real_card('Demon Form', index=22, base_cost=3, bottle='POWER')]
    gc = _sync_gc(seed, 0, _frame(records))
    hand = sts.create_battle(gc, 'Cultist').card_state()['hand']
    assert {20, 21, 22} <= {c['uid'] for c in hand}


def test_replay_maps_searing_java_misc_to_native_upgrade_count():
    gc = _sync_gc(42, 0, _frame([real_card('Searing Blow', upgrades=3, misc=0, base_cost=2)]))
    assert gc.deck_state() == [dict(id='SEARING_BLOW', upgrades=3, misc=3, bottle='NONE')]
    battle = sts.create_battle(gc, 'Cultist')
    hp = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == hp - 27


@pytest.mark.parametrize('missing', ['misc', 'uuid', 'simulator_card_state'])
def test_incomplete_frame_cannot_silently_construct_fresh_master_cards(missing):
    record = real_card('Bash')
    del record[missing]
    with pytest.raises(ValueError, match='missing fields'):
        _sync_gc(42, 0, _frame([record]))


def test_replay_rejects_duplicate_master_uuid():
    with pytest.raises(ValueError, match='duplicate master card UUID'):
        _sync_gc(42, 0, _frame([real_card('Bash'), real_card('Defend_R')]))


def test_replay_rejects_mismatched_bottle_type():
    with pytest.raises(ValueError, match='bottle type mismatches'):
        _sync_gc(42, 0, _frame([real_card('Bash', bottle='SKILL')]))
