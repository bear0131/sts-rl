"""Decision contracts: kill/block tradeoffs, source identity, partial actions."""
from copy import deepcopy

import pytest

from env.core import action_space as A
from rl.evaluation.baselines.combat_rules import (
    CombatView, card_score, damage, rank_actions, selection_score,
)


def card(name, cost=1, up=0):
    return dict(id=name, cost=cost, up=up)


def monster(hp=40, attack=10, **powers):
    return dict(id='Cultist', hp=hp, max_hp=hp, block=0, gone=False,
                adjusted_damage=attack, hits=1, powers=powers)


@pytest.fixture
def view():
    return CombatView(dict(hp=50, max_hp=80, block=0, energy=3, powers={}),
        [monster()], [card('Strike_R'), card('Defend_R')],
        dict(draw=[card('Strike_R')]*5, discard=[], exhaust=[], generated=[]),
        targets={('play', 0): [0], ('play', 1): [None]})


def pick(view, *actions):
    return rank_actions(view, actions)[0]['action']


def test_guaranteed_combat_kill_beats_defending(view):
    view.monsters[0]['hp'] = 6
    view.player['hp'] = 2
    assert pick(view, A._PLAY, A._PLAY+1, A.END_TURN) == A._PLAY


def test_block_is_valued_only_against_visible_threat(view):
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY+1
    view.player['block'] = 10
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY
    assert pick(view, A._PLAY+1, A.END_TURN) == A.END_TURN


def test_bash_and_flex_precede_attacks_when_followup_is_affordable(view):
    view.hand = [card('Strike_R'), card('Bash', 2), card('Flex', 0)]
    view.monsters[0]['adjusted_damage'] = 0
    view.targets = {('play', i): [0] for i in range(3)}
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY+1
    assert pick(view, A._PLAY, A._PLAY+2) == A._PLAY+2
    view.player['energy'] = 2
    # Bash without an immediate follow-up still carries future-turn value.
    assert card_score(view.hand[1], 0, view)[1]['vulnerable'] == 4


def test_target_index_does_not_compact_empty_or_dead_slots(view):
    view.monsters = [None, monster(hp=40), monster(hp=6)]
    view.pending = ('play', 0)
    assert pick(view, A._TARGET+1, A._TARGET+2) == A._TARGET+2


def test_pending_potion_keeps_its_identity(view):
    view.potions = ['Fire Potion']
    view.monsters = [monster(hp=20), monster(hp=6)]
    view.pending = ('potion', 0)
    assert pick(view, A._TARGET, A._TARGET+1) == A._TARGET
    view.pending = ('play', 0)
    assert pick(view, A._TARGET, A._TARGET+1) == A._TARGET+1


@pytest.mark.parametrize('context,pile', [('HEADBUTT','discard'), ('EXHUME','exhaust'),
    ('SECRET_WEAPON','draw'), ('DISCOVERY','generated')])
def test_select_uses_the_screen_candidate_region(view, context, pile):
    view.select = {'context': context}
    view.piles[pile] = [card('Wound'), card('Carnage', 2)]
    view.hand = [card('Carnage', 2), card('Wound')]
    assert pick(view, A._PICK, A._PICK+1) == A._PICK+1


def test_multi_exhaust_removes_junk_then_confirms(view):
    view.select = {'context': 'EXHAUST_MANY', 'multi': True}
    view.hand = [card('Wound'), card('Carnage', 2)]
    assert pick(view, A._PICK, A._PICK+1, A._CONFIRM) == A._PICK
    assert pick(view, A._PICK+1, A._CONFIRM) == A._CONFIRM


def test_missing_selection_candidate_fails_loudly(view):
    view.select = {'context': 'EXHUME'}
    with pytest.raises(ValueError, match='missing from exhaust'):
        selection_score(0, view)


def test_determinism_does_not_depend_on_unobserved_draw_order(view):
    view.hand.append(card('Havoc'))
    view.targets[('play', 2)] = [None]
    view.piles['draw'] = [card('Bludgeon'), card('Wound'), card('Strike_R')]
    before = deepcopy(view)
    first = rank_actions(view, [A._PLAY, A._PLAY+1, A._PLAY+2])
    assert view == before
    view.piles['draw'].reverse()
    assert rank_actions(view, [A._PLAY, A._PLAY+1, A._PLAY+2]) == first


def test_multihit_strength_weak_vulnerability_and_block(view):
    view.player['powers'] = {'Strength': 2, 'Weakened': 1}
    view.monsters[0]['powers'] = {'Vulnerable': 1}
    # floor((5 + 2)*.75*1.5) twice, not one rounding after summing hits.
    assert damage(card('Twin Strike'), view.monsters[0], view) == 14
    view.monsters[0]['block'] = 5
    assert card_score(card('Twin Strike'), 0, view)[1]['damage'] == pytest.approx(9*.85)


def test_self_damage_cannot_be_spent_as_if_hp_were_free(view):
    view.player['hp'] = 3
    assert pick(view, A.END_TURN, A._PLAY) == A._PLAY
    view.hand = [card('Bloodletting', 0), card('Bludgeon', 3)]
    assert pick(view, A.END_TURN, A._PLAY) == A.END_TURN


def test_chip_persistent_enemy_block_instead_of_ending_forever(view):
    # Development seed 10000, floor18 Spheric Guardian: all attacks were
    # incorrectly valued at zero while its Barricade block exceeded one hit.
    view.monsters[0].update(block=40, powers={'Barricade':1})
    view.player['block'] = 20
    assert pick(view, A.END_TURN, A._PLAY) == A._PLAY
