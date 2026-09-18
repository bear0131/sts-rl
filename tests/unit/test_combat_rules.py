"""Decision contracts: kill/block tradeoffs, source identity, partial actions."""
from copy import deepcopy

import pytest

from env.core import action_space as A
from rl.evaluation.baselines.combat_rules import (
    CombatView, card_score, damage, hide_intent, potion_score, rank_actions, selection_score,
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


def test_play_affordable_nonattacks_to_unlock_clash(view):
    view.player['energy'] = 4
    view.monsters[0].update(hp=109, block=8, adjusted_damage=0)
    view.hand = [card('Strike_R'), card('Strike_R'), card('Defend_R'),
                 card('Defend_R'), card('Clash', 0)]
    view.targets = {('play', i): [0] for i in range(4)}
    assert pick(view, *(A._PLAY+i for i in range(4)), A.END_TURN) == A._PLAY+2
    # An unplayable Wound prevents promising a Clash that cannot be unlocked.
    view.hand.append(card('Wound', -2))
    assert 'unlock_clash' not in card_score(view.hand[2], 0, view)[1]


def test_debuffs_do_not_reward_expensive_overkill(view):
    view.monsters = [monster(hp=2, attack=0), monster(hp=40)]
    view.hand = [card('Anger', 0), card('Bash', 2)]
    view.targets = {('play', 0): [0], ('play', 1): [0]}
    assert card_score(view.hand[1], 0, view)[1]['vulnerable'] == 0
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY


def test_inflame_precedes_attacks_when_both_fit(view):
    view.monsters[0].update(hp=420, max_hp=420)
    view.hand = [card('Strike_R', up=1), card('Inflame')]
    view.targets = {('play', 0): [0], ('play', 1): [None]}
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY+1


def test_long_fight_scaling_is_worth_setting_up(view):
    view.monsters[0].update(hp=420, max_hp=420)
    view.hand = [card('Demon Form', 3), card('Defend_R')]
    view.targets = {('play', 0): [None], ('play', 1): [None]}
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY


def test_plated_armor_covers_current_threat(view):
    view.player['powers'] = {'Plated Armor': 10}
    assert view.deficit == 0
    assert pick(view, A._PLAY+1, A.END_TURN) == A.END_TURN


def test_spore_cloud_penalty_requires_a_survivor_and_no_artifact(view):
    # Equal dictionaries still represent separate monster slots.
    view.monsters = [monster(hp=6, **{'Spore Cloud': 2}), monster(hp=6, **{'Spore Cloud': 2})]
    parts = card_score(view.hand[0], 0, view)[1]
    assert parts['spore_cloud'] < 0
    view.player['powers'] = {'Artifact': 1}
    assert 'spore_cloud' not in card_score(view.hand[0], 0, view)[1]
    view.player['powers'] = {}
    view.monsters.pop()
    assert 'spore_cloud' not in card_score(view.hand[0], 0, view)[1]


def test_mercury_does_not_cancel_current_enemy_attack(view):
    view.relics = {'Mercury Hourglass': 0}
    view.monsters[0]['hp'] = 3
    assert view.incoming == 10
    assert view.deficit == 10
    # A current combat lethal is still preferable to waiting for next turn.
    assert pick(view, A._PLAY, A._PLAY+1, A.END_TURN) == A._PLAY


def test_sleep_allows_setup_but_does_not_stall_forever(view):
    view.monsters[0].update(id='Lagavulin', move='LAGAVULIN_SLEEP',
        hp=109, block=8, powers={'Metallicize': 8}, adjusted_damage=0)
    view.hand = [card('Carnage', 2)]
    view.targets = {('play', 0): [0]}
    view.piles['draw'] = [card('Inflame')]
    assert pick(view, A._PLAY, A.END_TURN) == A.END_TURN
    view.turn = 3
    assert pick(view, A._PLAY, A.END_TURN) == A._PLAY


@pytest.mark.parametrize('potion', ['Regen Potion', 'Essence Of Steel', 'Heart Of Iron', 'Cultist Potion'])
def test_use_long_fight_potions_before_damage_accumulates(view, potion):
    view.monsters[0].update(hp=420, max_hp=420)
    view.potions = [potion]
    assert potion_score(0, None, view)[0] > 0


def test_flex_potion_requires_attacks_to_use_its_temporary_strength(view):
    view.potions = ['Flex Potion']
    view.hand = [card('Strike_R')]*3
    assert potion_score(0, None, view)[0] > 0
    view.hand = [card('Defend_R')]*3
    assert potion_score(0, None, view)[0] < 0


def test_upgraded_neutral_damage_and_block_are_not_base_values(view):
    assert damage(card('Flash Of Steel', 0, up=1), view.monsters[0], view) == 6
    assert card_score(card('Good Instincts', 0, up=1), None, view)[0] > card_score(card('Good Instincts', 0), None, view)[0]


def test_generated_skill_with_clash_hand_is_a_valid_selection(view):
    view.hand = [card('Clash', 0)]
    view.select = {'context': 'DISCOVERY'}
    view.piles['generated'] = [card('Defend_R'), card('Impervious', 2)]
    assert pick(view, A._PICK, A._PICK+1) in (A._PICK, A._PICK+1)


def test_apparition_covers_multihit_threat_and_avoids_excess_block(view):
    view.monsters[0].update(adjusted_damage=20, hits=3)
    view.hand = [card('Apparition'), card('Defend_R')]
    view.targets = {('play', 0): [None], ('play', 1): [None]}
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY
    view.player['powers'] = {'Intangible': 1}
    view.player['block'] = 3
    assert view.incoming == 3
    assert pick(view, A._PLAY+1, A.END_TURN) == A.END_TURN


def test_dome_hides_actual_intent_before_scoring(view):
    first = monster(attack=999)
    first.update(move='LAGAVULIN_ATTACK', move_id=87, base_damage=333, hits=3)
    second = monster(attack=0)
    second.update(move='LAGAVULIN_SLEEP', move_id=92, base_damage=0, hits=0)
    for m in (first, second):
        hide_intent(m, view.player, {'Runic Dome': 0})
    assert first == second
    assert first['intent_hidden'] and first['hits'] == first['adjusted_damage'] == 0
    view.monsters = [first]
    ranked = rank_actions(view, [A._PLAY, A._PLAY+1, A.END_TURN])
    view.monsters = [second]
    assert rank_actions(view, [A._PLAY, A._PLAY+1, A.END_TURN]) == ranked


def test_race_permanent_lagavulin_debuffs_without_ignoring_lethal(view):
    view.monsters[0].update(id='Lagavulin', hp=101, adjusted_damage=18,
                           powers={'Vulnerable': 1})
    view.player['hp'] = 80
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY
    view.player['hp'] = 3
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY+1


def test_do_not_exhaust_impervious_for_unused_block(view):
    view.monsters[0]['adjusted_damage'] = 0
    view.hand = [card('Impervious', 2)]
    view.targets = {('play', 0): [None]}
    assert pick(view, A._PLAY, A.END_TURN) == A.END_TURN


def test_block_setup_requires_affordable_body_slam(view):
    view.monsters[0]['adjusted_damage'] = 0
    view.hand = [card('Impervious', 2), card('Body Slam')]
    view.targets = {('play', 0): [None], ('play', 1): [0]}
    assert pick(view, A._PLAY, A._PLAY+1, A.END_TURN) == A._PLAY
    view.player['energy'] = 2
    assert pick(view, A._PLAY, A._PLAY+1, A.END_TURN) == A.END_TURN


def test_reserve_sleeping_turn_energy_for_barricade(view):
    view.monsters[0].update(id='Lagavulin', move='LAGAVULIN_SLEEP', hp=109,
                           adjusted_damage=0, block=8, powers={'Metallicize': 8})
    view.hand = [card('Barricade', 3), card('Power Through'), card('Body Slam')]
    view.targets = {('play', 0): [None], ('play', 1): [None], ('play', 2): [0]}
    assert pick(view, A._PLAY, A._PLAY+1, A._PLAY+2, A.END_TURN) == A._PLAY


def test_exhaust_selection_keeps_last_damage_source(view):
    view.monsters[0].update(hp=14, adjusted_damage=18)
    view.player['powers'] = {'Strength': -2}
    view.hand = [card('Strike_R'), card('Defend_R')]
    view.piles['draw'] = [card('Burning Pact')]
    view.select = {'context': 'EXHAUST_ONE'}
    assert pick(view, A._PICK, A._PICK+1) == A._PICK+1


def test_random_exhaust_can_destroy_the_only_remaining_attack(view):
    view.hand = [card('True Grit'), card('Strike_R')]
    view.piles['draw'] = [card('Defend_R')]
    view.targets = {('play', 0): [None], ('play', 1): [0]}
    assert pick(view, A._PLAY, A._PLAY+1) == A._PLAY+1


def test_second_debuff_lands_after_first_removes_last_artifact(view):
    view.monsters[0]['powers'] = {'Artifact': 1}
    view.hand = [card('Shockwave', 2), card('Strike_R')]
    parts = card_score(view.hand[0], 0, view)[1]
    assert parts['vulnerable'] == 7
    assert not parts.get('weak')
    view.monsters[0]['powers']['Artifact'] = 2
    parts = card_score(view.hand[0], 0, view)[1]
    assert not parts.get('vulnerable') and not parts.get('weak')


def test_artifact_opening_needs_visible_debuff_followup(view):
    view.monsters[0].update(hp=100, powers={'Artifact': 3})
    view.hand = [card('Shockwave', 2), card('Strike_R')]
    view.piles = dict(draw=[], discard=[], exhaust=[], generated=[])
    assert not card_score(view.hand[0], 0, view)[1].get('artifact_opening')
    view.hand.append(card('Uppercut', 2))
    view.player['energy'] = 4
    assert card_score(view.hand[0], 0, view)[1]['artifact_opening'] > 0
    # Artifact2 consumes both Shockwave effects; no pretend immediate Weak.
    view.monsters[0]['powers']['Artifact'] = 2
    assert not card_score(view.hand[0], 0, view)[1].get('weak')


def test_killed_enemy_gets_neither_debuff_nor_artifact_credit(view):
    view.monsters[0].update(hp=1, powers={'Artifact': 1})
    view.hand = [card('Uppercut', 2), card('Disarm')]
    parts = card_score(view.hand[0], 0, view)[1]
    assert not any(parts.get(k) for k in ('weak', 'vulnerable', 'artifact_opening'))
