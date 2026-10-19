"""BRIDGE42: actual DrugDealer two-pick transport, not generic transform flags."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bridge.event_contract import EventEffectAck
from bridge.real_game_env import RealGameAdapter
from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
from env.core import game_actions as ga
from tests.bridge_fixtures import real_card


def card(card_id, index, card_type='ATTACK', **kwargs):
    return dict(real_card(card_id, index=index, **kwargs), type=card_type)


@pytest.fixture
def augmenter():
    deck = [card('Strike_R', 0, bottle='ATTACK'), card('Strike_R', 1),
            card('Defend_R', 2, 'SKILL'), card('AscendersBane', 3, 'CURSE')]
    labels = ['[Test J.A.X.]', '[Become Test Subject]', '[Ingest Mutagens]']
    frame = dict(in_game=True, ready_for_command=True,
                 available_commands=['choose', 'state'], game_state=dict(
                     seed='42', floor=19, act=2, screen_type='EVENT', room_phase='EVENT',
                     current_hp=80, max_hp=80, gold=99, relics=[], potions=[], deck=deck,
                     choice_list=labels, screen_state=dict(event_id='Drug Dealer', options=[
                         dict(text=label, disabled=False, choice_index=i) for i, label in enumerate(labels)])))
    return frame


def begin(frame):
    ack = EventEffectAck()
    ack.arm(frame, SimpleNamespace(semantic=ga.event(1), command='CHOOSE 1'), 0)
    grid = deepcopy(frame)
    grid['game_state'].update(screen_type='GRID', screen_state=dict(
        cards=deepcopy(frame['game_state']['deck'][:3]), selected_cards=[], num_cards=2,
        any_number=False, for_upgrade=False, for_purge=False, for_transform=False,
        selection_source='com.megacrit.cardcrawl.events.city.DrugDealer',
        selection_context='TRANSFORM', simulator_parity_export=SIMULATOR_PARITY_EXPORT_VERSION))
    return ack, grid


def pick(ack, grid, index):
    assert not ack.waiting(grid, 1)
    adapter = RealGameAdapter()
    adapter.update(grid)
    action = ga.select(index)
    command = adapter.command_for_semantic(action)
    ack.arm(grid, SimpleNamespace(semantic=action, command=command), 2)
    # Stale selected list must wait, not reselect the same card.
    assert ack.waiting(grid, 3)
    grid['game_state']['screen_state']['selected_cards'].append(
        deepcopy(grid['game_state']['screen_state']['cards'][int(command.split()[1])]))
    return command


def result(frame, deck):
    done = deepcopy(frame)
    done['game_state']['deck'] = deepcopy(deck)
    done['game_state']['screen_state']['options'] = [dict(text='[Leave]', disabled=False, choice_index=0)]
    done['game_state']['choice_list'] = ['leave']
    return done


def test_augmenter_preserves_second_identical_card_and_waits_for_both_gains(augmenter):
    ack, grid = begin(augmenter)
    assert ack.waiting(augmenter, 0.5)
    assert pick(ack, grid, 1) == 'CHOOSE 1'
    # Remaining candidates are original slots 0 and 2, not a compact UI index.
    assert pick(ack, grid, 0) == 'CHOOSE 0'
    survivors = augmenter['game_state']['deck'][2:]
    partial = result(augmenter, survivors)
    assert ack.waiting(partial, 4)
    with pytest.raises(RuntimeError, match='still unverified'):
        ack.arm(partial, SimpleNamespace(semantic=None, command='CHOOSE 0'), 5)
    partial['game_state']['deck'].append(card('Anger', 10))
    assert ack.waiting(partial, 6)
    partial['game_state']['deck'].append(card('Cleave', 11))
    assert not ack.waiting(partial, 7)
    assert ack.pending is None


@pytest.mark.parametrize('mutation', ['missing_uuid', 'duplicate_uuid', 'wrong_selected', 'wrong_source', 'wrong_flags', 'missing_candidate'])
def test_augmenter_rejects_invalid_grid_identity_and_context(augmenter, mutation):
    ack, grid = begin(augmenter)
    ss = grid['game_state']['screen_state']
    if mutation == 'missing_uuid':
        del ss['cards'][0]['uuid']
    elif mutation == 'duplicate_uuid':
        ss['cards'][0]['uuid'] = ss['cards'][1]['uuid']
    elif mutation == 'wrong_selected':
        ss['selected_cards'] = [deepcopy(ss['cards'][0])]
    elif mutation == 'wrong_source':
        ss['selection_source'] = 'com.megacrit.cardcrawl.events.shrines.Transmogrifier'
    elif mutation == 'wrong_flags':
        ss['for_transform'] = True
    else:
        ss['cards'].pop()
    with pytest.raises((ValueError, RuntimeError)):
        ack.waiting(grid, 1)


@pytest.mark.parametrize('mutation', ['unselected_removed', 'survivor_changed', 'map_early', 'extra_gain', 'wrong_upgrade'])
def test_augmenter_rejects_wrong_results(augmenter, mutation):
    ack, grid = begin(augmenter)
    pick(ack, grid, 1)
    pick(ack, grid, 0)
    done = result(augmenter, augmenter['game_state']['deck'][2:] + [card('Anger', 10), card('Cleave', 11)])
    gs = done['game_state']
    if mutation == 'unselected_removed':
        gs['deck'].pop(0)
    elif mutation == 'survivor_changed':
        gs['deck'][0]['misc'] = 9
    elif mutation == 'map_early':
        gs['screen_type'] = 'MAP'
    elif mutation == 'extra_gain':
        gs['deck'].append(card('Anger', 12))
    else:
        gs['deck'][-1]['upgrades'] = 1
    with pytest.raises(RuntimeError):
        ack.waiting(done, 8)


@pytest.mark.parametrize('omamori', [0, 1, 2])
@pytest.mark.parametrize('bloom', [False, True])
def test_augmenter_curse_replacement_and_resource_effects(augmenter, omamori, bloom):
    gs = augmenter['game_state']
    gs['deck'][:2] = [card('Parasite', 0, 'CURSE'), card('Pain', 1, 'CURSE')]
    gs['relics'] = [dict(id=name, counter=count) for name, count in [
        ('Omamori', omamori), ('Darkstone Periapt', -1), ('CeramicFish', -1), ('Du-Vu Doll', 3)]]
    if bloom:
        gs['relics'].append(dict(id='Mark of the Bloom', counter=-1))
    ack, grid = begin(augmenter)
    pick(ack, grid, 0)
    pick(ack, grid, 0)
    gains = [card('Doubt', 10 + i, 'CURSE') for i in range(2 - omamori)]
    done = result(augmenter, gs['deck'][2:] + gains)
    after = done['game_state']
    after['relics'][0]['counter'] = 0
    after['relics'][3]['counter'] = 3 - omamori
    after['max_hp'] = 77 + 6 * len(gains)
    after['current_hp'] = 77 if bloom else after['max_hp']
    after['gold'] = 99 + 9 * len(gains)
    assert not ack.waiting(done, 8)


def test_augmenter_confirmation_is_transport_and_cannot_repeat(augmenter):
    ack, grid = begin(augmenter)
    pick(ack, grid, 0)
    pick(ack, grid, 0)
    confirm = SimpleNamespace(semantic=None, command='CONFIRM')
    ack.arm(grid, confirm, 4)
    assert ack.waiting(grid, 5)
    with pytest.raises(ValueError, match='repeated confirmation'):
        ack.arm(grid, confirm, 6)
    with pytest.raises(RuntimeError, match='timeout'):
        ack.waiting(grid, 31)


@pytest.mark.parametrize('eggs', [False, True])
def test_jax_card_waits_for_actual_gain_and_respects_egg(augmenter, eggs):
    gs = augmenter['game_state']
    gs['relics'] = [dict(id='CeramicFish', counter=-1)]
    if eggs:
        gs['relics'].append(dict(id='Toxic Egg 2', counter=-1))
    ack = EventEffectAck()
    ack.arm(augmenter, SimpleNamespace(semantic=ga.event(0), command='CHOOSE 0'), 0)
    done = result(augmenter, gs['deck'])
    assert ack.waiting(done, 1)
    done['game_state']['deck'].append(card('J.A.X.', 10, 'SKILL', upgrades=int(eggs)))
    assert ack.waiting(done, 2)  # No fake ACK before Ceramic Fish gold arrives.
    done['game_state']['gold'] += 9
    assert not ack.waiting(done, 3)


@pytest.mark.parametrize('circlets', [None, 0, 2])
def test_mutagenic_strength_duplicate_gives_or_stacks_circlet(augmenter, circlets):
    gs = augmenter['game_state']
    if circlets is not None:
        gs['relics'].append(dict(id='MutagenicStrength', counter=-1))
        if circlets:
            gs['relics'].append(dict(id='Circlet', counter=circlets))
    ack = EventEffectAck()
    ack.arm(augmenter, SimpleNamespace(semantic=ga.event(2), command='CHOOSE 2'), 0)
    done = result(augmenter, gs['deck'])
    assert ack.waiting(done, 1)
    if circlets is None:
        done['game_state']['relics'].append(dict(id='MutagenicStrength', counter=-1))
    elif not circlets:
        done['game_state']['relics'].append(dict(id='Circlet', counter=1))
    else:
        done['game_state']['relics'][-1]['counter'] += 1
    assert not ack.waiting(done, 2)
