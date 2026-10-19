import pytest

from bridge.real_game_env import RealGameAdapter
from bridge.schemas.entity_ids import NEOW_BONUS_IDS, neow_option_from_types


@pytest.mark.parametrize('bonus_type,bonus', NEOW_BONUS_IDS.items())
def test_all_stock_neow_bonus_keywords(bonus_type, bonus):
    assert neow_option_from_types(bonus_type, 'NONE') == {
        'bonus_id': bonus, 'drawback_id': 6 if bonus_type == 'BOSS_RELIC' else 1}


def _adapter(offers):
    adapter = RealGameAdapter(neow_observation=True)
    adapter.gs = {'max_hp': 80, 'simulator_neow_screen': 3,
        'simulator_neow_options': offers, 'choice_list': ['ignored localized label'] * len(offers)}
    adapter.frame = {'in_game': True, 'ready_for_command': True,
        'available_commands': ['choose', 'state'], 'game_state': adapter.gs}
    return adapter


@pytest.mark.parametrize('count', [2, 4])
def test_neow_adapter_uses_typed_objects_and_exact_ui_indices(count):
    offers = [{'choice_index': i, 'bonus_type': 'THREE_CARDS', 'drawback_type': 'NONE', 'hp_unit': 8}
              for i in range(count)]
    adapter = _adapter(offers)
    adapter.gs['screen_state'] = {'event_id': 'Neow Event'}
    from env.core import game_actions as ga
    assert adapter._legal_event_actions() == [ga.neow(i) for i in range(count)]
    assert adapter._neow_options({'options': [{'text': 'unknown cost and unknown bonus'}] * count}) == [
        {'bonus_id': 0, 'drawback_id': 1, 'hp_unit': 8}] * count
    adapter.gs['choice_list'].append('extra')
    with pytest.raises(ValueError, match='exactly cover'):
        adapter._neow_options({})


def test_legacy_text_cannot_be_used_to_guess_a_missing_typed_neow_offer():
    adapter = _adapter([])
    del adapter.gs['simulator_neow_options']
    with pytest.raises(ValueError, match='simulator_neow_options'):
        adapter._neow_options({'options': [{'text': 'Upgrade a Card'}] * 4})


@pytest.mark.parametrize('stage', [0, 1, 99])
def test_only_explicit_neow_dialogue_stages_can_be_auto_clicked(stage):
    adapter = _adapter([{}])
    adapter.gs.update(screen_type='EVENT', screen_state={'event_id': 'Neow Event'}, simulator_neow_screen=stage)
    assert adapter.auto_command() == ('CHOOSE 0', 'neow_intro')


@pytest.mark.parametrize('stage', [2, 10, 999, 42, True])
def test_unknown_or_unsupported_neow_stage_is_never_auto_clicked(stage):
    adapter = _adapter([{}])
    adapter.gs.update(screen_type='EVENT', screen_state={'event_id': 'Neow Event'}, simulator_neow_screen=stage)
    with pytest.raises(ValueError, match='Neow screen'):
        adapter.auto_command()


def test_typed_neow_does_not_hide_unknown_hp_cost_or_reordered_candidate():
    offers = [{'choice_index': i, 'bonus_type': 'THREE_CARDS', 'drawback_type': 'NONE', 'hp_unit': 8} for i in range(2)]
    adapter = _adapter(offers)
    offers[0]['hp_unit'] = 7
    with pytest.raises(ValueError, match='cached HP unit'):
        adapter._neow_options({})


@pytest.mark.parametrize('count', [2, 4])
def test_neow_actual_offer_count_reaches_observation_and_action_mask(count):
    import numpy as np
    from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
    from env.core import action_space as A
    offers = [{'choice_index': i, 'bonus_type': 'THREE_CARDS', 'drawback_type': 'NONE', 'hp_unit': 8}
              for i in range(count)]
    adapter = _adapter(offers)
    adapter.gs.update({
        'simulator_parity_export': SIMULATOR_PARITY_EXPORT_VERSION,
        'screen_type': 'EVENT', 'screen_state': {'event_id': 'Neow Event'},
        'floor': 0, 'current_hp': 80, 'gold': 99, 'class': 'IRONCLAD',
        'ascension_level': 0, 'act': 1, 'room_type': 'EmptyRoom', 'act_boss': 'Slime Boss',
        'has_ruby_key': False, 'has_emerald_key': False, 'has_sapphire_key': False,
        'deck': [], 'relics': [{'id': 'Burning Blood', 'counter': -1}],
        'potions': [{'id': 'Potion Slot'}] * 3,
    })
    canonical = adapter.obs_canonical()
    assert len(canonical['options']['neow']) == count
    observation = adapter.observation()
    # Bonus 0 is the real THREE_CARDS enum, not an absent offer; presence is
    # disambiguated by drawback 1 (NONE), versus 0 only in absent slots.
    assert np.count_nonzero(observation['neow_drawback_ids']) == count
    np.testing.assert_array_equal(observation['neow_bonus_ids'][:count], np.zeros(count))
    legal = np.flatnonzero(adapter.action_mask())
    assert legal.tolist() == list(range(A._NEOW, A._NEOW + count))
    offers[0]['hp_unit'] = 8
    offers[0]['choice_index'] = 1
    with pytest.raises(ValueError, match='reordered'):
        adapter._neow_options({})


def test_typed_neow_keywords_do_not_use_java_enum_ordinals_or_guess_none():
    from bridge.schemas.entity_ids import NEOW_BONUS_IDS, NEOW_DRAWBACK_IDS, neow_option_from_types
    assert len(NEOW_BONUS_IDS) == 19 and len(NEOW_DRAWBACK_IDS) == 5
    assert neow_option_from_types('RANDOM_COLORLESS_2', 'NONE') == {'bonus_id': 11, 'drawback_id': 1}
    assert neow_option_from_types('BOSS_RELIC', 'NONE') == {'bonus_id': 18, 'drawback_id': 6}
    with pytest.raises(ValueError, match='unknown Neow drawback'):
        neow_option_from_types('THREE_CARDS', 'UNKNOWN_COST')
    with pytest.raises(ValueError, match='implicit'):
        neow_option_from_types('BOSS_RELIC', 'CURSE')
    offers = [{'choice_index': i, 'bonus_type': 'THREE_CARDS', 'drawback_type': 'NONE', 'hp_unit': 8} for i in range(4)]
    adapter = _adapter(offers)
    for invalid in (True, '0', -1, 19):
        adapter.gs['simulator_neow_options'][0]['bonus_type'] = invalid
        with pytest.raises(ValueError, match='unknown Neow bonus'):
            adapter._neow_options({})


@pytest.mark.parametrize('offer', [{}, {'bonus_id': 0}, {'bonus_id': 0, 'drawback_id': 0},
                                 {'bonus_id': 19, 'drawback_id': 1}, {'bonus_id': True, 'drawback_id': 1}])
def test_encoder_never_turns_an_incomplete_offer_into_three_cards_or_pad(offer):
    from env.core import encode
    with pytest.raises(ValueError, match='Neow'):
        encode.encode({'phase': 'event', 'options': {'event': 'NEOW', 'neow': [offer]}}, neow_observation=True)


def test_complete_neow_keyword_table_matches_native_named_enums():
    import re
    from pathlib import Path
    from bridge.schemas.entity_ids import NEOW_BONUS_IDS, NEOW_DRAWBACK_IDS, NEOW_IMPLICIT_DRAWBACK_IDS
    source = (Path(__file__).resolve().parents[2] / 'sim/sts_lightspeed/include/game/Neow.h').read_text()
    def enums(name):
        body = re.search(r'enum class ' + name + r'\s*\{(.*?)\};', source, re.S).group(1)
        result, index = {}, 0
        for token, value in re.findall(r'\b([A-Z][A-Z0-9_]*)\s*(?:=\s*(\d+))?\s*,', body):
            if value:
                index = int(value)
            result[token] = index
            index += 1
        return result
    assert NEOW_BONUS_IDS == {key: value for key, value in enums('Bonus').items() if key != 'INVALID'}
    drawback = enums('Drawback')
    assert NEOW_DRAWBACK_IDS == {key: value for key, value in drawback.items() if key not in {'INVALID', 'LOSE_STARTER_RELIC'}}
    assert NEOW_IMPLICIT_DRAWBACK_IDS == {'BOSS_RELIC': drawback['LOSE_STARTER_RELIC']}
