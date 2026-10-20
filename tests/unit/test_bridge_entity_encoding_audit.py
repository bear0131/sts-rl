"""BRIDGE08: known real IDs must not become padding in checkpoint inputs."""
import pytest

from bridge.real_game_env import (
    _normalize_boss,
    _normalize_event,
    _potion_id,
    _relic_id,
)
from bridge.schemas.combat_state import _card
from env.core import encode
from env.core.cards_vocab import CARD_INDEX
from env.core.vocab import POTION_VOCAB_INDEX


@pytest.mark.parametrize("real_id,canonical", [
    ("SteroidPotion", "Flex Potion"),
    ("ColorlessPotion", "Colorless Potion"),
    ("SpeedPotion", "Speed Potion"),
    ("GamblersBrew", "Gamblers Brew"),
])
def test_real_potion_identity_matches_checkpoint_token(real_id, canonical):
    expected = POTION_VOCAB_INDEX[canonical]
    assert expected > 0
    assert POTION_VOCAB_INDEX.get(_potion_id(real_id), 0) == expected


@pytest.mark.parametrize("real_id,canonical", [
    ("HandOfGreed", "Hand Of Greed"),
    ("PanicButton", "Panic Button"),
    ("Flash of Steel", "Flash Of Steel"),
    ("Master of Strategy", "Master Of Strategy"),
])
def test_real_colorless_identity_matches_checkpoint_token(real_id, canonical):
    expected = CARD_INDEX[canonical]
    assert expected > 0
    assert CARD_INDEX.get(_card(real_id, 0)["id"], 0) == expected


def test_unknown_ids_fail_before_policy_instead_of_padding():
    with pytest.raises(ValueError, match="unknown real potion"):
        _potion_id("UnrecognizedPotion")
    with pytest.raises(ValueError, match="unknown card"):
        _card("UnrecognizedCard", 0)
    with pytest.raises(ValueError, match="unknown real relic"):
        _relic_id({"id": "UnrecognizedRelic"})
    with pytest.raises(ValueError, match="unknown real boss"):
        _normalize_boss("UnrecognizedBoss")
    with pytest.raises(ValueError, match="unknown real event"):
        _normalize_event("UnrecognizedEvent")
    with pytest.raises(ValueError, match="unknown categorical"):
        encode._idx(POTION_VOCAB_INDEX, "UnrecognizedPotion")
    with pytest.raises(ValueError, match="unknown card id"):
        encode._card_idx({"id": "UnrecognizedCard"})


def test_only_explicit_empty_potion_slots_become_padding():
    assert _potion_id({"id": "Potion Slot"}) == "PAD"
    with pytest.raises(ValueError, match="missing required field.*id"):
        _potion_id({})
    with pytest.raises(ValueError, match="PAD sentinel"):
        _potion_id("PAD")


@pytest.mark.parametrize('upgrades', [-1, 32768, True, 1.5, '1'])
def test_card_upgrade_transport_never_coerces_or_truncates(upgrades):
    with pytest.raises(ValueError, match='upgrade count'):
        _card('Searing Blow', upgrades)


def test_live_card_cannot_claim_the_empty_sentinel():
    with pytest.raises(ValueError, match='EMPTY sentinel'):
        _card('EMPTY', 0)


@pytest.mark.parametrize('counters', [None, [], [None], [-1], ['1'], [True]])
def test_relic_counter_absence_or_invalid_type_cannot_be_filled_with_zero(counters):
    canonical = {'phase': 'map', 'global': {'relics': ['Lizard Tail']}}
    if counters is not None:
        canonical['global']['relic_counters'] = counters
    with pytest.raises(ValueError, match='relic'):
        encode.encode(canonical)
    canonical['phase'] = 'combat'
    with pytest.raises(ValueError, match='relic'):
        encode.encode_combat(canonical)


def test_complete_codec_inventory_is_serializable_and_coverage_checked():
    import json
    from bridge.schemas.codec_contract import complete_word_tables, validate_codec_contract
    tables = complete_word_tables()
    counts = validate_codec_contract()
    assert counts['event_option_contracts'] == len(encode.V.EVENT_VOCAB) - 1
    assert counts['screen_types'] == len(tables['real_to_canonical']['screen'])
    assert tables['real_to_canonical']['character']['THE_SILENT'] == 'SILENT'
    assert tables['intent_components']['ATTACK_DEBUFF'] == ('attack', 'debuff')
    assert set(tables['relic_counter_contracts']) == set(encode.V.RELIC_VOCAB) - {'PAD'}
    assert counts['relic_counter_contracts'] == 180
    assert counts['java_relic_counter_users'] == 29
    assert counts['combat_counter_fields'] == 27
    assert counts['native_combat_root_fields'] == 16
    assert counts['real_room_classes'] == 11
    assert counts['real_card_wire_fields'] == 12
    assert tables['relic_counter_contracts']['Du-Vu Doll']['meaning'] == 'cached_master_deck_curse_count'
    assert tables['relic_counter_contracts']['Du-Vu Doll']['numeric_audit_status'] == 'mapped_persistent'
    assert tables['relic_counter_contracts']['Kunai']['combat_modulo'] == 3
    assert tables['relic_counter_contracts']['HornCleat']['combat_stop_turn'] == 2
    assert tables['relic_counter_contracts']['Burning Blood']['inherited_display_value_map'] == {-1: 0}
    assert json.loads(json.dumps(tables))['rejected_real_ids']['cards']


def test_card_field_inventory_declares_transport_gaps_and_all_native_special_data_users():
    import re
    from pathlib import Path
    from bridge.schemas.codec_contract import complete_word_tables
    tables = complete_word_tables()
    assert set(tables['real_card_wire_fields']) == {
        'id', 'name', 'uuid', 'misc', 'is_playable', 'cost', 'upgrades',
        'type', 'rarity', 'has_target', 'exhausts', 'ethereal',
    }
    source = (Path(__file__).resolve().parents[2] / 'sim/sts_lightspeed/src/combat/CardInstance.cpp').read_text()
    body = source.split('bool CardInstance::usesSpecialData() const {', 1)[1].split('default:', 1)[0]
    assert set(re.findall(r'case CardId::(\w+):', body)) == set(tables['card_special_data_fields'])
    assert tables['real_card_wire_fields']['misc']['status'] == 'settled_root_restored'
    assert tables['real_card_wire_fields']['uuid']['status'] == 'settled_root_restored_selection_pending'
    assert tables['card_state_field_contracts']['base_block']['status'] == 'mutation_audit_pending'
    assert 'free_to_play_once' in tables['card_state_field_contracts']
    assert 'base_cost' in tables['card_state_field_contracts']


def test_all_declared_aliases_preserve_nonzero_tokens():
    from bridge.schemas.entity_ids import CARD_ALIASES, POTION_ALIASES
    for real_id, canonical in CARD_ALIASES.items():
        assert CARD_INDEX[_card(real_id, 0)["id"]] == CARD_INDEX[canonical] > 0
    for real_id, canonical in POTION_ALIASES.items():
        assert POTION_VOCAB_INDEX[_potion_id(real_id)] == POTION_VOCAB_INDEX[canonical] > 0


def test_native_card_names_match_real_card_names():
    from bridge.schemas.combat_state import _sim_card_id, real_to_sim_card_enum
    for enum, raw in [('PANIC_BUTTON', 'PanicButton'), ('APPARITION', 'Ghostly'),
                      ('HAND_OF_GREED', 'HandOfGreed'), ('FLASH_OF_STEEL', 'Flash of Steel')]:
        canonical = _card(raw, 0)['id']
        assert _sim_card_id(enum) == canonical
        assert real_to_sim_card_enum(raw) == enum
        assert real_to_sim_card_enum(canonical) == enum


def test_entire_source_id_catalog_has_non_padding_identity():
    import re
    from pathlib import Path
    from env.core._gen_vocab import _string_table
    root = Path(__file__).resolve().parents[2] / 'sim/sts_lightspeed/include/constants'
    pairs = re.findall(r'\{CardId::(\w+), "([^"]+)"\}', (root/'SaveFileMappings.h').read_text())
    assert len(pairs) >= 370
    for enum, raw in pairs:
        if enum != 'INVALID':
            assert CARD_INDEX[_card(raw, 0)['id']] > 0, raw
    ids = _string_table((root/'Potions.h').read_text(), 'potionIds')[2:]
    assert len(ids) == 42
    assert all(POTION_VOCAB_INDEX[_potion_id(raw)] > 0 for raw in ids)


def test_shop_choices_resolve_identity_not_position():
    from bridge.real_game_env import RealGameAdapter
    from env.core import game_actions as ga
    a = RealGameAdapter()
    a.update({'game_state': {'screen_type': 'SHOP_SCREEN', 'gold': 100,
        'relics': [], 'potions': [],
        'choice_list': ['防御', '打击'], 'screen_state': {'cards': [
            {'id': 'Strike_R', 'name': '打击', 'price': 20},
            {'id': 'Defend_R', 'name': '防御', 'price': 20}],
            'relics': [], 'potions': [], 'purge_cost': 75}}})
    assert a._shop_choice_actions() == [(ga.shop_buy(ga.SHOP_CARD, 1), 0),
                                       (ga.shop_buy(ga.SHOP_CARD, 0), 1)]
    a.gs['choice_list'] = ['打击']
    with pytest.raises(ValueError, match='coverage mismatch'):
        a._shop_choice_actions()


def test_reward_claim_not_acknowledged_by_unrelated_screen_change():
    from copy import deepcopy
    from bridge.real_game_env import RealDecision
    from bridge.client.agent_orchestrator import RewardCommandAck
    frame = {'game_state': {'screen_type': 'COMBAT_REWARD', 'gold': 10,
        'potions': [{'id': 'Potion Slot'}], 'screen_state': {'rewards': [
            {'reward_type': 'POTION', 'potion': {'id': 'SteroidPotion'}}]}}}
    ack = RewardCommandAck()
    ack.arm(frame, RealDecision('CHOOSE 0', 'policy', semantic=('claim', 'potion', 0)), 0)
    changed = deepcopy(frame)
    changed['game_state']['screen_type'] = 'MAP'
    changed['game_state']['gold'] = 100
    assert ack.waiting(changed, 1)
    changed['game_state']['potions'] = [{'id': 'SteroidPotion'}]
    assert not ack.waiting(changed, 2)


def test_circlet_reward_acknowledges_exact_copy_counter_not_another_relic():
    from copy import deepcopy
    from bridge.real_game_env import RealDecision
    from bridge.client.agent_orchestrator import RewardCommandAck
    frame = {'game_state': {'screen_type': 'COMBAT_REWARD', 'relics': [
        {'id': 'Circlet', 'counter': 2}, {'id': 'Girya', 'counter': 0}],
        'screen_state': {'rewards': [
            {'reward_type': 'RELIC', 'relic': {'id': 'Circlet'}}]}}}
    ack = RewardCommandAck()
    ack.arm(frame, RealDecision('CHOOSE 0', 'policy', semantic=('claim', 'relic', 0)), 0)
    after = deepcopy(frame)
    after['game_state']['relics'][1]['counter'] = 1
    assert ack.waiting(after, 1)
    after['game_state']['relics'][0]['counter'] = 4
    assert ack.waiting(after, 2)
    after['game_state']['relics'][0]['counter'] = 3
    assert not ack.waiting(after, 3)
