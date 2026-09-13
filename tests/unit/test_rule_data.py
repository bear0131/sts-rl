from __future__ import annotations

import copy

import torch

from env.core.rule_language import (
    deck_rule_sentence,
    relic_loadout_sentence,
    validate_first_version_language,
)
from rl.rules.rule_data import (
    CALCULATIONS_BEGIN,
    CRITICAL_CALCULATION,
    CRITICAL_COUNTER,
    NEXT_STATE_BEGIN,
    RuleTokenizer,
    canonicalize_training_tokenizer,
    critical_target_number_mask,
    critical_target_number_regions,
    controlled_attack_calculations,
    derive_local_calculations,
    events_from_snapshots,
    sample_tokens,
    target_supervision_regions,
    transition_consistency_specs,
)
from rl.rules.rule_model import RuleModelConfig, RuleWorldModel
from rl.rules.training.train_rules import (
    _loss_and_counts,
    _split_samples,
    _transition_consistency_loss,
)


def _state(*, hp: int, monster_hp: int, hand=None, discard=None):
    return {
        "combat": {
            "turn": 1,
            "player": {
                "hp": hp,
                "max_hp": 80,
                "block": 0,
                "energy": 3,
                "gold": 99,
                "powers": {},
                "combat_counters": {"pen_nib": 0},
            },
            "monsters": [{
                "id": "Cultist", "hp": monster_hp, "max_hp": 50,
                "block": 0, "gone": False, "base_damage": 6,
                "adjusted_damage": 6, "hits": 1,
                "move": "CULTIST_ATTACK", "powers": {},
            }],
            "hand": hand or [],
            "draw": [],
            "discard": discard or [],
            "exhaust": [],
        },
        "global": {
            "ascension": 0, "act": 1, "floor": 1,
            "relics": ["Burning Blood", "Vajra"],
            "relic_counters": [0, 0],
            "deck": [("Strike_R", 0)],
        },
    }


def _sample():
    card = {"id": "Strike_R", "up": 0, "cost": 1}
    before = _state(hp=80, monster_hp=50, hand=[card])
    after = _state(
        hp=80,
        monster_hp=44,
        discard=[{"id": "Strike_R", "up": 0}],
    )
    return {
        "state": before,
        "action": {"type": "CARD", "source": 0, "target": 0, "card": card},
        "rules": {
            "deck": deck_rule_sentence(before["global"]["deck"]),
            "relics": relic_loadout_sentence(before["global"]["relics"]),
        },
        "effects": events_from_snapshots([before["combat"], after["combat"]]),
        "next_state": after,
    }


def test_rule_language_covers_first_version_random_loadout():
    validate_first_version_language(
        {"Strike_R", "Twin Strike", "Wound"},
        {"Burning Blood", "Vajra", "Pen Nib"},
    )


def test_sample_sequence_contains_materialized_rules_and_ordered_target():
    prefix, target = sample_tokens(_sample())
    assert "RULE_CONTEXT_BEGIN" in prefix
    assert "WORD_Vajra" in prefix
    assert "CARD_Strike_R" in prefix
    assert "WORD_Strike_R" not in prefix
    assert "VALUE_Strike_R" not in prefix + target
    assert "<RULE_STEPS_BEGIN>" not in target
    assert target.index("<EFFECTS_BEGIN>") < target.index(NEXT_STATE_BEGIN)
    assert "EFFECT_MONSTER_VALUE_CHANGE" in target


def test_legacy_v2_sequence_remains_available_for_checkpoint_diagnostics():
    prefix, target = sample_tokens(_sample(), output_version=2)
    assert "WORD_Strike_R" in prefix
    assert "CARD_Strike_R" in prefix
    assert "<RULE_STEPS_BEGIN>" in target
    assert target.count(CALCULATIONS_BEGIN) == 1
    assert "VALUE_Strike_R" in target


def test_legacy_v3_sequence_keeps_verbose_event_bookkeeping():
    _prefix, target = sample_tokens(_sample(), output_version=3)
    assert "FIELD_atomic" in target
    assert "STEP_INDEX" in target
    assert "STEP_RELATION" in target
    assert "STEP_CAUSE_STEP" in target
    assert "STEP_EVENT" in target
    assert "EFFECT_ACTION_COMPLETE" in target
    assert target.count(CALCULATIONS_BEGIN) == len(_sample()["effects"])


def test_legacy_tokenizer_migration_removes_unused_card_aliases():
    sample = _sample()
    current = RuleTokenizer.build([sample])
    legacy = RuleTokenizer((
        *current.tokens, "WORD_Strike_R", "VALUE_Strike_R",
    ))
    migrated = canonicalize_training_tokenizer(legacy, [sample])
    assert "CARD_Strike_R" in migrated.index
    assert "WORD_Strike_R" not in migrated.index
    assert "VALUE_Strike_R" not in migrated.index


def test_multiword_card_name_has_one_canonical_token_identity():
    sample = _sample()
    card = {"id": "Body Slam", "up": 0, "cost": 1}
    sample["state"]["combat"]["hand"] = [card]
    sample["state"]["global"]["deck"] = [("Body Slam", 0)]
    sample["action"]["card"] = card
    sample["rules"]["deck"] = deck_rule_sentence([("Body Slam", 0)])
    prefix, target = sample_tokens(sample)
    assert "CARD_Body_Slam" in prefix
    assert "WORD_Body" not in prefix
    assert "WORD_Slam" not in prefix
    assert "VALUE_Body_Slam" not in prefix + target


def _dropkick_sample():
    sample = _sample()
    card = {"id": "Dropkick", "up": 0, "cost": 1}
    before = _state(hp=80, monster_hp=52, hand=[card])
    before["combat"]["player"]["powers"] = {"Strength": 2}
    before["combat"]["monsters"][0]["block"] = 3
    before["combat"]["monsters"][0]["powers"] = {"Vulnerable": 2}
    before["global"]["deck"] = [("Dropkick", 0)]
    after = copy.deepcopy(before)
    after["combat"]["monsters"][0]["hp"] = 45
    after["combat"]["monsters"][0]["block"] = 0
    after["combat"]["hand"] = []
    after["combat"]["discard"] = [{"id": "Dropkick", "up": 0}]
    sample.update({
        "state": before,
        "action": {"type": "CARD", "source": 0, "target": 0, "card": card},
        "rules": {
            "deck": deck_rule_sentence(before["global"]["deck"]),
            "relics": relic_loadout_sentence(before["global"]["relics"]),
        },
        "effects": events_from_snapshots([before["combat"], after["combat"]]),
        "next_state": after,
    })
    return sample


def test_dropkick_calculation_is_local_and_precedes_hp_outcome():
    sample = _dropkick_sample()
    local = derive_local_calculations(sample)
    hp_index = next(
        index for index, effect in enumerate(sample["effects"])
        if effect.get("type") == "MONSTER_VALUE_CHANGE"
        and effect.get("field") == "hp"
    )
    hp_calculations = local[hp_index]
    by_type = {calculation["type"]: calculation for calculation in hp_calculations}
    assert by_type["ADD_STRENGTH"]["result"] == 7
    assert by_type["MULTIPLY_VULNERABLE"]["result"] == 21
    assert by_type["FLOOR_DIVIDE_VULNERABLE"]["result"] == 10
    assert by_type["SUBTRACT_BLOCK_DAMAGE"]["result"] == 7
    assert by_type["SUBTRACT_HP"] == {
        "type": "SUBTRACT_HP", "left": 52, "right": 7, "result": 45,
    }

    _prefix, target = sample_tokens(sample)
    effect_starts = [
        position for position, token in enumerate(target)
        if token == "EFFECT_BEGIN"
    ]
    serialized_effects = [
        effect for effect in sample["effects"]
        if effect.get("type") != "ACTION_COMPLETE"
    ]
    assert len(effect_starts) == len(serialized_effects)
    assert target.count(CALCULATIONS_BEGIN) == sum(
        bool(calculations)
        for effect, calculations in zip(sample["effects"], local)
        if effect.get("type") != "ACTION_COMPLETE"
    )
    for removed in (
        "FIELD_atomic", "STEP_INDEX", "STEP_RELATION", "STEP_CAUSE_STEP",
        "STEP_EVENT", "EFFECT_ACTION_COMPLETE",
    ):
        assert removed not in target
    for effect_start in effect_starts:
        effect_end = target.index("EFFECT_END", effect_start)
        has_calculations = CALCULATIONS_BEGIN in target[effect_start:effect_end]
        outcome_positions = [
            target.index(field, effect_start, effect_end)
            for field in ("FIELD_before", "FIELD_delta", "FIELD_after")
            if field in target[effect_start:effect_end]
        ]
        if has_calculations:
            calculation_begin = target.index(
                CALCULATIONS_BEGIN, effect_start, effect_end
            )
            calculation_end = target.index(
                "<CALCULATIONS_END>", calculation_begin, effect_end
            )
            assert all(calculation_end < position for position in outcome_positions)

    hp_effect = next(
        start for start in effect_starts
        if "EFFECT_MONSTER_VALUE_CHANGE" in target[start:target.index("EFFECT_END", start)]
        and "VALUE_hp" in target[start:target.index("EFFECT_END", start)]
    )
    hp_effect_end = target.index("EFFECT_END", hp_effect)
    assert target.index("CALCULATION_SUBTRACT_HP", hp_effect, hp_effect_end) \
        < target.index("FIELD_before", hp_effect, hp_effect_end)


def test_critical_number_mask_marks_effect_and_next_monster_values():
    _prefix, target = sample_tokens(_sample())
    mask = critical_target_number_mask(target)
    marked = [token for token, selected in zip(target, mask) if selected]
    assert marked
    assert "SIGN_NEG" in marked
    assert all(token == "SIGN_NEG" or token.startswith("DIGIT_") for token in marked)


def test_validation_counts_calculation_blocks_and_canonical_card_names():
    labels = torch.tensor([[
        10, 11, 12, 13, 14, 10, 15, 12, 0,
    ]])
    predicted = labels.clone()
    predicted[0, 1] = 16
    predicted[0, 3] = 16
    logits = torch.full((1, labels.shape[1], 17), -10.0)
    logits.scatter_(2, predicted.unsqueeze(-1), 10.0)

    _loss, counts = _loss_and_counts(
        logits, labels, pad_id=0, next_state_id=14,
        calculation_begin_id=10, calculation_end_id=12,
        card_token_ids=torch.tensor([13]),
    )

    assert counts["calculation_tokens"] == 6
    assert counts["calculation_correct"] == 5
    assert counts["card_name_tokens"] == 1
    assert counts["card_name_correct"] == 0


def test_transition_consistency_specs_and_loss_align_with_next_tokens():
    sample = _sample()
    tokenizer = RuleTokenizer.build([sample])
    prefix, target = sample_tokens(sample)
    specs = transition_consistency_specs(sample, target)
    hp_spec = next(spec for spec in specs if spec["field"] == "hp")
    assert hp_spec["before"] == 50
    assert hp_spec["next"]["value"] == 44
    assert [span["value"] for span in hp_spec["deltas"]] == [-6]

    target_ids = tokenizer.encode(target)
    logits = torch.full((1, len(target) - 1, len(tokenizer.tokens)), -20.0)
    for target_position in range(1, len(target)):
        logits[0, target_position - 1, target_ids[target_position]] = 20.0
    logits.requires_grad_()
    digit_ids = torch.tensor([
        tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)
    ])
    loss, terms = _transition_consistency_loss(
        logits, [specs], digit_ids, scale=10.0
    )
    assert terms >= 2
    assert float(loss.detach()) < 1e-6
    loss.backward()
    assert logits.grad is not None


def test_critical_number_regions_separate_effect_and_next_state():
    _prefix, target = sample_tokens(_sample())
    regions = critical_target_number_regions(target)
    effect_regions = set(regions[:target.index(NEXT_STATE_BEGIN)])
    next_regions = set(regions[target.index(NEXT_STATE_BEGIN):])
    assert 1 in effect_regions
    assert 2 in next_regions


def test_counter_identity_has_dedicated_supervision_region():
    target = [
        "<EFFECTS_BEGIN>", "EFFECT_BEGIN", "EFFECT_COUNTER_CHANGE",
        "FIELD_counter", "VALUE_attacks_played", "EFFECT_END",
        "<EFFECTS_END>",
    ]
    regions = target_supervision_regions(target)
    positions = [
        index + 1
        for index, token in enumerate(target[:-1])
        if token == "FIELD_counter"
    ]
    assert positions
    assert all(regions[position] == CRITICAL_COUNTER for position in positions)
    assert all(target[position].startswith("VALUE_") for position in positions)


def test_tiny_rule_model_forward_and_latent():
    sample = _sample()
    tokenizer = RuleTokenizer.build([sample])
    prefix, target = sample_tokens(sample)
    prefix_ids = torch.tensor([tokenizer.encode(prefix)])
    target_ids = torch.tensor([tokenizer.encode(target)])
    model = RuleWorldModel(RuleModelConfig(
        vocab_size=len(tokenizer.tokens), pad_id=tokenizer.pad_id,
        d_model=16, n_heads=4, encoder_layers=1, decoder_layers=1,
        feedforward=32, max_prefix_tokens=len(prefix) + 1,
        max_target_tokens=len(target) + 1, dropout=0.0,
    ))
    logits, latent = model(prefix_ids, target_ids[:, :-1])
    assert logits.shape == (1, len(target) - 1, len(tokenizer.tokens))
    assert latent.shape == (1, 16)


def test_controlled_attack_expands_arithmetic_and_uses_lsd_first_numbers():
    sample = _sample()
    sample["metadata"] = {
        "experiment": "random_strength_enemy_status_strike_bash",
        "strength": 17,
        "card": "Strike_R",
        "enemy_block": 17,
        "enemy_vulnerable": 0,
        "calculated_attack_damage": 23,
        "expected_block_loss": 17,
        "expected_hp_loss": 6,
    }
    sample["calculations"] = controlled_attack_calculations(sample)
    prefix, target = sample_tokens(sample)
    begin = target.index(CALCULATIONS_BEGIN)
    add = target.index("CALCULATION_ADD_STRENGTH", begin)
    result = target.index("CALC_FIELD_result", add)
    assert target[result + 1:result + 5] == [
        "NUM_LSD_BEGIN", "DIGIT_3", "DIGIT_2", "NUM_END"
    ]
    regions = critical_target_number_regions(target)
    assert regions[result + 2] == CRITICAL_CALCULATION
    assert regions[result + 3] == CRITICAL_CALCULATION


def test_number_place_embedding_tracks_decimal_significance():
    sample = _sample()
    sample["metadata"] = {
        "experiment": "random_strength_enemy_status_strike_bash",
        "strength": 17,
        "card": "Strike_R",
        "enemy_block": 17,
        "enemy_vulnerable": 0,
        "calculated_attack_damage": 23,
        "expected_block_loss": 17,
        "expected_hp_loss": 6,
    }
    sample["calculations"] = controlled_attack_calculations(sample)
    tokenizer = RuleTokenizer.build([sample])
    prefix, target = sample_tokens(sample)
    config = RuleModelConfig(
        vocab_size=len(tokenizer.tokens), pad_id=tokenizer.pad_id,
        d_model=16, n_heads=4, encoder_layers=1, decoder_layers=1,
        feedforward=32, max_prefix_tokens=2048, max_target_tokens=2048,
        dropout=0.0, num_begin_id=tokenizer.index["NUM_BEGIN"],
        num_lsd_begin_id=tokenizer.index["NUM_LSD_BEGIN"],
        num_end_id=tokenizer.index["NUM_END"],
        digit_token_ids=tuple(
            tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)
        ),
    )
    model = RuleWorldModel(config)
    places = model._number_places(
        torch.tensor([tokenizer.encode(target)], dtype=torch.long),
        allow_future=False,
    )[0].tolist()
    add = target.index("CALCULATION_ADD_STRENGTH")
    result = target.index("CALC_FIELD_result", add)
    lsd_begin = result + 1
    assert places[lsd_begin + 1:lsd_begin + 3] == [1, 2]
    assert "NUM_BEGIN" not in target
    prefix_places = model._number_places(
        torch.tensor([tokenizer.encode(prefix)], dtype=torch.long),
        allow_future=True,
    )[0].tolist()
    normal_begin = next(
        index for index, token in enumerate(prefix)
        if token == "NUM_BEGIN" and prefix[index + 1:index + 3] == [
            "DIGIT_8", "DIGIT_0"
        ]
    )
    assert prefix_places[normal_begin + 1:normal_begin + 3] == [2, 1]

    # Decoder digit positions cannot change when future tokens are appended.
    partial = target[:lsd_begin + 3]
    partial_places = model._number_places(
        torch.tensor([tokenizer.encode(partial)], dtype=torch.long),
        allow_future=False,
    )[0].tolist()
    assert partial_places[lsd_begin + 1:lsd_begin + 3] == [1, 2]


def test_training_split_keeps_combats_disjoint():
    samples = []
    for combat_index in range(3):
        for action_index in range(2):
            samples.append({
                "metadata": {
                    "combat_index": combat_index,
                    "decision": 0,
                    "action_index": action_index,
                }
            })
    train, val, unit = _split_samples(samples, 0.34, seed=0)
    train_combats = {sample["metadata"]["combat_index"] for sample in train}
    val_combats = {sample["metadata"]["combat_index"] for sample in val}
    assert unit == "combat"
    assert train_combats
    assert val_combats
    assert train_combats.isdisjoint(val_combats)
