from __future__ import annotations

import copy

import pytest
import torch

from env.core.cards_vocab import CARD_INDEX
from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    StateVectorConfig,
    HandCompositionConfig,
    HandCompositionEncoder,
    hand_composition_feature_dim,
    hand_inputs_from_state,
    hand_composition_delta,
    summarize_hand,
    VectorOracle,
    VectorQuery,
    VectorResponse,
    VectorValueBinder,
    VectorValueBinding,
    VectorValueRequest,
    VectorValueStore,
    VectorTranscriptBuilder,
    default_vector_model_config,
    relic_effect_tokens,
    serialize_havoc_skeleton,
    serialize_sword_boomerang,
    serialize_armaments_hand_operation,
    serialize_burning_pact_hand_operation,
    serialize_burning_pact_operations,
    vector_delta,
    vectorize_state,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
    VECTOR_TOKEN_INDEX,
    validate_vector_token_limits,
    EventDecoderOutput,
    _calculation_number,
)
from rl.render_rule_vector_examples import render_examples


def _state():
    return {
        "combat": {
            "turn": 1,
            "player": {
                "hp": 80, "max_hp": 80, "block": 3, "energy": 3, "gold": 99,
                "powers": {"Strength": 2},
                "combat_counters": {"cards_played": 1},
            },
            "monsters": [{
                "id": "Cultist", "hp": 50, "max_hp": 50, "block": 4,
                "base_damage": 6, "adjusted_damage": 8, "hits": 1,
                "gone": False, "powers": {"Vulnerable": 2},
            }],
            "hand": [{"id": "Strike_R", "up": 0}],
            "draw": [{"id": "Defend_R", "up": 0}, {"id": "Strike_R", "up": 0}],
            "discard": [],
            "exhaust": [],
        },
        "global": {
            "relics": ["Burning Blood"], "relic_counters": [2],
            "deck": [("Strike_R", 0), ("Defend_R", 0), ("Strike_R", 0)],
        },
    }


def test_discrete_numbers_are_serialized_least_significant_digit_first():
    assert _calculation_number(50) == [
        "NUM_LSD_BEGIN", "DIGIT_0", "DIGIT_5", "NUM_END",
    ]
    assert _calculation_number(-123) == [
        "NUM_LSD_BEGIN", "SIGN_NEG", "DIGIT_3", "DIGIT_2", "DIGIT_1",
        "NUM_END",
    ]
    assert _calculation_number(40.0) == [
        "NUM_LSD_BEGIN", "DIGIT_0", "DIGIT_4", "NUM_END",
    ]


def test_card_and_relic_program_inputs_are_name_free():
    cfg = default_vector_model_config(
        card_dim=12, relic_dim=12, card_token_dim=12, relic_token_dim=12,
        card_heads=3, relic_heads=3, world_dim=24, decoder_dim=24,
        world_heads=4, decoder_heads=4, world_feedforward=48,
        decoder_feedforward=48, delta_hidden=24,
    )
    model = VectorRuleWorldModel(cfg)
    assert isinstance(model.world_encoder.encoder, torch.nn.TransformerEncoder)
    assert isinstance(model.decoder.decoder, torch.nn.TransformerDecoder)
    card_id = torch.tensor([CARD_INDEX["Strike_R"]])
    vector = model.card_encoder.card_vectors(card_id)
    program = model.card_encoder.version_program_ids[card_id, torch.tensor([0])]
    direct = model.card_encoder.forward_programs(program)
    assert torch.allclose(vector, direct)
    assert not hasattr(model.card_encoder, "id_embedding")
    assert "Burning" not in relic_effect_tokens("Burning Blood")
    assert relic_effect_tokens("Sundial") == (
        "Every", "3", "times", "the", "draw", "pile", "is", "shuffled",
        ",", "gain", "2", "Energy", ".",
    )
    assert len(CARD_RULE_VOCAB) > 0 and len(RELIC_RULE_VOCAB) > 0


def test_state_and_delta_use_fixed_entity_layout():
    before = _state()
    after = copy.deepcopy(before)
    after["combat"]["player"]["energy"] = 2
    after["combat"]["player"]["powers"]["Strength"] = 3
    after["combat"]["monsters"][0]["hp"] = 44
    after["combat"]["monsters"][0]["block"] = 0
    after["combat"]["hand"] = []
    after["combat"]["draw"] = [{"id": "Strike_R", "up": 0}]
    after["combat"]["discard"] = [{"id": "Strike_R", "up": 0}]
    config = StateVectorConfig(max_enemies=3)
    left = vectorize_state(before, config)
    delta = vector_delta(before, after, config)
    assert left.player.shape == (config.player_dim,)
    assert left.enemies.shape == (3, config.enemy_dim)
    assert delta.flat.shape == (config.world_dim,)
    energy_index = config.player_scalar_names.index("energy")
    assert delta.player[energy_index].item() == pytest.approx(-0.1)
    enemy_hp_index = config.enemy_scalar_names.index("hp")
    assert delta.enemies[0, enemy_hp_index].item() == pytest.approx(-0.03)
    assert set(config.pile_names) == {"draw_count", "discard_count", "exhaust_count"}
    assert delta.piles[config.pile_names.index("draw_count")].item() == pytest.approx(-0.01)
    assert delta.piles[config.pile_names.index("discard_count")].item() == pytest.approx(0.01)
    assert left.player.shape == (config.player_dim,)
    relic_start = (
        len(config.player_scalar_names)
        + len(config.player_buff_names)
        + len(config.counter_names)
        + len(config.relic_summary_names)
    )
    burning_blood_index = relic_start + config.relic_presence_names.index("Burning Blood")
    assert left.player[burning_blood_index].item() == pytest.approx(1.0)
    assert left.player[relic_start + config.relic_presence_names.index("Sundial")].item() == pytest.approx(0.0)


def test_count_only_pile_contract_and_rendered_examples():
    config = StateVectorConfig()
    assert config.pile_names == ("draw_count", "discard_count", "exhaust_count")
    assert config.pile_dim == 3
    with pytest.raises(ValueError, match="draw_count, discard_count, exhaust_count"):
        vectorize_state(_state(), StateVectorConfig(pile_names=("hand_count",)))
    rendered = render_examples()
    assert "EVENT_PILE_SHUFFLE" in rendered
    assert "EVENT_COUNTER" in rendered
    assert "<CALCULATIONS_BEGIN>" in rendered
    assert "CALCULATION_READ_BASE_DAMAGE" in rendered
    assert "CALCULATION_FLOOR_DIVIDE_VULNERABLE" in rendered
    assert "SOURCE_RULE" not in rendered
    assert "piles.draw_count=+0.030000; piles.discard_count=-0.030000" in rendered
    assert "piles.draw_count=-0.010000" in rendered
    assert "hand_count" not in rendered
    assert "deck_count" not in rendered
    assert "AnonymousCard" not in rendered
    assert "示例一：飞身踢" in rendered
    assert "示例二：剑柄打击" in rendered
    lines = rendered.splitlines()
    calculation_line = next(
        index for index, line in enumerate(lines)
        if line.strip() == "<CALCULATIONS_BEGIN>"
    )
    assert lines[calculation_line].startswith("      <CALCULATIONS_BEGIN>")
    assert lines[calculation_line + 1].startswith("        CALCULATION_BEGIN")
    assert lines[calculation_line + 2].startswith("          CALCULATION_APPLY_DELTA")
    for line in rendered.splitlines():
        if line.strip().startswith("DELTA_SLOT"):
            assert line.count("DELTA_SLOT") == 1


def test_pile_card_identity_changes_have_no_vector_delta():
    before = _state()
    after = copy.deepcopy(before)
    after["combat"]["draw"][0]["id"] = "Bash"
    after["combat"]["discard"][0:0] = [{"id": "Strike_R", "up": 1}]
    after["combat"]["discard"].pop(0)
    delta = vector_delta(before, after)
    assert torch.count_nonzero(delta.piles) == 0
    assert torch.count_nonzero(delta.player) == 0
    assert torch.count_nonzero(delta.enemies) == 0


def test_sword_boomerang_has_one_query_per_hit_and_delta_slot():
    delta = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    serialized = serialize_sword_boomerang(3, delta, target_slots=[0, 1, 0])
    assert serialized.tokens.count("QUERY_RANDOM_TARGET") == 3
    assert serialized.tokens.count("DELTA_SLOT") == 3
    assert serialized.tokens.count("LOOP_COUNT_3") == 1
    assert serialized.delta_targets.count(tuple(delta[1])) == 1
    assert "SOURCE_RULE" not in serialized.tokens


def test_havoc_skeleton_is_nested_and_cost_free():
    tokens = serialize_havoc_skeleton()
    assert tokens.index("PLAY_TOP_DRAW_CARD") < tokens.index("QUERY_TOP_DRAW_CARD")
    assert "COST_MODE_IGNORE" in tokens
    assert tokens.count("SCOPE_BEGIN") == 2
    assert "OBS_CARD_INSTANCE" in tokens
    validate_vector_token_limits(tokens, max_events=0, max_queries=1, max_nesting_depth=2)


def test_hand_composition_is_anonymous_and_upgrade_sensitive():
    cfg = default_vector_model_config(
        card_dim=12, relic_dim=12, hand_dim=10, card_token_dim=12,
        relic_token_dim=12, card_heads=3, relic_heads=3, world_dim=24,
        decoder_dim=24, world_heads=4, decoder_heads=4, world_feedforward=48,
        decoder_feedforward=48, delta_hidden=24, hand_hidden=16,
    )
    model = VectorRuleWorldModel(cfg)
    vectors = torch.randn(1, 3, cfg.card_encoder.output_dim)
    base = summarize_hand(vectors, torch.zeros(1, 3))
    upgraded = summarize_hand(vectors, torch.tensor([[0.0, 1.0, 0.0]]))
    assert base.shape == (1, hand_composition_feature_dim(cfg.card_encoder.output_dim))
    assert not torch.equal(base, upgraded)
    encoded = model.hand_encoder(vectors, torch.tensor([[0.0, 1.0, 0.0]]))
    assert encoded.shape == (1, cfg.hand_encoder.output_dim)
    assert isinstance(model.hand_encoder.encoder, torch.nn.TransformerEncoder)
    model.hand_encoder.eval()
    permuted = model.hand_encoder(
        vectors[:, [2, 0, 1]], torch.tensor([[0.0, 0.0, 1.0]])
    )
    assert torch.allclose(encoded, permuted, atol=1e-5)
    empty = model.hand_encoder(torch.empty(1, 0, cfg.card_encoder.output_dim))
    assert empty.shape == (1, cfg.hand_encoder.output_dim)
    state_hand, upgrades = hand_inputs_from_state(_state(), model.card_encoder)
    assert state_hand.shape == (1, 1, cfg.card_encoder.output_dim)
    assert upgrades.shape == (1, 1)
    after = copy.deepcopy(_state())
    after["combat"]["hand"][0]["up"] = 1
    assert torch.count_nonzero(
        hand_composition_delta(_state(), after, model.card_encoder)
    ) > 0
    transformed_delta = hand_composition_delta(
        _state(), after, model.card_encoder, hand_encoder=model.hand_encoder
    )
    assert transformed_delta.shape == (cfg.hand_encoder.output_dim,)
    memory, _ = model.encode_world(
        vectorize_state(_state(), cfg.state),
        hand_composition=encoded,
    )
    assert memory.shape[0] == 1


def test_hand_operation_serialization_carries_delta_and_selection():
    hand_delta = (0.1, -0.2, 0.0)
    armaments = serialize_armaments_hand_operation(False, hand_delta)
    upgraded = serialize_armaments_hand_operation(True, hand_delta)
    pact = serialize_burning_pact_hand_operation(hand_delta)
    assert "QUERY_CHOSEN_HAND_CARD" in armaments.tokens
    assert "QUERY_CHOSEN_HAND_CARD" not in upgraded.tokens
    assert "HAND_EXHAUST" in pact.tokens
    assert pact.hand_delta_targets[pact.tokens.index("HAND_DELTA_SLOT")] == hand_delta
    full_pact = serialize_burning_pact_operations(
        hand_delta,
        (-0.1, 0.2, 0.0),
        draw_query_kinds=("QUERY_RANDOM_CARD", "QUERY_RANDOM_CARD"),
    )
    assert full_pact.tokens.count("EVENT_HAND_COMPOSITION") == 2
    assert full_pact.tokens.index("HAND_EXHAUST") < full_pact.tokens.index("HAND_DRAW")
    assert full_pact.tokens.count("QUERY_CHOSEN_HAND_CARD") == 1
    assert full_pact.tokens.count("QUERY_RANDOM_CARD") == 2


def test_transcript_oracle_marks_environment_responses():
    builder = VectorTranscriptBuilder()
    response = VectorResponse(
        query_id="q0", kind="QUERY_RANDOM_TARGET",
        response_token="ENTITY_ENEMY_SLOT_1", vector=(1.0, 2.0),
    )
    builder.exchange(VectorQuery("q0", "QUERY_RANDOM_TARGET"), response)
    transcript = builder.finish()
    assert transcript.tokens[0] == "<OUTPUT_BEGIN>"
    assert transcript.tokens[-1] == "<OUTPUT_END>"
    assert transcript.owners[transcript.tokens.index("<OBS_BEGIN>")] == 0
    oracle = VectorOracle([response])
    assert oracle.respond(VectorQuery("q0", "QUERY_RANDOM_TARGET")) == response
    assert oracle.complete
    inputs, labels = transcript.teacher_labels(VECTOR_TOKEN_INDEX)
    assert len(inputs) == len(labels)
    assert -100 in labels


def test_vector_model_forward_and_config_roundtrip(tmp_path):
    cfg = default_vector_model_config(
        card_dim=12, relic_dim=12, card_token_dim=12, relic_token_dim=12,
        card_heads=3, relic_heads=3, world_dim=24, decoder_dim=24,
        world_heads=4, decoder_heads=4, world_feedforward=48,
        decoder_feedforward=48, delta_hidden=24, decoder_max_tokens=16,
        response_dim=4,
    )
    model = VectorRuleWorldModel(cfg)
    state = vectorize_state(_state(), cfg.state)
    target = torch.tensor([[VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"]]])
    output = model(state, target)
    assert output.logits.shape == (1, 1, len(VECTOR_TOKEN_INDEX))
    assert output.delta.shape == (1, 1, cfg.state.world_dim)
    assert output.hand_delta.shape == (1, 1, 0)
    action_vector = torch.randn(1, cfg.card_encoder.output_dim)
    upgraded_output = model(
        state,
        target,
        action_card_vector=action_vector,
        action_upgrade=torch.tensor([1.0]),
    )
    assert upgraded_output.logits.shape == output.logits.shape
    path = tmp_path / "vector-model.json"
    cfg.save(str(path))
    restored = VectorRuleModelConfig.load(str(path))
    assert restored.decoder.d_model == cfg.decoder.d_model
    assert restored.state.world_dim == cfg.state.world_dim
    assert restored.hand_encoder.n_heads == cfg.hand_encoder.n_heads
    assert restored.hand_encoder.feedforward == cfg.hand_encoder.feedforward
    assert restored.world_encoder.action_upgrade_scale == cfg.world_encoder.action_upgrade_scale


def test_interactive_generation_injects_oracle_response():
    cfg = default_vector_model_config(
        card_dim=12, relic_dim=12, hand_dim=10, card_token_dim=12,
        relic_token_dim=12, card_heads=3, relic_heads=3, world_dim=24,
        decoder_dim=24, world_heads=4, decoder_heads=4, world_feedforward=48,
        decoder_feedforward=48, delta_hidden=24, response_dim=4,
        decoder_max_tokens=16,
    )
    model = VectorRuleWorldModel(cfg)

    class FakeDecoder(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.calls = 0

        def forward(self, target_input_ids, memory, memory_padding, *, response_vectors=None):
            batch, length = target_input_ids.shape
            logits = torch.full(
                (batch, length, self.config.token_vocab_size), -100.0,
                device=target_input_ids.device,
            )
            token = (
                VECTOR_TOKEN_INDEX["<QUERY_BEGIN>"]
                if self.calls == 0
                else VECTOR_TOKEN_INDEX["QUERY_RANDOM_TARGET"]
                if self.calls == 1
                else VECTOR_TOKEN_INDEX["<QUERY_END>"]
                if self.calls == 2
                else VECTOR_TOKEN_INDEX["<OUTPUT_END>"]
            )
            logits[:, -1, token] = 100.0
            self.calls += 1
            hidden = torch.zeros(batch, length, self.config.d_model, device=memory.device)
            delta = torch.zeros(batch, length, cfg.state.world_dim, device=memory.device)
            hand_delta = torch.zeros(batch, length, self.config.hand_delta_dim, device=memory.device)
            return EventDecoderOutput(logits, hidden, delta, hand_delta)

    model.decoder = FakeDecoder(model.decoder.config)
    response = VectorResponse(
        query_id="q0", kind="QUERY_RANDOM_TARGET",
        response_token="ENTITY_ENEMY_SLOT_0", vector=(0.0, 0.0, 0.0, 0.0),
        observation_token="OBS_RANDOM_TARGET",
    )
    transcript, deltas = model.generate_interactive(
        vectorize_state(_state(), cfg.state),
        output_begin_id=VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"],
        output_end_id=VECTOR_TOKEN_INDEX["<OUTPUT_END>"],
        token_index=VECTOR_TOKEN_INDEX,
        oracle=VectorOracle([response]),
        max_new_tokens=5,
    )
    assert "QUERY_RANDOM_TARGET" in transcript.tokens
    assert transcript.tokens[-1] == "<OUTPUT_END>"
    assert transcript.responses == (response,)
    obs = transcript.tokens.index("<OBS_BEGIN>")
    assert transcript.tokens[obs:obs + 4] == (
        "<OBS_BEGIN>", "OBS_RANDOM_TARGET", "ENTITY_ENEMY_SLOT_0", "<OBS_END>",
    )
    assert transcript.owners[transcript.tokens.index("<QUERY_END>")] == 1
    assert transcript.owners[transcript.tokens.index("<OBS_BEGIN>")] == 0
    assert len(deltas) == 4
    model.decoder.calls = 0
    transcript2, deltas2, hand_deltas = model.generate_interactive(
        vectorize_state(_state(), cfg.state),
        output_begin_id=VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"],
        output_end_id=VECTOR_TOKEN_INDEX["<OUTPUT_END>"],
        token_index=VECTOR_TOKEN_INDEX,
        oracle=VectorOracle([response]),
        max_new_tokens=5,
        return_hand_deltas=True,
    )
    assert transcript2.tokens == transcript.tokens
    assert len(deltas2) == len(hand_deltas) == 4


def test_interactive_generation_binds_reads_and_inlines_exact_values():
    cfg = default_vector_model_config(
        card_dim=12, relic_dim=12, hand_dim=10, card_token_dim=12,
        relic_token_dim=12, card_heads=3, relic_heads=3, world_dim=24,
        decoder_dim=24, world_heads=4, decoder_heads=4, world_feedforward=48,
        decoder_feedforward=48, delta_hidden=24, decoder_max_tokens=96,
    )
    model = VectorRuleWorldModel(cfg)
    predicted_tokens = (
        "<READ_REQUEST_BEGIN>", "READ_SOURCE", "VALUE_player_hp",
        "<CALCULATIONS_BEGIN>",
        "CALCULATION_BEGIN", "CALCULATION_SUBTRACT",
        "CALC_FIELD_left", "VALUE_reg_player_hp",
        "CALC_FIELD_right", "VALUE_reg_player_hp",
        "CALC_FIELD_result", "VALUE_reg_player_hp_after_subtract",
        "CALC_VALUE", "NUM_LSD_BEGIN", "DIGIT_0", "NUM_END",
        "CALCULATION_END",
        "CALCULATION_BEGIN", "CALCULATION_RELU",
        "CALC_FIELD_input", "VALUE_reg_player_hp_after_subtract",
        "CALC_FIELD_result", "VALUE_reg_player_hp_after",
        "CALC_VALUE", "NUM_LSD_BEGIN", "DIGIT_0", "NUM_END",
        "CALCULATION_END", "<CALCULATIONS_END>", "DELTA_SLOT",
        "<REGION_END>",
    )

    class FakeDecoder(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.calls = 0

        def forward(
            self, target_input_ids, memory, memory_padding,
            *, response_vectors=None,
        ):
            batch, length = target_input_ids.shape
            logits = torch.full(
                (batch, length, self.config.token_vocab_size), -100.0,
                device=target_input_ids.device,
            )
            token = predicted_tokens[self.calls]
            logits[:, -1, VECTOR_TOKEN_INDEX[token]] = 100.0
            self.calls += 1
            hidden = torch.zeros(
                batch, length, self.config.d_model, device=memory.device,
            )
            delta = torch.zeros(
                batch, length, cfg.state.world_dim, device=memory.device,
            )
            hand_delta = torch.zeros(
                batch, length, self.config.hand_delta_dim, device=memory.device,
            )
            return EventDecoderOutput(logits, hidden, delta, hand_delta)

    model.decoder = FakeDecoder(model.decoder.config)
    binding = VectorValueBinding(
        request_id="r0", source_token="VALUE_player_hp",
        binding_token="VALUE_reg_player_hp", value=50,
    )
    value_store = VectorValueStore(
        {"VALUE_player_hp": 50},
        {"VALUE_player_hp": "VALUE_reg_player_hp"},
    )
    transcript, deltas = model.generate_interactive(
        vectorize_state(_state(), cfg.state),
        output_begin_id=VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"],
        output_end_id=VECTOR_TOKEN_INDEX["<REGION_END>"],
        token_index=VECTOR_TOKEN_INDEX,
        value_binder=value_store,
        prefix_ids=(
            VECTOR_TOKEN_INDEX["<REGION_BEGIN>"],
            VECTOR_TOKEN_INDEX["REGION_KIND"],
            VECTOR_TOKEN_INDEX["REGION_SELF_DAMAGE"],
            VECTOR_TOKEN_INDEX["<NODE_EXECUTE>"],
        ),
        max_new_tokens=len(predicted_tokens),
    )

    assert transcript.value_bindings == (binding,)
    source = transcript.tokens.index("VALUE_player_hp")
    assert transcript.tokens[source + 1:source + 4] == (
        "READ_BIND", "VALUE_reg_player_hp", "<READ_REQUEST_END>",
    )
    constant = (
        "CONST_VALUE", "NUM_LSD_BEGIN", "DIGIT_0", "DIGIT_5", "NUM_END",
    )
    assert sum(
        transcript.tokens[index:index + len(constant)] == constant
        for index in range(len(transcript.tokens) - len(constant) + 1)
    ) == 2
    derived_constant = (
        "VALUE_reg_player_hp_after_subtract", "CONST_VALUE",
        "NUM_LSD_BEGIN", "DIGIT_0", "NUM_END",
    )
    assert sum(
        transcript.tokens[index:index + len(derived_constant)] == derived_constant
        for index in range(len(transcript.tokens) - len(derived_constant) + 1)
    ) == 1
    assert transcript.tokens[:4] == (
        "<REGION_BEGIN>", "REGION_KIND", "REGION_SELF_DAMAGE",
        "<NODE_EXECUTE>",
    )
    assert all(owner == 0 for owner in transcript.owners[:4])
    result = transcript.tokens.index("CALC_FIELD_result")
    assert transcript.tokens[result:result + 3] == (
        "CALC_FIELD_result", "VALUE_reg_player_hp_after_subtract", "CALC_VALUE",
    )
    assert all(
        transcript.owners[index] == 0
        for index, token in enumerate(transcript.tokens)
        if token in {"READ_BIND", "CONST_VALUE"}
    )
    assert len(deltas) == len(predicted_tokens)
    assert len(transcript.outputs_for_model_token(deltas, "DELTA_SLOT")) == 1


def test_value_binder_validates_order_and_value_store_reads_latest_value():
    first = VectorValueBinding(
        "r0", "VALUE_player_hp", "VALUE_reg_player_hp", 50,
    )
    binder = VectorValueBinder([first])
    assert binder.bind(VectorValueRequest("r0", "VALUE_player_hp")) == first
    assert binder.complete

    store = VectorValueStore(
        {"VALUE_player_hp": 50},
        {"VALUE_player_hp": "VALUE_reg_player_hp"},
    )
    store.write("VALUE_player_hp", 42)
    second = store.bind(VectorValueRequest("r1", "VALUE_player_hp"))
    assert second == VectorValueBinding(
        "r1", "VALUE_player_hp", "VALUE_reg_player_hp", 42,
    )

    live = VectorValueStore.from_state(
        _state(),
        {
            "player.hp": "reg.player_hp",
            "enemy[0].block": "reg.target_block",
            "relic[0].counter": "reg.relic_counter",
            "action.paid_energy": "reg.repeat_count",
        },
        extra_values={"action.paid_energy": 3},
    )
    assert live.bind(VectorValueRequest("r2", "VALUE_player_hp")).value == 80
    assert live.bind(VectorValueRequest("r3", "VALUE_enemy_0_block")).value == 4
    assert live.bind(VectorValueRequest("r4", "VALUE_relic_0_counter")).value == 2
    assert live.bind(VectorValueRequest("r5", "VALUE_action_paid_energy")).value == 3
