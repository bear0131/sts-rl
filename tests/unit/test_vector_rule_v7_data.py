from __future__ import annotations

import random

from rl.rules.data.make_vector_rule_v7_data import (
    V7_EXECUTOR_CONTRACT_VERSION,
    _action_name,
    build_v7_record,
    compile_v7_record,
)
from rl.rules.data.make_vector_rule_training_data import build_record
from rl.rules.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    StateVectorConfig,
    vectorize_state,
    vector_delta,
)
from rl.rules.vector_rule_v7_data import (
    V7TokenCodec,
    collate_vector_rule_v7_batch,
    examples_from_v7_record,
)


def test_v7_record_expands_to_outer_and_local_scopes():
    config = StateVectorConfig()
    record = build_v7_record(0, config, random.Random(7))
    assert record["executor_contract_version"] == V7_EXECUTOR_CONTRACT_VERSION
    codec = V7TokenCodec.from_records([record])
    examples = examples_from_v7_record(
        record, codec, config,
        card_vocab_size=len(CARD_RULE_VOCAB),
        relic_vocab_size=len(RELIC_RULE_VOCAB),
        max_cards=10, max_relics=32, max_tokens=2048,
    )
    assert examples[0].scope_kind == "OUTER_HEADERS"
    assert len(examples) == record["lengths"]["node_count"] + 1
    assert all(example.target_ids for example in examples)
    assert max(len(example.target_ids) for example in examples[1:]) <= 260
    assert record["lengths"]["constant_expansion_length_max"] >= 0
    assert record["lengths"]["constant_expansion_count_max"] >= 0
    assert record["lengths"]["calculation_length_max"] >= 1


def test_compile_v7_record_is_the_same_path_used_by_the_stream_builder():
    config = StateVectorConfig()
    expected = build_v7_record(5, config, random.Random(2718))
    v6 = build_record(5, config, random.Random(2718))
    assert compile_v7_record(v6, config) == expected


def test_v7_local_scope_conditions_on_outer_header_without_header_loss():
    config = StateVectorConfig()
    record = build_v7_record(1, config, random.Random(11))
    codec = V7TokenCodec.from_records([record])
    examples = examples_from_v7_record(record, codec, config)
    local = examples[1]
    tokens = codec.decode(local.target_ids)
    assert tokens[local.condition_length - 1] == "<NODE_EXECUTE>"
    assert "REGION_KIND" in tokens[:local.condition_length]
    assert "REGION_KIND" not in tokens[local.condition_length:]

    batch = collate_vector_rule_v7_batch([local], pad_id=codec.pad_id)
    label_start = local.condition_length - 1
    assert batch["labels"][0, :label_start].eq(codec.pad_id).all()
    assert batch["labels"][0, label_start].item() == local.target_ids[local.condition_length]


def test_v7_read_bind_is_context_without_prediction_loss():
    config = StateVectorConfig()
    record = build_v7_record(2, config, random.Random(19))
    codec = V7TokenCodec.from_records([record])
    local = next(
        example for example in examples_from_v7_record(record, codec, config)
        if "READ_BIND" in codec.decode(example.target_ids)
    )
    tokens = codec.decode(local.target_ids)
    batch = collate_vector_rule_v7_batch([local], pad_id=codec.pad_id)
    labels = batch["labels"][0]
    assert "<READ_RESPONSE_BEGIN>" not in tokens
    assert "<REGISTER_CONTEXT_BEGIN>" not in tokens
    assert "CONST_VALUE" in tokens
    # Binding names are assigned by the environment; READ_SOURCE remains the
    # only model-owned part of a runtime read request.
    for index, token in enumerate(tokens):
        if token == "READ_BIND" or token == "<READ_REQUEST_END>" or (
            index > 0 and tokens[index - 1] == "READ_BIND"
        ):
            assert not local.target_model_mask[index]


def test_v7_inline_constants_follow_all_valued_register_operands():
    config = StateVectorConfig()
    record = build_v7_record(2, config, random.Random(19))
    codec = V7TokenCodec.from_records([record])
    rows = [
        row for row in record["node_local_sequences"]
        if "CONST_VALUE" in row["tokens"]
    ]
    assert rows
    for row in rows:
        tokens = tuple(row["tokens"])
        mask = tuple(bool(value) for value in row["model_mask"])
        assert "<REGISTER_CONTEXT_BEGIN>" not in tokens
        assert "<READ_RESPONSE_BEGIN>" not in tokens
        calculation_begin = tokens.index("<CALCULATIONS_BEGIN>")
        valued = {
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == "READ_BIND"
        }
        result_fields = {
            "CALC_FIELD_result", "CALC_FIELD_source_result",
            "CALC_FIELD_destination_result",
        }
        for index, token in enumerate(tokens[:-1]):
            if token not in result_fields:
                continue
            calculation_end = tokens.index("CALCULATION_END", index + 1)
            if "CALC_VALUE" in tokens[index + 2:calculation_end]:
                valued.add(tokens[index + 1])
        for index, token in enumerate(tokens):
            if token != "CONST_VALUE":
                continue
            assert index > 0 and tokens[index - 1] in valued
            assert not mask[index]
            end = tokens.index("NUM_END", index + 1)
            number_tokens = list(range(index + 1, end + 1))
            assert number_tokens
            assert all(not mask[position] for position in number_tokens)
        for index, token in enumerate(tokens):
            if index <= calculation_begin or not token.startswith("VALUE_reg_") or token not in valued:
                continue
            field = tokens[index - 1][len("CALC_FIELD_"):] if (
                index > 0 and tokens[index - 1].startswith("CALC_FIELD_")
            ) else ""
            if field in {"result", "source_result", "destination_result"}:
                continue
            assert tokens[index + 1] == "CONST_VALUE"


def test_v7_relics_are_presence_only_and_not_rule_inputs():
    config = StateVectorConfig()
    record = build_v7_record(2, config, random.Random(19))
    assert all("rule" not in relic for relic in record["relics"])
    assert "relic_presence" in record["state"]
    assert len(record["state"]["relic_presence"]) == len(config.relic_presence_names)
    assert "relic_counters" not in record["state"]
    codec = V7TokenCodec.from_records([record])
    example = examples_from_v7_record(record, codec, config)[0]
    assert example.relic_programs == ()
    assert example.relic_counters == ()


def test_v7_multiple_independent_impacts_have_multiple_delta_slots():
    config = StateVectorConfig()

    def walk(node, path="root"):
        yield path, node
        for child_index, child in enumerate(node["children"]):
            yield from walk(child, f"{path}/{child_index}")

    for index in range(256):
        record = build_v7_record(index, config, random.Random(index))
        for row in record["node_local_sequences"]:
            if row["kind"] != "DAMAGE" or row["tokens"].count("DELTA_SLOT") < 2:
                continue
            node = row
            assert row["tokens"].count("DELTA_SLOT") == len(
                next(n for n in walk(record["effect_tree"]) if n[0] == row["path"])[1]["delta_slots"]
            )
            return
    raise AssertionError("bootstrap did not produce a multi-impact damage node")


def test_v7_inline_constant_keeps_operator_explicit():
    config = StateVectorConfig()
    rows = []
    rng = random.Random(77)
    for index in range(256):
        rows.extend(build_v7_record(index, config, rng)["node_local_sequences"])
    damage = next(row for row in rows if row["kind"] == "DAMAGE")
    tokens = damage["tokens"]
    calculation_begin = tokens.index("<CALCULATIONS_BEGIN>")
    operators = {
        token for token in tokens[calculation_begin:]
        if token.startswith("CALCULATION_")
        and token not in {"CALCULATIONS_BEGIN", "CALCULATIONS_END"}
    }
    assert "CALCULATION_SUBTRACT" in operators


def test_v7_node_specs_carry_executor_operation_parameters():
    config = StateVectorConfig()
    rng = random.Random(29)
    records = [build_v7_record(index, config, rng) for index in range(128)]
    headers_by_kind: dict[str, list[list[str]]] = {}
    for record in records:
        for row in record["node_local_sequences"]:
            condition = row["tokens"][:row["condition_length"]]
            headers_by_kind.setdefault(row["kind"], []).append(condition)

    required = {
        "CARD_PAYMENT": (),
        "DRAW": ("DRAW_COUNT",),
        "ENERGY": ("ENERGY_OP", "ENERGY_AMOUNT"),
        "CARD_DESTINATION": ("DESTINATION", "CARD_COUNT"),
        "STATE_CHANGE": ("FIELD_OR_STATUS", "CHANGE_AMOUNT"),
        "RELIC_COUNTER": (
            "COUNTER_REF", "COUNTER_EVENT", "COUNTER_INCREMENT",
            "COUNTER_THRESHOLD", "COUNTER_RESET", "TRIGGER_EFFECT",
            "TRIGGER_AMOUNT",
        ),
    }
    for kind, fields in required.items():
        assert kind in headers_by_kind
        assert all(all(field in header for field in fields) for header in headers_by_kind[kind])
    assert all(
        "PAYMENT_COST" in header or "PAYMENT_SOURCE" in header
        for header in headers_by_kind["CARD_PAYMENT"]
    )
    numeric_headers = [
        header for headers in headers_by_kind.values() for header in headers
        if any(token.startswith("NUM_") for token in header)
    ]
    assert numeric_headers
    assert all("VALUE_BINDING" not in header for header in numeric_headers)

    body_slam_rows = [
        row for record in records for row in record["node_local_sequences"]
        if row["kind"] == "DAMAGE"
        and "VALUE_PLAYER_BLOCK" in row["tokens"][:row["condition_length"]]
    ]
    assert body_slam_rows
    assert all(
        "BASE_DAMAGE" not in row["tokens"][:row["condition_length"]]
        and "VALUE_player_block" in row["tokens"][row["condition_length"]:]
        for row in body_slam_rows
    )

    repeat_rows = [
        row for record in records for row in record["node_local_sequences"]
        if row["kind"] == "REPEAT"
    ]
    call_rows = [
        row for record in records for row in record["node_local_sequences"]
        if row["kind"] == "CALL"
    ]
    assert repeat_rows and call_rows
    assert all(
        (
            "REPEAT_SOURCE" in row["tokens"][:row["condition_length"]]
            and "VALUE_action_paid_energy" in row["tokens"][row["condition_length"]:]
        )
        or (
            "REPEAT_COUNT" in row["tokens"][:row["condition_length"]]
            and "VALUE_action_paid_energy" not in row["tokens"][row["condition_length"]:]
        )
        for row in repeat_rows
    )
    assert all(
        "ARG_REMAINING" not in row["tokens"][:row["condition_length"]]
        and "VALUE_call_frame_remaining" in row["tokens"][row["condition_length"]:]
        for row in call_rows
    )


def test_v7_multihit_cards_use_one_recursive_repeat_body():
    config = StateVectorConfig()
    rng = random.Random(121)
    records = {}
    for index in range(2000):
        record = build_v7_record(index, config, rng)
        name = _action_name(record)
        if name in {"Twin Strike", "Pummel", "Sword Boomerang", "Whirlwind"}:
            records.setdefault(name, record)
        if len(records) == 4:
            break
    assert len(records) == 4

    def nodes(node):
        yield node
        for child in node["children"]:
            yield from nodes(child)

    for name, record in records.items():
        tree_nodes = list(nodes(record["effect_tree"]))
        repeats = [node for node in tree_nodes if node["kind"] == "REPEAT"]
        assert repeats
        assert all(
            sum(child["kind"] == "CALL" for child in repeat["children"]) == 1
            for repeat in repeats
        )
        if name == "Whirlwind":
            repeat = next(r for r in repeats if r["header_params"] == {"REPEAT_SOURCE": "PAID_ENERGY"})
            assert sum(node["kind"] == "AOE_DAMAGE_ROUND" for node in nodes(repeat)) == 1
        else:
            expected = {
                "Twin Strike": 2,
                "Pummel": 4 + int(record["action"]["upgrade"] > 0),
                "Sword Boomerang": 3 + int(record["action"]["upgrade"] > 0),
            }[name]
            repeat = next(r for r in repeats if r["header_params"] == {"REPEAT_COUNT": expected})
            assert sum(node["kind"] == "DAMAGE" for node in nodes(repeat)) == 1
        repeat_damage_nodes = [
            node for node in nodes(repeat) if node["kind"] == "DAMAGE"
        ]
        assert repeat_damage_nodes
        assert all(
            "READ_BASE_DAMAGE" in {
                row["type"] for row in damage["calculations"]
            }
            and not {
                "REUSE_ATTACK_DAMAGE", "COUNT_ATTACK_HITS",
            }.intersection(row["type"] for row in damage["calculations"])
            for damage in repeat_damage_nodes
        )

    sword = records["Sword Boomerang"]
    sword_nodes = list(nodes(sword["effect_tree"]))
    assert sum(node["kind"] == "RANDOM_HIT" for node in sword_nodes) == 1
    assert sum(node["kind"] == "QUERY" for node in sword_nodes) == 1
    sword_damage = next(node for node in sword_nodes if node["kind"] == "DAMAGE")
    assert sword_damage["entity_ref"] == "ENTITY_QUERY_TARGET"
    query_row = next(
        row for row in sword["node_local_sequences"] if row["kind"] == "QUERY"
    )
    assert "QUERY_RANDOM_TARGET" in query_row["tokens"]
    obs_begin = query_row["tokens"].index("<OBS_BEGIN>")
    obs_end = query_row["tokens"].index("<OBS_END>")
    assert all(not owned for owned in query_row["model_mask"][obs_begin:obs_end + 1])


def test_v7_local_scope_predicts_one_continuous_delta_per_realized_node_delta():
    config = StateVectorConfig()
    record = build_v7_record(2, config, random.Random(19))
    codec = V7TokenCodec.from_records([record])
    examples = examples_from_v7_record(record, codec, config)
    local_examples = examples[1:]
    assert any(delta is not None for example in local_examples for delta in example.target_delta)
    for example in local_examples:
        tokens = codec.decode(example.target_ids)
        target_positions = [
            index for index, delta in enumerate(example.target_delta)
            if delta is not None
        ]
        assert target_positions == [
            index for index, token in enumerate(tokens) if token == "DELTA_SLOT"
        ]
    batch = collate_vector_rule_v7_batch(local_examples, pad_id=codec.pad_id)
    assert int(batch["delta_mask"].sum().item()) == sum(
        "DELTA_SLOT" in codec.decode(example.target_ids)
        for example in local_examples
    )


def test_v7_training_length_limit_applies_to_outer_and_local_scopes_only():
    config = StateVectorConfig()
    record = build_v7_record(2, config, random.Random(19))
    codec = V7TokenCodec.from_records([record])
    longest_scope = max(
        len(record["tokens"]),
        *(row["length"] for row in record["node_local_sequences"]),
    )
    record["tokens"] = list(record["tokens"]) * 4
    examples = examples_from_v7_record(
        record, codec, config, max_tokens=longest_scope,
    )
    assert max(len(example.target_ids) for example in examples) <= longest_scope


def test_v7_model_calculations_read_runtime_numbers_but_keep_card_numbers_in_header(monkeypatch):
    # Guarantee a damage calculation instead of depending on a random power's
    # (previously incorrect) discard event to produce runtime reads.
    from rl.rules.data import make_vector_rule_training_data as teacher
    original = teacher._card_instance
    monkeypatch.setattr(teacher, "_card_instance", lambda rng, *, action=False:
                        {"id": "Bash", "up": 0} if action else original(rng))
    config = StateVectorConfig()
    record = build_v7_record(3, config, random.Random(23))
    assert "CALCULATION_BEGIN" in record["tokens"]
    assert "<READ_REQUEST_BEGIN>" in record["tokens"]
    assert "<READ_RESPONSE_BEGIN>" not in record["tokens"]
    assert "<REGISTER_CONTEXT_BEGIN>" not in record["tokens"]
    assert "CONST_VALUE" in record["tokens"]
    assert "DELTA_SLOT" in record["tokens"]
    assert "COMMIT_NODE_EFFECT" not in record["tokens"]
    saw_model_owned_result_number = False
    for row in record["node_local_sequences"]:
        condition = row["tokens"][:row["condition_length"]]
        body = row["tokens"][row["condition_length"]:]
        inside_constant = False
        for index, (token, model_owned) in enumerate(zip(row["tokens"], row["model_mask"])):
            if token == "CONST_VALUE":
                inside_constant = True
            if token.startswith(("NUM_", "DIGIT_", "SIGN_")):
                if index < row["condition_length"]:
                    assert not model_owned
                elif inside_constant:
                    assert not model_owned
                else:
                    assert model_owned
                    saw_model_owned_result_number = True
            if token == "NUM_END" and inside_constant:
                inside_constant = False
        assert not any(
            token.startswith("VALUE_node_param_") for token in body
        )
        if any(token.startswith("NUM_") for token in condition):
            assert "VALUE_BINDING" not in condition
        assert all(not any(char.isspace() for char in token) for token in row["tokens"])
    assert saw_model_owned_result_number


def test_v7_damage_expands_concrete_intermediate_results():
    config = StateVectorConfig()
    rng = random.Random(77)
    rows = []
    for index in range(256):
        rows.extend(build_v7_record(index, config, rng)["node_local_sequences"])
    damage = next(
        row for row in rows
        if row["kind"] == "DAMAGE"
        and all(token in row["tokens"] for token in (
            "CALCULATION_ADD",
            "CALCULATION_MULTIPLY",
            "CALCULATION_FLOOR_DIVIDE",
            "CALCULATION_MIN",
            "CALCULATION_SUBTRACT",
            "CALCULATION_RELU",
        ))
    )
    tokens = damage["tokens"]
    mask = damage["model_mask"]
    result_positions = [index for index, token in enumerate(tokens) if token == "CALC_VALUE"]
    assert len(result_positions) >= 6
    assert all(mask[index] for index in result_positions)
    assert all(
        any(
            tokens[next_index].startswith(("NUM_", "DIGIT_", "SIGN_"))
            and mask[next_index]
            for next_index in range(index + 1, min(index + 7, len(tokens)))
        )
        for index in result_positions
    )
    # A calculated result stays model-owned at its unique assignment, then the
    # environment supplies its exact value at every later operand reference.
    unblocked_assign = next(
        index for index, token in enumerate(tokens[:-1])
        if token == "CALC_FIELD_result"
        and tokens[index + 1] == "VALUE_reg_unblocked_damage"
    )
    assert tokens[unblocked_assign + 2] == "CALC_VALUE"
    unblocked_use = next(
        index for index in range(unblocked_assign + 2, len(tokens))
        if tokens[index] == "VALUE_reg_unblocked_damage"
    )
    assert tokens[unblocked_use + 1] == "CONST_VALUE"
    assert all(
        not mask[index]
        for index in range(
            unblocked_use + 1,
            tokens.index("NUM_END", unblocked_use + 2) + 1,
        )
    )
    hp_subtract_assign = next(
        index for index, token in enumerate(tokens[:-1])
        if token == "CALC_FIELD_result"
        and tokens[index + 1] == "VALUE_reg_target_hp_after_subtract"
    )
    hp_subtract_use = next(
        index for index in range(hp_subtract_assign + 2, len(tokens))
        if tokens[index] == "VALUE_reg_target_hp_after_subtract"
    )
    assert tokens[hp_subtract_use + 1] == "CONST_VALUE"


def test_v7_outer_predicts_card_rule_numbers_while_local_header_is_masked():
    config = StateVectorConfig()
    record = build_v7_record(4, config, random.Random(31))
    codec = V7TokenCodec.from_records([record])
    examples = examples_from_v7_record(record, codec, config)
    outer = examples[0]
    outer_tokens = codec.decode(outer.target_ids)
    assert any(token.startswith("NUM_") for token in outer_tokens)
    assert all(outer.target_model_mask)
    for local in examples[1:]:
        local_tokens = codec.decode(local.target_ids)
        for index, token in enumerate(local_tokens[:local.condition_length]):
            if token.startswith(("NUM_", "DIGIT_", "SIGN_")):
                assert not local.target_model_mask[index]


def test_world_vectors_encode_buff_presence_not_magnitude():
    config = StateVectorConfig()
    state = {
        "combat": {
            "player": {"block": 11, "powers": {"Strength": 7}},
            "monsters": [{"block": 4, "powers": {"Vulnerable": 3}}],
        },
        "global": {},
    }
    vector = vectorize_state(state, config)
    player_strength = (
        len(config.player_scalar_names) + config.player_buff_names.index("Strength")
    )
    enemy_vulnerable = (
        len(config.enemy_scalar_names)
        + config.monster_buff_names.index("Vulnerable")
    )
    assert vector.player[player_strength].item() == 1.0
    assert vector.enemies[0, enemy_vulnerable].item() == 1.0
    assert vector.player[config.player_scalar_names.index("block_present")].item() == 1.0
    assert vector.enemies[
        0, config.enemy_scalar_names.index("block_present")
    ].item() == 1.0

    stronger = {
        "combat": {
            "player": {"block": 11, "powers": {"Strength": 9}},
            "monsters": [{"block": 4, "powers": {"Vulnerable": 5}}],
        },
        "global": {},
    }
    removed = {
        "combat": {
            "player": {"block": 0, "powers": {}},
            "monsters": [{"block": 0, "powers": {}}],
        },
        "global": {},
    }
    magnitude_only = vector_delta(state, stronger, config)
    assert magnitude_only.player[player_strength].item() == 0.0
    assert magnitude_only.enemies[0, enemy_vulnerable].item() == 0.0
    presence_removed = vector_delta(state, removed, config)
    assert presence_removed.player[player_strength].item() == -1.0
    assert presence_removed.enemies[0, enemy_vulnerable].item() == -1.0


def test_v7_codec_reserves_rare_enemy_field_values():
    config = StateVectorConfig()
    record = build_v7_record(0, config, random.Random(7))
    codec = V7TokenCodec.from_records([record])
    assert "VALUE_enemy_2_present" in codec.token_to_id
    assert "VALUE_enemy_2_gone" in codec.token_to_id
