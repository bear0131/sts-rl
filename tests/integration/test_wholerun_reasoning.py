from __future__ import annotations

from dataclasses import asdict

import torch

from env.core import action_space as A
from env.core.cards_vocab import CARD_VOCAB
from env.core import encode
from rl.semantic_reasoning_policy import (
    AnonymousCardProgramEmbedding,
    CachedReasoningDecoder,
    ReasoningKVCache,
    SemanticReasoningPolicyCore,
)
from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    CardRuleEncoder,
    ProgramEncoderConfig,
    RuleProgramTransformer,
)
from rl.split_policy import SplitStSPolicyCore
from rl.wholerun_reasoning import (
    IGNORE_INDEX,
    TokenOwner,
    VOCAB,
    anonymous_card_tokens,
    build_interactive_transcript,
    compact_program_numbers,
    number_tokens,
)


def test_anonymous_card_surface_contains_rules_but_not_name():
    tokens = anonymous_card_tokens(
        {"id": "Twin Strike", "up": 0, "cost": 1}, reference=7
    )
    joined = " ".join(tokens)
    assert "Twin" not in joined
    assert "Strike" not in joined
    assert "TYPE_ATTACK" in tokens
    assert "DEAL_DAMAGE" in tokens
    assert "TIMES" in tokens
    assert "CARD_REF" in tokens


def test_numeric_encoding_uses_compact_atoms_and_fallback():
    assert number_tokens(0) == ["NUM_BEGIN", "DIGIT_0", "NUM_END"]
    assert number_tokens(-32) == [
        "NUM_BEGIN", "SIGN_NEG", "DIGIT_3", "DIGIT_2", "NUM_END"
    ]
    assert number_tokens(256) == [
        "NUM_BEGIN", "DIGIT_2", "DIGIT_5", "DIGIT_6", "NUM_END"
    ]
    assert "SIGN_POS" not in VOCAB.tokens
    assert compact_program_numbers(
        ("COST", "NUM_BEGIN", "DIGIT_1", "NUM_END")
    ) == ("COST", "NUM_BEGIN", "DIGIT_1", "NUM_END")
    assert compact_program_numbers(
        ("COST", "NUM_BEGIN", "VALUE_X", "NUM_END")
    ) == ("COST", "NUM_BEGIN", "VALUE_X", "NUM_END")


def test_environment_responses_are_inserted_and_not_supervised():
    events = [
        {
            "type": "MONSTER_VALUE_CHANGE",
            "slot": 2,
            "field": "hp",
            "before": 20,
            "delta": -3,
            "after": 17,
        },
        {
            "type": "CARD_MOVE",
            "source": "draw",
            "destination": "hand",
            "card": "Strike_R",
            "up": 0,
        },
        {"type": "ACTION_COMPLETE"},
    ]
    transcript = build_interactive_transcript(
        events=events,
        action_segment="PLAY_CARD",
        action_offset=0,
        action_card={"id": "Sword Boomerang", "up": 0, "cost": 1},
        random_target=True,
    )
    tokens = list(transcript.tokens)
    target_query = tokens.index("QUERY_RANDOM_TARGET")
    target_observation = tokens.index("OBS_RANDOM_TARGET")
    damage_event = tokens.index("EVENT_MONSTER_VALUE_CHANGE")
    draw_query = tokens.index("QUERY_DRAW_CARD")
    draw_observation = tokens.index("OBS_DRAW_CARD")
    card_event = tokens.index("EVENT_CARD_MOVE")
    assert target_query < target_observation < damage_event
    assert draw_query < draw_observation < card_event
    assert "Strike_R" not in tokens
    assert "Sword Boomerang" not in tokens

    _inputs, labels = transcript.teacher_labels()
    for index, owner in enumerate(transcript.owners[1:]):
        if owner == TokenOwner.ENVIRONMENT:
            assert labels[index] == IGNORE_INDEX
        else:
            assert labels[index] != IGNORE_INDEX


def test_local_calculation_precedes_visible_numeric_result():
    transcript = build_interactive_transcript(
        events=[{
            "type": "PLAYER_VALUE_CHANGE",
            "field": "energy",
            "before": 3,
            "delta": -1,
            "after": 2,
        }],
        action_segment="PLAY_CARD",
    )
    tokens = list(transcript.tokens)
    calc_begin = tokens.index("CALC_BEGIN")
    calc_end = tokens.index("CALC_END")
    visible_result = tokens.index("VALUE_AFTER", calc_end)
    assert calc_begin < calc_end < visible_result


def test_cached_decoder_matches_parallel_causal_forward():
    torch.manual_seed(7)
    decoder = CachedReasoningDecoder(
        len(VOCAB.tokens),
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_tokens=64,
        padding_idx=VOCAB.pad_id,
    ).eval()
    memory = torch.randn(2, 6, 32)
    memory_padding = torch.zeros(2, 6, dtype=torch.bool)
    token_ids = torch.randint(3, len(VOCAB.tokens), (2, 11))
    parallel = decoder(token_ids, memory, memory_padding)

    cache = ReasoningKVCache.empty(len(decoder.layers))
    incremental = []
    for position in range(token_ids.shape[1]):
        logits, cache = decoder.step(
            token_ids[:, position], memory, memory_padding, cache
        )
        incremental.append(logits)
    incremental = torch.stack(incremental, dim=1)
    assert cache.length == token_ids.shape[1]
    torch.testing.assert_close(incremental, parallel, atol=2e-5, rtol=2e-5)


def test_cached_decoder_supports_positions_beyond_initial_capacity():
    torch.manual_seed(17)
    decoder = CachedReasoningDecoder(
        len(VOCAB.tokens),
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_tokens=4,
        padding_idx=VOCAB.pad_id,
    ).eval()
    memory = torch.randn(1, 6, 32)
    memory_padding = torch.zeros(1, 6, dtype=torch.bool)
    token_ids = torch.randint(3, len(VOCAB.tokens), (1, 11))
    parallel = decoder(token_ids, memory, memory_padding)

    cache = ReasoningKVCache.empty(len(decoder.layers))
    incremental = []
    for position in range(token_ids.shape[1]):
        logits, cache = decoder.step(
            token_ids[:, position], memory, memory_padding, cache
        )
        incremental.append(logits)
    incremental = torch.stack(incremental, dim=1)
    assert cache.length == token_ids.shape[1]
    torch.testing.assert_close(incremental, parallel, atol=2e-5, rtol=2e-5)


def test_anonymous_embedding_never_reads_canonical_id_weight():
    torch.manual_seed(11)
    embedding = AnonymousCardProgramEmbedding(
        len(CARD_VOCAB), 32, n_heads=4, n_layers=1
    ).eval()
    ids = torch.tensor([CARD_VOCAB.index("Strike_R")])
    upgrades = torch.zeros_like(ids)
    before = embedding.versioned_forward(ids, upgrades)
    with torch.no_grad():
        embedding.weight.normal_(mean=1000.0, std=100.0)
    after = embedding.versioned_forward(ids, upgrades)
    torch.testing.assert_close(before, after)


def test_anonymous_embedding_returns_zero_for_padding_card():
    embedding = AnonymousCardProgramEmbedding(
        len(CARD_VOCAB), 32, n_heads=4, n_layers=1
    ).eval()
    ids = torch.zeros(4, dtype=torch.long)
    upgrades = torch.zeros_like(ids)
    output = embedding.versioned_forward(ids, upgrades)
    assert torch.isfinite(output).all()
    assert output.eq(0).all()


def test_anonymous_embedding_drops_padding_rows_before_transformer():
    embedding = AnonymousCardProgramEmbedding(
        len(CARD_VOCAB), 32, n_heads=4, n_layers=1
    ).eval()
    seen = []
    original = embedding.program_encoder.forward

    def record(source, *args, **kwargs):
        seen.append(int(source.shape[0]))
        return original(source, *args, **kwargs)

    embedding.program_encoder.forward = record
    card_id = CARD_VOCAB.index("Strike_R")
    embedding.versioned_forward(
        torch.tensor([[0, card_id, 0]]),
        torch.zeros(1, 3),
    )
    assert seen == [1]


def test_wholerun_can_load_and_freeze_v7_card_vectors(tmp_path):
    """The non-reasoning wholerun frontend loads and freezes v7 card vectors."""

    torch.manual_seed(23)
    card_config = ProgramEncoderConfig(
        token_vocab_size=len(CARD_RULE_VOCAB),
        token_dim=64,
        output_dim=128,
        n_heads=4,
        n_layers=2,
        feedforward=256,
        max_tokens=256,
        dropout=0.0,
    )
    backbone = RuleProgramTransformer(
        token_dim=card_config.token_dim,
        n_heads=card_config.n_heads,
        n_layers=card_config.n_layers,
        feedforward=card_config.feedforward,
        max_tokens=card_config.max_tokens,
        dropout=card_config.dropout,
    )
    source_encoder = CardRuleEncoder(card_config, backbone=backbone).eval()
    checkpoint = tmp_path / "card-v7.pt"
    torch.save(
        {
            "config": {"card_encoder": asdict(card_config)},
            "model": {
                f"card_encoder.{key}": value.detach().clone()
                for key, value in source_encoder.state_dict().items()
            },
        },
        checkpoint,
    )

    observation_space = encode.obs_space(
        card_language=False,
        deck_size_observation=True,
        neow_observation=True,
        event_option_observation=True,
        interactive_reasoning=False,
    )
    core = SplitStSPolicyCore(
        observation_space,
        d_model=256,
        n_layers=4,
        n_heads=4,
        card_language=False,
        card_vector_checkpoint=str(checkpoint),
    ).eval()

    card_id = torch.tensor([CARD_VOCAB.index("Strike_R")])
    upgrades = torch.zeros_like(card_id)
    expected = source_encoder.card_vectors(card_id, upgrades)
    actual = core.shared.card.versioned_forward(card_id, upgrades)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 128)
    assert core.shared.cards(card_id, upgrades).shape == (1, 256)
    assert core.shared.card_vector_cache_enabled
    assert core.shared.card_vector_cache is not None
    assert core.shared.card_vector_cache.shape == (len(CARD_VOCAB), 2, 128)
    original_lookup = core.shared.card.versioned_forward
    core.shared.card.versioned_forward = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("frozen card encoder was queried after caching"))
    )
    try:
        cached = core.shared.cards(card_id, upgrades)
    finally:
        core.shared.card.versioned_forward = original_lookup
    expected_projected = core.shared.card_projection(
        torch.cat([expected, upgrades.float().unsqueeze(-1)], dim=-1)
    )
    torch.testing.assert_close(cached, expected_projected)
    assert core.shared.card_projection.weight.shape == (256, 129)
    assert all(not parameter.requires_grad for parameter in core.shared.card.parameters())
    assert any(parameter.requires_grad for parameter in core.shared.card_projection.parameters())

    sampled = observation_space.sample()
    obs = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in sampled.items()
    }
    obs["phase"] = torch.tensor([[1]])
    obs["hand_ids"].zero_()
    obs["hand_ids"][0, 0] = card_id.item()
    obs["hand_upgrades"].zero_()
    obs["hand_costs"].zero_()
    logits, values = core.actor_critic(obs)
    assert logits.shape[0] == values.shape[0] == 1
    (logits.square().mean() + values.square().mean()).backward()
    assert all(parameter.grad is None for parameter in core.shared.card.parameters())


def test_selected_play_card_carries_runtime_cost():
    class SelectionOnly:
        _combat_phase = 0

    obs = {
        "phase": torch.zeros(1, 1),
        "hand_ids": torch.tensor([[CARD_VOCAB.index("Strike_R"), 0]]),
        "hand_upgrades": torch.tensor([[0.2, 0.0]]),
        "hand_costs": torch.tensor([[0.3, 0.0]]),
    }
    action = torch.tensor([A.seg_base("PLAY_CARD")])
    ids, upgrades, costs, cost_known = (
        SemanticReasoningPolicyCore._selected_card(
            SelectionOnly(), obs, action
        )
    )
    assert int(ids[0]) == CARD_VOCAB.index("Strike_R")
    torch.testing.assert_close(upgrades[0], torch.tensor(0.2))
    torch.testing.assert_close(costs[0], torch.tensor(0.3))
    assert bool(cost_known[0])


def test_pending_card_slot_is_opt_in_observation_contract():
    old_space = encode.obs_space()
    new_space = encode.obs_space(interactive_reasoning=True)
    assert "pending_card_slot" not in old_space.spaces
    assert "pending_card_slot" in new_space.spaces
    encoded = encode.encode(
        {"pending_card_slot": 4}, interactive_reasoning=True
    )
    assert int(encoded["pending_card_slot"][0]) == 4
