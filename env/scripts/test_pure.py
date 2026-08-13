"""Smoke test for the v2 backend-agnostic obs layer (vocab + encode + intents +
actions + render). Runs WITHOUT the sim built: it feeds hand-made canonical dicts
(the shape bridge.schemas produces) through the pure functions.

Run from the repo root:  python -m env.scripts.test_pure
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from env.core import encode
from env.core.cards_vocab import CARD_INDEX
from env.core import actions
from env.core import action_space as run_actions
from env.core import render
from env.core import intents
from env.core import vocab as V
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers
from env.core.card_language import (
    LANGUAGE_CARDS,
    MAX_PROGRAM_TOKENS,
    STATUS_CARDS,
    TOKEN_DOC,
    TOKEN_INDEX,
    TOKEN_VOCAB,
    card_program,
    encoded_card_programs,
)
from env.core import combat_language as CL


# ------------------------------ fixtures ------------------------------

def _fake_combat():
    """A combat-block canonical (normalize_sim shape) + phase/global/options."""
    return {
        "phase": "combat",
        "turn": 2,
        "player": {
            "hp": 68, "max_hp": 80, "block": 5, "energy": 3,
            "combat_counters": {
                "centennial_puzzle_active": True,
                "cards_played": 4,
                "combust_hp_loss": 2,
                "echo_form_cards_doubled": 1,
                "cards_discarded": 3,
                "times_damaged": 5,
                "bomb_2": 40,
                "orange_attack": True,
                "lizard_tail_active": False,
            },
            "powers": {"Strength": 2, "Vulnerable": 0, "Frail": 1},
        },
        "monsters": [
            {"id": "JawWorm", "hp": 40, "max_hp": 44, "block": 0, "gone": False,
             "base_damage": 11, "adjusted_damage": 14, "hits": 1,
             "powers": {"Strength": 3},
             "move": "JAW_WORM_CHOMP", "move_id": 87},
            {"id": "Cultist", "hp": 0, "max_hp": 48, "block": 0, "gone": True,
             "base_damage": 0, "hits": 0, "powers": {},
             "move": "INVALID", "move_id": 0},
        ],
        "hand": [{"id": "Strike_R", "up": 0, "cost": 0},
                 {"id": "Bash", "up": 0, "cost": 2},
                 {"id": "Defend_R", "up": 1, "cost": 1}],
        "draw": [{"id": "Strike_R", "up": 0}, {"id": "Strike_R", "up": 0}],
        "discard": [{"id": "Defend_R", "up": 0}],
        "exhaust": [],
        "global": {
            "character": "IRONCLAD", "ascension": 3, "act": 1, "floor": 6,
            "boss": "HEXAGHOST",
            "relics": [
                "Burning Blood", "Centennial Puzzle", "Lizard Tail", "Akabeko",
            ],
            "relic_counters": [0, -1, -1, 2],
            "potions": ["Fire Potion", "INVALID"], "gold": 99,
            "deck": [("Strike_R", 0), ("Strike_R", 0), ("Bash", 0),
                     ("Defend_R", 1)],
            "boss_relics": [],
        },
    }


def _fake_map():
    return {
        "phase": "map",
        "options": {
            "nodes": [(2, "MONSTER"), (3, "EVENT"), (4, "REST")],
            "full_map": [
                {"x": 2, "y": 0, "room": "MONSTER", "children": [2, 3],
                 "current": False, "choice": True, "reachable": True},
                {"x": 3, "y": 1, "room": "REST", "children": [3],
                 "current": False, "choice": False, "reachable": True},
            ],
        },
        "global": {
            "character": "IRONCLAD", "ascension": 0, "act": 1, "floor": 5,
            "hp": 61, "max_hp": 80,
            "boss": "HEXAGHOST", "relics": ["Burning Blood"], "potions": [],
            "gold": 50, "deck": [("Strike_R", 0)], "boss_relics": [],
        },
    }


# ------------------------------ vocab ------------------------------

def test_vocab_invariants():
    # id vocabs: index 0 is the PAD sentinel; no duplicates; index map consistent.
    for name in ("MONSTER_VOCAB", "MOVE_VOCAB", "RELIC_VOCAB", "POTION_VOCAB",
                 "EVENT_VOCAB", "BOSS_VOCAB"):
        voc = getattr(V, name)
        idx = getattr(V, name + "_INDEX")
        assert voc[0] == "PAD", (name, voc[0])
        assert len(voc) == len(set(voc)), f"{name} has duplicates"
        assert idx[voc[7]] == 7
        cap = getattr(V, name + "_CAPACITY")
        assert len(voc) <= cap, (name, len(voc), cap)
    # buff vocabs: dense, no PAD sentinel, fit in their reserved vector length.
    assert len(V.PLAYER_BUFF_VOCAB) <= V.PLAYER_BUFF_VEC
    assert len(V.MONSTER_BUFF_VOCAB) <= V.MONSTER_BUFF_VEC
    assert V.PLAYER_BUFF_VOCAB[0] == "Strength"
    assert V.PHASE_VOCAB[0] == "none" and V.CHARACTER_VOCAB[0] == "INVALID"
    assert V.INTENT_CATEGORY[0] == "unknown"
    print(f"vocab OK: monsters={len(V.MONSTER_VOCAB)} moves={len(V.MOVE_VOCAB)} "
          f"relics={len(V.RELIC_VOCAB)} buffs P{len(V.PLAYER_BUFF_VOCAB)}/"
          f"M{len(V.MONSTER_BUFF_VOCAB)}")


# ------------------------------ card table ------------------------------

def test_ironclad_card_table():
    from env.core.cards_vocab import CARD_INDEX

    assert len(IRONCLAD_CARDS) == 75
    assert all(card_id in CARD_INDEX for card_id in IRONCLAD_CARDS)
    assert all(info.effect_en and info.effect_zh for info in IRONCLAD_CARDS.values())
    assert card_numbers("Disarm", 1) == (1, 0, 0)
    assert card_numbers("Infernal Blade", 1) == (0, 0, 0)
    assert card_numbers("Searing Blow", 10) == (2, 97, 0)
    print("Ironclad card table OK (75 bilingual wiki-backed entries)")


def test_ironclad_card_language():
    from env.core.cards_vocab import CARD_INDEX, CARD_VOCAB

    assert len(TOKEN_VOCAB) == len(TOKEN_INDEX) == len(set(TOKEN_VOCAB))
    assert all(token in TOKEN_DOC for token in TOKEN_VOCAB)
    for card_id in IRONCLAD_CARDS:
        for upgraded in (False, True):
            program = card_program(card_id, upgraded)
            assert program[0] == "CARD_BEGIN" and program[-1] == "CARD_END"
            assert all(token in TOKEN_INDEX for token in program)
    rows = encoded_card_programs()
    assert len(rows) == len(CARD_VOCAB)
    assert all(len(row) == MAX_PROGRAM_TOKENS for row in rows)
    for card_id in STATUS_CARDS:
        assert any(rows[CARD_INDEX[card_id]]), card_id
    assert "END_TURN_IN_HAND" in card_program("Burn", False)
    assert "DIGIT_4" in card_program("Burn", True)
    assert "ETHEREAL" in card_program("Dazed", False)
    assert "EXHAUST" in card_program("Slimed", False)
    assert "ON_DRAW" in card_program("Void", False)
    assert "LOSE_ENERGY" in card_program("Void", False)
    assert "UNPLAYABLE_NORMALLY" in card_program("Wound", False)
    unrelated = next(card for card in CARD_VOCAB if card not in LANGUAGE_CARDS)
    assert not any(rows[CARD_INDEX[unrelated]])
    print(
        f"Card language OK (75 Ironclad + 5 Status cards, "
        f"{len(TOKEN_VOCAB)} tokens, "
        f"max program {MAX_PROGRAM_TOKENS})"
    )


def test_status_card_language_model_paths():
    """Status DSL must affect both per-card hand and aggregate pile inputs."""
    import torch

    from env.core.cards_vocab import CARD_INDEX, CARD_VOCAB
    from rl.model import CardLanguageEmbedding, StSTransformerExtractor

    status_ids = torch.tensor([CARD_INDEX[c] for c in STATUS_CARDS])
    card_embedding = CardLanguageEmbedding(len(CARD_VOCAB), 64)
    assert torch.all(card_embedding.program_ids[status_ids].ne(0).any(dim=1))
    raw = torch.nn.Embedding.forward(card_embedding, status_ids)
    language_delta = card_embedding(status_ids) - raw
    assert torch.all(language_delta.norm(dim=1) > 0)

    canon = _fake_combat()
    canon["hand"][0] = {"id": "Burn", "up": 0, "cost": -1}
    canon["draw"] = [{"id": "Void", "up": 0}]
    obs_np = encode.encode(canon, card_language=True)
    obs = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in obs_np.items()
    }
    encoder = StSTransformerExtractor(
        encode.obs_space(card_language=True), features_dim=32, d_model=64,
        n_layers=1, n_heads=4, card_language=True,
    )
    seen_inputs = []
    handle = encoder.shared["card"].register_forward_pre_hook(
        lambda _module, args: seen_inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        encoded, _ = encoder.encode_tokens(obs)
    handle.remove()
    burn_id = CARD_INDEX["Burn"]
    assert encoded.shape[1] == encoder.n_tokens
    assert any(x.shape == obs["hand_ids"].shape and (x == burn_id).any()
               for x in seen_inputs)

    assert obs["draw_card_ids"][0, 0] == CARD_INDEX["Void"]
    card_layer = encoder.shared["card"]
    program = card_layer.program_ids[burn_id]
    used = program[program != 0]
    reversed_program = torch.zeros_like(program)
    reversed_program[:used.numel()] = used.flip(0)
    ordered = card_layer._encode_programs(program.unsqueeze(0))
    reversed_vector = card_layer._encode_programs(reversed_program.unsqueeze(0))
    assert not torch.allclose(ordered, reversed_vector)
    assert card_layer.program_position.shape[0] == MAX_PROGRAM_TOKENS

    ids = torch.tensor([[
        CARD_INDEX["Strike_R"], CARD_INDEX["Bash"], CARD_INDEX["Defend_R"]
    ]])
    upgrades = torch.tensor([[0.0, 0.2, 0.0]])
    counts = torch.ones_like(upgrades)
    upgraded_counts = torch.tensor([[0.0, 1.0, 0.0]])
    pile_ordered = encoder.pile_card_language(
        ids, upgrades, counts, upgraded_counts, card_layer
    )
    pile_reversed = encoder.pile_card_language(
        ids.flip(1), upgrades.flip(1), counts.flip(1),
        upgraded_counts.flip(1), card_layer
    )
    assert pile_ordered.norm() > 0
    assert not torch.allclose(pile_ordered, pile_reversed)
    print("Status language reaches model hand and pile encoding paths")


# ------------------------------ intents ------------------------------

def test_intent_derivation():
    assert intents.derive("JAW_WORM_CHOMP", 11) == "attack"   # base_damage wins
    assert intents.derive("CENTURION_DEFEND", 0) == "defend"
    assert intents.derive("CULTIST_INCANTATION", 0) == "buff"
    assert intents.derive("THE_COLLECTOR_MEGA_DEBUFF", 0) == "strong_debuff"
    assert intents.derive("SNAKE_PLANT_ENFEEBLING_SPORES", 0) == "debuff"
    assert intents.derive("LAGAVULIN_SLEEP", 0) == "sleep"
    assert intents.derive("LOOTER_ESCAPE", 0) == "escape"
    assert intents.derive("BYRD_STUNNED", 0) == "stun"
    assert intents.derive(None, 0) == "unknown"
    assert intents.derive("INVALID", 0) == "unknown"
    print("intent derivation OK")


# ------------------------------ encode ------------------------------

def test_obs_matches_space():
    space = encode.obs_space()
    for canon in (_fake_combat(), _fake_map(), {}):  # combat / overworld / empty
        obs = encode.encode(canon)
        assert set(obs) == set(space.spaces), (set(obs) ^ set(space.spaces))
        for k, v in obs.items():
            sub = space.spaces[k]
            assert v.shape == sub.shape, (k, v.shape, sub.shape)
            assert v.dtype == sub.dtype, (k, v.dtype, sub.dtype)
            assert sub.contains(v), f"{k} out of declared bounds"
    print(f"obs matches space (OBS_VERSION={encode.OBS_VERSION}); "
          f"{len(space.spaces)} groups")

    language_space = encode.obs_space(card_language=True)
    language_obs = encode.encode(_fake_combat(), card_language=True)
    assert set(language_obs) == set(language_space.spaces)
    assert language_space.contains(language_obs)
    assert language_obs["draw_card_ids"][0] == CARD_INDEX["Strike_R"]
    assert language_obs["draw_card_counts"][0] == 2
    assert language_obs["deck_card_ids"][0] == CARD_INDEX["Strike_R"]
    defend_slot = next(
        i for i, card_id in enumerate(language_obs["deck_card_ids"])
        if card_id == CARD_INDEX["Defend_R"]
    )
    assert language_obs["deck_card_upgrades"][defend_slot] > 0
    print("card-language pile sequences preserve ids, order, and upgrades")


def test_combat_fields_land():
    from env.core.cards_vocab import CARD_INDEX
    o = encode.encode(_fake_combat())
    assert abs(o["player_scalars"][0] - 68 / encode.HP_SCALE) < 1e-6
    assert abs(o["player_scalars"][5] - 68 / 80) < 1e-6
    assert abs(o["player_buffs"][V.PLAYER_BUFF_VOCAB_INDEX["Strength"]]
               - 2 / encode.BUFF_SCALE) < 1e-6
    assert o["monster_id"][0] == V.MONSTER_VOCAB_INDEX["JawWorm"]
    assert o["monster_scalars"][0][0] == 1.0
    assert o["monster_move"][0] == 87
    assert o["monster_intent_cat"][0] == V.INTENT_CATEGORY_INDEX["attack"]
    assert o["hand_ids"][1] == CARD_INDEX["Bash"]
    assert o["hand_costs"][0] == 0.0  # current cost overrides printed cost
    assert abs(o["hand_damage"][0] - 6 / encode.DMG_SCALE) < 1e-6
    assert abs(o["hand_block"][2] - 8 / encode.BLOCK_SCALE) < 1e-6
    assert o["draw_counts"][CARD_INDEX["Strike_R"]] == 2.0
    counter = dict(zip(encode.COMBAT_COUNTER_KEYS, o["combat_counters"]))
    assert counter["cards_played"] == 4 / encode.COUNTER_SCALE
    assert counter["combust_hp_loss"] == 2 / encode.COUNTER_SCALE
    assert counter["echo_form_cards_doubled"] == 1 / encode.COUNTER_SCALE
    assert counter["cards_discarded"] == 3 / encode.COUNTER_SCALE
    assert counter["times_damaged"] == 5 / encode.COUNTER_SCALE
    assert counter["bomb_2"] == 40 / encode.COUNTER_SCALE
    assert counter["orange_attack"] == 1.0
    print("combat fields land OK")


def test_fixed_monster_slots_no_drift():
    """Dead monster must STAY in its slot with present=0 (no slot shift)."""
    o = encode.encode(_fake_combat())
    assert o["monster_scalars"][0][0] == 1.0       # JawWorm present
    assert o["monster_id"][1] == V.MONSTER_VOCAB_INDEX["Cultist"]
    assert o["monster_scalars"][1][0] == 0.0       # Cultist absent, slot kept
    print("fixed monster slots OK (no drift)")


def test_global_and_options():
    o = encode.encode(_fake_combat())
    assert o["phase"][0] == V.PHASE_VOCAB_INDEX["combat"]
    assert o["character"][0] == V.CHARACTER_VOCAB_INDEX["IRONCLAD"]
    assert o["next_boss"][0] == V.BOSS_VOCAB_INDEX["HEXAGHOST"]
    assert o["relics"][0] == V.RELIC_VOCAB_INDEX["Burning Blood"]
    assert o["relic_counters"][1] == 1
    assert o["relic_counters"][2] == 0
    assert o["potions"][0] == V.POTION_VOCAB_INDEX["Fire Potion"]
    assert o["potions"][1] == 0                    # 'INVALID' -> PAD
    assert abs(o["gold"][0] - 99 / encode.GOLD_SCALE) < 1e-6

    m = encode.encode(_fake_map())
    assert m["phase"][0] == V.PHASE_VOCAB_INDEX["map"]
    assert m["map_node_rooms"][0] == V.ROOM_VOCAB_INDEX["MONSTER"]
    assert m["map_node_rooms"][1] == V.ROOM_VOCAB_INDEX["EVENT"]
    assert abs(m["map_node_x"][0] - 2 / encode.MAP_X_SCALE) < 1e-6
    assert m["monster_id"].sum() == 0 and m["hand_ids"].sum() == 0
    print("global + options OK")


def test_optional_global_map_view():
    base = encode.encode(_fake_map())
    assert "map_full_rooms" not in base

    space = encode.obs_space(global_map_view=True)
    o = encode.encode(_fake_map(), global_map_view=True)
    assert space.contains(o)
    first = 2
    second = encode.MAP_COLS + 3
    assert o["map_full_rooms"][first] == V.ROOM_VOCAB_INDEX["MONSTER"]
    assert o["map_full_features"][first, 3] == 1.0
    assert o["map_full_features"][first, 5 + 3] == 1.0
    assert o["map_full_rooms"][second] == V.ROOM_VOCAB_INDEX["REST"]
    assert o["map_full_features"][second, 4] == 1.0
    print(f"optional global map obs OK (version {encode.GLOBAL_MAP_OBS_VERSION})")


def test_optional_deck_size_observation():
    canon = _fake_map()
    deck = canon["global"]["deck"]
    o = encode.encode(canon, deck_size_observation=True)
    space = encode.obs_space(deck_size_observation=True)
    assert space.contains(o)
    assert abs(o["deck_size"][0] - len(deck) / encode.DECK_SIZE_SCALE) < 1e-6
    assert np.count_nonzero(o["deck_ids"]) == len(deck)
    print(f"optional deck size obs OK (version {encode.DECK_SIZE_OBS_VERSION})")


def test_lossless_deck_sequence():
    canon = _fake_map()
    canon["global"]["deck"] = [
        ("Strike_R", 0), ("Strike_R", 1), ("Searing Blow", 4),
    ]
    o = encode.encode(canon)
    assert o["deck_ids"][:3].tolist() == [
        CARD_INDEX["Strike_R"], CARD_INDEX["Strike_R"],
        CARD_INDEX["Searing Blow"],
    ]
    assert np.allclose(
        o["deck_upgrades"][:3], [0, 1 / encode.UPGRADE_SCALE, 4 / encode.UPGRADE_SCALE]
    )
    assert "deck_counts" not in o

    canon["global"]["deck"] = [("Strike_R", 0)] * encode.DECK_SLOTS
    assert encode.obs_space().contains(encode.encode(canon))
    canon["global"]["deck"].append(("Defend_R", 0))
    try:
        encode.encode(canon)
    except ValueError as exc:
        assert "global.deck" in str(exc) and "capacity 96" in str(exc)
    else:
        raise AssertionError("oversized deck was silently truncated")
    print("lossless per-card deck sequence and hard capacity OK")


def test_upgrade_counts_are_required_for_overworld_cards():
    canon = _fake_map()
    canon["phase"] = "shop"
    canon["options"] = {"shop_cards": [("Strike_R", 1, 50)]}
    o = encode.encode(canon)
    assert o["shop_card_ids"][0] == CARD_INDEX["Strike_R"]
    assert o["shop_card_upgrades"][0] == 1 / encode.UPGRADE_SCALE

    canon["options"]["shop_cards"] = [("Strike_R", 50)]
    try:
        encode.encode(canon)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy shop card without upgrade count was accepted")

    canon = _fake_map()
    canon["global"]["deck"] = [{"id": "Strike_R"}]
    try:
        encode.encode(canon)
    except ValueError as exc:
        assert "upgrade count" in str(exc)
    else:
        raise AssertionError("deck card without upgrade count was accepted")
    print("overworld card upgrade counts are mandatory")


def test_complete_relic_sequence():
    canon = _fake_combat()
    canon["player"]["combat_counters"] = {}
    held = list(V.RELIC_VOCAB[1:42])
    assert len(held) > 30
    canon["global"]["relics"] = held
    canon["global"]["relic_counters"] = list(range(len(held)))
    o = encode.encode(canon)
    assert o["relics"][:len(held)].tolist() == [
        V.RELIC_VOCAB_INDEX[relic] for relic in held
    ]
    assert np.allclose(
        o["relic_counters"][:len(held)],
        np.arange(len(held)) / encode.COUNTER_SCALE,
    )

    canon["global"]["relics"] = [held[0]] * (encode.R_MAX + 1)
    canon["global"]["relic_counters"] = []
    try:
        encode.encode(canon)
    except ValueError as exc:
        assert "global.relics" in str(exc) and str(encode.R_MAX) in str(exc)
    else:
        raise AssertionError("oversized relic list was silently truncated")
    print("all held relics are encoded with a hard capacity check")


def test_combat_language():
    assert CL.ATTACK_MOVES <= set(V.MOVE_VOCAB)
    for move in V.MOVE_VOCAB[1:]:
        program = CL.intent_program(move)
        assert program[0] == "INTENT_BEGIN" and program[-1] == "INTENT_END"
        assert f"MOVE_{move}" in program

    o = encode.encode(_fake_combat(), combat_language=True)
    assert encode.obs_space(combat_language=True).contains(o)
    player = [CL.TOKEN_VOCAB[i] for i in o["player_status_language"] if i]
    intent = [CL.TOKEN_VOCAB[i] for i in o["monster_intent_language"][0] if i]
    assert "STATUS_Strength" in player and "STATUS_Frail" in player
    # Adjusted 14 damage, not raw 11, is digit-tokenized into the live intent.
    damage_at = intent.index("DEAL_DAMAGE")
    assert intent[damage_at + 1:damage_at + 5] == [
        "NUM_BEGIN", "DIGIT_1", "DIGIT_4", "NUM_END"
    ]
    assert abs(o["monster_scalars"][0, 4] - 14 / encode.DMG_SCALE) < 1e-6
    print(f"combat language OK ({len(V.MOVE_VOCAB) - 1} intents, "
          f"{len(CL.TOKEN_VOCAB)} tokens)")


# --------------------------- actions / render ---------------------------

def test_action_codec_roundtrip():
    seen = set()
    for aid in range(actions.N_ACTIONS):
        kind = actions.decode(aid)
        seen.add(aid)
        if kind[0] == actions.KIND_PLAY:
            assert actions.encode_play(kind[1], kind[2]) == aid
        elif kind[0] == actions.KIND_SELECT:
            assert actions.encode_select(kind[1]) == aid
    assert len(seen) == actions.N_ACTIONS
    print(f"N_ACTIONS = {actions.N_ACTIONS}  (version {actions.ACTION_VERSION})")


def test_monotonic_multi_select():
    desc = {"cards_in_hand": 4, "pick_max": 2}
    assert run_actions.multi_legal_ids(desc, set()) == [
        run_actions._PICK + h for h in range(4)
    ] + [run_actions._CONFIRM]

    selected, commit = run_actions.multi_toggle(run_actions._PICK + 1, set())
    assert selected == {1} and commit is None
    legal = run_actions.multi_legal_ids(desc, selected)
    assert run_actions._PICK + 1 not in legal
    assert all(run_actions._PICK + h in legal for h in (0, 2, 3))

    selected, commit = run_actions.multi_toggle(run_actions._PICK + 3, selected)
    assert selected == {1, 3} and commit is None
    assert run_actions.multi_legal_ids(desc, selected) == [run_actions._CONFIRM]
    selected, commit = run_actions.multi_toggle(run_actions._CONFIRM, selected)
    assert selected == {1, 3} and commit == (1, 3)

    try:
        run_actions.multi_toggle(run_actions._PICK + 1, selected)
    except ValueError as exc:
        assert "already selected" in str(exc)
    else:
        raise AssertionError("an already-selected MULTI card was accepted")
    print("monotonic combat MULTI selection OK")


def test_render():
    canon = _fake_combat()
    s = render.render_state(canon)
    assert "JawWorm" in s and "68/80" in s and "Strike_R" in s, s
    assert "Cultist" in s and "gone" in s, s
    assert render.render_action(actions.END_TURN) == "end turn"
    label = render.render_action(actions.encode_play(1, 0), canon)
    assert "Bash" in label and "JawWorm" in label, label
    print("render OK")


if __name__ == "__main__":
    test_vocab_invariants()
    test_ironclad_card_table()
    test_ironclad_card_language()
    test_status_card_language_model_paths()
    test_intent_derivation()
    test_obs_matches_space()
    test_combat_fields_land()
    test_fixed_monster_slots_no_drift()
    test_global_and_options()
    test_optional_global_map_view()
    test_optional_deck_size_observation()
    test_lossless_deck_sequence()
    test_upgrade_counts_are_required_for_overworld_cards()
    test_complete_relic_sequence()
    test_combat_language()
    test_action_codec_roundtrip()
    test_monotonic_multi_select()
    test_render()
    print("\nALL OK")
