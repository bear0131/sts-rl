"""Regression tests for the phase-specialized whole-run actor-critic.

Run from the repository root:

    python -m tests.unit.test_split_policy
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import torch
from gymnasium import spaces

from env.core import action_space as A
from env.core import encode
from env.core import vocab as V
from env.core.cards_vocab import CARD_INDEX
from rl.model import uses_card_language
from rl.split_policy import (
    COMBAT_OBSERVATION_KEYS,
    OVERWORLD_OBSERVATION_KEYS,
    OVERWORLD_PHASE_OBSERVATION_KEYS,
    CombatStSPointerPolicy,
    SplitStSPointerPolicy,
    SplitStSPolicyCore,
)
from rl.trainer import _checkpoint_policy_arch


torch.set_num_threads(1)


def _space():
    return encode.obs_space(
        global_map_view=True,
        card_language=True,
        combat_language=True,
        deck_size_observation=True,
        neow_observation=True,
        event_option_observation=True,
    )


def _core(observation_space):
    return SplitStSPolicyCore(
        observation_space,
        d_model=32,
        n_layers=2,
        n_heads=4,
        card_language=True,
        combat_language=True,
        map_dag=True,
        features_dim=32,
        vf_arch=(32,),
    ).eval()


def _empty(observation_space):
    out = {}
    for key, field in observation_space.spaces.items():
        dtype = torch.long if field.dtype.kind in ("i", "u") else torch.float32
        out[key] = torch.zeros((1, *field.shape), dtype=dtype)
    return out


def _index(mapping, preferred):
    value = mapping.get(preferred, 0)
    return value if value else 1


def _fixture(observation_space, phase_name):
    obs = _empty(observation_space)
    obs["phase"][0, 0] = V.PHASE_VOCAB_INDEX[phase_name]
    obs["character"][0, 0] = V.CHARACTER_VOCAB_INDEX["IRONCLAD"]
    obs["ascension"][0, 0] = 0.5
    obs["act"][0, 0] = 0.25
    obs["floor"][0, 0] = 0.2
    obs["gold"][0, 0] = 0.15
    obs["run_hp"][0] = torch.tensor([0.35, 0.4, 0.875])
    obs["deck_size"][0, 0] = 0.03

    cards = [
        _index(CARD_INDEX, "Strike_R"),
        _index(CARD_INDEX, "Bash"),
        _index(CARD_INDEX, "Defend_R"),
    ]
    obs["deck_ids"][0, :3] = torch.tensor(cards)
    obs["deck_upgrades"][0, 1] = 1 / encode.UPGRADE_SCALE
    obs["relics"][0, 0] = _index(V.RELIC_VOCAB_INDEX, "Burning Blood")
    obs["relic_counters"][0, 0] = 0.3
    obs["potions"][0, 0] = _index(V.POTION_VOCAB_INDEX, "Fire Potion")

    obs["player_scalars"][0] = torch.tensor([0.35, 0.4, 0.05, 0.3, 0.1, 0.875])
    obs["combat_counters"][0, 0] = 0.2
    obs["monster_id"][0, 0] = _index(V.MONSTER_VOCAB_INDEX, "Cultist")
    obs["monster_move"][0, 0] = 1
    obs["monster_intent_cat"][0, 0] = 1
    obs["monster_scalars"][0, 0] = torch.tensor([1.0, 0.24, 0.8, 0.0, 0.06, 0.1])
    obs["hand_ids"][0, :2] = torch.tensor(cards[:2])
    obs["hand_upgrades"][0, 1] = 1 / encode.UPGRADE_SCALE
    obs["hand_costs"][0, :2] = torch.tensor([0.1, 0.2])
    obs["draw_counts"][0, cards[0]] = 2
    obs["discard_counts"][0, cards[2]] = 1
    obs["draw_card_ids"][0, 0] = cards[0]
    obs["draw_card_counts"][0, 0] = 2
    obs["discard_card_ids"][0, 0] = cards[2]
    obs["discard_card_counts"][0, 0] = 1

    obs["select_card_ids"][0, :3] = torch.tensor(cards)
    obs["select_card_upgrades"][0, 1] = 1 / encode.UPGRADE_SCALE
    obs["selected_card_ids"][0, 0] = cards[0]
    obs["offer_card_ids"][0, :3] = torch.tensor(cards)
    obs["offer_card_upgrades"][0, 2] = 1 / encode.UPGRADE_SCALE
    obs["reward_gold"][0, 0] = 0.1
    obs["reward_relic_ids"][0, 0] = obs["relics"][0, 0]
    obs["reward_potion_ids"][0, 0] = obs["potions"][0, 0]

    obs["map_node_rooms"][0, :2] = torch.tensor([1, 2])
    obs["map_node_x"][0, :2] = torch.tensor([2 / encode.MAP_X_SCALE, 3 / encode.MAP_X_SCALE])
    obs["map_full_rooms"][0, :2] = torch.tensor([1, 2])
    obs["map_full_features"][0, 0, 0] = 2 / encode.MAP_X_SCALE
    obs["map_full_features"][0, 0, 3] = 1
    obs["map_full_features"][0, 1, 0] = 3 / encode.MAP_X_SCALE

    obs["shop_card_ids"][0, :2] = torch.tensor(cards[:2])
    obs["shop_card_upgrades"][0, 1] = 1 / encode.UPGRADE_SCALE
    obs["shop_card_prices"][0, :2] = torch.tensor([0.05, 0.08])
    obs["shop_relic_ids"][0, 0] = obs["relics"][0, 0]
    obs["shop_relic_prices"][0, 0] = 0.2
    obs["shop_potion_ids"][0, 0] = obs["potions"][0, 0]
    obs["shop_potion_prices"][0, 0] = 0.05
    obs["boss_relic_ids"][0, 0] = obs["relics"][0, 0]
    obs["event_option_ids"][0, :2] = torch.tensor([1, 2])
    obs["event_option_event_ids"][0, :2] = 1
    obs["event_option_features"][0, :, 1] = 1
    obs["neow_bonus_ids"][0, :2] = torch.tensor([1, 2])
    return obs


def _batch(*observations):
    return {
        key: torch.cat([observation[key] for observation in observations], dim=0)
        for key in observations[0]
    }


def _perturb(observation, observation_space, keys):
    changed = {key: value.clone() for key, value in observation.items()}
    for key in keys:
        if key == "phase" or key not in changed:
            continue
        if changed[key].dtype == torch.long:
            high = int(observation_space[key].high.max())
            changed[key].fill_(min(high, 1))
        else:
            changed[key].add_(0.731)
    return changed


def _assert_same(left, right):
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])


def _has_nonzero_grad(module):
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
        for parameter in module.parameters()
    )


def _has_no_grad(module):
    return all(parameter.grad is None for parameter in module.parameters())


def test_strict_field_routing():
    observation_space = _space()
    core = _core(observation_space)
    combat = _fixture(observation_space, "combat")
    card_select = _fixture(observation_space, "card_select")

    seen_combat = []
    seen_overworld = []
    combat_hook = core.combat.encoder.register_forward_pre_hook(
        lambda _module, args: seen_combat.append(set(args[0]))
    )
    overworld_hook = core.overworld.encoder.register_forward_pre_hook(
        lambda _module, args: seen_overworld.append(set(args[0]))
    )
    with torch.no_grad():
        combat_output = core.actor_critic(combat)
        changed = _perturb(
            combat,
            observation_space,
            set(observation_space.spaces) - COMBAT_OBSERVATION_KEYS,
        )
        _assert_same(combat_output, core.actor_critic(changed))

        active_overworld = (
            OVERWORLD_OBSERVATION_KEYS
            | OVERWORLD_PHASE_OBSERVATION_KEYS[V.PHASE_VOCAB_INDEX["card_select"]]
        )
        overworld_output = core.actor_critic(card_select)
        changed = _perturb(
            card_select,
            observation_space,
            set(observation_space.spaces) - active_overworld,
        )
        _assert_same(overworld_output, core.actor_critic(changed))
    combat_hook.remove()
    overworld_hook.remove()

    assert all(keys <= COMBAT_OBSERVATION_KEYS for keys in seen_combat)
    assert all("phase" not in keys and "deck_ids" not in keys for keys in seen_combat)
    assert all("hand_ids" not in keys and "draw_counts" not in keys for keys in seen_overworld)
    assert all("shop_card_ids" not in keys for keys in seen_overworld)


def test_mixed_batch_matches_individual_routes():
    observation_space = _space()
    core = _core(observation_space)
    observations = [
        _fixture(observation_space, "map"),
        _fixture(observation_space, "combat"),
        _fixture(observation_space, "shop"),
        _fixture(observation_space, "card_select"),
    ]
    with torch.no_grad():
        mixed = core.actor_critic(_batch(*observations))
        individual = [core.actor_critic(observation) for observation in observations]
    expected_logits = torch.cat([item[0] for item in individual])
    expected_values = torch.cat([item[1] for item in individual])
    torch.testing.assert_close(mixed[0], expected_logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mixed[1], expected_values, atol=1e-6, rtol=1e-6)


def test_combat_fast_path_matches_router_and_explicit_relic_width():
    observation_space = _space()
    core = _core(observation_space)
    combat = _fixture(observation_space, "combat")
    compact = {
        key: value
        for key, value in combat.items()
        if key in COMBAT_OBSERVATION_KEYS
    }

    with torch.no_grad():
        routed = core.actor_critic(combat)
        automatic = core.combat_actor_critic(compact)
        explicit = core.combat_actor_critic(compact, active_relic_width=1)

    _assert_same(routed, automatic)
    _assert_same(automatic, explicit)


def test_specialist_gradients_are_isolated_but_tokens_are_shared():
    observation_space = _space()
    core = _core(observation_space)

    logits, values = core.actor_critic(_fixture(observation_space, "combat"))
    (logits.square().mean() + values.square().mean()).backward()
    assert _has_nonzero_grad(core.shared)
    assert _has_nonzero_grad(core.combat)
    assert _has_no_grad(core.overworld)

    for parameter in core.parameters():
        parameter.grad = None
    logits, values = core.actor_critic(_fixture(observation_space, "map"))
    (logits.square().mean() + values.square().mean()).backward()
    assert _has_nonzero_grad(core.shared)
    assert _has_nonzero_grad(core.overworld)
    assert _has_no_grad(core.combat)

    card_weights = [
        name for name, _parameter in core.named_parameters()
        if name.endswith("shared.card.weight")
    ]
    assert card_weights == ["shared.card.weight"]


def test_overworld_card_upgrades_affect_value_and_pointer():
    observation_space = _space()
    core = _core(observation_space)
    base = _fixture(observation_space, "card_select")

    deck_upgraded = {key: value.clone() for key, value in base.items()}
    deck_upgraded["deck_upgrades"][0, 0] = 1 / encode.UPGRADE_SCALE
    candidate_upgraded = {key: value.clone() for key, value in base.items()}
    candidate_upgraded["select_card_upgrades"][0, 0] = 1 / encode.UPGRADE_SCALE
    with torch.no_grad():
        base_logits, base_value = core.actor_critic(base)
        _deck_logits, upgraded_value = core.actor_critic(deck_upgraded)
        candidate_logits, _candidate_value = core.actor_critic(candidate_upgraded)

    assert not torch.equal(base_value, upgraded_value)
    pick_action = A.seg_base("PICK_CARD")
    assert base_logits[0, pick_action] != candidate_logits[0, pick_action]


def test_combat_receives_all_held_relics():
    observation_space = _space()
    core = _core(observation_space)
    base = _fixture(observation_space, "combat")
    many = {key: value.clone() for key, value in base.items()}
    held_count = 40
    many["relics"][0, :held_count] = torch.arange(1, held_count + 1)
    many["relic_counters"][0, :held_count] = (
        torch.arange(held_count) / encode.COUNTER_SCALE
    )

    seen_ids = []
    handle = core.shared.relic.register_forward_pre_hook(
        lambda _module, args: seen_ids.append(args[0].detach().clone())
    )
    with torch.no_grad():
        base_output = core.actor_critic(base)
        many_output = core.actor_critic(many)
    handle.remove()

    assert seen_ids[0].shape[1] == 1
    assert seen_ids[1].shape[1] == held_count
    assert torch.equal(
        seen_ids[1][0], torch.arange(1, held_count + 1, dtype=torch.long)
    )
    assert not (
        torch.equal(base_output[0], many_output[0])
        and torch.equal(base_output[1], many_output[1])
    )


def test_state_dict_round_trip():
    observation_space = _space()
    core = _core(observation_space)
    observation = _batch(
        _fixture(observation_space, "combat"),
        _fixture(observation_space, "event"),
    )
    with torch.no_grad():
        expected = core.actor_critic(observation)
    payload = io.BytesIO()
    torch.save(core.state_dict(), payload)
    payload.seek(0)
    restored = _core(observation_space)
    restored.load_state_dict(torch.load(payload, weights_only=True))
    with torch.no_grad():
        actual = restored.actor_critic(observation)
    _assert_same(expected, actual)


def test_policy_metadata_and_layer_budget():
    observation_space = _space()
    action_space = spaces.Discrete(A.N_ACTIONS)
    split = SplitStSPointerPolicy(
        observation_space,
        action_space,
        lambda _progress: 3e-4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        card_language=True,
        combat_language=True,
        map_dag=True,
    )
    combat = CombatStSPointerPolicy(
        observation_space,
        action_space,
        lambda _progress: 3e-4,
        d_model=32,
        n_layers=1,
        n_heads=4,
        card_language=True,
        combat_language=True,
    )
    assert _checkpoint_policy_arch(SimpleNamespace(policy=split)) == "split-pointer"
    assert _checkpoint_policy_arch(SimpleNamespace(policy=combat)) == "combat-pointer"
    assert uses_card_language(split)
    assert sum(split.core.specialist_layers) == 2


if __name__ == "__main__":
    test_strict_field_routing()
    test_mixed_batch_matches_individual_routes()
    test_combat_fast_path_matches_router_and_explicit_relic_width()
    test_specialist_gradients_are_isolated_but_tokens_are_shared()
    test_overworld_card_upgrades_affect_value_and_pointer()
    test_combat_receives_all_held_relics()
    test_state_dict_round_trip()
    test_policy_metadata_and_layer_budget()
    print("split policy isolation, gradients, upgrades, and serialization OK")
