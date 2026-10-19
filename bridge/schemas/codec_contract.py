"""Complete simulator <-> real-game identity and selection codec contract.

PAD/EMPTY are reserved for structurally absent fixed-width slots.  They are
never an alias target for a live entity.  This module expands canonical
identity entries as well as the irregular aliases so audits can inspect one
complete table instead of relying on scattered fallback transformations.
"""

from __future__ import annotations

from functools import lru_cache

from env.core import vocab as V
from env.core.cards_vocab import CARD_VOCAB, EMPTY
from env.core.ironclad_cards import IRONCLAD_CARDS

from . import combat_state as CS
from .entity_ids import (
    BOSS_ALIASES,
    BOSS_CANONICAL_IDS,
    CARD_ALIASES,
    COMBAT_SELECTION_SOURCES,
    DYNAMIC_COMBAT_SELECTION_SOURCES,
    DYNAMIC_EVENT_OPTION_KINDS,
    DYNAMIC_EVENT_OPTION_KIND_ALIASES,
    LIVE_RELIC_STATE_FIELDS,
    LIVE_RELIC_COUNTER_FIELDS,
    LIVE_RELIC_COUNTER_MODULI,
    LIVE_RELIC_COUNTER_STOP_TURNS,
    RELIC_COUNTER_DOMAINS,
    PHASE_RELIC_COUNTER_DOMAINS,
    REAL_RELIC_COUNTER_USAGE,
    DERIVED_RELIC_COUNTER_SOURCES,
    FRESH_RELIC_INITIAL_DATA,
    REAL_ROOM_CLASS_IDS,
    NATIVE_GLOBAL_FIELDS,
    POTION_CAPACITY_RULES,
    COMPACT_PLAYER_SCALAR_FIELDS,
    FRESH_CARD_MISC,
    JSON_PRESENCE_CONTRACTS,
    REAL_CARD_WIRE_FIELDS,
    CARD_STATE_FIELD_CONTRACTS,
    CARD_SPECIAL_DATA_FIELDS,
    CARD_STATE_TRANSPORT_FIELDS,
    COMBAT_SELECTION_TRANSPORT_FIELDS,
    NATIVE_CARD_QUEUE_FIELDS,
    NATIVE_CARD_INSTANCE_FIELDS,
    NATIVE_CARD_SNAPSHOT_FIELDS,
    NEOW_BONUS_IDS,
    NEOW_DRAWBACK_IDS,
    NEOW_IMPLICIT_DRAWBACK_IDS,
    NEOW_SCREEN_CONTRACTS,
    RELIC_COUNTER_SENTINELS,
    COMBAT_RELIC_COUNTER_SENTINELS,
    EMPTY_REAL_POTION_IDS,
    EVENT_ALIASES,
    MONSTER_ALIASES,
    NATIVE_MONSTER_STATUS_IDS,
    NATIVE_PLAYER_STATUS_IDS,
    HIT_REACTION_CONTRACTS,
    CARD_DAMAGE_TIMING_CONTRACTS,
    COMBAT_VICTORY_ACTION_CONTRACTS,
    MONSTER_DEATH_CONTRACTS,
    MONSTER_LIFECYCLE_FIELDS,
    NATIVE_COMBAT_CARD_ENUM_IDS,
    OVERWORLD_SELECTION_EVENTS,
    OVERWORLD_SELECTION_FLAGS,
    OVERWORLD_SELECTION_RELICS,
    POTION_ALIASES,
    REAL_REWARD_TYPE_IDS,
    REAL_COMMAND_IDS,
    REAL_SCREEN_TYPE_IDS,
    REAL_INTENT_COMPONENTS,
    REAL_CHARACTER_IDS,
    REAL_ROOM_PHASES,
    REAL_ACTION_PHASES,
    REAL_CARD_TYPE_IDS,
    REAL_CARD_RARITY_IDS,
    REAL_CARD_TARGET_IDS,
    REWARD_CHOICE_TYPES,
    RUN_KEY_FIELDS,
    REAL_POWER_ALIASES,
    ROOM_ALIASES,
    SCREEN_PHASES,
    SIM_POWER_ALIASES,
    STRUCTURAL_POWER_FIELDS,
    TRANSPORT_ONLY_REAL_EVENT_IDS,
    UNSUPPORTED_REAL_CARD_IDS,
    UNSUPPORTED_REAL_POWER_IDS,
    UNSUPPORTED_REAL_RELIC_IDS,
    UNSUPPORTED_SELECTION_SOURCES,
    native_power_enum,
)


def _identity(values, *sentinels):
    excluded = set(sentinels)
    return {value: value for value in values if value not in excluded}


REAL_TO_CANONICAL = {
    "card": _identity(CARD_VOCAB, EMPTY) | CARD_ALIASES,
    "relic": _identity(V.RELIC_VOCAB, "PAD"),
    "potion": _identity(V.POTION_VOCAB, "PAD") | POTION_ALIASES,
    "monster": _identity(V.MONSTER_VOCAB, "PAD") | MONSTER_ALIASES,
    "player_power": (
        _identity(V.PLAYER_BUFF_VOCAB)
        | {source: target for source, target in REAL_POWER_ALIASES.items()
           if target in V.PLAYER_BUFF_VOCAB_INDEX}
    ),
    "monster_power": (
        _identity(V.MONSTER_BUFF_VOCAB)
        | {source: target for source, target in REAL_POWER_ALIASES.items()
           if target in V.MONSTER_BUFF_VOCAB_INDEX}
    ),
    "event": _identity(V.EVENT_VOCAB, "PAD") | EVENT_ALIASES,
    "boss": _identity(BOSS_CANONICAL_IDS) | BOSS_ALIASES,
    "room": _identity(V.ROOM_VOCAB, "NONE", "INVALID") | ROOM_ALIASES,
    "screen": dict(SCREEN_PHASES),
    "select_context": _identity(V.SELECT_CONTEXT_VOCAB, "PAD"),
    "character": dict(REAL_CHARACTER_IDS),
}

# Fully expanded simulator tables.  The card table has one entry per native
# CardId enum name; constructing it from every frozen canonical token makes the
# generic spelling rule auditable and collision-checked rather than a runtime
# fallback for arbitrary input.
SIM_TO_CANONICAL = {
    "card": {
        CS.real_to_sim_card_enum(canonical): canonical
        for canonical in CARD_VOCAB
        if canonical != EMPTY
    },
    "monster": {
        native: CS._canonical_real_monster_id(real)
        for native, real in CS._MONSTER_SIM2REAL.items()
    },
    "player_power": (
        _identity(V.PLAYER_BUFF_VOCAB)
        | {source: target for source, target in SIM_POWER_ALIASES.items()
           if target in V.PLAYER_BUFF_VOCAB_INDEX}
    ),
    "monster_power": (
        _identity(V.MONSTER_BUFF_VOCAB)
        | {source: target for source, target in SIM_POWER_ALIASES.items()
           if target in V.MONSTER_BUFF_VOCAB_INDEX}
    ),
    "move": _identity(V.MOVE_VOCAB, "PAD"),
    "relic": _identity(V.RELIC_VOCAB, "PAD"),
    "potion": _identity(V.POTION_VOCAB, "PAD"),
    "event": _identity(V.EVENT_VOCAB, "PAD"),
    "boss": _identity(BOSS_CANONICAL_IDS),
}

REAL_MONSTER_MOVE_TO_CANONICAL = {
    real_monster: dict(moves)
    for real_monster, moves in CS._REAL_MONSTER_MOVES.items()
}

SELECTION_SOURCE_TO_CONTEXT = {
    "combat": dict(COMBAT_SELECTION_SOURCES),
    "combat_dynamic": {
        source: dict(variants)
        for source, variants in DYNAMIC_COMBAT_SELECTION_SOURCES.items()
    },
    "overworld_flag": dict(OVERWORLD_SELECTION_FLAGS),
    "overworld_relic": dict(OVERWORLD_SELECTION_RELICS),
    "overworld_event": dict(OVERWORLD_SELECTION_EVENTS),
    "unsupported": dict(UNSUPPORTED_SELECTION_SOURCES),
}


def complete_word_tables():
    """Return the expanded source tables, including explicit unverified rows.

    A declaration is an inventory disposition, never evidence that gameplay
    effects or all localization variants have passed a real-game comparison.
    """
    from bridge import event_contract
    event_options = {}
    for real_event, rules in event_contract.CONTRACTS.items():
        canonical = EVENT_ALIASES.get(real_event, real_event)
        event_options[canonical] = [
            {"semantic": index, "labels": labels.split('|'), "effect": effect,
             "effect_status": "unverified" if effect.startswith('pending_') else "implemented"}
            for index, labels, effect in rules
        ]
    relic_counters = {}
    for relic in REAL_TO_CANONICAL["relic"]:
        usage = REAL_RELIC_COUNTER_USAGE.get(relic)
        relic_counters[relic] = {
            "java_field": "AbstractRelic.counter",
            "native_persistent_field": DERIVED_RELIC_COUNTER_SOURCES.get(relic, "RelicInstance.data"),
            "fresh_native_initial_data": FRESH_RELIC_INITIAL_DATA.get(relic, 0),
            "java_class_with_counter_usage": usage[0] if usage else "AbstractRelic (inherited)",
            "meaning": usage[1] if usage else "inherited_display_sentinel",
            "persistent_value_map": RELIC_COUNTER_SENTINELS.get(relic, {}),
            "combat_value_map": COMBAT_RELIC_COUNTER_SENTINELS.get(relic, {}),
            "combat_runtime_field": LIVE_RELIC_COUNTER_FIELDS.get(relic),
            "java_numeric_domain": RELIC_COUNTER_DOMAINS.get(relic) or PHASE_RELIC_COUNTER_DOMAINS.get(relic),
            "combat_modulo": LIVE_RELIC_COUNTER_MODULI.get(relic),
            "combat_stop_turn": LIVE_RELIC_COUNTER_STOP_TURNS.get(relic),
            "inherited_display_value_map": {-1: 0} if relic not in REAL_RELIC_COUNTER_USAGE or relic == "CeramicFish" else {},
            "numeric_audit_status": (
                "mapped_runtime" if relic in LIVE_RELIC_COUNTER_FIELDS else
                "mapped_derived" if relic in DERIVED_RELIC_COUNTER_SOURCES else
                "mapped_persistent" if relic in RELIC_COUNTER_SENTINELS or relic in RELIC_COUNTER_DOMAINS else
                "mapped_display_sentinel"
            ),
            "phase_audit_status": (
                "mapped_map_choices_and_charge_receipt" if relic == "WingedGreaves" else
                "chest_reward_order_source_reviewed_not_real_game_verified"
                if relic in {"Matryoshka", "NlothsMask"} else
                "legacy_use_not_supported" if relic == "CeramicFish" else
                "numeric_transport_reviewed_not_full_effect_acceptance"
            ),
        }
    return {
        "real_to_canonical": REAL_TO_CANONICAL,
        "json_presence_contracts": JSON_PRESENCE_CONTRACTS,
        "sim_to_canonical": SIM_TO_CANONICAL,
        "monster_moves_by_real_id": REAL_MONSTER_MOVE_TO_CANONICAL,
        "selection_sources": SELECTION_SOURCE_TO_CONTEXT,
        "event_options": event_options,
        "structured_event_contracts": sorted(event_contract.STRUCTURED_CONTRACTS),
        "dynamic_event_kinds": DYNAMIC_EVENT_OPTION_KINDS,
        "dynamic_event_kind_aliases": DYNAMIC_EVENT_OPTION_KIND_ALIASES,
        "commands": sorted(REAL_COMMAND_IDS),
        "reward_types": dict(REWARD_CHOICE_TYPES),
        "chest_reward_link_contracts": {
            "Matryoshka": {
                "java_source": "Matryoshka.onChestOpen",
                "native_source": "GameContext::openTreasureRoomChest",
                "effect": "append extra relic before base chest relic; consume one remaining charge",
            },
            "SAPPHIRE_KEY": {
                "java_source": "AbstractChest.open / RewardItem.relicLink",
                "native_source": "Rewards::sapphireKey / executeRewardsAction",
                "effect": "mutually exclusive with last/base chest relic, not first/Matryoshka relic",
            },
            "NlothsMask": {
                "java_source": "NlothsMask.onChestOpenAfter / AbstractRoom.removeOneRelicFromRewards",
                "native_source": "GameContext::openTreasureRoomChest",
                "effect": "consume charge after generation; remove first relic and only its linked key",
            },
        },
        "run_keys": RUN_KEY_FIELDS,
        "real_room_classes": REAL_ROOM_CLASS_IDS,
        "native_combat_root_fields": NATIVE_GLOBAL_FIELDS,
        "potion_capacity_rules": POTION_CAPACITY_RULES,
        "compact_player_scalar_fields": {
            name: {"index": index, "source_and_consumer": meaning}
            for index, (name, meaning) in enumerate(COMPACT_PLAYER_SCALAR_FIELDS.items())
        },
        "live_relic_state_fields": LIVE_RELIC_STATE_FIELDS,
        "persistent_relic_counter_sentinels": RELIC_COUNTER_SENTINELS,
        "combat_relic_counter_sentinels": COMBAT_RELIC_COUNTER_SENTINELS,
        "relic_counter_contracts": relic_counters,
        "combat_counter_fields": {key: key for key in CS._REAL_COMBAT_COUNTER_KEYS},
        "native_relic_snapshot_fields": {"id": "required stable identity", "counter": "required native integer; no Java sentinel/default"},
        "fresh_card_misc": FRESH_CARD_MISC,
        "native_fresh_card_misc": {
            native: {"canonical": canonical, "initial_misc": FRESH_CARD_MISC.get(canonical, 0)}
            for native, canonical in SIM_TO_CANONICAL['card'].items()
        },
        "real_card_wire_fields": REAL_CARD_WIRE_FIELDS,
        "card_state_field_contracts": CARD_STATE_FIELD_CONTRACTS,
        "card_special_data_fields": CARD_SPECIAL_DATA_FIELDS,
        "card_state_transport_fields": CARD_STATE_TRANSPORT_FIELDS,
        "combat_selection_transport_fields": COMBAT_SELECTION_TRANSPORT_FIELDS,
        "native_card_queue_fields": NATIVE_CARD_QUEUE_FIELDS,
        "native_card_instance_fields": NATIVE_CARD_INSTANCE_FIELDS,
        "native_card_snapshot_fields": NATIVE_CARD_SNAPSHOT_FIELDS,
        "native_master_deck_snapshot_fields": {'id': 'native CardId spelling', 'upgrades': 'explicit count', 'misc': 'explicit current native value', 'bottle': 'NONE/ATTACK/SKILL/POWER; physical slot kept'},
        "search_boundary_relic_fields": {"Pocketwatch": "pocketwatch_cards_at_boundary"},
        "neow_bonus_keywords": NEOW_BONUS_IDS,
        "neow_drawback_keywords": NEOW_DRAWBACK_IDS,
        "neow_implicit_drawbacks": NEOW_IMPLICIT_DRAWBACK_IDS,
        "neow_screen_contracts": NEOW_SCREEN_CONTRACTS,
        "neow_transport_fields": {
            "screen": "simulator_neow_screen", "offers": "simulator_neow_options",
            "choice_index": "choice_index", "bonus": "bonus_type", "drawback": "drawback_type",
            "cached_hp_unit": "hp_unit", "absent_offer_marker": "drawback_id=0 only in unoffered slots",
        },
        "map_transport_fields": {
            "next_nodes": "all legal next-room records; up to seven columns, never truncated",
            "next_nodes_order": "CHOOSE UI index; independent of physical x column",
            "x": "MAP_NODE action column 0..6; unique within the candidate row",
            "y": "row 0..14; all next_nodes must be in one row",
            "symbol": "exact ROOM_ALIASES room identity; no absent-room fallback",
            "children": "game_state.map complete physical DAG edges; required, never invented by Wing Boots",
            "current_node": "actual current coordinate/room only; Java first node is (0,-1); edges are looked up in full map",
            "boss_available": "Act1..3 uses current column/final-row DAG slot; Act4 uses the actual (3,3) Boss leaf",
            "simulator_current_node": "v12 actual room coordinate/class across all screens; Java boss button lands at (-1,15)",
            "simulator_map_leaf_nodes": "v12 actual nonempty no-edge map nodes omitted by stock export; no guessed Boss record",
            "terminal_map_nodes": "typed TrueVictoryRoom display leaf retained outside frozen actionable room tokens; not PAD",
            "map_full_rooms": "105 physical slots, room absent only at genuine empty coordinates",
            "map_full_features": "x/y/current/choice/reachable plus all seven outgoing columns",
            "map_node_rooms": "frozen three-token prefix preview; not the complete action list",
            "map_node_x": "same preview order/scaling; wide choices require lossless full-map coverage",
        },
        "intent_components": REAL_INTENT_COMPONENTS,
        "room_phases": sorted(REAL_ROOM_PHASES),
        "action_phases": sorted(REAL_ACTION_PHASES),
        "card_types": sorted(REAL_CARD_TYPE_IDS),
        "card_rarities": sorted(REAL_CARD_RARITY_IDS),
        "card_targets_require_enemy": REAL_CARD_TARGET_IDS,
        "structural_powers": STRUCTURAL_POWER_FIELDS,
        "hit_reaction_contracts": HIT_REACTION_CONTRACTS,
        "card_damage_timing_contracts": CARD_DAMAGE_TIMING_CONTRACTS,
        "combat_victory_action_contracts": COMBAT_VICTORY_ACTION_CONTRACTS,
        "monster_death_contracts": MONSTER_DEATH_CONTRACTS,
        "monster_lifecycle_fields": MONSTER_LIFECYCLE_FIELDS,
        "monster_intent_damage_contract": {
            "java_source": "AbstractMonster.getIntentDmg / intentDmg",
            "wire": "move_adjusted_damage",
            "native_restore": "strict monster adjusted_damage (required)",
            "native_inactive_field": "Monster::inactiveIntentDamage",
            "observation_and_key": "Monster::getIntentDamage; dict, packed and search hash",
            "active_rule": "derive from actual powers/target; never override attack calculation",
            "inactive_rule": "preserve explicit cached display after powers disappear; keep entity and lifecycle",
            "reset": "setMove or fresh construct/spawn; active restored roots clear inactive cache",
        },
        "rejected_real_ids": {
            "cards": UNSUPPORTED_REAL_CARD_IDS,
            "relics": UNSUPPORTED_REAL_RELIC_IDS,
            "powers": UNSUPPORTED_REAL_POWER_IDS,
        },
        "empty_real_potion_ids": sorted(EMPTY_REAL_POTION_IDS),
        "transport_only_events": TRANSPORT_ONLY_REAL_EVENT_IDS,
    }


def _targets(table):
    for value in table.values():
        if isinstance(value, dict):
            yield from value.values()
        else:
            yield value


@lru_cache(maxsize=1)
def validate_codec_contract() -> dict[str, int]:
    """Raise on an incomplete, colliding, PAD-producing, or stale word table."""
    canonical_sets = {
        "card": set(CARD_VOCAB) - {EMPTY},
        "relic": set(V.RELIC_VOCAB) - {"PAD"},
        "potion": set(V.POTION_VOCAB) - {"PAD"},
        "monster": set(V.MONSTER_VOCAB) - {"PAD"},
        "move": set(V.MOVE_VOCAB) - {"PAD"},
        "player_power": set(V.PLAYER_BUFF_VOCAB),
        "monster_power": set(V.MONSTER_BUFF_VOCAB),
        "event": set(V.EVENT_VOCAB) - {"PAD"},
        "boss": set(BOSS_CANONICAL_IDS),
        "room": set(V.ROOM_VOCAB) - {"NONE", "INVALID"},
        "screen": set(V.PHASE_VOCAB),
        "select_context": set(V.SELECT_CONTEXT_VOCAB) - {"PAD"},
        "character": set(V.CHARACTER_VOCAB) - {"INVALID"},
    }

    for kind, table in REAL_TO_CANONICAL.items():
        targets = set(_targets(table))
        unknown = targets - canonical_sets[kind]
        if unknown:
            raise RuntimeError(f"{kind} real codec targets unknown tokens: {sorted(unknown)!r}")
        missing = canonical_sets[kind] - targets
        if kind != "screen" and missing:
            raise RuntimeError(f"{kind} real codec omits canonical tokens: {sorted(missing)!r}")
        if any(value in {"PAD", EMPTY, "NONE", "INVALID"} for value in targets):
            raise RuntimeError(f"{kind} real codec maps live input to a sentinel")
    if set(SCREEN_PHASES) != set(REAL_SCREEN_TYPE_IDS):
        raise RuntimeError(
            "CommunicationMod screen word table is incomplete: "
            f"missing={sorted(set(REAL_SCREEN_TYPE_IDS) - set(SCREEN_PHASES))!r} "
            f"extra={sorted(set(SCREEN_PHASES) - set(REAL_SCREEN_TYPE_IDS))!r}"
        )
    if not REAL_COMMAND_IDS or any(
        not isinstance(command, str) or command != command.upper()
        for command in REAL_COMMAND_IDS
    ):
        raise RuntimeError("CommunicationMod command word table is malformed")

    # Every stock Java identity has one declared disposition: accepted and
    # mapped, structurally empty, or rejected with an explicit reason.  An
    # exception may never shadow a supported spelling because that would make
    # behavior depend on which consumer checked the value first.
    exceptional = {
        "card": set(UNSUPPORTED_REAL_CARD_IDS),
        "relic": set(UNSUPPORTED_REAL_RELIC_IDS),
        "potion": set(EMPTY_REAL_POTION_IDS),
        "player_power": set(UNSUPPORTED_REAL_POWER_IDS),
        "monster_power": set(UNSUPPORTED_REAL_POWER_IDS),
    }
    for kind, rejected in exceptional.items():
        overlap = set(REAL_TO_CANONICAL[kind]) & rejected
        if overlap:
            raise RuntimeError(
                f"{kind} identities are both supported and exceptional: "
                f"{sorted(overlap)!r}"
            )
    structural_overlap = (
        set(STRUCTURAL_POWER_FIELDS) & set(UNSUPPORTED_REAL_POWER_IDS)
    )
    if structural_overlap:
        raise RuntimeError(
            "real powers are both structurally represented and unsupported: "
            f"{sorted(structural_overlap)!r}"
        )
    transport_event_overlap = (
        set(REAL_TO_CANONICAL["event"]) & set(TRANSPORT_ONLY_REAL_EVENT_IDS)
    )
    if transport_event_overlap:
        raise RuntimeError(
            "real event IDs are both policy events and transport-only: "
            f"{sorted(transport_event_overlap)!r}"
        )
    if set(REWARD_CHOICE_TYPES.values()) != set(REAL_REWARD_TYPE_IDS):
        raise RuntimeError(
            "reward choice word table does not exactly cover RewardItem types"
        )
    if len(REWARD_CHOICE_TYPES) != len(set(REWARD_CHOICE_TYPES)):
        raise RuntimeError("reward choice word table contains duplicate labels")
    if set(RUN_KEY_FIELDS) != {"ruby", "emerald", "sapphire"}:
        raise RuntimeError("run-key word table is incomplete")
    if set(NEOW_BONUS_IDS.values()) != set(range(19)) or set(NEOW_DRAWBACK_IDS.values()) != set(range(1, 6)):
        raise RuntimeError("Neow keyword tables omit or collide native IDs")
    if NEOW_IMPLICIT_DRAWBACK_IDS != {"BOSS_RELIC": 6}:
        raise RuntimeError("Neow starting-relic loss must be an explicit bonus implication")
    for relic, field in LIVE_RELIC_STATE_FIELDS.items():
        if relic not in canonical_sets["relic"] or field not in CS._REAL_COMBAT_COUNTER_KEYS:
            raise RuntimeError(f"live relic state word table names unknown field: {relic!r}/{field!r}")
    for relic, field in LIVE_RELIC_COUNTER_FIELDS.items():
        if relic not in canonical_sets["relic"] or field not in (*CS._REAL_COMBAT_COUNTER_KEYS, "$turn"):
            raise RuntimeError(f"live relic counter table names unknown field: {relic!r}/{field!r}")
    if (set(REAL_RELIC_COUNTER_USAGE) | set(DERIVED_RELIC_COUNTER_SOURCES)) - canonical_sets["relic"]:
        raise RuntimeError("relic counter source inventory names unknown relics")
    if set(FRESH_RELIC_INITIAL_DATA) - canonical_sets["relic"] or any(
        type(value) is not int or value <= 0 for value in FRESH_RELIC_INITIAL_DATA.values()
    ):
        raise RuntimeError("fresh relic initial-data word table is malformed")
    if set(FRESH_CARD_MISC) - canonical_sets['card'] or any(
        type(value) is not int or value <= 0 for value in FRESH_CARD_MISC.values()
    ):
        raise RuntimeError('fresh card misc word table is malformed')
    classified_counters = (set(RELIC_COUNTER_SENTINELS) | set(RELIC_COUNTER_DOMAINS)
                           | set(PHASE_RELIC_COUNTER_DOMAINS) | {"CeramicFish"})
    if classified_counters != set(REAL_RELIC_COUNTER_USAGE):
        raise RuntimeError("Java counter users and numeric disposition word tables differ")
    if (set(LIVE_RELIC_COUNTER_MODULI) | set(LIVE_RELIC_COUNTER_STOP_TURNS)) - set(LIVE_RELIC_COUNTER_FIELDS):
        raise RuntimeError("live relic transforms name unknown runtime fields")
    for table in (RELIC_COUNTER_SENTINELS, COMBAT_RELIC_COUNTER_SENTINELS):
        for relic, values in table.items():
            if relic not in canonical_sets["relic"] or not values:
                raise RuntimeError(f"relic counter sentinel word table is malformed: {relic!r}")
    for semantic, fields in RUN_KEY_FIELDS.items():
        if set(fields) != {"real", "sim"} or not all(fields.values()):
            raise RuntimeError(f"run-key word table is malformed for {semantic!r}")
    for event, kinds in DYNAMIC_EVENT_OPTION_KINDS.items():
        if event not in canonical_sets["event"]:
            raise RuntimeError(f"dynamic event word table names unknown event {event!r}")
        if not kinds or len(kinds) != len(set(kinds.values())):
            raise RuntimeError(f"dynamic event word table is empty/colliding for {event!r}")
        if any(type(index) is not int or not 0 <= index < 8 for index in kinds.values()):
            raise RuntimeError(f"dynamic event word table has invalid index for {event!r}")
    for event, aliases in DYNAMIC_EVENT_OPTION_KIND_ALIASES.items():
        kinds = DYNAMIC_EVENT_OPTION_KINDS.get(event, {})
        if any(alias in kinds or target not in kinds for alias, target in aliases.items()):
            raise RuntimeError(f"dynamic event aliases are malformed for {event!r}")
    # Import lazily to keep the low-level identity module independent from the
    # event effect verifier.  Every canonical event has exactly one declared
    # static, structured/dynamic, or Neow option contract.
    from bridge import event_contract
    contract_events = event_contract.canonical_contract_events()
    expected_events = canonical_sets["event"]
    if contract_events != expected_events:
        raise RuntimeError(
            "event option word table is incomplete: "
            f"missing={sorted(expected_events - contract_events)!r} "
            f"extra={sorted(contract_events - expected_events)!r}"
        )

    if len(SIM_TO_CANONICAL["card"]) != len(CARD_VOCAB) - 1:
        raise RuntimeError("sim card codec contains a spelling collision")
    for native, canonical in SIM_TO_CANONICAL["card"].items():
        if CS._sim_card_id(native) != canonical:
            raise RuntimeError(
                f"sim card codec is not reversible: {native!r} -> {canonical!r}"
            )

    unknown_native_cards = (
        set(NATIVE_COMBAT_CARD_ENUM_IDS) - set(SIM_TO_CANONICAL["card"])
    )
    if unknown_native_cards:
        raise RuntimeError(
            "native combat capability table names unknown CardId values: "
            f"{sorted(unknown_native_cards)!r}"
        )
    ironclad_native = {
        CS.real_to_sim_card_enum(card_id) for card_id in IRONCLAD_CARDS
    }
    missing_ironclad_rules = ironclad_native - set(NATIVE_COMBAT_CARD_ENUM_IDS)
    if missing_ironclad_rules:
        raise RuntimeError(
            "native combat capability table omits Ironclad cards: "
            f"{sorted(missing_ironclad_rules)!r}"
        )

    for kind, table in SIM_TO_CANONICAL.items():
        allowed = canonical_sets[kind]
        unknown = set(_targets(table)) - allowed
        if unknown:
            raise RuntimeError(f"{kind} sim codec targets unknown tokens: {sorted(unknown)!r}")

    real_monsters = set(CS._MONSTER_SIM2REAL.values())
    if set(REAL_MONSTER_MOVE_TO_CANONICAL) != real_monsters:
        raise RuntimeError("real monster move tables do not cover every simulator monster")
    for monster, moves in REAL_MONSTER_MOVE_TO_CANONICAL.items():
        if not moves:
            raise RuntimeError(f"real monster has an empty move table: {monster!r}")
        unknown = set(moves.values()) - canonical_sets["move"]
        if unknown:
            raise RuntimeError(
                f"real monster {monster!r} maps unknown moves: {sorted(unknown)!r}"
            )

    selection_targets = set(COMBAT_SELECTION_SOURCES.values())
    selection_targets.update(_targets(DYNAMIC_COMBAT_SELECTION_SOURCES))
    selection_targets.update(OVERWORLD_SELECTION_FLAGS.values())
    selection_targets.update(OVERWORLD_SELECTION_RELICS.values())
    selection_targets.update(OVERWORLD_SELECTION_EVENTS.values())
    if selection_targets != canonical_sets["select_context"]:
        raise RuntimeError(
            "selection source table and SELECT_CONTEXT_VOCAB differ: "
            f"missing={sorted(canonical_sets['select_context'] - selection_targets)!r} "
            f"extra={sorted(selection_targets - canonical_sets['select_context'])!r}"
        )
    supported_sources = set(COMBAT_SELECTION_SOURCES) | set(DYNAMIC_COMBAT_SELECTION_SOURCES)
    overlap = supported_sources & set(UNSUPPORTED_SELECTION_SOURCES)
    if overlap:
        raise RuntimeError(f"selection sources are both supported and unsupported: {sorted(overlap)!r}")

    if set(CS._POWER_DROP) != set(STRUCTURAL_POWER_FIELDS):
        raise RuntimeError("a dropped power has no documented structural representation")

    native_player = {native_power_enum(power) for power in V.PLAYER_BUFF_VOCAB}
    if native_player != set(NATIVE_PLAYER_STATUS_IDS):
        raise RuntimeError(
            "player power vocabulary and native PlayerStatus differ: "
            f"missing={sorted(set(NATIVE_PLAYER_STATUS_IDS) - native_player)!r} "
            f"extra={sorted(native_player - set(NATIVE_PLAYER_STATUS_IDS))!r}"
        )
    native_monster = {native_power_enum(power) for power in V.MONSTER_BUFF_VOCAB}
    # ASLEEP is transported by the monster's move and intentionally absent
    # from the ordinary buff vector; every other native status must be present.
    expected_monster = set(NATIVE_MONSTER_STATUS_IDS) - {"ASLEEP"}
    if native_monster != expected_monster:
        raise RuntimeError(
            "monster power vocabulary and native MonsterStatus differ: "
            f"missing={sorted(expected_monster - native_monster)!r} "
            f"extra={sorted(native_monster - expected_monster)!r}"
        )
    if len(NATIVE_PLAYER_STATUS_IDS) != len(set(NATIVE_PLAYER_STATUS_IDS)):
        raise RuntimeError("native player status word table contains duplicates")
    if len(NATIVE_MONSTER_STATUS_IDS) != len(set(NATIVE_MONSTER_STATUS_IDS)):
        raise RuntimeError("native monster status word table contains duplicates")

    return {
        "real_tokens": sum(len(table) for table in REAL_TO_CANONICAL.values()),
        "sim_tokens": sum(len(table) for table in SIM_TO_CANONICAL.values()),
        "monster_moves": sum(len(table) for table in REAL_MONSTER_MOVE_TO_CANONICAL.values()),
        "selection_sources": (
            len(COMBAT_SELECTION_SOURCES)
            + len(DYNAMIC_COMBAT_SELECTION_SOURCES)
            + len(OVERWORLD_SELECTION_FLAGS)
            + len(OVERWORLD_SELECTION_RELICS)
            + len(OVERWORLD_SELECTION_EVENTS)
            + len(UNSUPPORTED_SELECTION_SOURCES)
        ),
        "native_player_statuses": len(NATIVE_PLAYER_STATUS_IDS),
        "native_monster_statuses": len(NATIVE_MONSTER_STATUS_IDS),
        "native_combat_cards": len(NATIVE_COMBAT_CARD_ENUM_IDS),
        "reward_types": len(REAL_REWARD_TYPE_IDS),
        "screen_types": len(REAL_SCREEN_TYPE_IDS),
        "command_keywords": len(REAL_COMMAND_IDS),
        "intent_keywords": len(REAL_INTENT_COMPONENTS),
        "room_phase_keywords": len(REAL_ROOM_PHASES),
        "action_phase_keywords": len(REAL_ACTION_PHASES),
        "card_type_keywords": len(REAL_CARD_TYPE_IDS),
        "card_rarity_keywords": len(REAL_CARD_RARITY_IDS),
        "card_target_keywords": len(REAL_CARD_TARGET_IDS),
        "run_key_fields": len(RUN_KEY_FIELDS),
        "native_combat_root_fields": len(NATIVE_GLOBAL_FIELDS),
        "real_room_classes": len(REAL_ROOM_CLASS_IDS),
        "compact_player_scalar_fields": len(COMPACT_PLAYER_SCALAR_FIELDS),
        "real_card_wire_fields": len(REAL_CARD_WIRE_FIELDS),
        "card_state_field_contracts": len(CARD_STATE_FIELD_CONTRACTS),
        "card_state_transport_fields": len(CARD_STATE_TRANSPORT_FIELDS),
        "combat_selection_transport_fields": len(COMBAT_SELECTION_TRANSPORT_FIELDS),
        "native_card_queue_fields": len(NATIVE_CARD_QUEUE_FIELDS),
        "hit_reaction_contracts": len(HIT_REACTION_CONTRACTS),
        "card_damage_timing_contracts": len(CARD_DAMAGE_TIMING_CONTRACTS),
        "combat_victory_action_contracts": len(COMBAT_VICTORY_ACTION_CONTRACTS),
        "monster_death_contracts": len(MONSTER_DEATH_CONTRACTS),
        "monster_lifecycle_fields": len(MONSTER_LIFECYCLE_FIELDS),
        "card_special_data_fields": len(CARD_SPECIAL_DATA_FIELDS),
        "relic_counter_contracts": len(canonical_sets["relic"]),
        "java_relic_counter_users": len(REAL_RELIC_COUNTER_USAGE),
        "combat_counter_fields": len(CS._REAL_COMBAT_COUNTER_KEYS),
        "live_relic_counter_fields": len(LIVE_RELIC_COUNTER_FIELDS),
        "neow_bonus_keywords": len(NEOW_BONUS_IDS),
        "neow_drawback_keywords": len(NEOW_DRAWBACK_IDS),
        "neow_screen_keywords": len(NEOW_SCREEN_CONTRACTS),
        "dynamic_event_kinds": sum(len(kinds) for kinds in DYNAMIC_EVENT_OPTION_KINDS.values()),
        "event_option_contracts": len(contract_events),
        "explicit_unsupported_real_ids": (
            len(UNSUPPORTED_REAL_CARD_IDS)
            + len(UNSUPPORTED_REAL_RELIC_IDS)
            + len(UNSUPPORTED_REAL_POWER_IDS)
            + len(TRANSPORT_ONLY_REAL_EVENT_IDS)
        ),
    }
