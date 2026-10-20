"""Canonical combat-state schema + normalizers for differential testing.

Differential testing compares the C++ sim (sts_lightspeed) against the real
Java game. The two expose combat state in very different shapes, so both get
normalized into the one canonical dict defined here; `diff()` then compares two
canonical dicts field by field.

Canonical combat state (every id is the game's stable English `id`, NOT the
localized `name` field, which is GBK-localized and unstable):

  {
    "turn": int,
    "player": {"hp", "max_hp", "block", "energy", "powers": {power_id: amount},
               "bombs": [{"turns": int, "damage": int}, ...]},
    "monsters": [
      {"id", "hp", "max_hp", "block", "gone", "base_damage",
       "adjusted_damage", "hits", "move", "move_id",
       "powers": {power_id: amount}}, ...],
    "hand": [{"id", "up", "cost"}, ...],
    "draw"/"discard"/"exhaust": [{"id", "up"}, ...],
  }

- normalize_real(game_state): from a CommunicationMod frame.
- normalize_sim(raw):         from sim BattleContext.to_dict().
"""


# ============================ real-game side ============================

from functools import lru_cache

from .entity_ids import (
    CARD_ALIASES,
    MONSTER_ALIASES,
    REAL_POWER_ALIASES,
    REAL_INTENT_COMPONENTS,
    LIVE_RELIC_STATE_FIELDS,
    native_combat_relic_counter,
    SIM_POWER_ALIASES,
    SIMULATOR_PARITY_EXPORT_VERSION,
    RUN_KEY_FIELDS,
    STRUCTURAL_POWER_FIELDS,
    THIEF_STATE_FIELDS,
    LAGAVULIN_STATE_FIELDS,
    MONSTER_MOVE_STATE_CONTRACTS,
    UNSUPPORTED_REAL_CARD_IDS,
    UNSUPPORTED_REAL_POWER_IDS,
    UNSUPPORTED_REAL_RELIC_IDS,
)

# Powers whose semantics the native simulator stores outside its status table
# are omitted from the public power comparison.  In particular, Exploder's
# countdown is encoded by its real move history/current move on every synced
# frame; the native rule uses that same history to select SLAM/EXPLODE and has
# no separate EXPLOSIVE status slot.  Keeping the Java-only display counter in
# the canonical power map would therefore reject an otherwise exact root.
# Pen Nib's armed power remains observable separately from its relic counter;
# OBS18 includes it in player_buffs.
_POWER_DROP = set(STRUCTURAL_POWER_FIELDS)
BOOLEAN_PLAYER_POWERS = frozenset({
    "Confused", "Hex", "Barricade", "Blasphemer",
    "Corruption", "Electro", "Surrounded", "Master Reality", "Pen Nib",
    "Wrath Next Turn", "No Draw",
})
BOOLEAN_MONSTER_POWERS = frozenset({
    "Barricade", "Minion", "Minion Leader", "Painful Stabs", "Regrow",
    "Shifting", "Stasis",
})
_REAL_POWER_CANON_ID = REAL_POWER_ALIASES

_REAL_COMBAT_COUNTER_KEYS = (
    "happy_flower", "incense_burner", "ink_bottle", "inserter",
    "nunchaku", "pen_nib", "sundial", "necronomicon_used",
    "cards_played", "attacks_played", "skills_played",
    "orange_attack", "orange_skill", "orange_power",
    "combust_hp_loss", "deva_form_energy_per_turn",
    "echo_form_cards_doubled", "panache_counter",
    "cards_discarded", "times_damaged",
    "bomb_1", "bomb_2", "bomb_3", "last_attack_unblocked_damage",
    *LIVE_RELIC_STATE_FIELDS.values(),
    "neow_lament",
)


def _validated_combat_counters(counters):
    unknown = set(counters) - set(_REAL_COMBAT_COUNTER_KEYS)
    if unknown:
        raise ValueError(f"combat counter word table has unknown fields: {sorted(unknown)!r}")
    binary = {"necronomicon_used", "orange_attack", "orange_skill", "orange_power",
              *LIVE_RELIC_STATE_FIELDS.values()}
    result = {}
    for key in _REAL_COMBAT_COUNTER_KEYS:
        value = counters[key]
        if key in binary:
            if type(value) not in (int, bool) or value not in (0, 1):
                raise ValueError(f"combat counter {key!r} is not boolean: {value!r}")
        elif type(value) is not int:
            raise ValueError(f"combat counter {key!r} is not an integer: {value!r}")
        result[key] = int(value)
    return result


def _validated_bombs(bombs, counters, *, real_powers=None):
    """Retain independent hits; the old three totals are only a projection."""
    if not isinstance(bombs, list):
        raise ValueError("bomb instances must be an explicit list, including when empty")
    wire = real_powers is not None
    keys = {"turns", "damage", "id"} if wire else {"turns", "damage"}
    result, ids, totals = [], [], [0, 0, 0]
    for record in bombs:
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError(f"bomb instance fields are incomplete or unmapped: {record!r}")
        turns, damage = record["turns"], record["damage"]
        if type(turns) is not int or not 1 <= turns <= 3:
            raise ValueError(f"bomb countdown outside 1..3: {turns!r}")
        if type(damage) is not int or not 0 <= damage <= 2147483647:
            raise ValueError(f"bomb damage outside Java integer range: {damage!r}")
        totals[turns - 1] += damage
        result.append({"turns": turns, "damage": damage})
        if wire:
            ids.append(record["id"])
    if wire:
        actual = [p for p in real_powers
                  if p["id"].startswith("TheBomb") and p["id"][7:].isdigit()]
        expected_ids = [p["id"] for p in actual]
        if ids != expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("bomb instances differ from ordered unique real power identities")
        if any(type(p["amount"]) is not int or p["amount"] != b["turns"]
               for p, b in zip(actual, result)):
            raise ValueError("bomb countdown differs from its actual power amount")
    for turns, total in enumerate(totals, 1):
        if total > 2147483647 or counters[f"bomb_{turns}"] != total:
            raise ValueError("bomb counter differs from explicit bomb instance totals")
    return result


def _powers_real(power_list, *, include_structural=False):
    # real game flattens every buff/debuff into a list; key by stable English id
    if not isinstance(power_list, list):
        raise ValueError(f"real powers must be a list, got {power_list!r}")
    out = {}
    for index, power in enumerate(power_list):
        if not isinstance(power, dict) or "id" not in power or "amount" not in power:
            raise ValueError(f"malformed real power record {index}: {power!r}")
        raw_name = str(power["id"])
        if raw_name.startswith("TheBomb") and raw_name[7:].isdigit():
            # Each copy has a generated ID (TheBomb0, TheBomb1, ...).  Its
            # countdown/private damage survive as ordered independent bombs.
            continue
        if raw_name in UNSUPPORTED_REAL_POWER_IDS:
            raise ValueError(
                f"unsupported real power ID {raw_name!r}: "
                f"{UNSUPPORTED_REAL_POWER_IDS[raw_name]}"
            )
        name = _REAL_POWER_CANON_ID.get(raw_name, raw_name)
        if name in _POWER_DROP and not include_structural:
            continue
        if name in out:
            raise ValueError(f"duplicate real power identity: {name!r}")
        out[name] = int(power["amount"])
    return out


def _player_powers_real(power_list):
    # Presence powers commonly use amount=-1 in CommunicationMod.
    powers = {name: 1 if name in BOOLEAN_PLAYER_POWERS else amount
              for name, amount in _powers_real(power_list).items()}
    from env.core.vocab import PLAYER_BUFF_VOCAB_INDEX
    unknown = set(powers) - set(PLAYER_BUFF_VOCAB_INDEX)
    if unknown:
        raise ValueError(f"unknown real player power IDs: {sorted(unknown)!r}")
    return powers


def _monster_powers_real(power_list):
    powers = {name: 1 if name in BOOLEAN_MONSTER_POWERS else amount
              for name, amount in _powers_real(power_list).items()}
    from env.core.vocab import MONSTER_BUFF_VOCAB_INDEX
    unknown = set(powers) - set(MONSTER_BUFF_VOCAB_INDEX)
    if unknown:
        raise ValueError(f"unknown real monster power IDs: {sorted(unknown)!r}")
    return powers


# status cards with no upgraded variant: the sim can mark them upgraded (e.g. an
# upgrade-all effect blindly flips the bit) while the real game keeps them at 0.
# The cards behave identically either way, so canonicalize the upgrade to 0.
_NO_UPGRADE = {"Dazed", "Wound", "Slimed", "Void"}  # Burn excluded (Burn+ exists)

_CARD_CANON_ID = CARD_ALIASES


def _card(card_id, up, cost=None):
    if not isinstance(card_id, str):
        raise ValueError(f"card ID must be a string: {card_id!r}")
    if type(up) is not int or not 0 <= up <= 32767:
        raise ValueError(f"card upgrade count must be an integer in native range: {up!r}")
    if card_id in UNSUPPORTED_REAL_CARD_IDS:
        raise ValueError(
            f"unsupported real card ID {card_id!r}: "
            f"{UNSUPPORTED_REAL_CARD_IDS[card_id]}"
        )
    card_id = _CARD_CANON_ID.get(card_id, card_id)
    from env.core.cards_vocab import CARD_INDEX
    if card_id == "EMPTY":
        raise ValueError("a live card identity cannot be the EMPTY sentinel")
    if card_id not in CARD_INDEX:
        raise ValueError(f"unknown card ID: {card_id}")
    card = {"id": card_id, "up": 0 if card_id in _NO_UPGRADE else up}
    if cost is not None:
        if type(cost) is not int or not -3 <= cost <= 127:
            raise ValueError(f"card cost must be an integer in native range: {cost!r}")
        value = cost
        # CommunicationMod follows AbstractCard.COST_UNPLAYABLE (-2), while
        # the native simulator distinguishes unplayable statuses (-2) from
        # unplayable curses (-3).  They have one public/action meaning here.
        card["cost"] = -2 if value in (-3, -2) else value
    return card


def _cards_real(pile, *, include_cost=False):
    if not isinstance(pile, list):
        raise ValueError(f"real card pile must be a list, got {pile!r}")
    cards = []
    for index, c in enumerate(pile):
        if not isinstance(c, dict):
            raise ValueError(f"malformed real card record {index}: {c!r}")
        for key in ("id", "upgrades"):
            if key not in c:
                raise ValueError(f"real card record is missing {key}: {c!r}")
        if include_cost and "cost" not in c:
            raise ValueError(f"real hand card is missing its current cost: {c!r}")
        cards.append(
            _card(c["id"], c["upgrades"], c["cost"] if include_cost else None)
        )
    return cards


def _attack(base, hits):
    # unify the "not attacking" case: base<=0 => (0 dmg, 0 hits) on both sides,
    # else (base, hits). Avoids -1/0 and 0/1 noise for non-attack intents.
    base = max(0, base)
    return base, (hits if base > 0 else 0)


def _monster_lifecycle(record, *, gone_key="gone"):
    fields = ("half_dead", "is_dying", "is_dead", "is_escaping", "escape_next")
    for key in (*fields, gone_key, "unawakened"):
        if key not in record or type(record[key]) is not bool:
            raise ValueError(f"monster lifecycle requires an explicit boolean {key!r}")
    gone = record["half_dead"] or record["is_dying"] or record["is_escaping"]
    if record[gone_key] != gone:
        raise ValueError("monster gone disagrees with lifecycle flags")
    return {key: record[key] for key in fields}


def _validated_thief_state(monster_id, state):
    if monster_id not in {'Looter', 'Mugger'}:
        if state is not None:
            raise ValueError(f"non-thief {monster_id!r} carries thief private state")
        return None
    if not isinstance(state, dict) or set(state) != set(THIEF_STATE_FIELDS):
        raise ValueError(f"thief {monster_id!r} requires complete explicit private state")
    if any(type(value) is not int for value in state.values()):
        raise ValueError("thief private state must contain actual integers")
    if (not 0 <= state['stolen_gold'] <= 2147483647
            or not 0 <= state['slash_count'] <= 3
            or state['gold_amount'] not in (15, 20)):
        raise ValueError(f"thief private state outside stock/native domain: {state!r}")
    return dict(state)


def _validated_lagavulin_state(monster_id, state):
    if monster_id != 'Lagavulin':
        if state is not None:
            raise ValueError(f"non-Lagavulin {monster_id!r} carries sleep state")
        return None
    if not isinstance(state, dict) or set(state) != set(LAGAVULIN_STATE_FIELDS):
        raise ValueError("Lagavulin requires explicit sleep state and idle count")
    if (type(state['sleeping']) is not bool or type(state['idle_count']) is not int
            or not 0 <= state['idle_count'] <= 3):
        raise ValueError(f"invalid Lagavulin gameplay state: {state!r}")
    return dict(state)


def _validated_wizard_charge(monster_id, value):
    if monster_id != 'GremlinWizard':
        if value is not None:
            raise ValueError('non-wizard carries a charge counter')
        return None
    if type(value) is not int or not 0 <= value <= 3:
        raise ValueError('GremlinWizard requires actual currentCharge in 0..3')
    return value


def _validated_writhing_mass_used_implant(monster_id, value):
    if monster_id != 'WrithingMass':
        if value is not None:
            raise ValueError('non-WrithingMass carries an implant flag')
        return None
    if type(value) is not bool:
        raise ValueError('WrithingMass requires actual usedMegaDebuff boolean')
    return value


def _validated_move_state(monster_id, state):
    fields = MONSTER_MOVE_STATE_CONTRACTS.get(monster_id)
    if fields is None:
        if state is not None:
            raise ValueError(f'{monster_id} cannot carry this species-specific move state')
        return None
    if not isinstance(state, dict) or set(state) != set(fields):
        raise ValueError(f'{monster_id} requires exact move state fields {sorted(fields)}')
    for key, contract in fields.items():
        value = state[key]
        if contract['type'] == 'bool':
            valid = type(value) is bool
        else:
            valid = type(value) is int and contract['min'] <= value <= contract['max']
        if not valid:
            raise ValueError(f'invalid {monster_id} move state {key}: {value!r}')
    return dict(state)


def _monster_real(m, *, intents_hidden=False):
    required = {
        "id", "current_hp", "max_hp", "block", "is_gone", "powers",
        "move_base_damage", "move_adjusted_damage", "move_hits", "move_id",
        "last_move_id", "second_last_move_id", "half_dead",
        "current_intent_export", "intent", "unawakened",
        "simulator_thief_state",
        "simulator_lagavulin_state",
        "simulator_wizard_charge",
        "simulator_writhing_mass_used_implant",
        "simulator_move_state",
    }
    missing = required - set(m)
    if missing:
        raise ValueError(
            f"real monster record is missing {sorted(missing)!r}: {m!r}"
        )
    if m["current_intent_export"] != SIMULATOR_PARITY_EXPORT_VERSION:
        raise ValueError(
            "real monster state was not produced by the required parity exporter: "
            f"{m['current_intent_export']!r}"
        )
    if m["intent"] not in REAL_INTENT_COMPONENTS:
        raise ValueError(f"unknown real monster intent keyword: {m['intent']!r}")
    canonical_id = _canonical_real_monster_id(m["id"])
    from env.core.vocab import MONSTER_VOCAB_INDEX
    if canonical_id not in MONSTER_VOCAB_INDEX or canonical_id == "PAD":
        raise ValueError(f"unknown real monster ID: {m['id']!r}")
    base, hits = _attack(m["move_base_damage"], m["move_hits"])
    adjusted, _ = _attack(m["move_adjusted_damage"], hits)
    hp = m["current_hp"]
    lifecycle = _monster_lifecycle(m, gone_key="is_gone")
    if m["unawakened"] and canonical_id != "AwakenedOne":
        raise ValueError("only Awakened One can have first-life protection")
    gone = m["is_gone"]
    from env.core.vocab import MOVE_VOCAB_INDEX
    moves = []
    for key in ("move_id", "last_move_id", "second_last_move_id"):
        name = real_to_sim_monster_move(m["id"], m[key])
        if name == "INVALID":
            index = 0  # The real -1/no-history sentinel, never an unknown move.
        else:
            try:
                index = MOVE_VOCAB_INDEX[name]
            except KeyError as exc:
                raise ValueError(f"mapped real monster move {name!r} is absent from MOVE_VOCAB") from exc
        moves.append((name, index))
    (move, move_id), (last_move, last_move_id), (second_last_move, second_last_move_id) = moves
    if move == "INVALID" and (not gone or lifecycle["half_dead"]):
        raise ValueError(f"active/reviving real monster {m['id']!r} has no current move")
    return {
        "id": canonical_id,
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": gone,
        **lifecycle,
        "base_damage": base,
        "adjusted_damage": adjusted,
        "hits": hits,
        "unawakened": m["unawakened"],
        "thief_state": _validated_thief_state(canonical_id, m['simulator_thief_state']),
        "lagavulin_state": _validated_lagavulin_state(canonical_id, m['simulator_lagavulin_state']),
        "wizard_charge": _validated_wizard_charge(canonical_id, m['simulator_wizard_charge']),
        "writhing_mass_used_implant": _validated_writhing_mass_used_implant(
            canonical_id, m['simulator_writhing_mass_used_implant']),
        "move_state": _validated_move_state(canonical_id, m['simulator_move_state']),
        # Java exposes Writhing Mass's attack-triggered intent reroll as the
        # presence power Compulsive. The native simulator stores the same
        # mechanism in the REACTIVE status bit while reusing its numeric slot
        # as a transient queued-hit counter. Carry presence separately so a
        # settled root does not inject a fake pending reroll count of one.
        "reactive": (
            m["id"] == "WrithingMass"
            and any(p["id"] == "Compulsive" for p in m["powers"])
        ),
        # Dead/halfDead entities can still have pending power callbacks.
        "powers": _monster_powers_real(m["powers"]),
        # CommunicationMod and the simulator use different numeric move-id
        # spaces. Translate the real byte through the explicit per-monster
        # table, then publish the actual MOVE_VOCAB index. A live monster with a
        # missing/unmapped move is an incomplete bridge frame, never PAD.
        "move": move,
        "move_id": move_id,
        "last_move": last_move,
        "last_move_id": last_move_id,
        "second_last_move": second_last_move,
        "second_last_move_id": second_last_move_id,
        "intent": m["intent"],
        "intent_hidden": False,
        "real_move_id": m["move_id"],
        "real_last_move_id": m["last_move_id"],
        "real_second_last_move_id": m["second_last_move_id"],
    }


def _real_energy_per_turn(game_state):
    cs = game_state["combat_state"]
    if "simulator_energy_per_turn" not in cs:
        raise ValueError(
            "real combat state is missing simulator_energy_per_turn; install "
            "the complete simulator-parity CommunicationMod exporter"
        )
    value = int(cs["simulator_energy_per_turn"])
    if value < 0 or value > 127:
        raise ValueError(f"real energy-per-turn is outside native range: {value}")
    return value


def _validated_draw_state(amount, just_applied, powers):
    if type(amount) is not int or not 0 <= amount <= 127:
        raise ValueError('base turn draw must be an explicit integer within native 0..127')
    if 'Draw Reduction' in powers:
        if (type(powers['Draw Reduction']) is not int or not 1 <= powers['Draw Reduction'] <= 32767
                or type(just_applied) is not bool):
            raise ValueError('Draw Reduction requires actual positive duration and boolean justApplied')
    elif just_applied is not None:
        raise ValueError('absent Draw Reduction requires explicit null justApplied')
    return {'card_draw_per_turn': amount, 'draw_reduction_just_applied': just_applied}


def normalize_real(game_state):
    """`game_state` is the dict under the top-level 'game_state' key of a
    CommunicationMod frame. Returns the canonical combat state, or None if the
    frame is not in combat."""
    cs = game_state.get("combat_state")
    if not cs:
        return None
    if cs.get("simulator_parity_export") != SIMULATOR_PARITY_EXPORT_VERSION:
        raise ValueError(
            "real combat state was not produced by the required parity exporter: "
            f"{cs.get('simulator_parity_export')!r}"
        )
    p = cs["player"]
    exported_counters = cs.get("simulator_combat_counters")
    if not isinstance(exported_counters, dict):
        raise ValueError(
            "real combat state is missing simulator_combat_counters; install "
            "the complete simulator-parity CommunicationMod exporter"
        )
    missing_counters = set(_REAL_COMBAT_COUNTER_KEYS) - set(exported_counters)
    if missing_counters:
        raise ValueError(
            "real combat counter export is incomplete: "
            f"{sorted(missing_counters)!r}"
        )
    combat_counters = _validated_combat_counters(exported_counters)
    combat_counters["pen_nib"] = native_combat_relic_counter("Pen Nib", combat_counters["pen_nib"])
    player_powers = _player_powers_real(p["powers"])
    bombs = _validated_bombs(cs.get("simulator_bombs"), combat_counters,
                            real_powers=p["powers"])
    power_overrides = cs.get("simulator_player_power_overrides")
    if not isinstance(power_overrides, dict):
        raise ValueError(
            "real combat state is missing simulator_player_power_overrides; "
            "install the complete simulator-parity CommunicationMod exporter"
        )
    unknown_overrides = set(power_overrides) - {"Panache"}
    if unknown_overrides:
        raise ValueError(
            f"unknown simulator player power overrides: {sorted(unknown_overrides)!r}"
        )
    if "Panache" in player_powers:
        if "Panache" not in power_overrides:
            raise ValueError("Panache is active but its private damage was not exported")
        player_powers["Panache"] = int(power_overrides["Panache"])
    elif "Panache" in power_overrides:
        raise ValueError("Panache override was exported without an active Panache power")
    from .card_state import real_card_state
    instances = {name: [real_card_state(card) for card in cs[wire]]
                 for name, wire in (('hand', 'hand'), ('draw', 'draw_pile'),
                                    ('discard', 'discard_pile'), ('exhaust', 'exhaust_pile'))}
    monsters = [_monster_real(m, intents_hidden=any(
        r["id"] == "Runic Dome" for r in game_state["relics"]
    )) for m in cs["monsters"]]
    stasis = []
    for monster in cs['monsters']:
        if 'simulator_stasis_card' not in monster:
            raise ValueError('real monster is missing simulator_stasis_card')
        held = monster['simulator_stasis_card']
        active = (not monster['is_gone'] and monster['current_hp'] > 0
                  and any(p['id'] == 'Stasis' for p in monster['powers']))
        if active != (held is not None) or (active and monster['id'] != 'BronzeOrb'):
            raise ValueError('stasis card export disagrees with its live Bronze Orb owner')
        stasis.append(None if held is None else real_card_state(held))
    return {
        "turn": cs["turn"],
        "post_enemy_turn_boundary": False,
        "player": {
            "hp": p["current_hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "gold": game_state["gold"],
            "combat_counters": combat_counters,
            "bombs": bombs,
            "powers": {k: v for k, v in player_powers.items()
                       if k != "Berserk"},
            "energy_per_turn": _real_energy_per_turn(game_state),
            **_validated_draw_state(cs['simulator_card_draw_per_turn'],
                                    cs['simulator_draw_reduction_just_applied'], player_powers),
        },
        "monsters": monsters,
        "card_instances": instances,
        "stasis_cards": stasis,
        **{name: [_card(card['id'], card['up'],
                       (0 if card['free_to_play_once'] else card['cost']) if name == 'hand' else None)
                  for card in pile] for name, pile in instances.items()},
    }


# ============================== sim side ===============================
# Map the sim's own enum-name strings to the real game's CommunicationMod ids.

# Cards: most Ironclad ids are Title-Case-with-spaces ("Burning Pact"), which
# the generic transform produces. Only the colored Strike/Defend are irregular.
_CARD_OVERRIDE = {
    "STRIKE_RED": "Strike_R", "DEFEND_RED": "Defend_R",
    "STRIKE_GREEN": "Strike_G", "DEFEND_GREEN": "Defend_G",
    "STRIKE_BLUE": "Strike_B", "DEFEND_BLUE": "Defend_B",
    "STRIKE_PURPLE": "Strike_P", "DEFEND_PURPLE": "Defend_P",
    "ASCENDERS_BANE": "AscendersBane",
    "CURSE_OF_THE_BELL": "CurseOfTheBell",
    # Ironclad/neutral/event cards: Java IDs are not display-name title case.
    "APPARITION": "Ghostly",
    "BLOOD_FOR_BLOOD": "Blood for Blood",
    "FLASH_OF_STEEL": "Flash of Steel",
    "HAND_OF_GREED": "HandOfGreed",
    "JAX": "J.A.X.",
    "MASTER_OF_STRATEGY": "Master of Strategy",
    "PANIC_BUTTON": "PanicButton",
    "RITUAL_DAGGER": "RitualDagger",
}
_CARD_OVERRIDE = {key: CARD_ALIASES.get(value, value) for key, value in _CARD_OVERRIDE.items()}


@lru_cache(maxsize=1024)
def _sim_card_id(name):
    if name in _CARD_OVERRIDE:
        return _CARD_OVERRIDE[name]
    return " ".join(w.capitalize() for w in name.split("_"))


_CARD_OVERRIDE_INV = {v: k for k, v in _CARD_OVERRIDE.items()}


def real_to_sim_card_enum(real_id):
    """Inverse of _sim_card_id: real CommunicationMod id -> sim enum name, for
    force-syncing the sim deck (e.g. 'Strike_R'->'STRIKE_RED', 'Searing Blow'->
    'SEARING_BLOW')."""
    real_id = CARD_ALIASES.get(real_id, real_id)
    if real_id in _CARD_OVERRIDE_INV:
        return _CARD_OVERRIDE_INV[real_id]
    return real_id.upper().replace(" ", "_")


# Monsters: real ids are irregular CamelCase ("AcidSlime_S", "SlaverRed",
# "FuzzyLouseNormal"); no clean transform, so map explicitly and raise on a miss
# (a missing entry should be loud, not a silent wrong compare).
_SIM_MONSTER_INVALID = "INVALID = 0"  # empty/not-yet-activated slot (cf. monsterIdStrings[0])
_MONSTER_SIM2REAL = {
    # act 1
    "ACID_SLIME_S": "AcidSlime_S", "ACID_SLIME_M": "AcidSlime_M", "ACID_SLIME_L": "AcidSlime_L",
    "SPIKE_SLIME_S": "SpikeSlime_S", "SPIKE_SLIME_M": "SpikeSlime_M", "SPIKE_SLIME_L": "SpikeSlime_L",
    "CULTIST": "Cultist", "JAW_WORM": "JawWorm",
    # The Java class names are counter-intuitive: Defensive is the green
    # weak-applying louse, while Normal is the red strength-gaining louse.
    "GREEN_LOUSE": "FuzzyLouseDefensive", "RED_LOUSE": "FuzzyLouseNormal",
    "GREMLIN_NOB": "GremlinNob", "HEXAGHOST": "Hexaghost",
    "LOOTER": "Looter", "SENTRY": "Sentry", "LAGAVULIN": "Lagavulin",
    "THE_GUARDIAN": "TheGuardian", "SLIME_BOSS": "SlimeBoss",
    "RED_SLAVER": "SlaverRed", "BLUE_SLAVER": "SlaverBlue",
    "FUNGI_BEAST": "FungiBeast",
    "FAT_GREMLIN": "GremlinFat", "MAD_GREMLIN": "GremlinWarrior",
    "SHIELD_GREMLIN": "GremlinTsundere", "SNEAKY_GREMLIN": "GremlinThief",
    "GREMLIN_WIZARD": "GremlinWizard",
    # act 2
    "BYRD": "Byrd", "CENTURION": "Centurion", "CHOSEN": "Chosen",
    "MYSTIC": "Healer", "SHELLED_PARASITE": "Shelled Parasite", "SNAKE_PLANT": "SnakePlant",
    "SNECKO": "Snecko", "SPHERIC_GUARDIAN": "SphericGuardian",
    "MUGGER": "Mugger", "EXPLODER": "Exploder", "REPULSOR": "Repulsor", "SPIKER": "Spiker",
    "GREMLIN_LEADER": "GremlinLeader", "TASKMASTER": "SlaverBoss",
    "BOOK_OF_STABBING": "BookOfStabbing", "THE_CHAMP": "Champ",
    "BRONZE_AUTOMATON": "BronzeAutomaton", "BRONZE_ORB": "BronzeOrb",
    "THE_COLLECTOR": "TheCollector", "TORCH_HEAD": "TorchHead",
    # masked-bandits event (sim names POINTY/ROMEO/BEAR -> real Bandit* ids)
    "POINTY": "BanditChild", "ROMEO": "BanditLeader", "BEAR": "BanditBear",
    # act 3
    "DARKLING": "Darkling", "ORB_WALKER": "Orb Walker", "SPIRE_GROWTH": "Serpent",
    "TRANSIENT": "Transient", "WRITHING_MASS": "WrithingMass", "THE_MAW": "Maw",
    "REPTOMANCER": "Reptomancer", "DAGGER": "Dagger",
    "GIANT_HEAD": "GiantHead", "NEMESIS": "Nemesis",
    "AWAKENED_ONE": "AwakenedOne", "TIME_EATER": "TimeEater",
    "DECA": "Deca", "DONU": "Donu",
    # act 4
    "SPIRE_SHIELD": "SpireShield", "SPIRE_SPEAR": "SpireSpear",
    "CORRUPT_HEART": "CorruptHeart",
}

# CommunicationMod mostly exposes the same monster identity used by the model
# vocabulary.  Spire Growth is the exception: the Java monster's internal ID is
# ``Serpent`` while the frozen observation vocabulary uses ``SpireGrowth``.
# Keep transport and model identities separate; feeding the transport ID into
# the encoder used to turn this live monster into PAD.
_MONSTER_REAL2CANON = MONSTER_ALIASES
_MONSTER_CANON2REAL = {canonical: raw for raw, canonical in MONSTER_ALIASES.items()}


def _canonical_real_monster_id(real_id):
    return _MONSTER_REAL2CANON.get(real_id, real_id)


# CommunicationMod exposes the Java monsters' private byte move ids.  They are
# local to each monster and are unrelated to the global C++ MonsterMoveId enum.
# This explicit codec was checked against the shipped game class constants; it
# is data only and lets the real-game bridge synchronize current and historical
# moves without guessing from damage (several moves share the same damage).
_REAL_MONSTER_MOVES = {
    "AcidSlime_L": {1: "ACID_SLIME_L_CORROSIVE_SPIT", 2: "ACID_SLIME_L_TACKLE", 3: "ACID_SLIME_L_SPLIT", 4: "ACID_SLIME_L_LICK"},
    "AcidSlime_M": {1: "ACID_SLIME_M_CORROSIVE_SPIT", 2: "ACID_SLIME_M_TACKLE", 4: "ACID_SLIME_M_LICK"},
    "AcidSlime_S": {1: "ACID_SLIME_S_TACKLE", 2: "ACID_SLIME_S_LICK"},
    "AwakenedOne": {1: "AWAKENED_ONE_SLASH", 2: "AWAKENED_ONE_SOUL_STRIKE", 3: "AWAKENED_ONE_REBIRTH", 5: "AWAKENED_ONE_DARK_ECHO", 6: "AWAKENED_ONE_SLUDGE", 8: "AWAKENED_ONE_TACKLE"},
    "BanditBear": {1: "BEAR_MAUL", 2: "BEAR_BEAR_HUG", 3: "BEAR_LUNGE"},
    "BanditChild": {1: "POINTY_ATTACK"},
    "BanditLeader": {1: "ROMEO_CROSS_SLASH", 2: "ROMEO_MOCK", 3: "ROMEO_AGONIZING_SLASH"},
    "BookOfStabbing": {1: "BOOK_OF_STABBING_MULTI_STAB", 2: "BOOK_OF_STABBING_SINGLE_STAB"},
    "BronzeAutomaton": {1: "BRONZE_AUTOMATON_FLAIL", 2: "BRONZE_AUTOMATON_HYPER_BEAM", 3: "BRONZE_AUTOMATON_STUNNED", 4: "BRONZE_AUTOMATON_SPAWN_ORBS", 5: "BRONZE_AUTOMATON_BOOST"},
    "BronzeOrb": {1: "BRONZE_ORB_BEAM", 2: "BRONZE_ORB_SUPPORT_BEAM", 3: "BRONZE_ORB_STASIS"},
    "Byrd": {1: "BYRD_PECK", 2: "BYRD_FLY", 3: "BYRD_SWOOP", 4: "BYRD_STUNNED", 5: "BYRD_HEADBUTT", 6: "BYRD_CAW"},
    "Centurion": {1: "CENTURION_SLASH", 2: "CENTURION_DEFEND", 3: "CENTURION_FURY"},
    "Champ": {1: "THE_CHAMP_HEAVY_SLASH", 2: "THE_CHAMP_DEFENSIVE_STANCE", 3: "THE_CHAMP_EXECUTE", 4: "THE_CHAMP_FACE_SLAP", 5: "THE_CHAMP_GLOAT", 6: "THE_CHAMP_TAUNT", 7: "THE_CHAMP_ANGER"},
    "Chosen": {1: "CHOSEN_ZAP", 2: "CHOSEN_DRAIN", 3: "CHOSEN_DEBILITATE", 4: "CHOSEN_HEX", 5: "CHOSEN_POKE"},
    "CorruptHeart": {1: "CORRUPT_HEART_BLOOD_SHOTS", 2: "CORRUPT_HEART_ECHO", 3: "CORRUPT_HEART_DEBILITATE", 4: "CORRUPT_HEART_BUFF"},
    "Cultist": {1: "CULTIST_DARK_STRIKE", 3: "CULTIST_INCANTATION"},
    "Dagger": {1: "DAGGER_STAB", 2: "DAGGER_EXPLODE"},
    "Darkling": {1: "DARKLING_CHOMP", 2: "DARKLING_HARDEN", 3: "DARKLING_NIP", 4: "DARKLING_REGROW", 5: "DARKLING_REINCARNATE"},
    "Deca": {0: "DECA_BEAM", 2: "DECA_SQUARE_OF_PROTECTION"},
    "Donu": {0: "DONU_BEAM", 2: "DONU_CIRCLE_OF_POWER"},
    "Exploder": {1: "EXPLODER_SLAM", 2: "EXPLODER_EXPLODE"},
    "FungiBeast": {1: "FUNGI_BEAST_BITE", 2: "FUNGI_BEAST_GROW"},
    "FuzzyLouseDefensive": {3: "GREEN_LOUSE_BITE", 4: "GREEN_LOUSE_SPIT_WEB"},
    "FuzzyLouseNormal": {3: "RED_LOUSE_BITE", 4: "RED_LOUSE_GROW"},
    "GiantHead": {1: "GIANT_HEAD_GLARE", 2: "GIANT_HEAD_IT_IS_TIME", 3: "GIANT_HEAD_COUNT"},
    "GremlinFat": {2: "FAT_GREMLIN_SMASH"},
    "GremlinLeader": {2: "GREMLIN_LEADER_RALLY", 3: "GREMLIN_LEADER_ENCOURAGE", 4: "GREMLIN_LEADER_STAB"},
    "GremlinNob": {1: "GREMLIN_NOB_RUSH", 2: "GREMLIN_NOB_SKULL_BASH", 3: "GREMLIN_NOB_BELLOW"},
    "GremlinThief": {1: "SNEAKY_GREMLIN_PUNCTURE"},
    "GremlinTsundere": {1: "SHIELD_GREMLIN_PROTECT", 2: "SHIELD_GREMLIN_SHIELD_BASH"},
    "GremlinWarrior": {1: "MAD_GREMLIN_SCRATCH"},
    "GremlinWizard": {1: "GREMLIN_WIZARD_ULTIMATE_BLAST", 2: "GREMLIN_WIZARD_CHARGING"},
    "Healer": {1: "MYSTIC_ATTACK_DEBUFF", 2: "MYSTIC_HEAL", 3: "MYSTIC_BUFF"},
    "Hexaghost": {1: "HEXAGHOST_DIVIDER", 2: "HEXAGHOST_TACKLE", 3: "HEXAGHOST_INFLAME", 4: "HEXAGHOST_SEAR", 5: "HEXAGHOST_ACTIVATE", 6: "HEXAGHOST_INFERNO"},
    "JawWorm": {1: "JAW_WORM_CHOMP", 2: "JAW_WORM_BELLOW", 3: "JAW_WORM_THRASH"},
    "Lagavulin": {1: "LAGAVULIN_SIPHON_SOUL", 3: "LAGAVULIN_ATTACK", 4: "LAGAVULIN_SLEEP", 5: "LAGAVULIN_SLEEP", 6: "LAGAVULIN_SLEEP"},
    "Looter": {1: "LOOTER_MUG", 2: "LOOTER_SMOKE_BOMB", 3: "LOOTER_ESCAPE", 4: "LOOTER_LUNGE"},
    "Maw": {2: "THE_MAW_ROAR", 3: "THE_MAW_SLAM", 4: "THE_MAW_DROOL", 5: "THE_MAW_NOM"},
    "Mugger": {1: "MUGGER_MUG", 2: "MUGGER_SMOKE_BOMB", 3: "MUGGER_ESCAPE", 4: "MUGGER_LUNGE"},
    "Nemesis": {2: "NEMESIS_ATTACK", 3: "NEMESIS_SCYTHE", 4: "NEMESIS_DEBUFF"},
    "Orb Walker": {1: "ORB_WALKER_LASER", 2: "ORB_WALKER_CLAW"},
    "Reptomancer": {1: "REPTOMANCER_SNAKE_STRIKE", 2: "REPTOMANCER_SUMMON", 3: "REPTOMANCER_BIG_BITE"},
    "Repulsor": {1: "REPULSOR_REPULSE", 2: "REPULSOR_BASH"},
    "Sentry": {3: "SENTRY_BOLT", 4: "SENTRY_BEAM"},
    "Serpent": {1: "SPIRE_GROWTH_QUICK_TACKLE", 2: "SPIRE_GROWTH_CONSTRICT", 3: "SPIRE_GROWTH_SMASH"},
    "Shelled Parasite": {1: "SHELLED_PARASITE_FELL", 2: "SHELLED_PARASITE_DOUBLE_STRIKE", 3: "SHELLED_PARASITE_SUCK", 4: "SHELLED_PARASITE_STUNNED"},
    "SlaverBlue": {1: "BLUE_SLAVER_STAB", 4: "BLUE_SLAVER_RAKE"},
    "SlaverBoss": {2: "TASKMASTER_SCOURING_WHIP"},
    "SlaverRed": {1: "RED_SLAVER_STAB", 2: "RED_SLAVER_ENTANGLE", 3: "RED_SLAVER_SCRAPE"},
    "SlimeBoss": {1: "SLIME_BOSS_SLAM", 2: "SLIME_BOSS_PREPARING", 3: "SLIME_BOSS_SPLIT", 4: "SLIME_BOSS_GOOP_SPRAY"},
    "SnakePlant": {1: "SNAKE_PLANT_CHOMP", 2: "SNAKE_PLANT_ENFEEBLING_SPORES"},
    "Snecko": {1: "SNECKO_PERPLEXING_GLARE", 2: "SNECKO_BITE", 3: "SNECKO_TAIL_WHIP"},
    "SphericGuardian": {1: "SPHERIC_GUARDIAN_SLAM", 2: "SPHERIC_GUARDIAN_ACTIVATE", 3: "SPHERIC_GUARDIAN_HARDEN", 4: "SPHERIC_GUARDIAN_ATTACK_DEBUFF"},
    "SpikeSlime_L": {1: "SPIKE_SLIME_L_FLAME_TACKLE", 3: "SPIKE_SLIME_L_SPLIT", 4: "SPIKE_SLIME_L_LICK"},
    "SpikeSlime_M": {1: "SPIKE_SLIME_M_FLAME_TACKLE", 4: "SPIKE_SLIME_M_LICK"},
    "SpikeSlime_S": {1: "SPIKE_SLIME_S_TACKLE"},
    "Spiker": {1: "SPIKER_CUT", 2: "SPIKER_SPIKE"},
    "SpireShield": {1: "SPIRE_SHIELD_BASH", 2: "SPIRE_SHIELD_FORTIFY", 3: "SPIRE_SHIELD_SMASH"},
    "SpireSpear": {1: "SPIRE_SPEAR_BURN_STRIKE", 2: "SPIRE_SPEAR_PIERCER", 3: "SPIRE_SPEAR_SKEWER"},
    "TheCollector": {1: "THE_COLLECTOR_SPAWN", 2: "THE_COLLECTOR_FIREBALL", 3: "THE_COLLECTOR_BUFF", 4: "THE_COLLECTOR_MEGA_DEBUFF", 5: "THE_COLLECTOR_SPAWN"},
    "TheGuardian": {1: "THE_GUARDIAN_DEFENSIVE_MODE", 2: "THE_GUARDIAN_FIERCE_BASH", 3: "THE_GUARDIAN_ROLL_ATTACK", 4: "THE_GUARDIAN_TWIN_SLAM", 5: "THE_GUARDIAN_WHIRLWIND", 6: "THE_GUARDIAN_CHARGING_UP", 7: "THE_GUARDIAN_VENT_STEAM"},
    "TimeEater": {2: "TIME_EATER_REVERBERATE", 3: "TIME_EATER_RIPPLE", 4: "TIME_EATER_HEAD_SLAM", 5: "TIME_EATER_HASTE"},
    "TorchHead": {1: "TORCH_HEAD_TACKLE"},
    "Transient": {1: "TRANSIENT_ATTACK"},
    "WrithingMass": {0: "WRITHING_MASS_STRONG_STRIKE", 1: "WRITHING_MASS_MULTI_STRIKE", 2: "WRITHING_MASS_FLAIL", 3: "WRITHING_MASS_WITHER", 4: "WRITHING_MASS_IMPLANT"},
}


def real_to_sim_monster_move(real_monster_id, real_move_id):
    """Translate a byte using either its transport or registered canonical ID.

    from_real calls this before identity normalization; search root restoration
    calls it afterwards. Both must address the same transport move table.
    """
    if type(real_move_id) is not int:
        raise ValueError(f"real monster move must be an integer byte: {real_move_id!r}")
    move_id = real_move_id
    if move_id == -1:
        return "INVALID"
    try:
        transport_id = _MONSTER_CANON2REAL.get(real_monster_id, real_monster_id)
        return _REAL_MONSTER_MOVES[transport_id][move_id]
    except KeyError as exc:
        raise KeyError(
            f"unmapped real monster move {real_monster_id!r} id={move_id}"
        ) from exc


def _sim_monster_id(name):
    if name not in _MONSTER_SIM2REAL:
        raise KeyError(f"unmapped sim monster id {name!r} - add it to _MONSTER_SIM2REAL")
    return _canonical_real_monster_id(_MONSTER_SIM2REAL[name])


# Powers: sim names are mostly identical to real ids; only the known diffs need
# overriding. Unknowns pass through (a wrong one will surface in the diff).
_POWER_SIM2REAL = SIM_POWER_ALIASES


def _sim_power_id(name):
    return _POWER_SIM2REAL.get(name, name)


def _powers_sim(pairs, vocabulary_index):
    if not isinstance(pairs, (list, tuple)):
        raise ValueError(f"sim powers must be a sequence, got {pairs!r}")
    out = {}
    known = set(vocabulary_index)
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"malformed sim power record {index}: {pair!r}")
        name = _sim_power_id(pair[0])
        if name in _POWER_DROP:
            continue
        if name not in known:
            raise ValueError(f"unknown sim power ID: {name!r}")
        if name in out:
            raise ValueError(f"duplicate sim power identity: {name!r}")
        out[name] = int(pair[1])
    return out


def _norm_sim_monster(m, *, allow_no_move=False):
    required = {
        "id", "hp", "max_hp", "block", "gone", "half_dead",
        "base_damage", "adjusted_damage", "hits", "powers", "move",
        "move_id", "last_move", "last_move_id", "second_last_move",
        "second_last_move_id", "unawakened", "reactive",
        "thief_state",
        "lagavulin_state",
        "wizard_charge",
        "writhing_mass_used_implant",
        "move_state",
    }
    missing = required - set(m)
    if missing:
        raise ValueError(f"sim monster is missing {sorted(missing)!r}: {m!r}")
    lifecycle = _monster_lifecycle(m)
    thief_state = _validated_thief_state(
        _sim_monster_id(m['id']) if m['id'] != _SIM_MONSTER_INVALID else 'PAD',
        m['thief_state'],
    )
    lagavulin_state = _validated_lagavulin_state(
        _sim_monster_id(m['id']) if m['id'] != _SIM_MONSTER_INVALID else 'PAD',
        m['lagavulin_state'],
    )
    wizard_charge = _validated_wizard_charge(
        _sim_monster_id(m['id']) if m['id'] != _SIM_MONSTER_INVALID else 'PAD', m['wizard_charge'])
    writhing_mass_used_implant = _validated_writhing_mass_used_implant(
        _sim_monster_id(m['id']) if m['id'] != _SIM_MONSTER_INVALID else 'PAD',
        m['writhing_mass_used_implant'])
    move_state = _validated_move_state(
        _sim_monster_id(m['id']) if m['id'] != _SIM_MONSTER_INVALID else 'PAD', m['move_state'])
    if m["id"] == _SIM_MONSTER_INVALID:
        if (any(m[key] != 0 for key in ("hp", "max_hp", "block"))
                or m["powers"] or not m["gone"] or not m["is_dying"]
                or any(m[key] for key in ("half_dead", "is_dead", "is_escaping", "escape_next",
                                         "unawakened", "reactive"))
                or any(m[key] != "INVALID" for key in ("move", "last_move", "second_last_move"))
                or any(m[key] != 0 for key in ("move_id", "last_move_id", "second_last_move_id"))):
            raise ValueError("invalid native monster ID carries entity state; cannot encode as PAD")
        return dict(id="PAD", hp=0, max_hp=0, block=0, gone=True,
                    base_damage=0, adjusted_damage=0, hits=0, powers={},
                    move="INVALID", move_id=0, last_move="INVALID",
                    last_move_id=0, second_last_move="INVALID",
                    second_last_move_id=0, unawakened=False,
                    reactive=False, thief_state=None, lagavulin_state=None, wizard_charge=None,
                    writhing_mass_used_implant=None, move_state=None, **lifecycle)
    try:
        move = m["move"]
        move_id = int(m["move_id"])
    except KeyError as exc:
        raise ValueError(f"live sim monster is missing {exc.args[0]}: {m!r}") from exc
    if move == "INVALID":
        if not allow_no_move and not m["gone"]:
            raise ValueError(
                f"live sim monster {m['id']!r} has no current move"
            )
        expected_move_id = 0
    else:
        from env.core.vocab import MOVE_VOCAB_INDEX
        try:
            expected_move_id = MOVE_VOCAB_INDEX[move]
        except KeyError as exc:
            raise ValueError(f"unknown sim monster move: {move!r}") from exc
    if move_id != expected_move_id:
        raise ValueError(
            f"sim monster move index mismatch: {move!r} exports {move_id}, "
            f"expected {expected_move_id}"
        )
    last_move = m["last_move"]
    if last_move == "INVALID":
        expected_last_move_id = 0
    else:
        from env.core.vocab import MOVE_VOCAB_INDEX
        try:
            expected_last_move_id = MOVE_VOCAB_INDEX[last_move]
        except KeyError as exc:
            raise ValueError(f"unknown sim previous monster move: {last_move!r}") from exc
    if int(m["last_move_id"]) != expected_last_move_id:
        raise ValueError(
            f"sim previous monster move index mismatch: {last_move!r} exports "
            f"{m['last_move_id']}, expected {expected_last_move_id}"
        )
    second_last_move = m["second_last_move"]
    if second_last_move == "INVALID":
        expected_second_last_move_id = 0
    else:
        from env.core.vocab import MOVE_VOCAB_INDEX
        try:
            expected_second_last_move_id = MOVE_VOCAB_INDEX[second_last_move]
        except KeyError as exc:
            raise ValueError(
                f"unknown sim second-previous monster move: {second_last_move!r}"
            ) from exc
    if int(m["second_last_move_id"]) != expected_second_last_move_id:
        raise ValueError(
            "sim second-previous monster move index mismatch: "
            f"{second_last_move!r} exports {m['second_last_move_id']}, "
            f"expected {expected_second_last_move_id}"
        )
    base, hits = _attack(m["base_damage"], m["hits"])
    adjusted, _ = _attack(m["adjusted_damage"], hits)
    if m["move"] == "EXPLODER_EXPLODE":
        # CommunicationMod publishes UNKNOWN, with no visible damage or hits.
        # This observable projection must not change native explosion damage.
        base, adjusted, hits = 0, 0, 0
    hp = m["hp"]
    from env.core.vocab import MONSTER_BUFF_VOCAB_INDEX
    return {
        "id": _sim_monster_id(m["id"]),
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": m["gone"],
        **lifecycle,
        "base_damage": base,
        "adjusted_damage": adjusted,
        "hits": hits,
        "unawakened": bool(m["unawakened"]),
        "reactive": bool(m["reactive"]),
        "thief_state": thief_state,
        "lagavulin_state": lagavulin_state,
        "wizard_charge": wizard_charge,
        "writhing_mass_used_implant": writhing_mass_used_implant,
        "move_state": move_state,
        "powers": _powers_sim(m["powers"], MONSTER_BUFF_VOCAB_INDEX),
        "move": move,
        "move_id": move_id,
        "last_move": last_move,
        "last_move_id": expected_last_move_id,
        "second_last_move": second_last_move,
        "second_last_move_id": expected_second_last_move_id,
    }


def normalize_sim(raw):
    """`raw` = sim BattleContext.to_dict(). Returns the canonical combat state."""
    required = {
        "turn", "post_enemy_turn_boundary", "player", "monsters", "hand",
        "draw", "discard", "exhaust",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"sim combat state is missing {sorted(missing)!r}")
    boundary_counters = {}
    if raw["post_enemy_turn_boundary"]:
        value = raw["pocketwatch_cards_at_boundary"]
        if type(value) is not int or value < 0:
            raise ValueError("native boundary Pocketwatch counter is not a nonnegative integer")
        boundary_counters["Pocketwatch"] = value
    p = raw["player"]
    player_required = {
        "hp", "max_hp", "block", "energy", "gold", "combat_counters",
        "powers", "energy_per_turn", "bombs", "card_draw_per_turn", "draw_reduction_just_applied",
    }
    missing_player = player_required - set(p)
    if missing_player:
        raise ValueError(
            f"sim player state is missing {sorted(missing_player)!r}"
        )
    missing_counters = set(_REAL_COMBAT_COUNTER_KEYS) - set(p["combat_counters"])
    if missing_counters:
        raise ValueError(
            f"sim combat counters are incomplete: {sorted(missing_counters)!r}"
        )
    from env.core.vocab import PLAYER_BUFF_VOCAB_INDEX
    combat_counters = _validated_combat_counters(p["combat_counters"])
    bombs = _validated_bombs(p["bombs"], combat_counters)
    return {
        # sim turn is 0-based at combat start; real game is 1-based.
        "turn": raw["turn"] + 1,
        # Native NOSL leaves may stop exactly after the enemy turn, before the
        # next-turn draw/energy setup. Preserve this search-only boundary bit
        # through the canonical layer so the combat encoder can expose it.
        "post_enemy_turn_boundary": bool(
            raw["post_enemy_turn_boundary"]
        ),
        "boundary_relic_counters": boundary_counters,
        "player": {
            "hp": p["hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "gold": p["gold"],
            "combat_counters": combat_counters,
            "bombs": bombs,
            "powers": _powers_sim(p["powers"], PLAYER_BUFF_VOCAB_INDEX),
            "energy_per_turn": p["energy_per_turn"],
            **_validated_draw_state(p['card_draw_per_turn'], p['draw_reduction_just_applied'],
                                    dict(p['powers'])),
        },
        "monsters": [
            # Preserve native target slots, including reserved empty positions.
            _norm_sim_monster(
                m, allow_no_move=bool(raw["post_enemy_turn_boundary"])
            )
            for m in raw["monsters"]
        ],
        "hand": [_card(_sim_card_id(n), u, cost) for n, u, cost in raw["hand"]],
        "draw": [_card(_sim_card_id(n), u) for n, u in raw["draw"]],
        "discard": [_card(_sim_card_id(n), u) for n, u in raw["discard"]],
        "exhaust": [_card(_sim_card_id(n), u) for n, u in raw["exhaust"]],
    }


# ========================= overworld (out of combat) =========================
# Coarser than combat: just the persistent run state that the next combat reads
# from (deck contents, relics, hp, gold, floor). Used to catch a desync the
# moment an out-of-combat decision (reward/event/shop/rest) diverges.

def normalize_real_overworld(gs):
    required = {
        "floor", "current_hp", "max_hp", "gold", "deck", "relics",
        "simulator_parity_export",
    }
    required.update(fields["real"] for fields in RUN_KEY_FIELDS.values())
    missing = required - set(gs)
    if missing:
        raise ValueError(f"real overworld state is missing {sorted(missing)!r}")
    if gs["simulator_parity_export"] != SIMULATOR_PARITY_EXPORT_VERSION:
        raise ValueError(
            "real overworld state was not produced by the required parity exporter: "
            f"{gs['simulator_parity_export']!r}"
        )
    if not isinstance(gs["deck"], list) or not isinstance(gs["relics"], list):
        raise ValueError("real overworld deck/relics must be lists")
    deck = []
    for index, card in enumerate(gs["deck"]):
        if not isinstance(card, dict) or "id" not in card or "upgrades" not in card:
            raise ValueError(f"malformed overworld card record {index}: {card!r}")
        canonical = _card(card["id"], card["upgrades"])
        deck.append((canonical["id"], canonical["up"]))
    from env.core.vocab import RELIC_VOCAB_INDEX
    relics = []
    for index, relic in enumerate(gs["relics"]):
        if not isinstance(relic, dict) or "id" not in relic:
            raise ValueError(f"malformed overworld relic record {index}: {relic!r}")
        relic_id = str(relic["id"])
        if relic_id in UNSUPPORTED_REAL_RELIC_IDS:
            raise ValueError(
                f"unsupported real relic ID {relic_id!r}: "
                f"{UNSUPPORTED_REAL_RELIC_IDS[relic_id]}"
            )
        if relic_id == "PAD" or relic_id not in RELIC_VOCAB_INDEX:
            raise ValueError(f"unknown real relic ID: {relic_id!r}")
        relics.append(relic_id)
    keys = {}
    for semantic, fields in RUN_KEY_FIELDS.items():
        value = gs[fields["real"]]
        if type(value) is not bool:
            raise ValueError(
                f"real run key {fields['real']!r} must be boolean, got {value!r}"
            )
        keys[semantic] = value
    return {
        "floor": gs["floor"],
        "hp": gs["current_hp"],
        "max_hp": gs["max_hp"],
        "gold": gs["gold"],
        "deck": sorted(deck),
        "relics": sorted(relics),
        "keys": keys,
    }


# sim screen_state_name suffix -> semantic phase string. The names look like
# "ScreenState.REST_ROOM"; we key on the bare tail.
_SCREEN2PHASE = {
    "EVENT_SCREEN": "event",     # incl. Neow, distinguished by cur_event_name in env
    "REWARDS": "combat_reward",
    "BOSS_RELIC_REWARDS": "boss_relic",
    "CARD_SELECT": "card_select",
    "MAP_SCREEN": "map",
    "TREASURE_ROOM": "chest",
    "REST_ROOM": "rest",
    "SHOP_ROOM": "shop",
    "BATTLE": "combat",
    "INVALID": "invalid",
}


def sim_phase(gc):
    """Semantic phase for the gc's current screen (bare-name lookup)."""
    tail = gc.screen_state_name.split(".")[-1]
    try:
        return _SCREEN2PHASE[tail]
    except KeyError as exc:
        raise ValueError(f"unknown native screen-state keyword: {tail!r}") from exc


def _sim_outcome_name(gc):
    return str(getattr(gc, "outcome", "")).split(".")[-1]


def _sim_screen_options(gc, phase):
    """The current screen's decision context, read from the gc itself (no map
    reachability here — that needs the rebuilt SpireMap and is added by the env,
    which keeps this schema backend-symmetric and free of an env dependency).

    Only fields the screen actually owns are populated; callers branch on phase.
    """
    if phase == "combat_reward":
        info = gc.get_reward_info()
        card_count = int(info["card_count"])
        cards = []
        if card_count and _sim_outcome_name(gc) == "UNDECIDED":
            cards = [_card_reward_entry(c) for c in gc.get_card_reward()]
        elif _sim_outcome_name(gc) != "UNDECIDED":
            card_count = 0
        return {
            "gold": list(info["gold"]),
            "relics": list(info["relics"]),
            "potions": list(info["potions"]),
            "keys": list(info["keys"]),
            "card_count": card_count,
            # the actual card choices, so the player/agent sees WHICH cards are on
            # offer, not just how many. get_card_reward returns Card objects;
            # str(card.id) is 'CardId.FLASH_OF_STEEL' -> take the bare enum name.
            "cards": cards,
        }
    if phase == "card_select":
        # The grid candidates PLUS the select-screen semantics/progress
        # (obs_design §4.1): WHAT this grid is for (context), how many to pick,
        # and which cards are already picked THIS screen. The sim removes a picked
        # card from the grid and into haveSelectedCards, so already-picked cards
        # are NOT in get_select_cards() anymore -> we surface their ids in
        # select["selected"] instead. Overworld grids require exactly pick_count
        # picks (no pick-zero / pick-any), unlike some combat screens.
        si = gc.select_screen_info()
        return {
            "cards": [(_sim_card_id(n), u) for n, u in gc.get_select_cards()],
            "select": _select_options(
                si["context"], si["pick_count"], si["have_selected"],
                [(_sim_card_id(n), u) for n, u in si["selected"]],
                can_pick_zero=False, can_pick_any=False),
        }
    if phase == "event":
        options = {"event": gc.cur_event_name}
        get_details = getattr(gc, "event_option_details", None)
        if get_details is not None:
            details = []
            for raw in get_details():
                detail = dict(raw)
                if "card" in detail:
                    name, upgrades = detail["card"]
                    detail["card"] = (_sim_card_id(name), int(upgrades))
                details.append(detail)
            options["event_choices"] = details
        return options
    if phase == "shop":
        # The shop shelf, POSITIONAL (slot i == the buy action's id, so obs slots
        # line up with the action mask). Sold/empty slots have price -1 from the
        # sim; the encoder maps those to PAD id + price 0 (a shop has no "originally
        # empty" slot, so -1 unambiguously means "already bought"). Card ids are
        # normalized like the deck; relic/potion ids are the sim's stable strings.
        s = gc.shop_contents()
        return {
            "shop_cards": [
                (_sim_card_id(c["id"]), c["upgrades"], c["price"])
                for c in s["cards"]
            ],
            "shop_relics": [(r["id"], r["price"]) for r in s["relics"]],
            "shop_potions": [(p["id"], p["price"]) for p in s["potions"]],
            "shop_remove_cost": s["remove_cost"],
        }
    # map/rest/chest/boss_relic options are enumerated by the env (map needs
    # SpireMap; the rest are fixed-shape) -> nothing gc-readable to add here.
    return {}


def _select_options(context, pick_count, have_selected, selected_ids,
                    can_pick_zero, can_pick_any, multi=False, pick_max=0,
                    cards_in_hand=0):
    """The card-select screen's semantics + progress, in one backend-symmetric
    shape (obs_design §4.1). Shared by the overworld grid and the combat sub-
    screen; the encoder reads it into select_context / pick_remaining /
    can_pick_zero / can_pick_any / selected_card_ids.

      context        : SELECT_CONTEXT_VOCAB name (CardSelectTask for combat or
                       CardSelectScreenType for overworld); "INVALID" off-screen.
      pick_remaining : how many MORE to pick = pick_count - have_selected.
      selected       : canonical ids picked THIS screen. Mandatory combat GRID
                       choices accumulate in native state, without moving cards
                       until completion. MULTI hand selections accumulate in
                       the thin layer and submit one bitset on confirmation.
      multi          : True for Exhaust Many / Gamble — the thin layer drives a
                       select+confirm machine over `cards_in_hand` hand indices,
                       capped at `pick_max` (obs_design §4.2). False otherwise.
    """
    return {
        "context": context,
        "pick_remaining": max(0, pick_count - have_selected),
        "pick_min": 0 if (can_pick_zero or can_pick_any) else int(pick_count),
        "can_pick_zero": bool(can_pick_zero),
        "can_pick_any": bool(can_pick_any),
        "selected": selected_ids,
        "multi": bool(multi),
        "pick_max": int(pick_max),
        "cards_in_hand": int(cards_in_hand),
    }


def generated_selection_parameters(info):
    """Non-tensor parameters distinguish normal Toolbox from free Discovery."""
    count = info["generated_count"]
    free = info["generated_free_for_turn"]
    if type(count) is not int or count not in (1, 2) or type(free) is not bool:
        raise ValueError("generated-card selection requires exact count1/2 and boolean cost mode")
    return {"generated_count": count, "generated_free_for_turn": free}


def combat_select_options(select_info):
    """Canonical 'select' block for an in-combat card-select sub-screen, from
    bc.select_info() (context = CardSelectTask name, pick_count, can_pick_zero,
    can_pick_any, multi, pick_max, cards_in_hand).

    SINGLE/GRID use PICK_CARD directly; mandatory GRID screens expose the
    ordered pending selections and settle only after the last required pick.
    MULTI hand screens resolve a whole bitset; their thin prefix stays separate
    from GRID's source-pile identities. All flags/counts are explicit."""
    required = {
        "context", "pick_count", "can_pick_zero", "can_pick_any", "multi",
        "pick_max", "cards_in_hand", "zone", "candidates", "selected",
    }
    missing = required - set(select_info)
    if missing:
        raise ValueError(f"native combat selection is missing {sorted(missing)!r}")
    multi = bool(select_info["multi"])
    pick_count = int(select_info["pick_count"])
    selected = [(_sim_card_id(name), up) for name, up in select_info["selected"]]
    if (pick_count < 0 or (not multi and len(selected) >= pick_count)
            or (multi and selected)):
        raise ValueError("native live selection has invalid count/progress")
    result = _select_options(
        select_info["context"], pick_count, len(selected), selected,
        can_pick_zero=select_info["can_pick_zero"],
        can_pick_any=select_info["can_pick_any"],
        multi=multi,
        pick_max=select_info["pick_max"],
        cards_in_hand=select_info["cards_in_hand"])
    result["zone"] = select_info["zone"]
    if select_info["context"] == "DISCOVERY":
        result.update(generated_selection_parameters(select_info))
    result["candidates"] = [
        {"native_index": int(item["native_index"]),
         "card": None if item["skip"] else _card(
             _sim_card_id(item["card"][0]), item["card"][1],
             item["card"][2])}
        for item in select_info["candidates"]
    ]
    return result


def _card_reward_entry(card):
    """A Card object from get_card_reward() -> (canonical id, upgrade_count)."""
    enum_name = _enum_tail(card.id)
    return (_sim_card_id(enum_name), card.upgrade_count)


def normalize_sim_overworld(gc):
    phase = sim_phase(gc)
    return {
        "floor": gc.floor_num,
        "hp": gc.cur_hp,
        "max_hp": gc.max_hp,
        "gold": gc.gold,
        "deck": sorted((_sim_card_id(n), u) for n, u in gc.deck_ids()),
        "relics": sorted(gc.relic_ids()),
        "keys": {
            semantic: bool(getattr(gc, fields["sim"]))
            for semantic, fields in RUN_KEY_FIELDS.items()
        },
        "phase": phase,
        "options": _sim_screen_options(gc, phase),
    }


# ----------------------------- obs global block -----------------------------
# The persistent run state the OBS needs in EVERY phase (combat included): the
# agent decides combat moves with the whole run in view (which boss is next, what
# relics/potions it holds, how big the deck is). All of this is pure-Python off
# the gc — the env still holds gc mid-combat. Distinct from normalize_sim_overworld
# (which is the differential-testing snapshot) so the obs contract can evolve
# without touching the diff path. Ids are canonical (cards/relics/potions/boss).

@lru_cache(maxsize=512, typed=True)
def _fixed_enum_tail(v):
    return str(v).split(".")[-1]


def _enum_tail(v):
    """'MonsterEncounter.HEXAGHOST' / 'CharacterClass.IRONCLAD' -> bare tail."""
    # Native enum properties can reference mutable C++ fields. Their hash
    # changes in place: retaining them as LRU keys can corrupt eviction,
    # return unrelated names, and crash CPython. Only cache owned strings.
    if type(v) is str:
        return _fixed_enum_tail(v)
    return str(v).split(".")[-1]


def canonical_sim_potion(value):
    """Translate native empty-slot sentinels at the backend boundary only."""
    name = _enum_tail(value)
    if name == "EMPTY_POTION_SLOT":
        return "PAD"
    if name == "INVALID":
        raise ValueError("native held-potion slot contains INVALID instead of EMPTY_POTION_SLOT")
    return name


def normalize_sim_global(gc, *, combat=None):
    """Run-global obs block (obs_design.md §3/§4). Backend-agnostic shape; the
    sim fills it from gc. `boss_relics` is only meaningful on a boss-relic
    screen (empty/duplicate otherwise) — callers gate on phase."""
    # boss_relic_ids() reads gc.info.bossRelics[], which is only populated on a
    # BOSS_RELIC_REWARDS screen; off-screen it holds uninitialized RelicId values
    # whose relicIds[] lookup can be a null string (-> RuntimeError from the
    # binding). Gate on phase so we only touch it when it is meaningful; PAD
    # otherwise. (AttributeError guards an older .so without the binding.)
    boss_relics = []
    if sim_phase(gc) == "boss_relic":
        if not hasattr(gc, "boss_relic_ids"):
            raise RuntimeError(
                "boss-relic screen requires a native boss_relic_ids exporter"
            )
        boss_relics = list(gc.boss_relic_ids())
    relic_state = list(gc.relic_state())
    if combat is None:
        combat = sim_phase(gc) == "combat"
    # Native persistence is committed on battle exit; Java consumes Tea Set
    # at the first turn start. Its observable pending charge is already spent
    # throughout combat, even while native GC retains the exit-time journal.
    if combat:
        relic_state = [
            dict(item, counter=0) if item["id"] == "Ancient Tea Set" else item
            for item in relic_state
        ]
    potions = [canonical_sim_potion(potion) for potion in gc.overworld_potions]
    return {
        # True only for the search-only state immediately after the final
        # enemy action of End Turn, before next-turn setup/draw.
        # The boundary marker is a combat-search-only feature.  The run-global
        # block has no raw observation mapping, so ordinary environment states
        # are always outside that boundary.
        "post_enemy_turn_boundary": False,
        "character": _enum_tail(gc.character) if hasattr(gc, "character")
                     else _enum_tail(gc.cc),
        "ascension": gc.ascension,
        "act": gc.act,
        "floor": gc.floor_num,
        "hp": gc.cur_hp,
        "max_hp": gc.max_hp,
        "boss": _enum_tail(gc.boss),
        "combat_is_boss": _enum_tail(gc.cur_room) == "BOSS",
        "relics": [item["id"] for item in relic_state],
        "relic_counters": [item["counter"] for item in relic_state],
        "potions": potions,
        "keys": {
            semantic: bool(getattr(gc, fields["sim"]))
            for semantic, fields in RUN_KEY_FIELDS.items()
        },
        "gold": gc.gold,
        "deck": [(_sim_card_id(n), u) for n, u in gc.deck_ids()],
        "deck_state": [dict(row, id=_sim_card_id(row['id'])) for row in gc.deck_state()],
        "boss_relics": boss_relics,
    }


def diff_overworld(real, sim):
    out = []
    for k in ("floor", "hp", "max_hp", "gold", "deck", "relics", "keys"):
        if real[k] != sim[k]:
            out.append((k, real[k], sim[k]))
    return out


# ================================ diff =================================

_MONSTER_CMP_FIELDS = (
    "id", "hp", "max_hp", "block", "gone", "half_dead", "base_damage",
    "is_dying", "is_dead", "is_escaping", "escape_next",
    "adjusted_damage", "hits", "move", "move_id", "last_move",
    "last_move_id", "second_last_move", "second_last_move_id", "unawakened",
    "reactive", "powers", "thief_state", "lagavulin_state", "wizard_charge",
    "writhing_mass_used_implant",
    "move_state",
)


def diff(real, sim, pile_mode="auto"):
    """Compare two canonical states. Returns a list of (path, real_val, sim_val)
    for every mismatch; empty list means identical.

    pile_mode: 'ordered' compares piles position-by-position; 'multiset' ignores
    order; 'auto' = hand ordered, draw/discard/exhaust as multisets (their
    cross-engine ordering is not guaranteed equal -- see the draw-pile caveat)."""
    out = []

    def cmp(path, a, b):
        if a != b:
            out.append((path, a, b))

    cmp("turn", real["turn"], sim["turn"])
    cmp(
        "post_enemy_turn_boundary",
        bool(real["post_enemy_turn_boundary"]),
        bool(sim["post_enemy_turn_boundary"]),
    )
    for k in ("hp", "max_hp", "block", "energy", "gold"):
        cmp(f"player.{k}", real["player"][k], sim["player"][k])
    if "energy_per_turn" in real["player"]:
        cmp("player.energy_per_turn", real["player"]["energy_per_turn"],
            sim["player"]["energy_per_turn"])
    for field in ('card_draw_per_turn', 'draw_reduction_just_applied'):
        if field in real['player']:
            cmp('player.' + field, real['player'][field], sim['player'][field])
    cmp("player.powers", real["player"]["powers"], sim["player"]["powers"])
    cmp("player.bombs", real["player"]["bombs"], sim["player"]["bombs"])
    cmp(
        "player.combat_counters",
        real["player"]["combat_counters"],
        sim["player"]["combat_counters"],
    )

    # Callers must supply an explicit entity mapping. Only actual empty slots
    # are absent; corpses, escaping entities and halfDead retain their state.
    rms = [m for m in real["monsters"] if m["id"] != "PAD"]
    sms = [m for m in sim["monsters"] if m["id"] != "PAD"]
    if len(rms) != len(sms):
        out.append(("monsters.len", len(rms), len(sms)))
    for i, (rm, sm) in enumerate(zip(rms, sms)):
        for k in _MONSTER_CMP_FIELDS:
            cmp(f"monsters[{i}].{k}", rm[k], sm[k])

    for pile in ("hand", "draw", "discard", "exhaust"):
        mode = ("ordered" if pile == "hand" else "multiset") if pile_mode == "auto" else pile_mode
        rp, sp = real[pile], sim[pile]
        if mode == "multiset":
            rs = sorted((c["id"], c["up"]) for c in rp)
            ss = sorted((c["id"], c["up"]) for c in sp)
            if rs != ss:
                out.append((f"{pile}(multiset)", rs, ss))
        else:
            if len(rp) != len(sp):
                out.append((f"{pile}.len", len(rp), len(sp)))
            for i, (rc, sc) in enumerate(zip(rp, sp)):
                cmp(f"{pile}[{i}]", rc, sc)
    return out


if __name__ == "__main__":
    # Smoke test: normalize every combat frame in an orchestrator.log and dump
    # the id inventory (the exact ids the sim-side mapping must cover so far).
    import json
    import re
    import sys

    log_path = sys.argv[1]
    cards, monsters, powers = set(), set(), set()
    n_frames = 0
    sample = None
    for line in open(log_path, encoding="gbk", errors="replace"):
        m = re.search(r"RECV (\{.*\})", line)
        if not m:
            continue
        try:
            state = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        canon = normalize_real(state.get("game_state", {}))
        if canon is None:
            continue
        n_frames += 1
        sample = canon
        for pile in ("hand", "draw", "discard", "exhaust"):
            cards.update(c["id"] for c in canon[pile])
        for mob in canon["monsters"]:
            monsters.add(mob["id"])
            powers.update(mob["powers"])
        powers.update(canon["player"]["powers"])

    print(f"normalized {n_frames} combat frames")
    print("card ids:   ", sorted(cards))
    print("monster ids:", sorted(monsters))
    print("power ids:  ", sorted(powers))
    print("last frame sample:")
    print(json.dumps(sample, ensure_ascii=False, indent=2))
