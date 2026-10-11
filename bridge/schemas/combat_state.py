"""Canonical combat-state schema + normalizers for differential testing.

Differential testing compares the C++ sim (sts_lightspeed) against the real
Java game. The two expose combat state in very different shapes, so both get
normalized into the one canonical dict defined here; `diff()` then compares two
canonical dicts field by field.

Canonical combat state (every id is the game's stable English `id`, NOT the
localized `name` field, which is GBK-localized and unstable):

  {
    "turn": int,
    "player": {"hp", "max_hp", "block", "energy", "powers": {power_id: amount}},
    "monsters": [
      {"id", "hp", "max_hp", "block", "gone", "base_damage",
       "adjusted_damage", "hits",
       "powers": {power_id: amount}}, ...],   # real side also carries intent/move_id (diff ignores)
    "hand": [{"id", "up", "cost"}, ...],
    "draw"/"discard"/"exhaust": [{"id", "up"}, ...],
  }

- normalize_real(game_state): from a CommunicationMod frame.
- normalize_sim(raw):         from sim BattleContext.to_dict().
"""


# ============================ real-game side ============================

# Powers whose semantics the native simulator stores outside its status table
# are omitted from the public power comparison.  In particular, Exploder's
# countdown is encoded by its real move history/current move on every synced
# frame; the native rule uses that same history to select SLAM/EXPLODE and has
# no separate EXPLOSIVE status slot.  Keeping the Java-only display counter in
# the canonical power map would therefore reject an otherwise exact root.
# Pen Nib's armed power remains observable separately from its relic counter;
# OBS18 includes it in player_buffs.
_POWER_DROP = {"Asleep", "Split", "Explosive"}
BOOLEAN_PLAYER_POWERS = frozenset({
    "Confused", "Hex", "Draw Reduction", "Barricade", "Blasphemer",
    "Corruption", "Electro", "Surrounded", "Master Reality", "Pen Nib",
    "Wrath Next Turn", "No Draw",
})
BOOLEAN_MONSTER_POWERS = frozenset({
    "Barricade", "Minion", "Minion Leader", "Painful Stabs", "Regrow",
    "Shifting", "Stasis",
})
_REAL_POWER_CANON_ID = {
    # CommunicationMod uses the Java power ID; the native simulator and its
    # observation vocabulary use the displayed/stable semantic name.
    "Confusion": "Confused",
    "Flex": "Lose Strength",
    "Generic Strength Up Power": "Generic Strength Up",
    "Life Link": "Regrow",
    "Regenerate": "Regen",
    "Regeneration": "Regen",
}


def _powers_real(power_list):
    # real game flattens every buff/debuff into a list; key by stable English id
    return {
        _REAL_POWER_CANON_ID.get(p["id"], p["id"]): p["amount"]
        for p in power_list
        if _REAL_POWER_CANON_ID.get(p["id"], p["id"]) not in _POWER_DROP
    }


def _player_powers_real(power_list):
    # Presence powers commonly use amount=-1 in CommunicationMod.
    return {name: 1 if name in BOOLEAN_PLAYER_POWERS else amount
            for name, amount in _powers_real(power_list).items()}


def _monster_powers_real(power_list):
    return {name: 1 if name in BOOLEAN_MONSTER_POWERS else amount
            for name, amount in _powers_real(power_list).items()}


# status cards with no upgraded variant: the sim can mark them upgraded (e.g. an
# upgrade-all effect blindly flips the bit) while the real game keeps them at 0.
# The cards behave identically either way, so canonicalize the upgrade to 0.
_NO_UPGRADE = {"Dazed", "Wound", "Slimed", "Void"}  # Burn excluded (Burn+ exists)

# The generated model vocabulary title-cases enum words. Normalize the one
# CommunicationMod id whose preposition casing differs from that convention.
_CARD_CANON_ID = {"Blood for Blood": "Blood For Blood"}


def _card(card_id, up, cost=None):
    card_id = _CARD_CANON_ID.get(card_id, card_id)
    card = {"id": card_id, "up": 0 if card_id in _NO_UPGRADE else up}
    if cost is not None:
        value = int(cost)
        # CommunicationMod follows AbstractCard.COST_UNPLAYABLE (-2), while
        # the native simulator distinguishes unplayable statuses (-2) from
        # unplayable curses (-3).  They have one public/action meaning here.
        card["cost"] = -2 if value <= -2 else value
    return card


def _cards_real(pile, *, include_cost=False):
    return [
        _card(c["id"], c["upgrades"], c.get("cost") if include_cost else None)
        for c in pile
    ]


def _attack(base, hits):
    # unify the "not attacking" case: base<=0 => (0 dmg, 0 hits) on both sides,
    # else (base, hits). Avoids -1/0 and 0/1 noise for non-attack intents.
    base = max(0, base)
    return base, (hits if base > 0 else 0)


def _monster_real(m):
    base, hits = _attack(m.get("move_base_damage", -1), m.get("move_hits", 1))
    adjusted, _ = _attack(m.get("move_adjusted_damage", base), hits)
    hp = m["current_hp"]
    return {
        "id": m["id"],
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": m["is_gone"],
        "base_damage": base,
        "adjusted_damage": adjusted,
        "hits": hits,
        # a dead monster's powers are meaningless and reported inconsistently
        # (real clears them on death, sim keeps them) -> drop them once hp<=0.
        "powers": _monster_powers_real(m["powers"]) if hp > 0 else {},
        # OBS move fields. NOTE: the obs MOVE_VOCAB is keyed on the sim's
        # MonsterMoveId enum; CommunicationMod's move_id is a DIFFERENT id space,
        # so we do NOT pass it as the obs `move_id` (that would index the wrong
        # vocab). Real backend leaves move/move_id PAD and lets the encoder derive
        # INTENT_CATEGORY from base_damage (attack) — a real intent->sim-move map
        # is future work when the real inference backend is wired (see obs_design
        # §6 / DEVNOTES). The raw real fields are kept under real_* for debugging.
        "move": "INVALID",
        "move_id": 0,
        "intent": m.get("intent"),
        "real_move_id": m.get("move_id", -1),
        "real_last_move_id": m.get("last_move_id", -1),
        "real_second_last_move_id": m.get("second_last_move_id", -1),
    }


def normalize_real(game_state):
    """`game_state` is the dict under the top-level 'game_state' key of a
    CommunicationMod frame. Returns the canonical combat state, or None if the
    frame is not in combat."""
    cs = game_state.get("combat_state")
    if not cs:
        return None
    p = cs["player"]
    combat_counters = {
        # These two counters are exported directly by CommunicationMod. Other
        # simulator-internal counters are unavailable in the real-game schema
        # and intentionally remain zero-filled by the observation encoder.
        "cards_discarded": int(cs.get("cards_discarded_this_turn", 0) or 0),
        "times_damaged": int(cs.get("times_damaged", 0) or 0),
    }
    return {
        "turn": cs["turn"],
        "player": {
            "hp": p["current_hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "gold": game_state.get("gold", 0),
            "combat_counters": combat_counters,
            "powers": _player_powers_real(p["powers"]),
        },
        "monsters": [_monster_real(m) for m in cs["monsters"]],
        "hand": _cards_real(cs["hand"], include_cost=True),
        "draw": _cards_real(cs["draw_pile"]),
        "discard": _cards_real(cs["discard_pile"]),
        "exhaust": _cards_real(cs["exhaust_pile"]),
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
}


def _sim_card_id(name):
    if name in _CARD_OVERRIDE:
        return _CARD_OVERRIDE[name]
    return " ".join(w.capitalize() for w in name.split("_"))


_CARD_OVERRIDE_INV = {v: k for k, v in _CARD_OVERRIDE.items()}


def real_to_sim_card_enum(real_id):
    """Inverse of _sim_card_id: real CommunicationMod id -> sim enum name, for
    force-syncing the sim deck (e.g. 'Strike_R'->'STRIKE_RED', 'Searing Blow'->
    'SEARING_BLOW')."""
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
    """Translate a CommunicationMod per-monster byte into MonsterMoveId name."""
    move_id = int(real_move_id)
    if move_id < 0:
        return "INVALID"
    try:
        return _REAL_MONSTER_MOVES[real_monster_id][move_id]
    except KeyError as exc:
        raise KeyError(
            f"unmapped real monster move {real_monster_id!r} id={move_id}"
        ) from exc


def _sim_monster_id(name):
    if name not in _MONSTER_SIM2REAL:
        raise KeyError(f"unmapped sim monster id {name!r} - add it to _MONSTER_SIM2REAL")
    return _MONSTER_SIM2REAL[name]


# Powers: sim names are mostly identical to real ids; only the known diffs need
# overriding. Unknowns pass through (a wrong one will surface in the diff).
_POWER_SIM2REAL = {
    "Weak": "Weakened",
    "Enrage": "Anger",  # Gremlin Nob's growing-strength power (sim name vs real id)
}


def _sim_power_id(name):
    return _POWER_SIM2REAL.get(name, name)


def _powers_sim(pairs):
    return {_sim_power_id(name): amt for name, amt in pairs
            if _sim_power_id(name) not in _POWER_DROP}


def _norm_sim_monster(m):
    if m["id"] == _SIM_MONSTER_INVALID:
        return dict(id="PAD", hp=0, max_hp=0, block=0, gone=True,
                    base_damage=0, adjusted_damage=0, hits=0, powers={},
                    move="INVALID", move_id=0)
    base, hits = _attack(m["base_damage"], m["hits"])
    adjusted, _ = _attack(m.get("adjusted_damage", base), hits)
    hp = m["hp"]
    return {
        "id": _sim_monster_id(m["id"]),
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": m["gone"],
        "base_damage": base,
        "adjusted_damage": adjusted,
        "hits": hits,
        "powers": _powers_sim(m["powers"]) if hp > 0 else {},
        # current intent: the sim now exports the rolled MonsterMoveId as an
        # enum-name string + int (see bindings to_dict). Carried for the obs
        # MOVE_VOCAB / derived INTENT_CATEGORY; the differential `diff()` ignores
        # it (real-side move ids are not aligned and not compared).
        "move": m.get("move", "INVALID"),
        "move_id": m.get("move_id", 0),
    }


def normalize_sim(raw):
    """`raw` = sim BattleContext.to_dict(). Returns the canonical combat state."""
    p = raw["player"]
    return {
        # sim turn is 0-based at combat start; real game is 1-based.
        "turn": raw["turn"] + 1,
        # Native NOSL leaves may stop exactly after the enemy turn, before the
        # next-turn draw/energy setup. Preserve this search-only boundary bit
        # through the canonical layer so the combat encoder can expose it.
        "post_enemy_turn_boundary": bool(
            raw.get("post_enemy_turn_boundary", False)
        ),
        "player": {
            "hp": p["hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "gold": p.get("gold", 0),
            "combat_counters": p.get("combat_counters", {}),
            "powers": _powers_sim(p["powers"]),
        },
        "monsters": [
            # Preserve native target slots, including reserved empty positions.
            _norm_sim_monster(m) for m in raw["monsters"]
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
    return {
        "floor": gs.get("floor"),
        "hp": gs["current_hp"],
        "max_hp": gs["max_hp"],
        "gold": gs["gold"],
        "deck": sorted((c["id"], c["upgrades"]) for c in gs["deck"]),
        "relics": sorted(r["id"] for r in gs["relics"]),
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
    return _SCREEN2PHASE.get(tail, tail.lower())


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
      selected       : canonical ids already picked THIS screen. The combat
                       SINGLE sub-screen never accumulates across env steps (one
                       pick resolves) -> []. The combat MULTI screen (Exhaust
                       Many / Gamble) DOES accumulate, but in the THIN layer's
                       own monotonic selection set (the sim is told the whole
                       bitset at once on confirm) -> the bridge leaves it [] and
                       the thin layer
                       patches pick_remaining/selected from its accumulator.
      multi          : True for Exhaust Many / Gamble — the thin layer drives a
                       select+confirm machine over `cards_in_hand` hand indices,
                       capped at `pick_max` (obs_design §4.2). False otherwise.
    """
    return {
        "context": context,
        "pick_remaining": max(0, pick_count - have_selected),
        "can_pick_zero": bool(can_pick_zero),
        "can_pick_any": bool(can_pick_any),
        "selected": selected_ids,
        "multi": bool(multi),
        "pick_max": int(pick_max),
        "cards_in_hand": int(cards_in_hand),
    }


def combat_select_options(select_info):
    """Canonical 'select' block for an in-combat card-select sub-screen, from
    bc.select_info() (context = CardSelectTask name, pick_count, can_pick_zero,
    can_pick_any, multi, pick_max, cards_in_hand).

    SINGLE screens: one pick resolves, no accumulation -> 'selected' is empty and
    the thin layer maps each candidate to a PICK_CARD id directly.

    MULTI screens (Exhaust Many / Gamble): the sim resolves the whole screen with
    one bitset Action, so the thin layer selects hand indices and confirms. The
    sim leaves canPickZero/canPickAnyNumber false for these, but a MULTI screen
    can ALWAYS confirm zero (chooseExhaust/GambleCards early-return on empty) and
    Gamble has no upper bound — so we derive can_pick_zero=multi, and can_pick_any
    from pick_max being the whole hand (Gamble) vs a fixed cap (Exhaust Many)."""
    multi = bool(select_info.get("multi"))
    result = _select_options(
        select_info["context"], select_info["pick_count"], 0, [],
        can_pick_zero=select_info["can_pick_zero"] or multi,
        can_pick_any=select_info["can_pick_any"],
        multi=multi,
        pick_max=select_info.get("pick_max", 0),
        cards_in_hand=select_info.get("cards_in_hand", 0))
    result["zone"] = select_info.get("zone", "hand" if multi else "none")
    result["candidates"] = [
        {"native_index": int(item["native_index"]),
         "card": None if item.get("skip") else _card(
             _sim_card_id(item["card"][0]), item["card"][1],
             item["card"][2])}
        for item in select_info.get("candidates", ())
    ]
    return result


def _card_reward_entry(card):
    """A Card object from get_card_reward() -> (canonical id, upgrade_count)."""
    enum_name = str(card.id).split(".")[-1]
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

def _enum_tail(v):
    """'MonsterEncounter.HEXAGHOST' / 'CharacterClass.IRONCLAD' -> bare tail."""
    return str(v).split(".")[-1]


def normalize_sim_global(gc):
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
        try:
            boss_relics = list(gc.boss_relic_ids())
        except (AttributeError, RuntimeError):
            boss_relics = []
    relic_state = list(gc.relic_state())
    return {
        # True only for the search-only state immediately after the final
        # enemy action of End Turn, before next-turn setup/draw.
        # The boundary marker is a combat-search-only feature.  The run-global
        # block has no raw observation mapping, so ordinary environment states
        # are always outside that boundary.
        "post_enemy_turn_boundary": False,
        "character": _enum_tail(gc.character) if hasattr(gc, "character")
                     else _enum_tail(gc.cc) if hasattr(gc, "cc") else "INVALID",
        "ascension": getattr(gc, "ascension", 0),
        "act": gc.act,
        "floor": gc.floor_num,
        "hp": gc.cur_hp,
        "max_hp": gc.max_hp,
        "boss": _enum_tail(gc.boss),
        "combat_is_boss": _enum_tail(getattr(gc, "cur_room", "")) == "BOSS",
        "relics": [item["id"] for item in relic_state],
        "relic_counters": [item["counter"] for item in relic_state],
        "potions": list(gc.overworld_potions),
        "gold": gc.gold,
        "deck": [(_sim_card_id(n), u) for n, u in gc.deck_ids()],
        "boss_relics": boss_relics,
    }


def diff_overworld(real, sim):
    out = []
    for k in ("floor", "hp", "max_hp", "gold", "deck", "relics"):
        if real[k] != sim[k]:
            out.append((k, real[k], sim[k]))
    return out


# ================================ diff =================================

_MONSTER_CMP_FIELDS = ("id", "hp", "max_hp", "block", "gone", "base_damage", "hits", "powers")


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
    for k in ("hp", "max_hp", "block", "energy"):
        cmp(f"player.{k}", real["player"][k], sim["player"][k])
    cmp("player.powers", real["player"]["powers"], sim["player"]["powers"])

    # Drop monsters that have left combat (gone=True): when a slime splits, the
    # real game keeps the dead parent slot in the list while the sim removes it,
    # which would shift indices and break the positional compare. The live
    # monsters that remain still align in order.
    rms = [m for m in real["monsters"] if not m["gone"]]
    sms = [m for m in sim["monsters"] if not m["gone"]]
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
