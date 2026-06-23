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
      {"id", "hp", "max_hp", "block", "gone", "base_damage", "hits",
       "powers": {power_id: amount}}, ...],   # real side also carries intent/move_id (diff ignores)
    "hand"/"draw"/"discard"/"exhaust": [{"id", "up"}, ...],
  }

- normalize_real(game_state): from a CommunicationMod frame.
- normalize_sim(raw):         from sim BattleContext.to_dict().
"""


# ============================ real-game side ============================

# powers that one side tracks only cosmetically -> drop on both sides so they
# don't show as spurious diffs. "Pen Nib" is a real-game attack counter shown as
# a power; the sim keeps it inside the relic (its doubled hit still lands, so the
# effect is still validated via monster hp). "Asleep" is a sim-internal flag the
# real CommunicationMod never exposes.
_POWER_DROP = {"Pen Nib", "Asleep", "Split"}


def _powers_real(power_list):
    # real game flattens every buff/debuff into a list; key by stable English id
    return {p["id"]: p["amount"] for p in power_list if p["id"] not in _POWER_DROP}


# status cards with no upgraded variant: the sim can mark them upgraded (e.g. an
# upgrade-all effect blindly flips the bit) while the real game keeps them at 0.
# The cards behave identically either way, so canonicalize the upgrade to 0.
_NO_UPGRADE = {"Dazed", "Wound", "Slimed", "Void"}  # Burn excluded (Burn+ exists)


def _card(card_id, up):
    return {"id": card_id, "up": 0 if card_id in _NO_UPGRADE else up}


def _cards_real(pile):
    return [_card(c["id"], c["upgrades"]) for c in pile]


def _attack(base, hits):
    # unify the "not attacking" case: base<=0 => (0 dmg, 0 hits) on both sides,
    # else (base, hits). Avoids -1/0 and 0/1 noise for non-attack intents.
    base = max(0, base)
    return base, (hits if base > 0 else 0)


def _monster_real(m):
    base, hits = _attack(m.get("move_base_damage", -1), m.get("move_hits", 1))
    hp = m["current_hp"]
    return {
        "id": m["id"],
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": m["is_gone"],
        "base_damage": base,
        "hits": hits,
        # a dead monster's powers are meaningless and reported inconsistently
        # (real clears them on death, sim keeps them) -> drop them once hp<=0.
        "powers": _powers_real(m["powers"]) if hp > 0 else {},
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
    }


def normalize_real(game_state):
    """`game_state` is the dict under the top-level 'game_state' key of a
    CommunicationMod frame. Returns the canonical combat state, or None if the
    frame is not in combat."""
    cs = game_state.get("combat_state")
    if not cs:
        return None
    p = cs["player"]
    return {
        "turn": cs["turn"],
        "player": {
            "hp": p["current_hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "powers": _powers_real(p["powers"]),
        },
        "monsters": [_monster_real(m) for m in cs["monsters"]],
        "hand": _cards_real(cs["hand"]),
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
    "ACID_SLIME_S": "AcidSlime_S", "ACID_SLIME_M": "AcidSlime_M", "ACID_SLIME_L": "AcidSlime_L",
    "SPIKE_SLIME_S": "SpikeSlime_S", "SPIKE_SLIME_M": "SpikeSlime_M", "SPIKE_SLIME_L": "SpikeSlime_L",
    "CULTIST": "Cultist", "JAW_WORM": "JawWorm",
    "GREEN_LOUSE": "FuzzyLouseNormal", "RED_LOUSE": "FuzzyLouseDefensive",
    "GREMLIN_NOB": "GremlinNob", "HEXAGHOST": "Hexaghost",
    "LOOTER": "Looter", "MUGGER": "Mugger", "SENTRY": "Sentry", "LAGAVULIN": "Lagavulin",
    "THE_GUARDIAN": "TheGuardian", "SLIME_BOSS": "SlimeBoss",
    "RED_SLAVER": "SlaverRed", "BLUE_SLAVER": "SlaverBlue",
    "FUNGI_BEAST": "FungiBeast",
    "SPIKER": "Spiker", "BYRD": "Byrd", "CENTURION": "Centurion", "CHOSEN": "Chosen",
    "MYSTIC": "Healer", "SHELLED_PARASITE": "Shelled Parasite", "SNAKE_PLANT": "Snake Plant",
    "SNECKO": "Snecko", "SPHERIC_GUARDIAN": "SphericGuardian", "REPULSOR": "Repulsor",
    "POINTY": "Repulsor", "EXPLODER": "Exploder", "TASKMASTER": "SlaverBoss",
    "FAT_GREMLIN": "GremlinFat", "MAD_GREMLIN": "GremlinWarrior",
    "SHIELD_GREMLIN": "GremlinTsundere", "SNEAKY_GREMLIN": "GremlinThief",
    "GREMLIN_WIZARD": "GremlinWizard", "TORCH_HEAD": "TorchHead",
}


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
    base, hits = _attack(m["base_damage"], m["hits"])
    hp = m["hp"]
    return {
        "id": _sim_monster_id(m["id"]),
        "hp": hp,
        "max_hp": m["max_hp"],
        "block": m["block"],
        "gone": m["gone"],
        "base_damage": base,
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
        "player": {
            "hp": p["hp"],
            "max_hp": p["max_hp"],
            "block": p["block"],
            "energy": p["energy"],
            "powers": _powers_sim(p["powers"]),
        },
        "monsters": [
            # Skip empty placeholder slots: the sim's monsterCount can include
            # not-yet-activated slots (e.g. Slime Boss reserves a slot for its
            # split halves) which surface as id "INVALID = 0", gone=true. The
            # real game never reports these.
            _norm_sim_monster(m) for m in raw["monsters"]
            if m["id"] != _SIM_MONSTER_INVALID
        ],
        "hand": [_card(_sim_card_id(n), u) for n, u in raw["hand"]],
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


def _sim_screen_options(gc, phase):
    """The current screen's decision context, read from the gc itself (no map
    reachability here — that needs the rebuilt SpireMap and is added by the env,
    which keeps this schema backend-symmetric and free of an env dependency).

    Only fields the screen actually owns are populated; callers branch on phase.
    """
    if phase == "combat_reward":
        info = gc.get_reward_info()
        return {
            "gold": list(info["gold"]),
            "relics": list(info["relics"]),
            "potions": list(info["potions"]),
            "card_count": info["card_count"],
            # the actual card choices, so the player/agent sees WHICH cards are on
            # offer, not just how many. get_card_reward returns Card objects;
            # str(card.id) is 'CardId.FLASH_OF_STEEL' -> take the bare enum name.
            "cards": ([_card_reward_entry(c) for c in gc.get_card_reward()]
                      if info["card_count"] else []),
        }
    if phase == "card_select":
        return {"cards": [(_sim_card_id(n), u) for n, u in gc.get_select_cards()]}
    if phase == "event":
        return {"event": gc.cur_event_name}
    # map/rest/chest/boss_relic/shop options are enumerated by the env (map needs
    # SpireMap; the rest are fixed-shape) -> nothing gc-readable to add here.
    return {}


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
    return {
        "character": _enum_tail(gc.character) if hasattr(gc, "character")
                     else _enum_tail(gc.cc) if hasattr(gc, "cc") else "INVALID",
        "ascension": getattr(gc, "ascension", 0),
        "act": gc.act,
        "floor": gc.floor_num,
        "boss": _enum_tail(gc.boss),
        "relics": list(gc.relic_ids()),
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
