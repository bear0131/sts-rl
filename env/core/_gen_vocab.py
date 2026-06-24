"""Generator for env/core/vocab.py — the categorical id-space contract for obs.

Run from the repo root:  python -m env.core._gen_vocab

Why a generator (cf. cards_vocab.py's header note): the id vocabularies must
match the sim's own enums exactly, and must stay append-only so old checkpoints'
embedded indices never shift. The sim's C++ enums are themselves append-only
(the upstream project only ever appends ids), so we mirror their *source order*
here — that inherits their append-only guarantee for free. This script parses
the sim headers and the bridge canonical id maps, then writes vocab.py.

It does NOT import slaythespire (parses headers as text), so it runs on a box
without the sim built. It DOES import the bridge schema maps so the obs id space
is the SAME canonical id space the normalizers speak (sim names -> real ids).

Provenance (parsed files, all under sim/sts_lightspeed/include/constants/):
  Relics.h               relicIds[]            -> RELIC_VOCAB   (already canonical)
  MonsterMoves.h         enum MonsterMoveId    -> MOVE_VOCAB    (sim enum names)
  Potions.h              potionNames[]         -> POTION_VOCAB  (display names; what
                                                  bc.potions / overworld_potions return)
  PlayerStatusEffects.h  playerStatusStrings[] -> PLAYER_BUFF_VOCAB (via bridge power map)
  MonsterStatusEffects.h enemyStatusStrings[]  -> MONSTER_BUFF_VOCAB (via bridge power map)
  Events.h               eventIdStrings[]      -> EVENT_VOCAB
  MonsterIds.h           monsterIdStrings[]    -> MONSTER_VOCAB (via bridge monster map)
  MonsterEncounters.h    enum MonsterEncounter -> BOSS_VOCAB    (enum names; str(gc.boss))

Hand-fixed (small, construction-time / sim-enum-fixed) vocabularies are written
literally: ROOM_VOCAB, PHASE_VOCAB, CHARACTER_VOCAB, INTENT_CATEGORY, ARG_STEP_VOCAB.
"""

import os
import re

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CONST = os.path.join(_REPO, "sim", "sts_lightspeed", "include", "constants")
_OUT = os.path.join(_HERE, "vocab.py")

# bridge canonical id maps — the obs id space must equal the normalizers' space.
import sys
sys.path.insert(0, os.path.join(_REPO, "bridge"))
from schemas import combat_state as cs  # noqa: E402


def _read(name):
    with open(os.path.join(_CONST, name), encoding="utf-8") as f:
        return f.read()


def _string_table(text, table_name):
    """Extract the ordered string literals of  ... <table_name>[] ... { "a","b",... };"""
    m = re.search(re.escape(table_name) + r"\s*\[\s*\]\s*(?:=\s*)?\{(.*?)\}\s*;",
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"table {table_name} not found")
    return re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))


def _enum_names(text, enum_name):
    """Ordered member names of  enum class <enum_name> ... { A, B=0, C, ... };"""
    m = re.search(r"enum\s+class\s+" + re.escape(enum_name) + r"[^{]*\{(.*?)\}\s*;",
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"enum {enum_name} not found")
    body = re.sub(r"//[^\n]*", "", m.group(1))          # strip line comments
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)  # strip block comments
    names = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name = tok.split("=")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            names.append(name)
    return names


def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build():
    relics = _read("Relics.h")
    moves = _read("MonsterMoves.h")
    potions = _read("Potions.h")
    pstatus = _read("PlayerStatusEffects.h")
    mstatus = _read("MonsterStatusEffects.h")
    events = _read("Events.h")
    monsters = _read("MonsterIds.h")
    encounters = _read("MonsterEncounters.h")
    cardselect = _read(os.path.join("..", "combat", "CardSelectInfo.h"))
    gamecontext = _read(os.path.join("..", "game", "GameContext.h"))

    # --- relics: relicIds[] are already canonical stable ids; drop trailing INVALID.
    relic_ids = [r for r in _string_table(relics, "relicIds") if r != "INVALID"]

    # --- moves: sim MonsterMoveId enum names (INVALID handled as PAD slot 0).
    move_ids = [n for n in _enum_names(moves, "MonsterMoveId") if n != "INVALID"]

    # --- potions: potionNames[] display strings; drop INVALID + EMPTY_POTION_SLOT
    # (those are the PAD/empty-slot sentinels, represented by index 0 here).
    potion_ids = [p for p in _string_table(potions, "potionNames")
                  if p not in ("INVALID", "EMPTY_POTION_SLOT")]

    # --- player buffs: the four first-class powers, then statusMap entries, mapped
    # through the bridge power map into canonical (real) ids, dropping the same
    # cosmetic powers the normalizers drop. Source order = append-only.
    raw_pstatus = [s for s in _string_table(pstatus, "playerStatusStrings")
                   if s != "INVALID"]
    player_buffs = _dedup_keep_order(
        ["Strength", "Dexterity", "Focus", "Artifact"]
        + [cs._sim_power_id(s) for s in raw_pstatus]
    )
    player_buffs = [b for b in player_buffs if b not in cs._POWER_DROP]

    raw_mstatus = [s for s in _string_table(mstatus, "enemyStatusStrings")
                   if s != "INVALID"]
    monster_buffs = _dedup_keep_order(cs._sim_power_id(s) for s in raw_mstatus)
    monster_buffs = [b for b in monster_buffs if b not in cs._POWER_DROP]

    # --- events: eventIdStrings[]; drop INVALID + the room-type sentinels
    # (MONSTER/REST/SHOP/TREASURE are room markers, not real events).
    event_ids = [e for e in _string_table(events, "eventIdStrings")
                 if e not in ("INVALID", "MONSTER", "REST", "SHOP", "TREASURE")]

    # --- monsters: map sim enum-name strings -> canonical real ids via the bridge
    # map (the id space normalize_sim/normalize_real emit). Only the mapped set is
    # reachable on the sim backend today; append-only as the map grows.
    sim_monster_names = [m for m in _string_table(monsters, "monsterIdStrings")
                         if m != "INVALID = 0"]
    monster_ids = _dedup_keep_order(
        cs._MONSTER_SIM2REAL[n] for n in sim_monster_names
        if n in cs._MONSTER_SIM2REAL
    )

    # --- boss: MonsterEncounter enum names (str(gc.boss) -> 'MonsterEncounter.X').
    boss_ids = [b for b in _enum_names(encounters, "MonsterEncounter")
                if b != "INVALID"]

    # --- select context: the WHAT-is-this-card-select-screen-for vocab (obs_design
    # §4.1). One MERGED vocab over the combat CardSelectTask (Armaments/Exhaust/
    # Gamble/Headbutt/...) and the overworld CardSelectScreenType (Transform/Upgrade/
    # Remove/...). The two enums' names don't collide, so a single vocab keyed on
    # the bare enum name serves both; the binding's select_info/combat select_info
    # return these exact strings. NONE (slot 0) = not on a card-select screen.
    select_tasks = [t for t in _string_table(cardselect, "cardSelectTaskStrings")
                    if t != "INVALID"]
    select_screens = [s for s in _enum_names(gamecontext, "CardSelectScreenType")
                      if s != "INVALID"]
    select_context = _dedup_keep_order(select_tasks + select_screens)

    return {
        "RELIC_VOCAB": relic_ids,
        "MOVE_VOCAB": move_ids,
        "POTION_VOCAB": potion_ids,
        "PLAYER_BUFF_VOCAB": player_buffs,
        "MONSTER_BUFF_VOCAB": monster_buffs,
        "EVENT_VOCAB": event_ids,
        "MONSTER_VOCAB": monster_ids,
        "BOSS_VOCAB": boss_ids,
        "SELECT_CONTEXT_VOCAB": select_context,
    }


# --- hand-fixed small vocabularies (not parsed; construction/sim-enum fixed) ---
# Room enum (sts.Room) bare names, the strings RunMap.room_type emits.
ROOM_VOCAB = ["NONE", "MONSTER", "ELITE", "EVENT", "REST", "SHOP", "TREASURE",
              "BOSS", "BOSS_TREASURE", "INVALID"]
# semantic phases SimGameEnv.phase() / sim_phase() can report (PAD = "none").
PHASE_VOCAB = ["none", "combat", "map", "combat_reward", "card_select", "rest",
               "event", "chest", "boss_relic", "shop", "neow", "invalid"]
# CharacterClass enum (INVALID kept as PAD slot 0).
CHARACTER_VOCAB = ["INVALID", "IRONCLAD", "SILENT", "DEFECT", "WATCHER"]
# intent categories derived from a move id (slot 0 = unknown / not-yet-exported).
INTENT_CATEGORY = ["unknown", "attack", "defend", "buff", "debuff",
                   "strong_debuff", "sleep", "stun", "escape"]
# autoregressive arg-step the thin SlayTheSpireEnv is on (action_design.md §1).
# It tells the model WHY the current id menu looks the way it does: the ROOT step
# (pick a main action) vs a parameter step (pick the target of the verb already
# chosen). Slot 0 = "root" so the obs PAD default is the (overwhelmingly common)
# root step — a backend that never sets arg_step (e.g. the single-combat
# SimCombatEnv) reads as root. The pick_card_2/pick_discard/pick_scry slots are
# RESERVED for the deferred C++ select_context batch (multi-pick screens); filling
# a reserved slot is not an OBS bump (it is a fixed-width categorical).
ARG_STEP_VOCAB = ["root", "pick_target", "pick_card", "pick_card_2",
                  "pick_discard", "pick_scry", "pick_exhaust", "pick_hand"]


_HEADER = '''"""Categorical id-space vocabularies for the v2 observation (AUTO-GENERATED).

Do not hand-edit. Regenerate with:  python -m env.core._gen_vocab

Every vocabulary is a list in CANONICAL id space (the stable ids the bridge
normalizers speak — see bridge/schemas/combat_state.py). Index 0 is always a
PAD / EMPTY / NONE / unknown sentinel so an absent or out-of-vocab value encodes
as 0. Lists are APPEND-ONLY: they mirror the sim's own (append-only) enum source
order, so an existing id's index never shifts and old checkpoints stay valid.

Two kinds of consumer (see encode.py):
  - id vocabs (card/monster/move/relic/potion/boss/event): obs stores an int
    index; appending a new id does NOT change obs shape (only the model's
    embedding-table size). Use *_CAPACITY as the embedding-table size with
    headroom so the model needn't resize when the vocab grows.
  - buff vocabs (player/monster): obs stores a DENSE amount-vector whose length
    is the *_BUFF_VEC capacity (reserved, > current len). Appending a buff fills
    a reserved slot and does NOT change obs shape. Only changing a *_VEC / shape
    bumps OBS_VERSION (see encode.py).
"""

'''


def _fmt_list(name, items):
    body = ",\n    ".join(repr(x) for x in items)
    return f"{name} = [\n    {body},\n]\n"


def main():
    vocabs = build()
    # capacities: id-vocab embedding-table size (headroom), per obs_design.md §1.
    capacities = {
        "MONSTER_VOCAB": 96, "MOVE_VOCAB": 224, "RELIC_VOCAB": 208,
        "POTION_VOCAB": 56, "EVENT_VOCAB": 72, "BOSS_VOCAB": 72,
        "SELECT_CONTEXT_VOCAB": 40,
    }
    # reserved dense buff-vector lengths (must exceed current vocab len), §2.1/§2.2.
    buff_vecs = {"PLAYER_BUFF_VEC": 110, "MONSTER_BUFF_VEC": 50}

    parts = [_HEADER]
    # id vocabs: PREPEND a PAD sentinel at index 0 (an absent / out-of-vocab id
    # encodes as 0), mirroring CARD_VOCAB's EMPTY-at-0 convention.
    id_order = ["MONSTER_VOCAB", "MOVE_VOCAB", "RELIC_VOCAB", "POTION_VOCAB",
                "EVENT_VOCAB", "BOSS_VOCAB", "SELECT_CONTEXT_VOCAB"]
    for name in id_order:
        parts.append(_fmt_list(name, ["PAD"] + vocabs[name]))
    # buff vocabs: dense amount-vectors keyed by buff; NO sentinel (absence = 0
    # amount, not a reserved index). Each slot is a real power id.
    for name in ["PLAYER_BUFF_VOCAB", "MONSTER_BUFF_VOCAB"]:
        parts.append(_fmt_list(name, vocabs[name]))
    # hand-fixed lists (each already starts with its own PAD/NONE/unknown at 0).
    parts.append(_fmt_list("ROOM_VOCAB", ROOM_VOCAB))
    parts.append(_fmt_list("PHASE_VOCAB", PHASE_VOCAB))
    parts.append(_fmt_list("CHARACTER_VOCAB", CHARACTER_VOCAB))
    parts.append(_fmt_list("INTENT_CATEGORY", INTENT_CATEGORY))
    parts.append(_fmt_list("ARG_STEP_VOCAB", ARG_STEP_VOCAB))

    # capacities + buff vector lengths
    cap_lines = "\n".join(f"{k}_CAPACITY = {v}" for k, v in capacities.items())
    vec_lines = "\n".join(f"{k} = {v}" for k, v in buff_vecs.items())
    parts.append("# ---- embedding-table capacities (id vocabs; headroom) ----\n"
                 + cap_lines + "\n")
    parts.append("# ---- reserved dense buff-vector lengths ----\n"
                 + vec_lines + "\n")

    # name -> index maps (built once, here, so every consumer shares them).
    index_names = id_order + ["PLAYER_BUFF_VOCAB", "MONSTER_BUFF_VOCAB",
                              "ROOM_VOCAB", "PHASE_VOCAB", "CHARACTER_VOCAB",
                              "INTENT_CATEGORY", "ARG_STEP_VOCAB"]
    idx_lines = "\n".join(
        f"{n}_INDEX = {{name: i for i, name in enumerate({n})}}"
        for n in index_names
    )
    parts.append("# ---- name -> index maps ----\n" + idx_lines + "\n")

    with open(_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))

    # report
    print(f"wrote {_OUT}")
    for name in id_order:
        print(f"  {name:20s} {len(vocabs[name]) + 1:4d} (incl PAD)")
    for name in ["PLAYER_BUFF_VOCAB", "MONSTER_BUFF_VOCAB"]:
        print(f"  {name:20s} {len(vocabs[name]):4d}")
    for k, v in capacities.items():
        n = len(vocabs[k]) + 1  # +1 PAD sentinel
        assert n <= v, f"{k} len {n} (incl PAD) exceeds capacity {v}"
    for k, v in buff_vecs.items():
        src = "PLAYER_BUFF_VOCAB" if "PLAYER" in k else "MONSTER_BUFF_VOCAB"
        assert len(vocabs[src]) <= v, f"{src} exceeds {k}"


if __name__ == "__main__":
    main()
