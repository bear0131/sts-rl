"""Act-1 encounter name lists for create_battle.

These are the exact strings create_battle accepts. IMPORTANT: create_battle
matches against sim's `monsterEncounterStrings` (MonsterEncounters.h), NOT the
SaveFileMappings display names — the two differ ("Two Louse" vs "2 Louse",
"Lots Of Slimes" vs "Lots of Slimes", etc). A name that doesn't match resolves
to INVALID and spawns ZERO monsters (an unwinnable, non-terminating combat), so
these MUST stay verbatim from monsterEncounterStrings. encounter_names() returns
this same list — treat it as the source of truth.

Scope is act-1 only for now: the combat RNG (Random(seed+floor)) is the only
RNG verified aligned with the real game, and act-1 is where our differential
testing lives. Extend per-act when later acts are validated.
"""

# Normal combats (the game's "weak" + "strong" act-1 pools). create_battle
# forces the encounter, so map-RNG ordering does not matter here — we just need
# a valid name. Excludes elites and bosses (separate pools below).
ACT1_NORMAL = [
    "Cultist",
    "Jaw Worm",
    "Two Louse",
    "Small Slimes",
    "Blue Slaver",
    "Gremlin Gang",
    "Looter",
    "Large Slime",
    "Lots Of Slimes",
    "Exordium Thugs",
    "Exordium Wildlife",
    "Red Slaver",
    "Three Louse",
    "Two Fungi Beasts",
]

ACT1_ELITE = [
    "Gremlin Nob",
    "Lagavulin",
    "Three Sentries",
]

ACT1_BOSS = [
    "Slime Boss",
    "The Guardian",
    "Hexaghost",
]

# Floor ranges per pool, used only to vary the combat seed (Random(seed+floor)).
# These are the typical act-1 floors each room type occupies; they need not be
# exact since the floor only perturbs RNG, it does not change the encounter.
ACT1_NORMAL_FLOORS = (1, 8)   # inclusive range
ACT1_ELITE_FLOORS = (6, 14)
ACT1_BOSS_FLOORS = (16, 16)

POOLS = {
    "normal": (ACT1_NORMAL, ACT1_NORMAL_FLOORS),
    "elite": (ACT1_ELITE, ACT1_ELITE_FLOORS),
    "boss": (ACT1_BOSS, ACT1_BOSS_FLOORS),
}
