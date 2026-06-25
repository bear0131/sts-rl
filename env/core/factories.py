"""gc_factory builders for SimCombatEnv.

A gc_factory is a zero-arg callable returning (GameContext, encounter_name).
SimCombatEnv calls it each reset() to get a fresh combat. This is the ONLY
place besides combat_env.py that imports slaythespire.

The act-1 factory builds a vanilla Ironclad GameContext — which already carries
the starter deck + Burning Blood (GameContext ctor calls initPlayer/initRelics)
— then forces a randomly chosen act-1 encounter via create_battle.

What "random encounter+seed" means here: we independently sample a 64-bit seed,
a floor, and an encounter name. The combat RNG is Random(seed + floor) (see
sim BattleContext.cpp), so (seed, floor) together pick the shuffle/AI stream;
the encounter name is forced and is decoupled from map RNG (which is NOT aligned
with the real game). For training this is fine — we want coverage of the combat
distribution, and the in-combat RNG stays the aligned real value.
"""

# `import env.core` runs the package __init__ which puts the sim build dir on
# sys.path; that must happen before `import slaythespire`.
from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts

from . import encounters

# 64-bit seed space for the sim's Random.
_SEED_HI = (1 << 63) - 1


def random_act1_factory(np_random, pool="normal", ascension=0):
    """Return a gc_factory that samples a random act-1 combat each call.

    np_random: a numpy Generator (SimCombatEnv passes its gym self.np_random,
        so env.reset(seed=...) makes the episode stream reproducible).
    pool: "normal" | "elite" | "boss" (see encounters.POOLS).
    ascension: ascension level for the GameContext.
    """
    names, (flo, fhi) = encounters.POOLS[pool]

    def factory():
        seed = int(np_random.integers(0, _SEED_HI))
        floor = int(np_random.integers(flo, fhi + 1))
        name = names[int(np_random.integers(0, len(names)))]
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.floor_num = floor
        return gc, name

    return factory


# Ironclad's obtainable colored card pool (sim enum strings; the 72 cards of
# TransformCardPool.colorCardPool[0] in CardPools.h). Starter Strikes/Defends and
# curses/colorless are intentionally excluded — these are the cards a real run can
# add to the deck. We sample from here to build random training decks.
_IRONCLAD_CARDS = [
    "ANGER", "CLEAVE", "WARCRY", "FLEX", "IRON_WAVE", "BODY_SLAM", "TRUE_GRIT",
    "SHRUG_IT_OFF", "CLASH", "THUNDERCLAP", "POMMEL_STRIKE", "TWIN_STRIKE",
    "CLOTHESLINE", "ARMAMENTS", "HAVOC", "HEADBUTT", "WILD_STRIKE", "HEAVY_BLADE",
    "PERFECTED_STRIKE", "SWORD_BOOMERANG", "EVOLVE", "UPPERCUT", "GHOSTLY_ARMOR",
    "FIRE_BREATHING", "DROPKICK", "CARNAGE", "BLOODLETTING", "RUPTURE",
    "SECOND_WIND", "SEARING_BLOW", "BATTLE_TRANCE", "SENTINEL", "ENTRENCH",
    "RAGE", "FEEL_NO_PAIN", "DISARM", "SEEING_RED", "DARK_EMBRACE", "COMBUST",
    "WHIRLWIND", "SEVER_SOUL", "RAMPAGE", "SHOCKWAVE", "METALLICIZE",
    "BURNING_PACT", "PUMMEL", "FLAME_BARRIER", "BLOOD_FOR_BLOOD", "INTIMIDATE",
    "HEMOKINESIS", "RECKLESS_CHARGE", "INFERNAL_BLADE", "DUAL_WIELD",
    "POWER_THROUGH", "INFLAME", "SPOT_WEAKNESS", "DOUBLE_TAP", "DEMON_FORM",
    "BLUDGEON", "FEED", "LIMIT_BREAK", "CORRUPTION", "BARRICADE", "FIEND_FIRE",
    "BERSERK", "IMPERVIOUS", "JUGGERNAUT", "BRUTALITY", "REAPER", "EXHUME",
    "OFFERING", "IMMOLATE",
]

# Ironclad starter deck in sim enum strings: 5x Strike, 4x Defend, 1x Bash.
_STARTER_DECK = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"]

# Curated combat-relevant relic pool (stable ids from relicIds[]). Kept to relics
# whose effect is read at battle start / during combat and is mechanically simple
# — coverage for the obs's relic slots without out-of-combat-only or save-state
# edge cases. Burning Blood (starter) is always kept on top of these.
_RELIC_POOL = [
    "Vajra", "Oddly Smooth Stone", "Anchor", "Bronze Scales", "Akabeko",
    "Bag of Marbles", "Bag of Preparation", "Blood Vial", "Kunai", "Shuriken",
    "Pen Nib", "Ornamental Fan", "Letter Opener", "Lantern", "Happy Flower",
    "Mercury Hourglass", "Red Skull", "Paper Crane", "Paper Frog",
    "StrikeDummy", "Gremlin Horn", "Nunchaku",
]


def random_combat_factory(np_random, pool="normal", ascension=0,
                          extra_cards=(0, 25), n_relics=(0, 5),
                          hp_frac=(0.2, 1.0)):
    """Return a gc_factory that samples a random combat loadout each call, for
    combat-head pretraining (see doc/night_0625/combat_pretrain_decision.md).

    Each call randomizes, on top of a fresh Ironclad GameContext:
      - deck: starter + a random number of random Ironclad cards (random upgrade),
      - relics: Burning Blood + a random subset of _RELIC_POOL,
      - starting HP: a random fraction of max_hp (so the model learns to read its
        own HP bar; dying then pays the death penalty in CombatRewardWrapper),
      - encounter: a random combat from `pool` (as in random_act1_factory).

    Ranges are (lo, hi) inclusive. This is a "card literacy" distribution (random
    legal cards), NOT a realistic-synergy distribution — by design for stage 1.
    """
    names, (flo, fhi) = encounters.POOLS[pool]

    def factory():
        seed = int(np_random.integers(0, _SEED_HI))
        floor = int(np_random.integers(flo, fhi + 1))
        name = names[int(np_random.integers(0, len(names)))]
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.floor_num = floor

        # deck = starter + N random Ironclad cards (each unupgraded or +1).
        deck = [(c, 0) for c in _STARTER_DECK]
        n_extra = int(np_random.integers(extra_cards[0], extra_cards[1] + 1))
        for _ in range(n_extra):
            card = _IRONCLAD_CARDS[int(np_random.integers(0, len(_IRONCLAD_CARDS)))]
            deck.append((card, int(np_random.integers(0, 2))))
        gc.set_deck(deck)

        # relics = Burning Blood + a random subset of the curated pool.
        k = int(np_random.integers(n_relics[0], n_relics[1] + 1))
        extra_relics = []
        if k:
            idx = np_random.choice(len(_RELIC_POOL), size=k, replace=False)
            extra_relics = [_RELIC_POOL[int(i)] for i in idx]
        gc.set_relics(["Burning Blood"] + extra_relics)

        # starting HP = a random fraction of max_hp (>=1 so we never start dead).
        frac = float(np_random.uniform(hp_frac[0], hp_frac[1]))
        gc.cur_hp = max(1, int(round(gc.max_hp * frac)))

        return gc, name

    return factory


def fixed_factory(seed, encounter, floor=1, ascension=0):
    """Return a gc_factory that always opens the same combat. Handy for tests
    and for overfitting a single combat to sanity-check the training loop."""

    def factory():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.floor_num = floor
        return gc, encounter

    return factory
