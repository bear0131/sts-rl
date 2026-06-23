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

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
_BUILD = os.path.join(_HERE, "..", "sim", "sts_lightspeed", "build")
if hasattr(os, "add_dll_directory") and os.path.isdir(_BUILD):
    os.add_dll_directory(os.path.abspath(_BUILD))
sys.path.insert(0, os.path.abspath(_BUILD))

import slaythespire as sts  # noqa: E402

import encounters  # noqa: E402

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


def fixed_factory(seed, encounter, floor=1, ascension=0):
    """Return a gc_factory that always opens the same combat. Handy for tests
    and for overfitting a single combat to sanity-check the training loop."""

    def factory():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.floor_num = floor
        return gc, encounter

    return factory
