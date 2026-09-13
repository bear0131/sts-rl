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
import math

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


# Ironclad training card pool. Starter types are included with equal slot odds;
# shuffling still randomizes the initial hand. Curses and colorless cards remain
# outside this uniform pool.
_IRONCLAD_CARDS = [
    "STRIKE_RED", "DEFEND_RED", "BASH",
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


def _uniform_ironclad_deck(np_random, size: int) -> list[tuple[str, int]]:
    """Build a random deck using every Ironclad card with equal slot odds.

    Sampling is with replacement because a training deck represents a random
    legal loadout, not a draft without duplicates. Curses are deliberately
    excluded from this distribution.
    """
    return [
        (
            _IRONCLAD_CARDS[int(np_random.integers(0, len(_IRONCLAD_CARDS)))],
            int(np_random.integers(0, 2)),
        )
        for _ in range(size)
    ]

# The "common" obtainable curses (events / Cursed Key). Excludes the special ones
# tied to ascension/relics/card-gen (Ascender's Bane, Curse of the Bell,
# Necronomicurse, Pride). A real run sometimes carries one, so we sprinkle them in.
_CURSE_POOL = [
    "CLUMSY", "DECAY", "DOUBT", "INJURY", "NORMALITY",
    "PAIN", "PARASITE", "REGRET", "SHAME", "WRITHE",
]

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

# Boss relics whose effect is read at battle START / DURING combat (no at-pickup
# deck transform like Pandora's Box / Astrolabe, no purely out-of-combat effect),
# so they can be attached to a sandbox combat via set_relics without us running any
# pickup logic. Used to make the gauntlet loadout a bit more in-domain (deeper
# floors carry boss relics; ~10% of runs swap Burning Blood for one).
_BOSS_RELIC_POOL = [
    "Snecko Eye", "Runic Pyramid", "Velvet Choker", "Mark of Pain",
    "Coffee Dripper", "Fusion Hammer", "Runic Dome", "Sozu", "Cursed Key",
    "Ice Cream", "Busted Crown", "Philosopher's Stone", "Runic Cube",
    "SlaversCollar", "Gambling Chip",
]


def _act_of(floor):
    """The act a floor belongs to: 1 (<=16), 2 (17-33), 3 (>=34)."""
    return 1 if floor <= 16 else 2 if floor <= 33 else 3


def _mean_added(floor):
    """Mean total cards added to the deck BY this floor. Saturating curve: early
    slope ~0.85/floor (you open a run grabbing lots of cards), flattening to a
    total mean ~15 by floor 50 (by act 3 you barely add). 15*(1-e^-0.06f)."""
    return 15.0 * (1.0 - math.exp(-0.06 * floor))


def _around(np_random, mean, hi=None):
    """A heavily-randomized non-negative count whose mean is ~`mean`: a uniform
    integer in [0, round(2*mean)], optionally capped at `hi`. Wide on purpose —
    the spread (weak decks deep / strong decks shallow) is what lets the value head
    disentangle floor-effect from deck-effect when it learns combat-power."""
    top = int(round(2.0 * mean))
    if hi is not None:
        top = min(top, hi)
    return int(np_random.integers(0, top + 1)) if top > 0 else 0


def _build_loadout_gc(np_random, ascension, extra_cards, n_relics, hp_frac):
    """A fresh Ironclad GameContext with a random deck/relics/starting-HP loadout,
    no floor/encounter set. Shared by the single-combat and gauntlet factories.

      - deck: starter + a random number of random Ironclad cards (random upgrade),
      - relics: Burning Blood + a random subset of _RELIC_POOL,
      - starting HP: a random fraction of max_hp (so the model learns to read its
        own HP bar).

    Ranges are (lo, hi) inclusive. This is a "card literacy" distribution (random
    legal cards), NOT a realistic-synergy distribution — by design for stage 1.
    """
    seed = int(np_random.integers(0, _SEED_HI))
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)

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
    return gc


def random_combat_factory(np_random, pool="normal", ascension=0,
                          extra_cards=(0, 25), n_relics=(0, 5),
                          hp_frac=(0.2, 1.0)):
    """Return a gc_factory that samples a random combat loadout + a random `pool`
    encounter each call (one combat = one episode)."""
    names, (flo, fhi) = encounters.POOLS[pool]

    def factory():
        gc = _build_loadout_gc(np_random, ascension, extra_cards, n_relics, hp_frac)
        gc.floor_num = int(np_random.integers(flo, fhi + 1))
        return gc, names[int(np_random.integers(0, len(names)))]

    return factory


def _build_gauntlet_gc(
    np_random, floor, ascension, hp_frac, *, uniform_ironclad_deck: bool = False,
):
    """A GameContext for a gauntlet episode, anchored on `floor` (= level L): the
    act, encounter pool, and loadout richness all derive from it, so the loadout is
    a plausible (if randomized) snapshot of a real run AT that floor — in-domain but
    with wide spread. Floor is FIXED for the whole episode (stable obs feature).

      - deck: by default, starter minus random Strike/Defend removals plus random
        Ironclad cards; with ``uniform_ironclad_deck``, every slot is sampled
        uniformly from the Ironclad card pool and the starter is omitted,
      - relics: a common subset (mean 0.20*L) + one boss relic per act-boss passed
        (act-1), and ~10% of runs swap Burning Blood for an extra boss relic,
      - starting HP: a random fraction of max_hp (carries across the gauntlet).
    """
    act = _act_of(floor)
    seed = int(np_random.integers(0, _SEED_HI))
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    gc.floor_num = floor
    gc.act = act

    # Keep the old floor-dependent deck-size distribution while changing the
    # card identity distribution for rule-data collection.
    starters = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4
    n_remove = _around(np_random, 0.10 * floor, hi=len(starters))
    n_add = _around(np_random, _mean_added(floor))
    if uniform_ironclad_deck:
        deck_size = len(starters) - n_remove + 1 + n_add
        deck = _uniform_ironclad_deck(np_random, deck_size)
    else:
        if n_remove:
            drop = set(np_random.choice(len(starters), size=n_remove, replace=False).tolist())
            starters = [c for i, c in enumerate(starters) if i not in drop]
        deck = [(c, 0) for c in starters] + [("BASH", 0)]
        for _ in range(n_add):
            card = _IRONCLAD_CARDS[int(np_random.integers(0, len(_IRONCLAD_CARDS)))]
            deck.append((card, int(np_random.integers(0, 2))))
        # ~5% of runs carry a random curse (events / Cursed Key in a real run).
        if float(np_random.uniform(0.0, 1.0)) < 0.05:
            deck.append((_CURSE_POOL[int(np_random.integers(0, len(_CURSE_POOL)))], 0))
    gc.set_deck(deck)

    # relics: Burning Blood (90%) + common subset + boss relics (one per boss passed,
    # plus one more in lieu of Burning Blood on the 10% swap).
    keep_burning = float(np_random.uniform(0.0, 1.0)) >= 0.10
    relics = ["Burning Blood"] if keep_burning else []
    n_common = _around(np_random, 0.20 * floor, hi=len(_RELIC_POOL))
    if n_common:
        idx = np_random.choice(len(_RELIC_POOL), size=n_common, replace=False)
        relics += [_RELIC_POOL[int(i)] for i in idx]
    n_boss = min((act - 1) + (0 if keep_burning else 1), len(_BOSS_RELIC_POOL))
    if n_boss:
        idx = np_random.choice(len(_BOSS_RELIC_POOL), size=n_boss, replace=False)
        relics += [_BOSS_RELIC_POOL[int(i)] for i in idx]
    gc.set_relics(relics)

    frac = float(np_random.uniform(hp_frac[0], hp_frac[1]))
    gc.cur_hp = max(1, int(round(gc.max_hp * frac)))
    return gc


def random_gauntlet_factory(np_random, ascension=0, hp_frac=(0.2, 1.0),
                            floor_range=(1, 50),
                            type_probs=(("normal", 0.75), ("elite", 0.20),
                                        ("boss", 0.05)),
                            *, uniform_ironclad_deck: bool = False):
    """For the combat GAUNTLET (SimCombatBackend): ONE random loadout fights an
    endless stream of random encounters until it dies (HP carries over, no heal).

    Each episode samples a level L in `floor_range`; the loadout (_build_gauntlet_gc)
    and the encounter pool both derive from L, and L stays FIXED for the episode.
    Each fight independently rolls an encounter TYPE per `type_probs` and draws a
    random encounter from act(L)'s pool of that type.

    Returns (make_gc, next_encounter):
      - make_gc() -> the per-episode loadout GameContext (samples L internally),
      - next_encounter(gc) -> the next fight's encounter name (reads gc's fixed L).

    Reward is "+1 per fight cleared, episode ends on death" — surviving longer (i.e.
    spending HP wisely) is what pays off, so HP value is learned implicitly with no
    explicit HP/death reward term (see worklog/archive/doc/night_0625/combat_pretrain_decision.md).
    """
    flo, fhi = floor_range
    types = [t for t, _ in type_probs]
    probs = [float(p) for _, p in type_probs]

    def make_gc():
        floor = int(np_random.integers(flo, fhi + 1))
        return _build_gauntlet_gc(
            np_random, floor, ascension, hp_frac,
            uniform_ironclad_deck=uniform_ironclad_deck,
        )

    def next_encounter(gc):
        pool = encounters.ACT_POOLS[_act_of(gc.floor_num)][
            types[int(np_random.choice(len(types), p=probs))]
        ]
        return pool[int(np_random.integers(0, len(pool)))]

    return make_gc, next_encounter


def random_rule_combat_factory(
    np_random,
    ascension=0,
    hp_frac=(0.2, 1.0),
    floor_range=(1, 50),
    type_probs=(("normal", 0.60), ("elite", 0.25), ("boss", 0.15)),
):
    """Return independent random combats for rule-data collection.

    Loadouts intentionally reuse the combat-gauntlet distribution, but every
    call creates a new loadout and exactly one encounter. There is no HP carry
    or repeated-fight episode. Bosses are oversampled relative to PPO combat
    training because rule coverage, rather than realistic room frequency, is
    the objective here.
    """
    make_gc, next_encounter = random_gauntlet_factory(
        np_random,
        ascension=ascension,
        hp_frac=hp_frac,
        floor_range=floor_range,
        type_probs=type_probs,
        uniform_ironclad_deck=True,
    )

    def factory():
        gc = make_gc()
        return gc, next_encounter(gc)

    return factory


def fixed_factory(seed, encounter, floor=1, ascension=0):
    """Return a gc_factory that always opens the same combat. Handy for tests
    and for overfitting a single combat to sanity-check the training loop."""

    def factory():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.floor_num = floor
        return gc, encounter

    return factory
