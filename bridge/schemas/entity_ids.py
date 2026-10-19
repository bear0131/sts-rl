"""Explicit Java ID aliases; preserve the frozen checkpoint token order.

Source: SaveFileMappings.h and Potions.h, checked against frozen vocabularies.
Runtime does not parse simulator headers.
"""

SIMULATOR_PARITY_EXPORT_VERSION = "simulator-parity-v5"

# These are active runtime abilities, distinct from the relic's persistent
# display counter. Both are represented by native Player relic bits.
LIVE_RELIC_STATE_FIELDS = {
    "Centennial Puzzle": "centennial_puzzle_active",
    "Lizard Tail": "lizard_tail_active",
}

# Java counter sentinel -> native persistent RelicInstance.data.  The mapping
# is explicit so a used-up relic cannot become true merely because -2 != 0.
RELIC_COUNTER_SENTINELS = {
    "Lizard Tail": {-1: 1, -2: 0},
    "MawBank": {-1: 1, -2: 0},
}

# Native combat uses -1 for the armed tenth attack; persistent native and
# Java counters both use 9. Do not apply this transform to run/save data.
COMBAT_RELIC_COUNTER_SENTINELS = {"Pen Nib": {9: -1}}


def native_relic_counter(relic_id, value):
    """Translate explicit Java sentinel states; never clamp unknown values."""
    if type(value) is not int:
        raise ValueError(f"relic {relic_id!r} counter is not an integer: {value!r}")
    sentinels = RELIC_COUNTER_SENTINELS.get(relic_id)
    if sentinels is not None:
        if value not in sentinels:
            raise ValueError(f"relic {relic_id!r} counter outside its sentinel word table: {value}")
        return sentinels[value]
    return value


def native_combat_relic_counter(relic_id, value):
    return COMBAT_RELIC_COUNTER_SENTINELS.get(relic_id, {}).get(value, value)

CARD_ALIASES = {
    "Venomology": "Alchemize",
    "Ghostly": "Apparition",
    "BattleHymn": "Battle Hymn",
    "BecomeAlmighty": "Become Almighty",
    "Blood for Blood": "Blood For Blood",
    "BootSequence": "Boot Sequence",
    "BowlingBash": "Bowling Bash",
    "Lockon": "Bullseye",
    "CarveReality": "Carve Reality",
    "Conserve Battery": "Charge Battery",
    "Gash": "Claw",
    "ConjureBlade": "Conjure Blade",
    "Creative AI": "Creative Ai",
    "Crippling Poison": "Crippling Cloud",
    "CrushJoints": "Crush Joints",
    "CutThroughFate": "Cut Through Fate",
    "DeceiveReality": "Deceive Reality",
    "DeusExMachina": "Deus Ex Machina",
    "DevaForm": "Deva Form",
    "Dodge and Roll": "Dodge And Roll",
    "Doom and Gloom": "Doom And Gloom",
    "EmptyBody": "Empty Body",
    "EmptyFist": "Empty Fist",
    "EmptyMind": "Empty Mind",
    "Undo": "Equilibrium",
    "FameAndFortune": "Fame And Fortune",
    "Fasting2": "Fasting",
    "FearNoEvil": "Fear No Evil",
    "Flash of Steel": "Flash Of Steel",
    "FlurryOfBlows": "Flurry Of Blows",
    "FlyingSleeves": "Flying Sleeves",
    "FollowUp": "Follow Up",
    "ForeignInfluence": "Foreign Influence",
    "Wireheading": "Foresight",
    "FTL": "Ftl",
    "Go for the Eyes": "Go For The Eyes",
    "HandOfGreed": "Hand Of Greed",
    "InnerPeace": "Inner Peace",
    "J.A.X.": "Jax",
    "Judgement": "Judgment",
    "JustLucky": "Just Lucky",
    "LessonLearned": "Lesson Learned",
    "LikeWater": "Like Water",
    "LiveForever": "Live Forever",
    "Master of Strategy": "Master Of Strategy",
    "MasterReality": "Master Reality",
    "MentalFortress": "Mental Fortress",
    "Multi-Cast": "Multi Cast",
    "Night Terror": "Nightmare",
    "Steam Power": "Overclock",
    "PanicButton": "Panic Button",
    "PiercingWail": "Piercing Wail",
    "PathToVictory": "Pressure Points",
    "ReachHeaven": "Reach Heaven",
    "Redo": "Recursion",
    "Rip and Tear": "Rip And Tear",
    "RitualDagger": "Ritual Dagger",
    "Adaptation": "Rushdown",
    "SandsOfTime": "Sands Of Time",
    "SashWhip": "Sash Whip",
    "SignatureMove": "Signature Move",
    "Vengeance": "Simmering Fury",
    "Underhanded Strike": "Sneaky Strike",
    "SpiritShield": "Spirit Shield",
    "Steam": "Steam Barrier",
    "Storm of Steel": "Storm Of Steel",
    "TalkToTheHand": "Talk To The Hand",
    "ThirdEye": "Third Eye",
    "ThroughViolence": "Through Violence",
    "Tools of the Trade": "Tools Of The Trade",
    "ClearTheMind": "Tranquility",
    "WaveOfTheHand": "Wave Of The Hand",
    "WheelKick": "Wheel Kick",
    "WindmillStrike": "Windmill Strike",
    "Wraith Form v2": "Wraith Form",
    "WreathOfFlame": "Wreath Of Flame"
}

POTION_ALIASES = {
    "AttackPotion": "Attack Potion",
    "BlessingOfTheForge": "Blessing Of The Forge",
    "BloodPotion": "Blood Potion",
    "BottledMiracle": "Bottled Miracle",
    "ColorlessPotion": "Colorless Potion",
    "CultistPotion": "Cultist Potion",
    "CunningPotion": "Cunning Potion",
    "DistilledChaos": "Distilled Chaos",
    "DuplicationPotion": "Duplication Potion",
    "ElixirPotion": "Elixir Potion",
    "EntropicBrew": "Entropic Brew",
    "EssenceOfDarkness": "Essence Of Darkness",
    "EssenceOfSteel": "Essence Of Steel",
    "FairyPotion": "Fairy Potion",
    "FearPotion": "Fear Potion",
    "SteroidPotion": "Flex Potion",
    "FocusPotion": "Focus Potion",
    "GamblersBrew": "Gamblers Brew",
    "GhostInAJar": "Ghost In A Jar",
    "HeartOfIron": "Heart Of Iron",
    "LiquidBronze": "Liquid Bronze",
    "LiquidMemories": "Liquid Memories",
    "PotionOfCapacity": "Potion Of Capacity",
    "PowerPotion": "Power Potion",
    "SkillPotion": "Skill Potion",
    "SmokeBomb": "Smoke Bomb",
    "SneckoOil": "Snecko Oil",
    "SpeedPotion": "Speed Potion",
    "StancePotion": "Stance Potion"
}

# Complete exceptional dispositions from desktop-1.0.jar.  Every other Java
# card/relic/potion ID is covered by the expanded canonical identity/alias
# tables in codec_contract.py.  Keeping unavailable/developer-only entities
# explicit prevents a future broad "unknown -> PAD" fallback.
UNSUPPORTED_REAL_CARD_IDS = {
    "Discipline": "unused Watcher beta card absent from the native CardId enum",
    "Impulse": "unused Defect beta card absent from the native CardId enum",
    "Unraveling": "unused Watcher beta card absent from the native CardId enum",
}

UNSUPPORTED_REAL_RELIC_IDS = {
    "Discerning Monocle": "deprecated relic absent from the native relic enum",
    "Test 1": "developer-only relic",
    "Test 3": "developer-only relic",
    "Test 4": "developer-only relic",
    "Test 5": "developer-only relic",
    "Test 6": "developer-only relic",
}

# This is the one stock CommunicationMod identity that denotes a real empty
# physical potion slot.  Strings such as EMPTY/None/PAD are not accepted.
EMPTY_REAL_POTION_IDS = frozenset({"Potion Slot"})

# RewardItem.RewardType enum values from desktop-1.0.jar and the exact
# CommunicationMod choice labels that address them.  Keys are normalized only
# for case/whitespace; no localized/display-name inference is permitted.
REAL_REWARD_TYPE_IDS = frozenset({
    "CARD", "GOLD", "RELIC", "POTION", "STOLEN_GOLD",
    "EMERALD_KEY", "SAPPHIRE_KEY",
})
REWARD_CHOICE_TYPES = {
    "card": "CARD",
    "gold": "GOLD",
    "stolen_gold": "STOLEN_GOLD",
    "relic": "RELIC",
    "potion": "POTION",
    "emerald_key": "EMERALD_KEY",
    "sapphire_key": "SAPPHIRE_KEY",
}

# One semantic name, one exact CommunicationMod wire field, and one exact
# native GameContext field for each Act-4 key.  Red/green/blue are simulator
# implementation names; ruby/emerald/sapphire are the public game identities.
RUN_KEY_FIELDS = {
    "ruby": {"real": "has_ruby_key", "sim": "red_key"},
    "emerald": {"real": "has_emerald_key", "sim": "green_key"},
    "sapphire": {"real": "has_sapphire_key", "sim": "blue_key"},
}

# Non-card identity codecs are kept here as data so every bridge consumer and
# the contract audit use exactly the same spelling table.  Identity entries are
# expanded from the frozen vocabularies by codec_contract.py; only transport
# aliases and source-to-semantic mappings need literals here.
MONSTER_ALIASES = {
    "Serpent": "SpireGrowth",
}

REAL_POWER_ALIASES = {
    "BattleHymn": "Battle Hymn",
    "BeatOfDeath": "Beat Of Death",
    "BlockReturnPower": "Block Return",
    "Confusion": "Confused",
    "CorpseExplosionPower": "Corpse Explosion",
    "Creative AI": "Creative Ai",
    "DevaForm": "Deva",
    "DevotionPower": "Devotion",
    "Draw Card": "Draw Card Next Turn",
    "DuplicationPower": "Duplication",
    "EndTurnDeath": "Blasphemer",
    "EnergizedBlue": "Energized",
    "EstablishmentPower": "Establishment",
    "FreeAttackPower": "Free Attack Power",
    "Hello": "Hello World",
    "IntangiblePlayer": "Intangible",
    "LikeWaterPower": "Like Water",
    "Lockon": "Lock On",
    "MasterRealityPower": "Master Reality",
    "NoBlockPower": "No Block",
    "OmegaPower": "Omega",
    "PathToVictoryPower": "Mark",
    "StaticDischarge": "Static Discharge",
    "WaveOfTheHandPower": "Wave Of The Hand",
    "WireheadingPower": "Foresight",
    "Wraith Form v2": "Wraith Form",
    "WrathNextTurnPower": "Wrath Next Turn",
    "Flex": "Lose Strength",
    "DexLoss": "Lose Dexterity",
    "Generic Strength Up Power": "Generic Strength Up",
    "Life Link": "Regrow",
    "Regenerate": "Regen",
    "Regeneration": "Regen",
}

SIM_POWER_ALIASES = {
    "Weak": "Weakened",
    "Enrage": "Anger",
}

# Exact cross-language enum spellings accepted by the native combat-state
# importer.  These lists deliberately duplicate the frozen C++ enum boundary:
# codec_contract validates that every canonical power maps to one entry, and
# C++ compile-time assertions validate that its name arrays match the enums.
NATIVE_PLAYER_STATUS_IDS = (
    "DOUBLE_DAMAGE", "DRAW_REDUCTION", "FRAIL", "INTANGIBLE",
    "VULNERABLE", "WEAK", "BIAS", "CONFUSED", "CONSTRICTED",
    "ENTANGLED", "FASTING", "HEX", "LOSE_DEXTERITY", "LOSE_STRENGTH",
    "NO_BLOCK", "NO_DRAW", "WRAITH_FORM", "BARRICADE", "BLASPHEMER",
    "CORRUPTION", "ELECTRO", "SURROUNDED", "MASTER_REALITY", "PEN_NIB",
    "WRATH_NEXT_TURN", "AMPLIFY", "BLUR", "BUFFER", "COLLECT",
    "DOUBLE_TAP", "DUPLICATION", "ECHO_FORM", "FREE_ATTACK_POWER",
    "REBOUND", "MANTRA", "ACCURACY", "AFTER_IMAGE", "BATTLE_HYMN",
    "BRUTALITY", "BURST", "COMBUST", "CREATIVE_AI", "DARK_EMBRACE",
    "DEMON_FORM", "DEVA", "DEVOTION", "DRAW_CARD_NEXT_TURN",
    "ENERGIZED", "ENVENOM", "ESTABLISHMENT", "EVOLVE", "FEEL_NO_PAIN",
    "FIRE_BREATHING", "FLAME_BARRIER", "FOCUS", "FORESIGHT",
    "HELLO_WORLD", "INFINITE_BLADES", "JUGGERNAUT", "LIKE_WATER", "LOOP",
    "MAGNETISM", "MAYHEM", "METALLICIZE", "NEXT_TURN_BLOCK",
    "NOXIOUS_FUMES", "OMEGA", "PANACHE", "PHANTASMAL", "PLATED_ARMOR",
    "RAGE", "REGEN", "RITUAL", "RUPTURE", "SADISTIC",
    "STATIC_DISCHARGE", "THORNS", "THOUSAND_CUTS", "TOOLS_OF_THE_TRADE",
    "VIGOR", "WAVE_OF_THE_HAND", "EQUILIBRIUM", "ARTIFACT", "DEXTERITY",
    "STRENGTH", "THE_BOMB",
)

NATIVE_MONSTER_STATUS_IDS = (
    "ARTIFACT", "BLOCK_RETURN", "CHOKED", "CORPSE_EXPLOSION", "LOCK_ON",
    "MARK", "METALLICIZE", "PLATED_ARMOR", "POISON", "REGEN",
    "SHACKLED", "STRENGTH", "VULNERABLE", "WEAK", "ANGRY",
    "BEAT_OF_DEATH", "CURIOSITY", "CURL_UP", "ENRAGE", "FADING",
    "FLIGHT", "GENERIC_STRENGTH_UP", "INTANGIBLE", "MALLEABLE",
    "MODE_SHIFT", "RITUAL", "SLOW", "SPORE_CLOUD", "THIEVERY", "THORNS",
    "TIME_WARP", "INVINCIBLE", "REACTIVE", "SHARP_HIDE", "ASLEEP",
    "BARRICADE", "MINION", "MINION_LEADER", "PAINFUL_STABS", "REGROW",
    "SHIFTING", "STASIS",
)

NATIVE_POWER_ENUM_ALIASES = {
    "WEAKENED": "WEAK",
    "ANGER": "ENRAGE",
    "CONFUSION": "CONFUSED",
    "FLEX": "LOSE_STRENGTH",
    "REGENERATE": "REGEN",
    "GENERIC_STRENGTH_UP_POWER": "GENERIC_STRENGTH_UP",
}


def native_power_enum(value):
    """Map one already-canonical power token to its declared native enum."""
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    return NATIVE_POWER_ENUM_ALIASES.get(text, text)

# These Java display powers are not thrown away: their future-relevant meaning
# is carried by the named structural field.  The contract validator rejects a
# dropped power unless this table documents where the information lives.
STRUCTURAL_POWER_FIELDS = {
    "Asleep": "monster.move",
    "Split": "monster.move",
    "Explosive": "monster.move_history",
    "Unawakened": "monster.unawakened",
    "Compulsive": "monster.reactive",
    # AbstractMonster derives this display-only power from the player's
    # Surrounded state and relative hitbox position.  Native intent damage
    # performs the same derivation; there is no independent mutable counter.
    "BackAttack": "player.surrounded + monster position/adjusted intent damage",
    "Berserk": "player.energy_per_turn",
    # Java creates a unique TheBomb<N> power for each copy.  Countdown and
    # damage are losslessly folded into the three native bomb damage buckets.
    "TheBomb*": "player.combat_counters.bomb_1/bomb_2/bomb_3",
}

# Exact Java Power IDs present in desktop-1.0.jar but intentionally unsupported
# by the native status/structural contracts.  Supported non-structural IDs are
# covered by REAL_POWER_ALIASES plus the frozen player/monster vocabularies.
# Seeing one of these in a live frame is a capability error, never a request to
# delete the power or encode it as PAD.
UNSUPPORTED_REAL_POWER_IDS = {
    "Adaptation": "Watcher Rushdown power",
    "AlwaysMad": "unused/deprecated monster power",
    "AngelForm": "unused/deprecated player power",
    "Attack Burn": "unused/deprecated player power",
    "CannotChangeStancePower": "Watcher stance rule",
    "Conserve": "unused/deprecated Defect power",
    "Controlled": "Watcher Mental Fortress power",
    "DEPRECATEDCondense": "deprecated Defect power",
    "DisciplinePower": "unused/deprecated Watcher power",
    "Draw": "unused/deprecated draw power",
    "EmotionalTurmoilPower": "unused/deprecated Watcher power",
    "EnergyDownPower": "unused/deprecated energy power",
    "FlickPower": "unused/deprecated Watcher power",
    "FlowPower": "unused/deprecated Watcher power",
    "Grounded": "unused/deprecated Watcher power",
    "GrowthPower": "unused/deprecated monster power",
    "Heatsink": "Defect Heatsinks power",
    "HotHot": "unused/deprecated player power",
    "Lightning Mastery": "unused/deprecated Defect power",
    "Mastery": "unused/deprecated Watcher power",
    "Night Terror": "unused/deprecated Nightmare power",
    "Nirvana": "Watcher Nirvana power",
    "NoSkills": "unused/deprecated combat restriction",
    "Nullify Attack": "unused/deprecated monster power",
    "OmnisciencePower": "Watcher Omniscience power",
    "RechargingCore": "unused/deprecated Defect power",
    "Repair": "unused/deprecated Defect power",
    "Retain Cards": "Watcher retain counter",
    "Retribution": "unused/deprecated Watcher power",
    "Serenity": "unused/deprecated Watcher power",
    "Skill Burn": "unused/deprecated player power",
    "Storm": "Defect Storm power",
    "StrikeUp": "unused/deprecated player power",
    "Study": "Watcher Study power",
    "TimeMazePower": "unused/deprecated Time Maze power",
    "Vault": "Watcher Vault turn-boundary power",
    "Winter": "unused/deprecated Defect power",
}

# CardId values with implemented play rules, plus unplayable statuses/curses
# whose in-hand/draw/discard hooks are implemented elsewhere.  This is a
# capability table, distinct from the complete identity table: recognizing a
# foreign card name must not imply that native search can execute it.
NATIVE_COMBAT_CARD_ENUM_IDS = frozenset({
    "ANGER", "APOTHEOSIS", "APPARITION", "ARMAMENTS", "BANDAGE_UP",
    "BARRICADE", "BASH", "BATTLE_TRANCE", "BERSERK", "BITE", "BLIND",
    "BLOODLETTING", "BLOOD_FOR_BLOOD", "BLUDGEON", "BODY_SLAM",
    "BRUTALITY", "BURNING_PACT", "CARNAGE", "CHRYSALIS", "CLASH",
    "CLEAVE", "CLOTHESLINE", "COMBUST", "CORRUPTION", "DARK_EMBRACE",
    "DARK_SHACKLES", "DEEP_BREATH", "DEFEND_BLUE", "DEFEND_GREEN",
    "DEFEND_PURPLE", "DEFEND_RED", "DEMON_FORM", "DISARM", "DISCOVERY",
    "DOUBLE_TAP", "DRAMATIC_ENTRANCE", "DROPKICK", "DUAL_WIELD",
    "ENLIGHTENMENT", "ENTRENCH", "EVOLVE", "EXHUME", "FEED",
    "FEEL_NO_PAIN", "FIEND_FIRE", "FINESSE", "FIRE_BREATHING",
    "FLAME_BARRIER", "FLASH_OF_STEEL", "FLEX", "FORETHOUGHT",
    "GHOSTLY_ARMOR", "GOOD_INSTINCTS", "HAND_OF_GREED", "HAVOC",
    "HEADBUTT", "HEAVY_BLADE", "HEMOKINESIS", "IMMOLATE", "IMPATIENCE",
    "IMPERVIOUS", "INFERNAL_BLADE", "INFLAME", "INTIMIDATE", "IRON_WAVE",
    "JACK_OF_ALL_TRADES", "JAX", "JUGGERNAUT", "LIMIT_BREAK", "MADNESS",
    "MAGNETISM", "MASTER_OF_STRATEGY", "MAYHEM", "METALLICIZE",
    "METAMORPHOSIS", "MIND_BLAST", "OFFERING", "PANACEA", "PANACHE",
    "PANIC_BUTTON", "PERFECTED_STRIKE", "POMMEL_STRIKE", "POWER_THROUGH",
    "PUMMEL", "PURITY", "RAGE", "RAMPAGE", "REAPER", "RECKLESS_CHARGE",
    "RITUAL_DAGGER", "RUPTURE", "SADISTIC_NATURE", "SEARING_BLOW",
    "SECOND_WIND", "SECRET_TECHNIQUE", "SECRET_WEAPON", "SEEING_RED",
    "SENTINEL", "SEVER_SOUL", "SHOCKWAVE", "SHRUG_IT_OFF",
    "SPOT_WEAKNESS", "STRIKE_BLUE", "STRIKE_GREEN", "STRIKE_PURPLE",
    "STRIKE_RED", "SWIFT_STRIKE", "SWORD_BOOMERANG", "THE_BOMB",
    "THINKING_AHEAD", "THUNDERCLAP", "TRANSMUTATION", "TRIP", "TRUE_GRIT",
    "TWIN_STRIKE", "UPPERCUT", "VIOLENCE", "WARCRY", "WHIRLWIND",
    "WILD_STRIKE", "WRAITH_FORM",
    # Passive/unplayable status and curse semantics.
    "BURN", "DAZED", "SLIMED", "VOID", "WOUND", "ASCENDERS_BANE",
    "CLUMSY", "CURSE_OF_THE_BELL", "DECAY", "DOUBT", "INJURY",
    "NECRONOMICURSE", "NORMALITY", "PAIN", "PARASITE", "PRIDE",
    "REGRET", "SHAME", "WRITHE",
})

SCREEN_PHASES = {
    "EVENT": "event",
    "CARD_REWARD": "combat_reward",
    "COMBAT_REWARD": "combat_reward",
    "GRID": "card_select",
    # HAND_SELECT is a combat sub-screen in normal play, so RealGameAdapter
    # reports phase=combat while combat_state is present.  Keep its standalone
    # transport disposition explicit as well: a transient frame must never be
    # rejected or guessed merely because combat_state arrived one frame late.
    "HAND_SELECT": "card_select",
    "MAP": "map",
    "REST": "rest",
    "CHEST": "chest",
    "BOSS_REWARD": "boss_relic",
    "SHOP_ROOM": "shop",
    "SHOP_SCREEN": "shop",
    "NONE": "none",
    "COMPLETE": "none",
    "GAME_OVER": "none",
}

# ChoiceScreenUtils.ChoiceType and CommandExecutor.getAvailableCommands from
# the installed CommunicationMod jar.  These are source vocabularies, not just
# the subset currently emitted in one run.  Unknown additions fail validation
# instead of being silently ignored by the bridge.
REAL_SCREEN_TYPE_IDS = frozenset({
    "EVENT", "CHEST", "SHOP_ROOM", "REST", "CARD_REWARD",
    "COMBAT_REWARD", "MAP", "BOSS_REWARD", "SHOP_SCREEN", "GRID",
    "HAND_SELECT", "GAME_OVER", "COMPLETE", "NONE",
})

REAL_COMMAND_IDS = frozenset({
    "PLAY", "CHOOSE", "END", "POTION", "CONFIRM", "PROCEED",
    "SKIP", "CANCEL", "RETURN", "LEAVE", "START", "KEY", "CLICK",
    "WAIT", "STATE",
})

# Full Java enum keyword sets, checked against the installed game bytecode.
# Composite intents keep every component here; the frozen observation's coarse
# intent category is still derived identically from the canonical native move
# on both backends, while the exact move identity is retained separately.
REAL_INTENT_COMPONENTS = {
    "ATTACK": ("attack",), "ATTACK_BUFF": ("attack", "buff"),
    "ATTACK_DEBUFF": ("attack", "debuff"),
    "ATTACK_DEFEND": ("attack", "defend"),
    "BUFF": ("buff",), "DEBUFF": ("debuff",),
    "STRONG_DEBUFF": ("strong_debuff",), "DEBUG": ("debug",),
    "DEFEND": ("defend",), "DEFEND_DEBUFF": ("defend", "debuff"),
    "DEFEND_BUFF": ("defend", "buff"), "ESCAPE": ("escape",),
    "MAGIC": ("magic",), "NONE": (), "SLEEP": ("sleep",),
    "STUN": ("stun",), "UNKNOWN": ("unknown",),
}
REAL_CHARACTER_IDS = {
    "IRONCLAD": "IRONCLAD", "THE_SILENT": "SILENT", "SILENT": "SILENT",
    "DEFECT": "DEFECT", "WATCHER": "WATCHER",
}
REAL_ROOM_PHASES = frozenset({"COMBAT", "EVENT", "COMPLETE", "INCOMPLETE"})
REAL_ACTION_PHASES = frozenset({"WAITING_ON_USER", "EXECUTING_ACTIONS"})
REAL_CARD_TYPE_IDS = frozenset({"ATTACK", "SKILL", "POWER", "STATUS", "CURSE"})
REAL_CARD_RARITY_IDS = frozenset({"BASIC", "SPECIAL", "COMMON", "UNCOMMON", "RARE", "CURSE"})
REAL_CARD_TARGET_IDS = {
    "ENEMY": True, "ALL_ENEMY": False, "SELF": False,
    "NONE": False, "SELF_AND_ENEMY": True, "ALL": False,
}

ROOM_ALIASES = {
    "m": "MONSTER",
    "monster": "MONSTER",
    "enemy": "MONSTER",
    "e": "ELITE",
    "elite": "ELITE",
    "?": "EVENT",
    "event": "EVENT",
    "$": "SHOP",
    "shop": "SHOP",
    "r": "REST",
    "rest": "REST",
    "t": "TREASURE",
    "treasure": "TREASURE",
    "chest": "TREASURE",
    "boss": "BOSS",
}

BOSS_ALIASES = {
    "The Guardian": "THE_GUARDIAN",
    "Guardian": "THE_GUARDIAN",
    "Slime Boss": "SLIME_BOSS",
    "SlimeBoss": "SLIME_BOSS",
    "Hexaghost": "HEXAGHOST",
    "Bronze Automaton": "AUTOMATON",
    "BronzeAutomaton": "AUTOMATON",
    "The Collector": "COLLECTOR",
    "Collector": "COLLECTOR",
    "The Champ": "CHAMP",
    "Champ": "CHAMP",
    "Awakened One": "AWAKENED_ONE",
    "AwakenedOne": "AWAKENED_ONE",
    "Time Eater": "TIME_EATER",
    "TimeEater": "TIME_EATER",
    "Donu and Deca": "DONU_AND_DECA",
    "Donu & Deca": "DONU_AND_DECA",
    "DonuDeca": "DONU_AND_DECA",
    "Corrupt Heart": "THE_HEART",
    "CorruptHeart": "THE_HEART",
}

BOSS_CANONICAL_IDS = frozenset({
    "SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST",
    "AUTOMATON", "COLLECTOR", "CHAMP",
    "AWAKENED_ONE", "TIME_EATER", "DONU_AND_DECA", "THE_HEART",
})

EVENT_ALIASES = {
    "FaceTrader": "Face Trader",
    "SecretPortal": "Secret Portal",
    "Match and Keep!": "Match and Keep",
    "N'loth": "Nloth",
    "NoteForYourself": "Note For Yourself",
    "MindBloom": "Mindbloom",
    "SensoryStone": "Sensory Stone",
    "Neow Event": "NEOW",
    "Neow": "NEOW",
    "We Meet Again": "WeMeetAgain",
    "WeMeetAgain": "WeMeetAgain",
    "Bonfire Spirits": "Bonfire Elementals",
}

# Typed dynamic event-option metadata -> frozen native EVENT_OPT index.  The
# wire ``choice_index`` is only the current UI ordinal and may compress disabled
# choices, so it must never be reused as the simulator semantic ID.  These are
# the dynamic events whose visible resource identity changes per run.
DYNAMIC_EVENT_OPTION_KINDS = {
    "WeMeetAgain": {
        "lose_potion": 0,
        "lose_gold": 1,
        "lose_card": 2,
        "leave": 3,
    },
    "Note For Yourself": {
        "gain_card_then_remove": 0,
        "leave": 1,
    },
    "Designer": {
        "upgrade_card": 0,
        "upgrade_random_cards": 1,
        "remove_card": 2,
        "transform_random_cards": 3,
        "remove_card_upgrade_random": 4,
        "lose_hp": 5,
    },
}

# Stock screens that reuse the event transport but are not policy events and
# therefore have no checkpoint EVENT_VOCAB token.  They must be consumed only
# by the exact transport contract, never encoded as PAD or guessed as an event.
TRANSPORT_ONLY_REAL_EVENT_IDS = {
    "Spire Heart": "act-three ending dialogue before the actual terminal frame",
}

# Current AbstractGameAction class -> simulator CardSelectTask.  Dynamic sources
# are listed separately because their private fields distinguish two contracts.
COMBAT_SELECTION_SOURCES = {
    "ArmamentsAction": "ARMAMENTS",
    "CodexAction": "CODEX",
    "DiscoveryAction": "DISCOVERY",
    "DualWieldAction": "DUAL_WIELD",
    "ExhumeAction": "EXHUME",
    "ForethoughtAction": "FORETHOUGHT",
    "GamblingChipAction": "GAMBLE",
    "DiscardPileToTopOfDeckAction": "HEADBUTT",
    "MeditateAction": "MEDITATE",
    "NightmareAction": "NIGHTMARE",
    "RecycleAction": "RECYCLE",
    "SkillFromDeckToHandAction": "SECRET_TECHNIQUE",
    "AttackFromDeckToHandAction": "SECRET_WEAPON",
    "BetterDrawPileToHandAction": "SEEK",
    "SeekAction": "SEEK",  # legacy game class; shipped Seek uses BetterDraw...
    "SetupAction": "SETUP",
    "PutOnDeckAction": "WARCRY",
}

DYNAMIC_COMBAT_SELECTION_SOURCES = {
    "BetterDiscardPileToHandAction": {
        "setCost=false": "HOLOGRAM",
        "setCost=true": "LIQUID_MEMORIES_POTION",
    },
    "ExhaustAction": {
        "one": "EXHAUST_ONE",
        "many": "EXHAUST_MANY",
    },
}

OVERWORLD_SELECTION_FLAGS = {
    "for_upgrade": "UPGRADE",
    "for_purge": "REMOVE",
    "for_transform": "TRANSFORM",
}

OVERWORLD_SELECTION_RELICS = {
    "Astrolabe": "TRANSFORM_UPGRADE",
    "Bottled Flame": "BOTTLE",
    "Bottled Lightning": "BOTTLE",
    "Bottled Tornado": "BOTTLE",
    "DollysMirror": "DUPLICATE",
}

OVERWORLD_SELECTION_EVENTS = {
    "Bonfire": "BONFIRE_SPIRITS",
    "Duplicator": "DUPLICATE",
    "TheLibrary": "OBTAIN",
    "NoteForYourself": "OBTAIN",
    "NeowEvent": "OBTAIN",
}

# Explicitly unsupported real screens are still part of the source vocabulary:
# the exporter emits UNMAPPED and Python fails closed instead of manufacturing
# PAD or reusing a semantically different task.
UNSUPPORTED_SELECTION_SOURCES = {
    "ScryAction": "native simulator has no Scry CardSelectTask",
    "ClarityAction": "deprecated Scry implementation has no native task",
    "OmniscienceAction": "native simulator has no Omniscience CardSelectTask",
    "CacheAction": "unused legacy action has no native task",
    "DiscardPileToHandAction": "unused legacy action has no native task",
    "DeckToHandAction": "unused legacy action has no native task",
}

