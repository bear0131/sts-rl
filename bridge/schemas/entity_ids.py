"""Explicit Java ID aliases; preserve the frozen checkpoint token order.

Source: SaveFileMappings.h and Potions.h, checked against frozen vocabularies.
Runtime does not parse simulator headers.
"""

SIMULATOR_PARITY_EXPORT_VERSION = "simulator-parity-v10"

# Java enum order differs from native Neow::Bonus. Never use enum ordinals.
NEOW_BONUS_IDS = {name: index for index, name in enumerate((
    "THREE_CARDS", "ONE_RANDOM_RARE_CARD", "REMOVE_CARD", "UPGRADE_CARD",
    "TRANSFORM_CARD", "RANDOM_COLORLESS", "THREE_SMALL_POTIONS", "RANDOM_COMMON_RELIC",
    "TEN_PERCENT_HP_BONUS", "THREE_ENEMY_KILL", "HUNDRED_GOLD", "RANDOM_COLORLESS_2",
    "REMOVE_TWO", "ONE_RARE_RELIC", "THREE_RARE_CARDS", "TWO_FIFTY_GOLD",
    "TRANSFORM_TWO_CARDS", "TWENTY_PERCENT_HP_BONUS", "BOSS_RELIC",
))}
NEOW_DRAWBACK_IDS = {"NONE": 1, "TEN_PERCENT_HP_LOSS": 2, "NO_GOLD": 3, "CURSE": 4, "PERCENT_DAMAGE": 5}
NEOW_IMPLICIT_DRAWBACK_IDS = {"BOSS_RELIC": 6}
NEOW_SCREEN_CONTRACTS = {
    0: "intro_transport", 1: "intro_transport", 2: "unsupported_blessing_or_skip",
    3: "typed_offers", 10: "unsupported_daily", 99: "completion_transport",
    999: "unsupported_endless",
}


def neow_option_from_types(bonus_type, drawback_type):
    if not isinstance(bonus_type, str) or bonus_type not in NEOW_BONUS_IDS:
        raise ValueError(f"unknown Neow bonus keyword: {bonus_type!r}")
    if not isinstance(drawback_type, str) or drawback_type not in NEOW_DRAWBACK_IDS:
        raise ValueError(f"unknown Neow drawback keyword: {drawback_type!r}")
    if bonus_type in NEOW_IMPLICIT_DRAWBACK_IDS:
        if drawback_type != "NONE":
            raise ValueError("BOSS_RELIC has only its implicit starting-relic loss")
        drawback_id = NEOW_IMPLICIT_DRAWBACK_IDS[bonus_type]
    else:
        drawback_id = NEOW_DRAWBACK_IDS[drawback_type]
    return {"bonus_id": NEOW_BONUS_IDS[bonus_type], "drawback_id": drawback_id}

# These are active runtime abilities, distinct from the relic's persistent
# display counter. Both are represented by native Player relic bits.
LIVE_RELIC_STATE_FIELDS = {
    "Centennial Puzzle": "centennial_puzzle_active",
    "Lizard Tail": "lizard_tail_active",
}

LIVE_RELIC_COUNTER_FIELDS = {
    "Centennial Puzzle": "centennial_puzzle_active",
    "Happy Flower": "happy_flower",
    "Incense Burner": "incense_burner",
    "InkBottle": "ink_bottle",
    "Inserter": "inserter",
    "Lizard Tail": "lizard_tail_active",
    "Nunchaku": "nunchaku",
    "Pen Nib": "pen_nib",
    "Sundial": "sundial",
    "Kunai": "attacks_played",
    "Letter Opener": "skills_played",
    "Ornamental Fan": "attacks_played",
    "Pocketwatch": "cards_played",
    "Shuriken": "attacks_played",
    "Velvet Choker": "cards_played",
    "CaptainsWheel": "$turn",
    "HornCleat": "$turn",
    "StoneCalendar": "$turn",
    "NeowsBlessing": "neow_lament",
}
LIVE_RELIC_COUNTER_MODULI = {"Kunai": 3, "Letter Opener": 3, "Ornamental Fan": 3, "Shuriken": 3}
LIVE_RELIC_COUNTER_STOP_TURNS = {"CaptainsWheel": 3, "HornCleat": 2}

# Every stock relic class in desktop-1.0.jar was checked for reads/writes of
# AbstractRelic.counter (including setCounter calls). These 29 classes use it;
# all other supported identities inherit the display sentinel. This is a
# source-use inventory, NOT a claim that every phase's semantics is verified.
REAL_RELIC_COUNTER_USAGE = {
    "Ancient Tea Set": ("AncientTeaSet", "pending_next_combat_energy"),
    "CaptainsWheel": ("CaptainsWheel", "combat_turn_until_activation"),
    "CeramicFish": ("CeramicFish", "legacy_use_method_not_obtain_card_counter"),
    "Circlet": ("Circlet", "copies_obtained"),
    "Du-Vu Doll": ("DuVuDoll", "master_deck_curse_count"),
    "Girya": ("Girya", "lifts_completed"),
    "Happy Flower": ("HappyFlower", "turns_modulo_three"),
    "HornCleat": ("HornCleat", "combat_turn_until_activation"),
    "Incense Burner": ("IncenseBurner", "turns_modulo_six"),
    "InkBottle": ("InkBottle", "cards_modulo_ten"),
    "Inserter": ("Inserter", "turns_modulo_two"),
    "Kunai": ("Kunai", "attacks_this_turn_modulo_three"),
    "Letter Opener": ("LetterOpener", "skills_this_turn_modulo_three"),
    "Lizard Tail": ("LizardTail", "revival_available"),
    "Matryoshka": ("Matryoshka", "extra_chest_relic_charges"),
    "MawBank": ("MawBank", "floor_gold_available_until_shop_spend"),
    "NeowsBlessing": ("NeowsLament", "one_hp_combat_charges"),
    "NlothsMask": ("NlothsMask", "chest_relic_removal_charges"),
    "Nunchaku": ("Nunchaku", "attacks_modulo_ten"),
    "Omamori": ("Omamori", "curse_prevention_charges"),
    "Ornamental Fan": ("OrnamentalFan", "attacks_this_turn_modulo_three"),
    "Pen Nib": ("PenNib", "attacks_modulo_ten_armed_at_nine"),
    "Pocketwatch": ("Pocketwatch", "cards_this_turn"),
    "Shuriken": ("Shuriken", "attacks_this_turn_modulo_three"),
    "StoneCalendar": ("StoneCalendar", "combat_turn"),
    "Sundial": ("Sundial", "shuffles_modulo_three"),
    "Tiny Chest": ("TinyChest", "question_rooms_modulo_four"),
    "Velvet Choker": ("VelvetChoker", "cards_this_turn"),
    "WingedGreaves": ("WingBoots", "disconnected_map_move_charges"),
}

DERIVED_RELIC_COUNTER_SOURCES = {"Du-Vu Doll": "Deck::getCurseCount()"}
FRESH_RELIC_INITIAL_DATA = {
    "Lizard Tail": 1, "MawBank": 1, "NlothsMask": 1, "Circlet": 1,
    "Matryoshka": 2, "Omamori": 2, "NeowsBlessing": 3, "WingedGreaves": 3,
}

# Inclusive Java display-counter domains at stable decision/save boundaries;
# unbounded counters use the Java signed-int limit, never clipping.
RELIC_COUNTER_DOMAINS = {
    "Circlet": (1, 2147483647), "Du-Vu Doll": (0, 2147483647), "Girya": (0, 3),
    "Happy Flower": (0, 2), "Incense Burner": (0, 5), "InkBottle": (0, 9),
    "Inserter": (0, 1), "Nunchaku": (0, 9), "Pen Nib": (0, 9),
    "Sundial": (0, 2), "Tiny Chest": (0, 3),
}
PHASE_RELIC_COUNTER_DOMAINS = {
    "CaptainsWheel": (-1, 2), "HornCleat": (-1, 1), "Kunai": (-1, 2),
    "Letter Opener": (-1, 2), "Ornamental Fan": (-1, 2),
    "Pocketwatch": (-1, 2147483647), "Shuriken": (-1, 2),
    "StoneCalendar": (-1, 2147483647), "Velvet Choker": (-1, 6),
}

# Java counter sentinel -> native persistent RelicInstance.data.  The mapping
# is explicit so a used-up relic cannot become true merely because -2 != 0.
RELIC_COUNTER_SENTINELS = {
    "Ancient Tea Set": {-1: 0, -2: 1},
    "Lizard Tail": {-1: 1, -2: 0},
    "MawBank": {-1: 1, -2: 0},
    "Matryoshka": {-2: 0, 1: 1, 2: 2},
    "NlothsMask": {-2: 0, 1: 1},
    "NeowsBlessing": {-2: 0, 0: 0, 1: 1, 2: 2, 3: 3},
    "Omamori": {0: 0, 1: 1, 2: 2},
    "WingedGreaves": {-2: 0, 0: 0, 1: 1, 2: 2, 3: 3},
}

# Native combat uses -1 for the armed tenth attack; persistent native and
# Java counters both use 9. Do not apply this transform to run/save data.
COMBAT_RELIC_COUNTER_SENTINELS = {"Pen Nib": {9: -1}}


def native_relic_counter(relic_id, value):
    """Translate explicit Java sentinel states; never clamp unknown values."""
    if type(value) is not int:
        raise ValueError(f"relic {relic_id!r} counter is not an integer: {value!r}")
    from env.core.vocab import RELIC_VOCAB_INDEX
    if relic_id not in RELIC_VOCAB_INDEX or relic_id == "PAD":
        raise ValueError(f"unknown live relic identity: {relic_id!r}")
    sentinels = RELIC_COUNTER_SENTINELS.get(relic_id)
    if sentinels is not None:
        if value not in sentinels:
            raise ValueError(f"relic {relic_id!r} counter outside its sentinel word table: {value}")
        return sentinels[value]
    domain = RELIC_COUNTER_DOMAINS.get(relic_id) or PHASE_RELIC_COUNTER_DOMAINS.get(relic_id)
    if domain is not None:
        if not domain[0] <= value <= domain[1]:
            raise ValueError(f"relic {relic_id!r} counter outside its numeric word table: {value}")
        return 0 if value == -1 else value
    # CeramicFish.use() is a legacy unused counter path; ordinary obtain-card
    # gold does not change counter. All remaining classes inherit -1.
    if value != -1:
        raise ValueError(f"relic {relic_id!r} has no supported display counter: {value}")
    return 0


def native_combat_relic_counter(relic_id, value):
    if relic_id == "Pen Nib" and (type(value) is not int or not 0 <= value <= 9):
        raise ValueError(f"real Pen Nib counter outside 0..9: {value!r}")
    return COMBAT_RELIC_COUNTER_SENTINELS.get(relic_id, {}).get(value, value)


def live_relic_counter_value(relic_id, canonical):
    if relic_id == "Pocketwatch" and canonical.get("post_enemy_turn_boundary", False):
        return canonical["boundary_relic_counters"][relic_id]
    source = LIVE_RELIC_COUNTER_FIELDS[relic_id]
    value = (canonical["turn"] if source == "$turn" else
             canonical["player"]["combat_counters"][source])
    modulo = LIVE_RELIC_COUNTER_MODULI.get(relic_id)
    stop = LIVE_RELIC_COUNTER_STOP_TURNS.get(relic_id)
    if modulo is not None:
        value %= modulo
    if stop is not None and value >= stop:
        value = 0
    return value

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

# Concrete stock room classes in desktop-1.0.jar. AbstractRoom and CampfireUI
# are not concrete rooms. Victory pages have no native combat equivalent.
REAL_ROOM_CLASS_IDS = {
    "EmptyRoom": "NONE", "EventRoom": "EVENT", "MonsterRoom": "MONSTER",
    "MonsterRoomBoss": "BOSS", "MonsterRoomElite": "ELITE", "RestRoom": "REST",
    "ShopRoom": "SHOP", "TreasureRoom": "TREASURE", "TreasureRoomBoss": "BOSS_TREASURE",
    "TrueVictoryRoom": None, "VictoryRoom": None,
}
NATIVE_COMBAT_ROOM_IDS = frozenset({"MONSTER", "ELITE", "BOSS", "EVENT"})
POTION_CAPACITY_RULES = {"ascension_0_to_10": 3, "ascension_11_to_20": 2, "Potion Belt": 2, "native_max": 5}
COMPACT_PLAYER_SCALAR_FIELDS = {
    "hp": "BattleContext.player.curHp -> player_scalars[0,5]",
    "max_hp": "BattleContext.player.maxHp -> player_scalars[1,5]",
    "block": "BattleContext.player.block -> player_scalars[2]",
    "energy": "BattleContext.player.energy -> player_scalars[3]",
    "turn": "BattleContext.turn (zero-based) -> player_scalars[4] (one-based)",
    "gold": "BattleContext.player.gold -> gold[0], overrides root journal",
}
FRESH_CARD_MISC = {'Ritual Dagger': 15, 'Genetic Algorithm': 1}
NATIVE_CARD_INSTANCE_FIELDS = {
    'id': 'native CardId spelling; INVALID is not a physical card',
    'upgrades': 'CardInstance.getUpgradeCount; complete Searing Blow count',
    'uid': 'CardInstance.uniqueId; stable identity, master cards use physical deck slot',
    'special_data': 'CardInstance.specialData; current growth, not fresh constructor value',
    'base_cost': 'CardInstance.cost; survives end-of-turn reset',
    'cost': 'CardInstance.costForTurn; underlying cost, not effective free-once cost',
    'free_to_play_once': 'CardInstance.freeToPlayOnce; explicit boolean',
    'retain': 'CardInstance.retain; explicit boolean',
    'exhausts': 'CardInstance.exhaust; actual persistent instance property, separate from one-use queue flag',
}
NATIVE_CARD_SNAPSHOT_FIELDS = {
    'version': 'integer 2', 'next_uid': 'allocation cursor; greater than every active uid',
    'hand': 'ordered complete records; at most 10',
    'draw': 'ordered complete records', 'discard': 'ordered complete records',
    'exhaust': 'ordered complete records',
    'stasis': 'exactly two native physical slots; null only for actual empty slots',
}
COMBAT_SELECTION_TRANSPORT_FIELDS = {
    'selection_context': 'explicit native semantic task, crosschecked against source class',
    'selection_source': 'current Java action class; never guessed from hand or text',
    'any_number': 'HAND_SELECT private HandCardSelectScreen.anyNumber; GRID original serializer; explicit boolean',
    'max_cards': 'HAND_SELECT.numCardsToSelect; 99 is normalized to actual selectable hand size',
    'num_cards': 'GRID.numCards; actual requested count, not always one',
    'can_pick_zero': 'HAND_SELECT.canPickZero; explicit boolean, independent of any_number',
    'selected': 'HAND_SELECT.selectedCards; complete typed instances; UUID and full instance hand-removal/selected-append acknowledgement',
    'selected_cards': 'GRID.selectedCards; complete typed instances in selection order; Liquid Memories stages 1/2 mandatory picks without moving source cards',
    'instance': 'parsed real_card_state sidecar per candidate; native selection_card_state validates all representable members and physical UID',
    'selected_instances': 'parsed GRID records; known UUID to stable native UID, checked against selected_card_state; not hand MULTI prefix bits',
    'grid_selection_slots': 'compact layout9 capacity 2; categorical columns 224/225, select-relative scalar columns 196/197; Python/native decoders fill existing OBS20 selected_card_ids/upgrades slots',
    'grid_settlement': 'Liquid Memories receipt validates all four ordered piles and complete instances; actual moved cards get cost0, full hand leaves remainder in discard; applyPowers refreshes Body Slam.base_damage=block, Mind Blast.base_damage=draw size, Ritual Dagger.base_block=misc',
}
# Internal pending-use inventory, not a claim that arbitrary Java queues can
# already be restored. These values survive native clones and pending keys;
# full real queue/action transport remains tracked by BRIDGE28.
NATIVE_CARD_QUEUE_FIELDS = {
    'card': 'full CardInstance; same-instance copies retain UUID but are distinct objects',
    'target': 'explicit monster slot, not compacted visible-monster ordinal',
    'isEndTurn': 'null Java card/end-turn queue marker; not an EMPTY playable card',
    'triggerOnUse': 'inverse of dontTriggerOnUseCard for status/end-turn settlement',
    'ignoreEnergyTotal': 'AbstractCard.ignoreEnergyOnUse; preserve original X-cost energy',
    'energyOnUse': 'CardQueueItem.energyOnUse; captured use budget',
    'freeToPlay': 'queue-level free play, separate from instance freeToPlayOnce',
    'randomTarget': 'choose a target when this queued item executes',
    'autoplay': 'CardQueueItem.autoplayCard / AbstractCard.isInAutoplay',
    'regretCardCount': 'captured hand size used by queued Regret HP loss',
    'purgeOnUse': 'temporary replay object removed after use; no physical pile move',
    'inLimbo': 'queued power copy belongs to Java limbo and receives same-UUID growth; Necronomicon copy does not',
    'exhaustOnUse': 'one-use exhaust outcome, distinct from persistent CardInstance.exhaust',
}
CARD_STATE_TRANSPORT_FIELDS = {
    'version': 'integer 1', 'base_cost': 'AbstractCard.cost',
    'free_to_play_once': 'AbstractCard.freeToPlayOnce', 'retain': 'AbstractCard.retain',
    'self_retain': 'AbstractCard.selfRetain', 'innate': 'AbstractCard.isInnate',
    'base_damage': 'AbstractCard.baseDamage', 'base_block': 'AbstractCard.baseBlock',
    'bottled_flame': 'AbstractCard.inBottleFlame',
    'bottled_lightning': 'AbstractCard.inBottleLightning',
    'bottled_tornado': 'AbstractCard.inBottleTornado',
}
# Every stock convertCardToJson key has a disposition, even when its consumer
# is still missing. Missing transport is not permission to substitute zero.
REAL_CARD_WIRE_FIELDS = {
    'name': {'source': 'AbstractCard.name', 'consumer': 'presentation only; never identity', 'status': 'display_only'},
    'uuid': {'source': 'AbstractCard.uuid', 'consumer': 'CombatCardIdentityMap: current physical master slot; stable temporary IDs >=96', 'status': 'settled_root_restored_selection_pending'},
    'misc': {'source': 'AbstractCard.misc; parity exporter always exports zero too', 'consumer': 'Card.misc / CardInstance.specialData; independent unsupported nonzero values rejected', 'status': 'settled_root_restored'},
    'is_playable': {'source': 'AbstractCard.canUse(player,null)', 'consumer': 'real legal mask crosschecked against native action mask', 'status': 'action_validation'},
    'cost': {'source': 'AbstractCard.costForTurn', 'consumer': 'CardInstance.costForTurn in all four piles and stasis; hand projection uses free-once', 'status': 'settled_root_restored'},
    'upgrades': {'source': 'AbstractCard.timesUpgraded', 'consumer': 'Card.getUpgraded / CardInstance.getUpgradeCount', 'status': 'transported'},
    'id': {'source': 'AbstractCard.cardID', 'consumer': 'card identity word tables', 'status': 'transported'},
    'type': {'source': 'AbstractCard.type.name()', 'consumer': 'native card-type metadata', 'status': 'identity_metadata'},
    'rarity': {'source': 'AbstractCard.rarity.name()', 'consumer': 'native card-rarity metadata', 'status': 'identity_metadata'},
    'has_target': {'source': 'AbstractCard.target ENEMY/SELF_AND_ENEMY', 'consumer': 'real/native legal target masks', 'status': 'action_validation'},
    'exhausts': {'source': 'AbstractCard.exhaust', 'consumer': 'CardInstance.exhaust; includes Blue Candle / Medical Kit mutations', 'status': 'settled_root_restored'},
    'ethereal': {'source': 'AbstractCard.isEthereal', 'consumer': 'checked against CardInstance.isEthereal; different dynamic values rejected', 'status': 'printed_validated_dynamic_pending'},
}
CARD_STATE_FIELD_CONTRACTS = {
    'base_cost': {'source': 'AbstractCard.cost', 'native': 'CardInstance.cost', 'status': 'settled_root_restored', 'reason': 'Madness/combat-permanent cost survives turn reset'},
    'free_to_play_once': {'source': 'AbstractCard.freeToPlayOnce', 'native': 'CardInstance.freeToPlayOnce', 'status': 'settled_root_restored', 'reason': 'effective hand cost is zero without erasing underlying cost'},
    'bottled_flame': {'source': 'AbstractCard.inBottleFlame', 'native': 'Deck.bottleIdxs[ATTACK]', 'status': 'master_restored', 'reason': 'master deck candidate eligibility and initial draw'},
    'bottled_lightning': {'source': 'AbstractCard.inBottleLightning', 'native': 'Deck.bottleIdxs[SKILL]', 'status': 'master_restored', 'reason': 'master deck candidate eligibility and initial draw'},
    'bottled_tornado': {'source': 'AbstractCard.inBottleTornado', 'native': 'Deck.bottleIdxs[POWER]', 'status': 'master_restored', 'reason': 'master deck candidate eligibility and initial draw'},
    'retain': {'source': 'AbstractCard.retain', 'native': 'CardInstance.retain', 'status': 'settled_root_restored', 'reason': 'turn-end retention is separate from printed self-retain'},
    'self_retain': {'source': 'AbstractCard.selfRetain', 'native': 'CardInstance.hasSelfRetain()', 'status': 'printed_validated_dynamic_pending', 'reason': 'different dynamic values rejected rather than dropped'},
    'innate': {'source': 'AbstractCard.isInnate', 'native': 'Card innate metadata at initial draw', 'status': 'initial_draw_audit_pending', 'reason': 'settled root already contains actual drawn cards; independent mutations need source audit'},
    'base_damage': {'source': 'AbstractCard.baseDamage', 'native': 'Rampage growth / fixed or recomputed card damage', 'status': 'growth_restored_other_mutations_audit_pending', 'reason': 'do not confuse cached Body Slam / Perfected Strike values with independent instance growth'},
    'base_block': {'source': 'AbstractCard.baseBlock', 'native': 'fixed card block / unsupported GeneticAlgorithm', 'status': 'mutation_audit_pending', 'reason': 'not yet a general native instance field'},
    'rampage_growth': {'source': 'Rampage.baseDamage - printed base damage', 'native': 'CardInstance.specialData', 'status': 'settled_root_restored', 'reason': 'not encoded in AbstractCard.misc'},
}
CARD_SPECIAL_DATA_FIELDS = {
    'SEARING_BLOW': {'real': 'AbstractCard.timesUpgraded (not misc)', 'native': 'Card.misc / CardInstance.specialData', 'status': 'upgrade_count_transported'},
    'RAMPAGE': {'real': 'baseDamage minus printed damage (not misc)', 'native': 'CardInstance.specialData', 'status': 'settled_root_restored'},
    'RITUAL_DAGGER': {'real': 'AbstractCard.misc, fresh15, fatal growth3/5', 'native': 'Card.misc / CardInstance.specialData', 'status': 'settled_root_restored'},
    'GENETIC_ALGORITHM': {'real': 'AbstractCard.misc, fresh1', 'native': 'Card.misc / CardInstance.specialData', 'status': 'fresh_rule_fixed_combat_unsupported'},
}
NATIVE_GLOBAL_FIELDS = {
    "character": {"real": "class", "native": "cc", "support": "IRONCLAD only"},
    "ascension": {"real": "ascension_level", "native": "ascension", "domain": "integer 0..20"},
    "act": {"real": "act", "native": "act", "domain": "integer 1..4"},
    "floor": {"real": "floor", "native": "floor_num", "domain": "nonnegative signed int32"},
    "hp": {"real": "current_hp", "native": "cur_hp", "domain": "0..max_hp, no clamping"},
    "max_hp": {"real": "max_hp", "native": "max_hp", "domain": "positive signed int32"},
    "gold": {"real": "gold", "native": "gold / BattleContext.player.gold", "domain": "0..32767 (native combat int16); reject overflow"},
    "boss": {"real": "act_boss", "native": "boss", "codec": "boss identity"},
    "deck": {"real": "deck", "native": "set_deck_state identity/upgrades checked against deck_state", "codec": "ordered identity/upgrade model projection; never used alone for current-state restore"},
    "deck_state": {"real": "deck[*].misc + simulator_card_state.bottled_*", "native": "set_deck_state", "codec": "complete ordered id/upgrades/native-misc/bottle snapshot; Searing Blow misc=upgrades"},
    "relics": {"real": "relics[*].id", "native": "set_relic_state[*].id", "codec": "relic identity"},
    "relic_counters": {"real": "relics[*].counter", "native": "set_relic_state[*].counter", "codec": "relic_counter_contracts"},
    "potions": {"real": "potions (including empty slots)", "native": "potion_capacity + set_potion_state", "codec": "potion identity; slot count must equal capacity rule"},
    "keys": {"real": "has_*_key", "native": "red_key / green_key / blue_key", "codec": "run_keys; explicit booleans"},
    "combat_is_boss": {"real": "room_type", "native": "cur_room", "codec": "real_room_classes; no substring matching"},
    "boss_relics": {"real": "BOSS_REWARD screen", "native": "not a combat-root field", "support": "must be empty at combat entry"},
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

# Source-audited hit callbacks, not an assertion that every pending Java
# action can be restored. Behavior regression remains tracked by BRIDGE34;
# complete action transport remains BRIDGE28. Amount means post-block,
# post-Boot and post-Invincible damage unless the entry says otherwise.
HIT_REACTION_CONTRACTS = {
    'ThornsPower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / THORNS',
        'guard': 'NORMAL hit from another non-null owner; includes amount0 and fatal hits',
        'queue': 'top THORNS DamagePlayer; survives combat-victory filtering',
    },
    'AngryPower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / ANGRY, not ENRAGE',
        'guard': 'NORMAL hit from non-null owner, amount>0',
        'queue': 'top Strength application; reject dead/escaped recipient when it executes',
    },
    'InvinciblePower.onAttackedToChangeDamage': {
        'native': 'Monster attack/damage helpers / INVINCIBLE',
        'guard': 'cap post-block damage before HP-loss-dependent callbacks',
        'queue': 'immediate cap and remaining-budget subtraction',
    },
    'EnvenomPower.onAttack': {
        'native': 'Monster::attackedUnblockedHelper / player.ENVENOM',
        'guard': 'NORMAL hit on another creature, amount>0 after Invincible cap',
        'queue': 'top Poison application',
    },
    'PlatedArmorPower.wasHPLost': {
        'native': 'Monster::attackedUnblockedHelper / PLATED_ARMOR',
        'guard': 'NORMAL hit from another non-null owner, amount>0',
        'queue': 'bottom reduction; last stack queues bottom ARMOR_BREAK for Shelled Parasite',
    },
    'CurlUpPower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / CURL_UP',
        'guard': 'not triggered, NORMAL hit, 0<amount<currentHealth',
        'queue': 'bottom block; native clears trigger immediately; distinct Java triggered/removal pending state still BRIDGE28',
    },
    'FlightPower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / FLIGHT',
        'guard': 'NORMAL hit, amount>0, float(amount)/2<currentHealth (Java reuses final-receive calculation)',
        'queue': 'bottom reduction; last stack queues bottom GROUNDED for Byrd',
    },
    'MalleablePower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / MALLEABLE',
        'guard': 'NORMAL hit from non-null owner, 0<amount<currentHealth',
        'queue': 'monster bottom block; increment next block amount immediately',
    },
    'ReactivePower.onAttacked': {
        'native': 'Monster::attackedUnblockedHelper / REACTIVE; Java ID Compulsive',
        'guard': 'NORMAL hit from non-null owner, 0<amount<currentHealth',
        'queue': 'one bottom RollMove per hit at original position and monster slot; native counter is queued roll count',
    },
    'DamageAction.update': {
        'native': 'Actions::AttackPlayer / DamagePlayer; VampireDamageAction is a separate implementation',
        'guard': 'NORMAL attack cancels for dying/halfDead source; THORNS retaliation does not use that source guard',
        'queue': 'each hit checked when executed; DAMAGE victory-filter classification in COMBAT_VICTORY_ACTION_CONTRACTS',
    },
    'Whirlwind.use/WhirlwindAction.update': {
        'native': 'BattleContext::useAttackCard / Actions::WhirlwindAction / AttackAllMonsterRecursive',
        'guard': 'capture integer multiDamage at use; waves=energyOnUse+Chemical X; no uint16 clipping',
        'queue': 'append waves behind existing callbacks; each wave permits addToTop retaliation before next wave',
    },
}


# Source-level damage timing for the current Ironclad/colorless batch. These
# contracts do not imply complete dynamic baseDamage/pending transport (BRIDGE28)
# or completed behavior regression (BRIDGE35).
CARD_DAMAGE_TIMING_CONTRACTS = {
    'CLEAVE': {'java': 'Cleave.use / DamageAllEnemiesAction', 'native': 'calculateCardDamageMatrix / AttackAllEnemy', 'capture': 'per-slot multiDamage at use; no settlement recalculation'},
    'DRAMATIC_ENTRANCE': {'java': 'DramaticEntrance.use / DamageAllEnemiesAction', 'native': 'calculateCardDamageMatrix / AttackAllEnemy', 'capture': 'per-slot multiDamage at use; no settlement recalculation'},
    'IMMOLATE': {'java': 'Immolate.use / DamageAllEnemiesAction', 'native': 'calculateCardDamageMatrix / AttackAllEnemy', 'capture': 'per-slot multiDamage at use; Burn generation remains after damage'},
    'THUNDERCLAP': {'java': 'ThunderClap.use / DamageAllEnemiesAction', 'native': 'calculateCardDamageMatrix / AttackAllEnemy', 'capture': 'per-slot multiDamage at use; Vulnerable applied after damage'},
    'REAPER': {'java': 'Reaper.use / VampireDamageAllEnemiesAction', 'native': 'calculateCardDamageMatrix / ReaperAction', 'capture': 'per-slot multiDamage at use; heal actual HP loss; enqueue heal before victory filtering'},
    'WHIRLWIND': {'java': 'Whirlwind.use / WhirlwindAction', 'native': 'calculateCardDamageMatrix / AttackAllMonsterRecursive', 'capture': 'one matrix at use for all waves; Vigor included once'},
    'DROPKICK': {'java': 'Dropkick.use / DropkickAction', 'native': 'calculateCardDamage / DropkickAction', 'capture': 'DamageInfo at use; Vulnerable checked at action update, even for a non-null dead target'},
    'SWORD_BOOMERANG': {'java': 'SwordBoomerang.use / AttackDamageRandomEnemyAction', 'native': 'SwordBoomerangAction', 'capture': 'choose target and recalculate each hit with actual card instance; queued Vigor removal follows hits'},
    'FIEND_FIRE': {'java': 'FiendFire.use / FiendFireAction / ExhaustAction', 'native': 'FiendFireAction / ExhaustRandomCardInHand', 'capture': 'damage at use, hit count from hand at action update; exhaust each random card before hits, anyNumber=true even for the last card'},
}


# Java GameActionManager.clearPostCombatActions is a class/type rule, not an
# action-name or general BLOCK rule. This is the source-audited BRIDGE36 batch;
# the full Java action-queue transport is still incomplete (BRIDGE28).
COMBAT_VICTORY_ACTION_CONTRACTS = {
    'PoisonLoseHpAction': {'native': 'PoisonLoseHpAction(target, amount)', 'keep': True, 'reason': 'DAMAGE; capture poison at turn start, HP_LOSS bypasses block, decrement surviving power after death callbacks'},
    'DamageAction': {'native': 'AttackEnemy/DamageEnemy/AttackPlayer/DamagePlayer', 'keep': True, 'reason': 'ActionType.DAMAGE'},
    'DamageAllEnemiesAction': {'native': 'AttackAllEnemy/DamageAllEnemy/AttackAllMonsterRecursive', 'keep': True, 'reason': 'ActionType.DAMAGE; recursive helper represents already-enqueued waves'},
    'DamageRandomEnemyAction': {'native': 'DamageRandomEnemy', 'keep': True, 'reason': 'ActionType.DAMAGE; select target then queue DamageEnemy at top'},
    'VampireDamageAction': {'native': 'VampireAttack', 'keep': True, 'reason': 'ActionType.DAMAGE, distinct from NORMAL DamageAction source guard'},
    'VampireDamageAllEnemiesAction': {'native': 'ReaperAction', 'keep': True, 'reason': 'ActionType.DAMAGE'},
    'FeedAction': {'native': 'FeedAction', 'keep': True, 'reason': 'ActionType.DAMAGE'},
    'GreedAction': {'native': 'HandOfGreedAction', 'keep': True, 'reason': 'ActionType.DAMAGE'},
    'RitualDaggerAction': {'native': 'RitualDaggerAction', 'keep': True, 'reason': 'ActionType.DAMAGE'},
    'SuicideAction': {'native': 'SuicideAction', 'keep': True, 'reason': 'ActionType.DAMAGE; bypass damage and do not itself filter queue'},
    'GainBlockAction': {'native': 'GainBlock/MonsterGainBlock', 'keep': True, 'reason': 'explicit class exception; dying/dead recipient rejects block at execution'},
    'HealAction': {'native': 'HealPlayer', 'keep': True, 'reason': 'explicit class exception'},
    'UseCardAction': {'native': 'OnAfterCardUsed/TimeEaterPlayCardQueueItem', 'keep': True, 'reason': 'explicit class exception'},
    'DropkickAction': {'native': 'DropkickAction', 'keep': False, 'reason': 'BLOCK type is not the GainBlockAction class'},
    'FiendFireAction': {'native': 'FiendFireAction', 'keep': False, 'reason': 'WAIT setup; already-enqueued DamageAction hits are different'},
    'WhirlwindAction': {'native': 'WhirlwindAction', 'keep': False, 'reason': 'SPECIAL setup; already-enqueued AOE waves are different'},
    'AttackDamageRandomEnemyAction': {'native': 'SwordBoomerangAction', 'keep': False, 'reason': 'default action type; not DamageRandomEnemyAction'},
    'EscapeAction': {'native': 'EscapeAction', 'keep': False, 'reason': 'TEXT; no independent victory filtering'},
    'ApplyPowerToRandomEnemyAction': {'native': 'The Specimen queued random-target callback', 'keep': False, 'reason': 'default type; captured Poison then cardRandomRng and top ApplyPower'},
    'MakeTempCardInHandAction/MakeTempCardInDiscardAction': {'native': 'returnStasisCard queued callback', 'keep': False, 'reason': 'CARD_MANIPULATION; no immediate card return from onDeath'},
}

MONSTER_DEATH_CONTRACTS = {
    'BRONZE_AUTOMATON': {'java': 'BronzeAutomaton.die', 'queue': 'top Suicide for each not-dead/not-dying slot after base death callbacks'},
    'THE_COLLECTOR': {'java': 'TheCollector.die', 'queue': 'top Suicide for each not-dead/not-dying slot after base death callbacks'},
    'REPTOMANCER': {'java': 'Reptomancer.die', 'queue': 'top Suicide for each not-dead/not-dying slot after base death callbacks'},
    'GREMLIN_LEADER': {'java': 'GremlinLeader.die', 'queue': 'bottom Escape for each not-dying slot; preserve intervening effects'},
    'AWAKENED_ONE': {'java': 'AwakenedOne.damage/die', 'queue': 'first life prevents victory and queues top ClearCardQueue; final death queues bottom Escape for remaining Cultists; full revival/root flags still BRIDGE37'},
    'target_group': {'java': 'MonsterGroup.getRandomMonster', 'queue': 'sample actual non-halfDead/non-dying/non-escaping slots; none means no target and no RNG, never fallback to last slot'},
    'StasisPower': {'java': 'StasisPower.onDeath / MakeTempCard actions', 'queue': 'choose hand-vs-discard at death; return same-UID stat-equivalent copy later; hand action may overflow at execution'},
    'TheSpecimen': {'java': 'TheSpecimen.onMonsterDeath / ApplyPowerToRandomEnemyAction', 'queue': 'only actual Poison and remaining battle; capture amount, later random target; never open player selection'},
    'GremlinHorn/SporeCloud/CorpseExplosion': {'java': 'onMonsterDeath/onDeath', 'queue': 'guard actual battle-ending condition; do not stop all death processing solely on MINION_LEADER'},
}

MONSTER_LIFECYCLE_FIELDS = {
    'gone': {'java': 'AbstractCreature.isDeadOrEscaped()', 'native': 'Monster::isDeadOrEscaped', 'contract': 'is_dying OR half_dead OR is_escaping; not an absent slot and not inferred from HP'},
    'half_dead': {'java': 'AbstractCreature.halfDead', 'native': 'Monster.halfDead', 'contract': 'preserve even at positive HP during queued revival; not itself battle victory'},
    'is_dying': {'java': 'AbstractCreature.isDying', 'native': 'Monster.isDyingB', 'contract': 'explicit death flag; with escaping drives areMonstersBasicallyDead'},
    'is_dead': {'java': 'AbstractCreature.isDead', 'native': 'Monster.isDeadB', 'contract': 'preserve animation-completed death flag independently; GainBlock/Regenerate guards consume it'},
    'is_escaping': {'java': 'AbstractMonster.isEscaping', 'native': 'Monster.isEscapingB', 'contract': 'HP/block/powers/history retained; do not run death callbacks'},
    'escape_next': {'java': 'AbstractMonster.escapeNext', 'native': 'Monster.escapeNext', 'contract': 'preserve upcoming escape separately from currently escaping'},
    'unawakened': {'java': 'AwakenedOne + room.cannotLose', 'native': 'AwakenedOne !miscInfo', 'contract': 'death protection survives first-life removal of UnawakenedPower; private form1/firstTurn and pending CanLose remain BRIDGE37'},
    'powers': {'java': 'AbstractCreature.powers', 'native': 'Monster status fields/bits', 'contract': 'canonical retains powers at zero HP; OBS20 masks gone powers only in fixed policy projection'},
    'move_history': {'java': 'nextMove + two moveHistory entries', 'native': 'Monster.moveHistory[0..2]', 'contract': 'never erase on gone/zero HP; only actual no-move sentinel maps INVALID'},
    'present': {'java': 'complete real entity list plus explicit slot mapping', 'native': 'INVALID Monster empty slot', 'contract': 'present=false only after proving full real entity coverage; reject corpse loss/identity replacement without complete private state'},
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

