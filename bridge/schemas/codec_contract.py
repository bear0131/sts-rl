"""Complete simulator <-> real-game identity and selection codec contract.

PAD/EMPTY are reserved for structurally absent fixed-width slots.  They are
never an alias target for a live entity.  This module expands canonical
identity entries as well as the irregular aliases so audits can inspect one
complete table instead of relying on scattered fallback transformations.
"""

from __future__ import annotations

from functools import lru_cache

from env.core import vocab as V
from env.core.cards_vocab import CARD_VOCAB, EMPTY
from env.core.ironclad_cards import IRONCLAD_CARDS

from . import combat_state as CS
from .entity_ids import (
    BOSS_ALIASES,
    BOSS_CANONICAL_IDS,
    REAL_ACT_BOSS_KEYS,
    CARD_ALIASES,
    COMBAT_SELECTION_SOURCES,
    DISCOVERY_SELECTION_PARAMETERS,
    DYNAMIC_COMBAT_SELECTION_SOURCES,
    DYNAMIC_EVENT_OPTION_KINDS,
    DYNAMIC_EVENT_OPTION_KIND_ALIASES,
    LIVE_RELIC_STATE_FIELDS,
    LIVE_RELIC_COUNTER_FIELDS,
    LIVE_RELIC_COUNTER_MODULI,
    LIVE_RELIC_COUNTER_STOP_TURNS,
    RELIC_COUNTER_DOMAINS,
    PHASE_RELIC_COUNTER_DOMAINS,
    REAL_RELIC_COUNTER_USAGE,
    DERIVED_RELIC_COUNTER_SOURCES,
    FRESH_RELIC_INITIAL_DATA,
    REAL_ROOM_CLASS_IDS,
    REWARD_ROOM_EXIT_CONTRACTS,
    CHEST_REWARD_SCREEN_CONTRACTS,
    NATIVE_GLOBAL_FIELDS,
    ON_USE_CARD_RELIC_IDS,
    ON_PLAYER_END_TURN_RELIC_IDS,
    ON_EXHAUST_POWER_IDS,
    END_TURN_POWER_PRIORITIES,
    POTION_CAPACITY_RULES,
    COMPACT_PLAYER_SCALAR_FIELDS,
    FRESH_CARD_MISC,
    JSON_PRESENCE_CONTRACTS,
    REAL_ANONYMOUS_ACTION_CONTRACTS,
    THIEF_STATE_FIELDS,
    LAGAVULIN_STATE_FIELDS,
    MONSTER_MOVE_STATE_CONTRACTS,
    REAL_CARD_WIRE_FIELDS,
    CARD_STATE_FIELD_CONTRACTS,
    CARD_SPECIAL_DATA_FIELDS,
    CARD_STATE_TRANSPORT_FIELDS,
    COMBAT_SELECTION_TRANSPORT_FIELDS,
    NATIVE_CARD_QUEUE_FIELDS,
    NATIVE_CARD_INSTANCE_FIELDS,
    NATIVE_CARD_SNAPSHOT_FIELDS,
    NEOW_BONUS_IDS,
    NEOW_DRAWBACK_IDS,
    NEOW_IMPLICIT_DRAWBACK_IDS,
    NEOW_SCREEN_CONTRACTS,
    RELIC_COUNTER_SENTINELS,
    COMBAT_RELIC_COUNTER_SENTINELS,
    EMPTY_REAL_POTION_IDS,
    EVENT_ALIASES,
    MONSTER_ALIASES,
    NATIVE_MONSTER_STATUS_IDS,
    NATIVE_PLAYER_STATUS_IDS,
    HIT_REACTION_CONTRACTS,
    CARD_DAMAGE_TIMING_CONTRACTS,
    COMBAT_VICTORY_ACTION_CONTRACTS,
    MONSTER_DEATH_CONTRACTS,
    MONSTER_LIFECYCLE_FIELDS,
    NATIVE_COMBAT_CARD_ENUM_IDS,
    OVERWORLD_SELECTION_EVENTS,
    OVERWORLD_SELECTION_FLAGS,
    OVERWORLD_SELECTION_RELICS,
    POTION_ALIASES,
    REAL_REWARD_TYPE_IDS,
    REAL_COMMAND_IDS,
    REAL_SCREEN_TYPE_IDS,
    REAL_INTENT_COMPONENTS,
    REAL_CHARACTER_IDS,
    REAL_ROOM_PHASES,
    REAL_ACTION_PHASES,
    REAL_CARD_TYPE_IDS,
    REAL_CARD_RARITY_IDS,
    REAL_CARD_TARGET_IDS,
    REWARD_CHOICE_TYPES,
    RUN_KEY_FIELDS,
    REAL_POWER_ALIASES,
    ROOM_ALIASES,
    SCREEN_PHASES,
    SIM_POWER_ALIASES,
    STRUCTURAL_POWER_FIELDS,
    TRANSPORT_ONLY_REAL_EVENT_IDS,
    UNSUPPORTED_REAL_CARD_IDS,
    UNSUPPORTED_REAL_POWER_IDS,
    UNSUPPORTED_REAL_RELIC_IDS,
    UNSUPPORTED_SELECTION_SOURCES,
    native_power_enum,
)


def _identity(values, *sentinels):
    excluded = set(sentinels)
    return {value: value for value in values if value not in excluded}


REAL_TO_CANONICAL = {
    "card": _identity(CARD_VOCAB, EMPTY) | CARD_ALIASES,
    "relic": _identity(V.RELIC_VOCAB, "PAD"),
    "potion": _identity(V.POTION_VOCAB, "PAD") | POTION_ALIASES,
    "monster": _identity(V.MONSTER_VOCAB, "PAD") | MONSTER_ALIASES,
    "player_power": (
        _identity(V.PLAYER_BUFF_VOCAB)
        | {source: target for source, target in REAL_POWER_ALIASES.items()
           if target in V.PLAYER_BUFF_VOCAB_INDEX}
    ),
    "monster_power": (
        _identity(V.MONSTER_BUFF_VOCAB)
        | {source: target for source, target in REAL_POWER_ALIASES.items()
           if target in V.MONSTER_BUFF_VOCAB_INDEX}
    ),
    "event": _identity(V.EVENT_VOCAB, "PAD") | EVENT_ALIASES,
    "boss": _identity(BOSS_CANONICAL_IDS) | BOSS_ALIASES,
    "room": _identity(V.ROOM_VOCAB, "NONE", "INVALID") | ROOM_ALIASES,
    "screen": dict(SCREEN_PHASES),
    "select_context": _identity(V.SELECT_CONTEXT_VOCAB, "PAD"),
    "character": dict(REAL_CHARACTER_IDS),
}

# Fully expanded simulator tables.  The card table has one entry per native
# CardId enum name; constructing it from every frozen canonical token makes the
# generic spelling rule auditable and collision-checked rather than a runtime
# fallback for arbitrary input.
SIM_TO_CANONICAL = {
    "card": {
        CS.real_to_sim_card_enum(canonical): canonical
        for canonical in CARD_VOCAB
        if canonical != EMPTY
    },
    "monster": {
        native: CS._canonical_real_monster_id(real)
        for native, real in CS._MONSTER_SIM2REAL.items()
    },
    "player_power": (
        _identity(V.PLAYER_BUFF_VOCAB)
        | {source: target for source, target in SIM_POWER_ALIASES.items()
           if target in V.PLAYER_BUFF_VOCAB_INDEX}
    ),
    "monster_power": (
        _identity(V.MONSTER_BUFF_VOCAB)
        | {source: target for source, target in SIM_POWER_ALIASES.items()
           if target in V.MONSTER_BUFF_VOCAB_INDEX}
    ),
    "move": _identity(V.MOVE_VOCAB, "PAD"),
    "relic": _identity(V.RELIC_VOCAB, "PAD"),
    "potion": _identity(V.POTION_VOCAB, "PAD"),
    "event": _identity(V.EVENT_VOCAB, "PAD"),
    "boss": _identity(BOSS_CANONICAL_IDS),
}

REAL_MONSTER_MOVE_TO_CANONICAL = {
    real_monster: dict(moves)
    for real_monster, moves in CS._REAL_MONSTER_MOVES.items()
}

SELECTION_SOURCE_TO_CONTEXT = {
    "combat": dict(COMBAT_SELECTION_SOURCES),
    "combat_dynamic": {
        source: dict(variants)
        for source, variants in DYNAMIC_COMBAT_SELECTION_SOURCES.items()
    },
    "overworld_flag": dict(OVERWORLD_SELECTION_FLAGS),
    "overworld_relic": dict(OVERWORLD_SELECTION_RELICS),
    "overworld_event": dict(OVERWORLD_SELECTION_EVENTS),
    "unsupported": dict(UNSUPPORTED_SELECTION_SOURCES),
}


def complete_word_tables():
    """Return the expanded source tables, including explicit unverified rows.

    A declaration is an inventory disposition, never evidence that gameplay
    effects or all localization variants have passed a real-game comparison.
    """
    from bridge import event_contract
    event_options = {}
    for real_event, rules in event_contract.CONTRACTS.items():
        canonical = EVENT_ALIASES.get(real_event, real_event)
        event_options[canonical] = [
            {"semantic": index, "labels": labels.split('|'), "effect": effect,
             "effect_status": "unverified" if effect.startswith('pending_') else "implemented"}
            for index, labels, effect in rules
        ]
    relic_counters = {}
    for relic in REAL_TO_CANONICAL["relic"]:
        usage = REAL_RELIC_COUNTER_USAGE.get(relic)
        relic_counters[relic] = {
            "java_field": "AbstractRelic.counter",
            "native_persistent_field": DERIVED_RELIC_COUNTER_SOURCES.get(relic, "RelicInstance.data"),
            "fresh_native_initial_data": FRESH_RELIC_INITIAL_DATA.get(relic, 0),
            "java_class_with_counter_usage": usage[0] if usage else "AbstractRelic (inherited)",
            "meaning": usage[1] if usage else "inherited_display_sentinel",
            "persistent_value_map": RELIC_COUNTER_SENTINELS.get(relic, {}),
            "combat_value_map": COMBAT_RELIC_COUNTER_SENTINELS.get(relic, {}),
            "combat_runtime_field": LIVE_RELIC_COUNTER_FIELDS.get(relic),
            "java_numeric_domain": RELIC_COUNTER_DOMAINS.get(relic) or PHASE_RELIC_COUNTER_DOMAINS.get(relic),
            "combat_modulo": LIVE_RELIC_COUNTER_MODULI.get(relic),
            "combat_stop_turn": LIVE_RELIC_COUNTER_STOP_TURNS.get(relic),
            "inherited_display_value_map": {-1: 0} if relic not in REAL_RELIC_COUNTER_USAGE or relic == "CeramicFish" else {},
            "numeric_audit_status": (
                "mapped_runtime" if relic in LIVE_RELIC_COUNTER_FIELDS else
                "mapped_derived" if relic in DERIVED_RELIC_COUNTER_SOURCES else
                "mapped_persistent" if relic in RELIC_COUNTER_SENTINELS or relic in RELIC_COUNTER_DOMAINS else
                "mapped_display_sentinel"
            ),
            "phase_audit_status": (
                "mapped_map_choices_and_charge_receipt" if relic == "WingedGreaves" else
                "chest_reward_order_source_reviewed_not_real_game_verified"
                if relic in {"Matryoshka", "NlothsMask"} else
                "legacy_use_not_supported" if relic == "CeramicFish" else
                "numeric_transport_reviewed_not_full_effect_acceptance"
            ),
        }
    return {
        "real_to_canonical": REAL_TO_CANONICAL,
        "real_act_boss_keys": REAL_ACT_BOSS_KEYS,
        "json_presence_contracts": JSON_PRESENCE_CONTRACTS,
        "anonymous_action_contracts": REAL_ANONYMOUS_ACTION_CONTRACTS,
        "thief_private_state_fields": THIEF_STATE_FIELDS,
        "lagavulin_state_fields": LAGAVULIN_STATE_FIELDS,
        "turn_start_power_effect_contracts": {
            "Magnetism": {
                "java": "MagnetismPower.atStartOfTurn / MakeTempCardInHandAction constructor",
                "native": "Player::applyStartOfTurnPowers / MakeTempCardInHand",
                "effect": "if monsters remain, sample one non-healing colorless card per stack now; capture MasterReality upgrade now; enqueue individual creations in sample order, overflow to discard",
                "audit": "BRIDGE55 source-corrected; full power ordering/pending restore still BRIDGE28; not behavior-verified",
            },
            "Mayhem": {
                "java": "MayhemPower.atStartOfTurn / MayhemPower$1.update",
                "native": "Player::applyStartOfTurnPowers / MayhemPlayTopCard / PlayTopCard",
                "effect": "enqueue one wrapper per stack; each wrapper later samples a living target and appends non-exhausting auto-play behind existing actions",
                "audit": "BRIDGE54 source-corrected; full power ordering/pending restore still BRIDGE28; not behavior-verified",
            },
        },
        "blood_potion_heal_contract": {
            "java": "BloodPotion.use / AbstractPotion.getPotency",
            "native": "BattleContext::drinkPotion / GameContext::drinkPotion",
            "amount": "truncate float(max_hp) * (SacredBark ? 0.40f : 0.20f)",
            "timing": "combat queues HealPlayer; overworld calls playerHeal; normal healing modifiers remain in their respective heal path",
            "audit": "BRIDGE55 combat multiplier/source arithmetic corrected, not behavior-verified",
        },
        "smoke_potion_contract": {
            "java": "SmokeBomb.canUse/use -> AbstractPlayer.updateEscapeAnimation -> AbstractRoom.endBattle -> CombatRewardScreen.openCombat(smoked=true)",
            "native": "BattleContext.canDrinkPotion/drinkPotion/exitBattle -> GameContext.afterEscape",
            "legality": "reject all actual BOSS species including dead/event bosses; Slime Boss split retains its BOSS corpse and still forbids smoke; Act4 BackAttack encounter remains unsupported",
            "outcome": "PLAYER_ESCAPE=3, distinct from victory=1/loss=2; pre-settlement HP target retained, win/loss distribution supervision excludes escape",
            "settlement": "consume potion; clear combat queues without killing monsters; ordinary endBattle relic effects; no displayed drops or card reward RNG; preserve room drop RNG/pools and player gold/deck; Colosseum first fight rolls hidden potion and returns to POST_COMBAT choices instead of map/reward screen",
            "transport": "issued SmokeBomb waits for same seed/act/floor, consumed real slot and settled non-combat frame; no new search during escape animation",
            "compatibility": "native outcome enum appends 3; existing values, OBS20/ACTION8/layout9 and weight tensors unchanged; rebuild native; no new Java field",
            "audit": "BRIDGE71 source implementation; real escape branch not yet verified",
        },
        "combat_exit_healing_contract": {
            "java": "AbstractRoom.endBattle: MeatOnTheBone.onTrigger before AbstractPlayer.onVictory relic order; room still COMBAT",
            "native": "BattleContext.updateRelicsOnExit used by exitBattle and projected_exit_hp; FaceOfCleric no longer duplicated in afterBattle/afterEscape",
            "order": "surviving victory/escape: Meat when 0 < HP <= maxHP/2, then BurningBlood/BlackBlood/Face in relic order",
            "amount": "Meat=12, BurningBlood=6, BlackBlood=12, Face=maxHP+1 and heal1; MagicFlower positive float *1.5 rounded, Bloom blocks healing not maxHP",
            "compatibility": "final mode0 combat-end-hp-v1 target unchanged; actual next-room HP and optional legacy projected exit corrected; skipBattles preview keeps prior convention",
            "audit": "BRIDGE72 source reviewed and native rebuilt; real combination coverage pending",
        },
        "card_use_relic_order_contract": {
            "source": "UseCardAction constructor iterates the actual acquisition-ordered player.relics list; no sorted/bitset callback order",
            "membership_not_order": sorted(ON_USE_CARD_RELIC_IDS),
            "independent ArtOfWar hook": "Java only clears gainEnergyNext on attack, without enqueuing an action; native derives this from the preceding attacksPlayedThisTurn in Player.applyStartOfTurnRelics before resetting that counter, so it is not an ordered queue callback",
            "native": "BattleContext.onUseRelicOrder filters the ordered GC inventory; four card-type paths call shared onUseCardRelics after powers and before enemy powers",
            "root": "required on_use_relic_order stable IDs, unique/exactly all owned supported hooks; raw readback checked, clone preserves; real source is the existing ordered global relic list, no new Java field/default",
            "effects": "Pellets before Ink removes NoDraw/Confusion before Draw1; Pellets vs Kunai/Shuriken can change negative-stat cleanse outcomes; preserve callback synchronous/BOTTOM/card-queue behavior",
            "compatibility": "OBS20/ACTION8/layout9/packed policy projection unchanged; this is transition metadata, not a model channel. Multi-hook NOSL keys include marker/ordered IDs; zero/one-hook old keys unchanged. Pair Python with rebuilt native; no special-combination coverage implied (BRIDGE129)",
        },
        "exhaust_power_order_contract": {
            "source": "CardGroup.moveToExhaustPile iterates actual player.powers; DarkEmbracePower and FeelNoPainPower both have priority5, so stable sorting retains first-application order",
            "membership_not_order": sorted(ON_EXHAUST_POWER_IDS),
            "native": "Player records relative order only when either hook is first added; stacking does not reorder, removal/re-add tracks the new application. Both BOTTOM callbacks run after relic hooks and before card-specific exhaust hooks",
            "root": "canonical player.on_exhaust_power_order filters the existing actual ordered powers; native readback preserves exactly the active hooks once. A strict root with both hooks and no order is rejected; zero/one hook has only one possible order",
            "effect": "FNP->DarkEmbrace versus reverse changes the queued order of Juggernaut random damage and FireBreathing AOE when exhaust draws a status/curse; legal random targets and surviving HP can differ",
            "compatibility": "paired rebuilt native/Python, no Java field or OBS20/ACTION8/layout9 change; Player copy preserves order and nondefault two-hook NOSL futures get a distinct key. No offline test or natural-combination coverage claimed (BRIDGE144)",
        },
        "end_turn_relic_order_contract": {
            "source": "AbstractRoom.applyEndOfTurnRelics iterates actual acquisition-ordered player.relics; NilrysCodex appends CodexAction, StoneCalendar at counter7 appends DamageAllEnemies52. Orichalcum prepends block; keep existing TOP/BOTTOM semantics",
            "membership_not_order": sorted(ON_PLAYER_END_TURN_RELIC_IDS),
            "native": "BattleContext.initRelics filters ordered GC inventory into onEndTurnRelicOrder; callEndOfTurnActions traverses it rather than fixed CloakClasp/FrozenCore/Codex/Orichalcum/Calendar order. Other-class effect implementations are not expanded",
            "root": "canonical/raw/root on_end_turn_relic_order from existing real relic list; strict explicit list, stable supported IDs exactly all owned hooks once, atomic complete/initial-selection replacement and readback, default clone preserves. No fake order from a set or enum",
            "effect": "Calendar acquired before Codex can kill the final enemy on turn7 before the choice; clearPostCombatActions cancels CARD_MANIPULATION, so no Codex selection may be invented first. CodexAction independently rechecks actual areMonstersBasicallyDead at execution, never HP0 alone or a half-dead reviving enemy",
            "compatibility": "paired rebuilt native and Python; legacy valid callback order keeps existing NOSL key, nonlegacy order gets its own marker/list. No new Java fields, OBS20/ACTION8/layout9, weight or budget change. Source/build is not natural Calendar+Codex lethal-combination acceptance (BRIDGE147)",
        },
        "end_turn_power_order_contract": {
            "source": "AbstractRoom.endTurn calls inherited AbstractCreature.applyEndOfTurnTriggers over actual powers; Combust/TheBomb/default supported hooks priority5, Establishment25 and Constricted105. Stable order within one priority",
            "priorities": dict(END_TURN_POWER_PRIORITIES),
            "native": "Player.endTurnPowerOrder tracks first additions/removals; THE_BOMB entries each bind the next independent ordered bomb. Snapshot iteration tolerates immediate Rage removal; expiry removes matching bomb entries only",
            "root": "player.on_end_turn_power_order from existing actual wire power list, with TheBomb<N> converted to ordered The Bomb markers; exact active singleton identities, bomb count and priorities required. Strict roots with multiple hooks cannot omit order",
            "effects": "Buffer may prevent Combust HP_LOSS or the larger Constricted THORNS depending on order, changing actual HP/survival. Damage actions survive clearPostCombatActions, so do not assume a lethal AOE erases later self-damage. Preserve ordered bomb positions and TOP regen versus BOTTOM effects, not other-class hook implementations",
            "compatibility": "paired native/Python raw-state change, no Java fields or OBS20/ACTION8/layout9/weight changes; copy and NOSL distinguish nonlegacy callback sequences. No offline behavior or natural-combination coverage claimed (BRIDGE145)",
        },
        "potion_use_relic_contract": {
            "java": "PotionPopUp: potion.use -> relic.onUsePotion -> destroyPotion; EntropicBrew/ObtainPotionAction/ToyOrnithopter",
            "native": "BattleContext.drinkPotion queues potion effects then HealPlayer(5); Smoke escape terminal follows healing; GameContext.drinkPotion applies overworld heal5",
            "healing": "Toy is not doubled by SacredBark; combat MagicFlower rounds 5*1.5 to 8; Bloom blocks; automatic Fairy and discard do not trigger",
            "entropic": "roll capacity times and fill first empty slots; combat rolls even with Sozu but queued obtain is blocked; overworld Sozu blocks generation/RNG entirely",
            "compatibility": "no new IDs or tensor layout; native rebuild required; overworld invalid slot/context/potion rejects before consumption",
            "audit": "BRIDGE73 source reviewed and built; real combination coverage pending",
        },
        "event_combat_rule_contracts": {
            "Colosseum": {
                "java": "city.Colosseum.buttonEffect/reopen + AbstractRoom.update",
                "native": "GameContext.setupEvent/chooseEventOption/afterEscape; GameAction.getValidEventSelectBits",
                "phases": "native0 Fight -> Slavers -> native1 Cowardice(0)/Victory(1) -> map or Taskmaster+Nob; native2 is finished",
                "drops": "first fight no displayed drops/cards but potion RNG advances; second pre-rolls Rare then Uncommon relic and gold100 with Idol, then postcombat potion before Event card reward",
                "escape": "first fight reopens second choice page; second fight empty reward then map, no card RNG",
                "compatibility": "disableColosseum remains true in default simulator event pool; existing IDs/layout unchanged; bridge already has typed two-fight choices",
            },
            "Dead Adventurer": {
                "java": "exordium.DeadAdventurer.buttonEffect",
                "native": "GameContext.chooseEventOption DEAD_ADVENTURER",
                "drops": "base gold25..35 plus remaining30 merged before Idol bonus; remaining relic selected precombat; potion before Event cards",
            },
            "Masked Bandits": {
                "java": "city.MaskedBandits.buttonEffect",
                "native": "GameContext.chooseEventOption MASKED_BANDITS",
                "drops": "gold25..35 with Idol; RedMask or Circlet if already owned; potion before Event cards",
            },
            "Mushrooms": {
                "java": "exordium.Mushrooms.buttonEffect",
                "native": "GameContext.chooseEventOption HYPNOTIZING_COLORED_MUSHROOMS",
                "drops": "gold20..30 with Idol; OddMushroom or Circlet if already owned; potion before Event cards",
            },
            "MindBloom": {
                "java": "beyond.MindBloom.buttonEffect",
                "native": "GameContext.chooseEventOption MINDBLOOM",
                "drops": "gold25 at A13+ else50 with Idol; rare relic precombat; potion before Event cards; remains EventRoom despite actual Boss species",
            },
            "Mysterious Sphere": {
                "java": "beyond.MysteriousSphere.buttonEffect",
                "native": "GameContext.chooseEventOption/afterBattle MYSTERIOUS_SPHERE",
                "drops": "gold45..55 with Idol and screenless rare relic precombat; potion before Event cards",
            },
            "elite_and_boss_relics": {
                "java": "PreservedInsect.atBattleStart / Sling.atBattleStart / SlaversCollar.beforeEnergyPrep",
                "native": "BattleContext.initRelics / shared hasBossMonster",
                "eliteTrigger": "elite rooms, DeadAdventurer ambush, Colosseum second fight only; not a loot-room reclassification",
                "effects": "Insect caps current HP at floor(maxHP*.75), never heals Neow1HP; Sling adds2 Strength; Collar adds1 energy for eliteTrigger OR any actual Boss type including MindBloom",
            },
            "audit": "BRIDGE74/75 source reviewed and native built; real F39 MindBloom fight/reward chain passed without Idol/Collar; new special branches still uncovered; no new checkpoint/OBS/ACTION/Java fields",
        },
        "native_reserved_monster_slot_contract": {
            "slime_boss_double_split": "retain SlimeBoss corpse plus both medium children of each species, dead or alive; only the two completed large-parent corpses leave the native projection and remain in the real historical ledger. Exact seven-to-five roster and species-relative order map real indices, never HP guesses/PAD; native split preserves boss identity for Smoke Bomb (BRIDGE107)",
            "java": "PhilosopherStone/NeowsLament/PreservedInsect and MonsterRoomElite iterate actual monster objects, not summon reservations",
            "native": "BattleContext.init/initRelics + MonsterGroup.applyEmeraldEliteBuff skip id=INVALID before writing stats; burning buff requires ELITE room",
            "mapping": "keep all physical target slots; actual summons initialize their slot and receive onSpawn effects; never clear a corrupt slot merely to normalize as PAD",
            "validation": "_norm_sim_monster still rejects INVALID carrying HP/block/powers/history/lifecycle; _find_shadow_battle adds failing encounter/nonce context and raises",
            "audit": "BRIDGE76 real F46 same-save attachment and Four Shapes win passed after fix; continued to F50 natural loss with zero bridge errors; Neow/burning-elite combinations still uncovered",
        },
        "card_use_and_empty_pile_contracts": {
            "Pantograph event bosses": "at battle start heal25 once when any actual monster species is BOSS, including MindBloom's first-act boss in EventRoom. Reuse the same complete boss predicate as SlaversCollar and SmokeBomb, not Room::BOSS; existing MagicFlower/MarkOfBloom healing and frozen model geometry remain (BRIDGE127)",
            "Gambler's Brew/empty hand and discard count": "queue GamblingChipAction only when hand is nonempty at potion use, not a blanket change to the shared action. Each selected discard increments existing cards_discarded once after its move and before card-specific manual-discard hooks; zero selection does not increment, ordinary end-turn counts and new-turn reset remain. Raw/packed/model counter and NOSL key consume the real value, overflow rejects before group moves (BRIDGE126)",
            "legal use versus empty effect": "SecretTechnique/SecretWeapon.canUse explicitly require a matching SKILL/ATTACK in draw even for autoplay, while their retrieval actions separately early-return for0 candidates, auto-retrieve1 or expose real GRID for multiple. Do not infer legal use from an action's empty branch. Conversely empty-hand Gambler'sBrew and Elixir remain consumable: the former omits its action at use, the latter's ExhaustAction returns without a selection. Native guards remain at those actual boundaries, not a blanket potion mask change. Actual can_use/requires_target/slot identities and full-state command progress remain explicit; do not fabricate a selection page/CHOOSE for a legal empty effect or default unknown records to PAD",
            "Corruption/onCardDrawOrDiscard": "after each actual draw, discard/exhaust move or generated-hand arrival, refresh only still-in-hand SKILL cards whose turn cost is nonzero using modifyCostForCombat(-9), updating base and turn cost together. Shared formula with initial four-pile application; do not refresh immediately after Snecko Oil, on plain deck/hand transfers or generated discard effects. HAND_SELECT exhaust/gamble remove all chosen physical cards before callbacks, preserving their original costs and canonical subset order (BRIDGE125/146)",
            "generated hand versus retrieval": "ShowCardAndAddToHandEffect first adds the actual instance, calls onCardDrawOrDiscard for the whole hand, then applies Corruption's explicit turn-only discount to that new skill. Shared native moveGeneratedCardToHand covers MakeTempCard(s), Discovery/Toolbox, DualWield and Stasis returned copies; overflow/direct generated discard has no refresh. CardManager assigns exactly one new UID for fresh instances, Stasis retains its original UID. Attack/SkillFromDeckToHand and BetterDiscardPileToHand perform plain transfers without this callback or a blanket Corruption override; Liquid Memories keeps its explicit turn-cost0 and Exhume alone keeps its explicit Corruption discount. No model/wire/layout change; source/build evidence is not natural-combination acceptance (BRIDGE146)",
            "audited turn and copy rules": "Brutality atStartOfTurnPostDraw queues draw then HP loss; DuplicationPower atEndOfRound reduces one unused charge rather than clearing all; DarkShackles checks Artifact synchronously in use() before its queued negative Strength resolves, and queues Shackled restoration only when Artifact is absent then. Preserve these source-confirmed rules, not intuitive text-only replacements. Rage/Panache plus Juggernaut alone has not proved an effectful order bug: Juggernaut appends random damage at BOTTOM and victory filtering preserves GainBlock/DAMAGE actions",
            "Snecko Oil/RandomizeHandCostAction": "sample once for each nonnegative-base-cost hand card; overwrite base and turn costs only if the sampled cost differs from base cost, otherwise preserve existing turn-only discounts. Ordinary Confusion draw already uses the same comparison; no additional RNG, model channels or layout changes (BRIDGE124)",
            "Impatience/ConditionalDrawAction": "queue the condition at BOTTOM in use(); inspect the actual hand for ATTACK when the condition action executes, then TOP queue the existing DrawCards(2/3). Pain/RunicCube TOP draws can change eligibility between use and execution; preserve NoDraw/full-hand/draw triggers and frozen model geometry (BRIDGE122)",
            "Draw Reduction duration": "actual positive duration, never a boolean1; first application reduces hand size once, stacks extend amount without resetting justApplied, ticks remove one duration after next draw amount has been captured, final removal/cleanse restores hand size once. Canonical, native export and packed observation retain amount; single-layer frozen inputs/IDs/layout unchanged (BRIDGE123)",
            "base turn draw restore": "v32 actual AbstractPlayer.gameHandSize -> simulator_card_draw_per_turn -> canonical player.card_draw_per_turn -> native Player.cardDrawPerTurn, plus nullable actual DrawReductionPower.justApplied -> exact native bit/readback. Transition-only metadata, not a new model counter/channel; no shadow/relic/default5 inference (BRIDGE123)",
            "Infernal Blade": "sample the actual ATTACK identity and capture turn-cost0 in use(), then queue BOTTOM MakeTempCardInHand. Do not defer sampling behind Pain/RunicCube/Confusion TOP reactions. Unique generated instances, full-hand discard and existing card pool/amount stay unchanged (BRIDGE121)",
            "fresh generated BloodForBlood": "BloodForBlood.makeCopy applies updateCost(-player.damagedThisCombat) to a fresh printed4-cost card. Native BattleContext.makeCardCopy supplies this same base/turn cost to public Codex/Discovery candidate records, accepted generated instances and pool-based DeadBranch/InfernalBlade/Metamorphosis creation. Capture at the original source's creation time; then retain InfernalBlade turn0 and Metamorphosis combat0 overrides. Existing physical/master-deck/stat-equivalent copies preserve their own cost and must never be discounted twice. MasterfulStab.makeCopy is ordinary fresh0, not damage-scaled. Reuse existing times_damaged state/key; keep full generated-instance validation, frozen model columns and action handles unchanged (BRIDGE152)",
            "Necronomicon eligibility": "onUseCard checks ATTACK, once-per-turn activated, and either costForTurn>=2 with freeToPlayOnce=false OR base cost==-1 with captured energyOnUse>=2. The X branch is independent of free-once; no blanket autoplay/purge gate. Havoc/Mayhem fixed2-cost cards can trigger unless explicitly free-once, Forethought free-once fixed2-cost cards cannot, and autoplay Whirlwind with energyOnUse>=2 can. Preserve source callback order, same-UUID copy without limbo and the existing activation counter; do not confuse queue free-to-play with the actual card flag (BRIDGE153)",
            "autoplay X-cost ownership": "AbstractPlayer.useCard sets freeToPlayOnce for every autoplayed base-X card before its use/callbacks, including DoubleTap/Duplication copies; ordinary fixed-cost autoplay does not set that flag. GameActionManager ignores current-energy down-clamping during autoplay. Whirlwind keeps captured energyOnUse, Transmutation retains its explicit max(captured,current) rule; neither free autoplay action consumes later gained energy. Shared use entry and both X lower clamps match this, UseCardAction clears free-once before even purge-on-use return. No queue fields/model columns/weights/action handles change (BRIDGE153)",
            "generated choice receipt": "after combat CARD_REWARD CHOOSE/SKIP, identical seed/act/floor/source/context and three actual card UUIDs remain pending even if monster intent/rendered damage changes; wait for the actual page to close/change before another model/search decision. Toolbox/Discovery/Codex share this receipt, existing pause-adjusted10s timeout remains (BRIDGE120)",
            "Toolbox/ChooseOneColorless": "v30 maps this exact stock action to the frozen DISCOVERY choice token, with explicit generated_count=1/generated_free_for_turn=false. Native Toolbox selection retains normal cost; DiscoveryAction retains its explicit true/1-or-2 mode. Full metadata/strict pending comparison/clone/search key preserve the distinction, no new model token or PAD substitution (BRIDGE119)",
            "specific exhaust free-once reset": "ExhaustSpecificCardAction moves the exact hand instance and runs onExhaust, then clears freeToPlayOnce on that same exhausted card; applies to ethereal end-turn/SeverSoul/SecondWind, so Exhume cannot revive the prior Forethought free-once flag. Generic ExhaustAction preserves its own semantics and must not be changed (BRIDGE114)",
            "Exhume single candidate": "only original exhaust pile size1 with a non-Exhume card auto-retrieves; a larger original pile opens GRID after excluding Exhumes even when one eligible physical candidate remains. Native retains excluded inert Exhumes internally and filters exact candidates/UIDs, so real/native selection phases agree without forced transport shortcuts (BRIDGE117)",
            "TrueGrit/FiendFire random exhaust": "ExhaustAction with anyNumber=false and hand<=amount moves the original whole hand from top without RNG; TrueGrit uses this path with one remaining card. FiendFire explicitly uses anyNumber=true, retaining per-exhaust random selection even for index0; generic free-once/onExhaust semantics unchanged (BRIDGE118)",
            "FiendFire versus fixed exhaust candidates": "FiendFireAction snapshots only hand count N at action execution, prepends N captured-damage actions then N random ExhaustAction(anyNumber=true) actions. Each exhaust reads the then-current hand, so DarkEmbrace draws can legitimately become later random candidates; do not freeze an original card list or collapse RNG draws. SecondWind/BlockPerNonAttackAction and SeverSoul/ExhaustAllNonAttackAction instead capture original eligible physical instances and prepend specific exhausts, never newly drawn replacements. SecondWind's captured exhausts all precede its prequeued blocks. Existing exact UID lookup and victory action filtering remain, no new model/strategy rules",
            "Sentinel versus Necronomicurse exhaust queues": "CardGroup.moveToExhaustPile invokes relics, player powers, then the card's own trigger before moving the instance and onCardDrawOrDiscard. Sentinel explicitly prepends GainEnergy2/3, not BOTTOM; native immediate gain is before the other queued exhaust reactions in the supported path. The later draw/discard hook refreshes Corruption costs and hand-only damage caches without consuming energy. Necronomicurse instead appends its replacement copy. Do not use an old inline comment as the oracle or unify the two queue positions",
            "Thinking Ahead empty-hand autoplay": "ThinkingAhead.use queues Draw2, then queues PutOnDeck only if hand is nonempty at use time (played hand card is still present until AbstractPlayer.useCard removes it); empty-hand Havoc/Mayhem autoplay must not create a later WARCRY selection after drawing. Warcry itself always queues PutOnDeck (BRIDGE113)",
            "Sever Soul": "ExhaustAllNonAttackAction walks the original hand forward and queues each captured physical instance at TOP; reverse-order exhaust and its TOP reactions precede the already-pending attack/UseCardAction. Do not append exhaust after damage or exhaust newly generated/drawn cards; preserve normal victory queue filtering (BRIDGE106)",
            "Dead Branch/Dark Embrace after lethal": "onExhaust checks areMonstersBasicallyDead before sampling a generated card/queuing draw; UseCardAction still survives victory and resolves the actual exhaust, Charon's Ashes, Feel No Pain and card-specific hooks. Half-dead reviving monsters do not satisfy the terminal predicate; do not add this gate to general DrawCardAction (BRIDGE94)",
            "failed autoplay and purge copies": "GameActionManager checks canUse even for purgeOnUse; autoplay bypasses energy, not turnHasEnded/VelvetChoker/Normality/Entangled/Clash. Failed autoplay queues BOTTOM ResolveUnplayedCard with triggerOnUse=false; preserve exhaustOnUse/doesExhaust, StrangeSpoon and purge disappearance, no card-use count or Burn/curse end-turn effects (BRIDGE93)",
            "admitted queue item with invalidated target": "AbstractCard.cardPlayable rejects a targeted isDying monster and any basically-dead group, but not an explicit null, halfDead or escaping target. GameActionManager first admits/calls onPlay/increments cardsPlayedThisTurn, then skips ENEMY use for null/isDeadOrEscaped; this is not failed canUse and must not enqueue ResolveUnplayedCard, discard/exhaust or onUse callbacks. Darkling half-death can leave DoubleTap/Duplication/Necro copies queued; AwakenedOne's separate ClearCardQueue is preserved. Choker/Normality/Pocketwatch retain the admitted count; onUse-only counters/effects do not advance. Policy/AnyTarget legality still requires an actual targetable slot; -1 is a known null target, never a fallback for out-of-range or INVALID slots. A removed non-purge PlayTop physical card disappears without exhaust hooks; a purge copy never removes the original physical card or decrements its pile cache (BRIDGE158)",
            "admission versus actual-use type counters": "v41 observes the two-argument UseCardAction constructor after actual onUse callbacks, counting ATTACK/SKILL only when dontTriggerOnUseCard is false. The MTS constructor hook declares UseCardAction __instance then the original card/target arguments; omitting the instance fails startup binding. One-argument delegation is not patched twice; successful purge copies count, failed autoplay/cancelled no-trigger actions do not. Actual preBattlePrep binds player/action-manager and clears counts; an observed turn change resets only this observer, unknown ownership/turn regression and uses exceeding admitted types fail explicitly. cards_played remains the GAM admitted-list size; attacks_played/skills_played preserve native onUse meanings for Kunai/Shuriken/Fan/LetterOpener/ArtOfWar, root restore/readback, NOSL key and frozen24 columns. Do not derive use counts from the admitted list or modulo counters, move native type increments to admission, or alter gameplay/RNG (BRIDGE159)",
            "Havoc/Mayhem shuffle": "PlayTopCard with empty draw and nonempty discard triggers onShuffle when constructing EmptyDeckShuffle, just like ordinary draw/DeepBreath; Abacus queues6block and Sundial advances modulo3/queues2energy; TOP shuffle+retry remain ahead of BOTTOM gains, both-empty does not trigger (BRIDGE91)",
            "Corruption/Mummified Hand": "first Corruption use constructs ApplyPower and immediately applies modifyCostForCombat(-9) to skills in hand/draw/discard/exhaust, before Mummified Hand selects positive base+turn cost non-freeOnce targets; power itself remains BOTTOM, existing Corruption skips reapplication, X costs stay negative (BRIDGE90)",
            "Unceasing Top": "empty-hand draw must settle queued shuffle/DrawCard/Evolve/FireBreathing/Void effects before PLAYER_NORMAL; NoDraw or empty draw+discard enqueue nothing and return without spinning (BRIDGE89)",
            "Void": "triggerWhenDrawn appends BOTTOM LoseEnergyAction(1), clamped to0 at execution; pre-existing Warcry/ThinkingAhead selection sees pre-loss energy; do not deduct inside draw (BRIDGE88)",
            "Chrysalis/Metamorphosis": "use() samples 3/5 skill/attack identities, reduces only positive base/turn costs to0 (Whirlwind stays X), then queues individual BOTTOM MakeTempCardInDrawPile; preserve actual CardGroup random spot 0..size-1 and no RNG for empty pile (BRIDGE87)",
            "Limit Break": "LimitBreakAction captures current Strength at execution and queues TOP ApplyPower; positive adds normally, negative uses DEBUFF/Artifact gate, zero creates no new status (BRIDGE85)",
            "empty_piles": "no automatic loss when hand/draw/discard are empty; potions, passive effects and generated cards remain available until actual terminal or existing budget",
            "Anger": "Java AngerPower -> native ENRAGE; every occupied monster slot, skill use, queued TOP Strength, dead/escaped owner rejected when executed",
            "Curiosity": "Java CuriosityPower -> native CURIOSITY; every occupied monster slot, power use, queued BOTTOM Strength; dead/halfDead/escaping owner rejected after any earlier Panache form transition",
            "Sharp Hide": "Java SharpHidePower -> native SHARP_HIDE; attack use queues damage for each actual owner, no slot0 assumption",
            "consumers": "BattleContext outcome -> native search terminal/value and GameContext; existing status IDs, observation/action layout and checkpoint stay unchanged",
            "audit": "BRIDGE77 deployed in natural seed200206 F42-to-F51 real victory with zero bridge errors; empty-pile/Nob/power-Panache lethal combinations still not specifically observed",
        },
        "player_hp_loss_contracts": {
            "java": "AbstractPlayer.damage / LoseHPAction / BufferPower / IntangiblePlayerPower / TungstenRod",
            "native": "Player.loseHp -> hpWasLost -> wouldDie; BattleContext.triggerOnOtherCardPlayed and Player.applyStartOfTurnPostDrawPowers",
            "mitigation": "HP_LOSS bypasses block and Torii, caps positive damage at1 with Intangible, consumes Buffer before Tungsten, and only positive final loss triggers wasHPLost/cards.tookDamage",
            "damage_counter_and_card_costs": "AbstractPlayer.damage increments damagedThisCombat only for positive final loss during COMBAT, including HP_LOSS; updateCardsOnDamage visits existing hand/discard/draw instances, not exhaust or limbo. Native hpWasLost/onTookDamage matches these independent per-card updates, while fresh BloodForBlood makeCopy separately inherits past hits (BRIDGE152). updateCost preserves base-minus-turn difference and clamps at0; upgradeBaseCost preserves a positive turn-cost offset but keeps an already-zero turn cost at0. Do not count fully prevented damage, double-discount existing copies, reset live counters on a selection, or replace actual cost with printed traits. Existing times_damaged strict restore, model column and search key are retained; source review is not natural coverage of every combination",
            "self_damage_sources": "Offering, Hemokinesis, Bloodletting, JAX, BlueCandle, Regret, Combust, Pain and Brutality use player source; Pain queues TOP, Brutality queues draw then loss at BOTTOM; actual self HP loss enables Rupture",
            "revival": "MarkOfTheBloom prevents revival; first held Fairy before available LizardTail; Tail heals max(1,maxHP/2) through shared heal modifiers, consumes availability once",
            "last_attack_unblocked_damage": "Java DamagePatch v21 retains NORMAL enemy lastDamageTaken=min(final positive damage, HP before loss/revival), not Constricted THORNS; native Player.attacked uses same cap, VampireAttack consumes it rather than healing overkill",
            "consumers": "existing Buffer/Rupture/Strength/status counters and LizardTail availability flow through native snapshot/search/key/bridge; no new observation/action layout or weight",
            "audit": "BRIDGE78 source reviewed; combinations still need natural real coverage",
        },
        "player_bloodied_latch_contract": {
            "java": "v40 AbstractCreature.isBloodied and actual RedSkull.isActive are distinct stored booleans; absent owned Red Skull explicitly exports inactive, reflection failures are not replaced with a default",
            "wire": "simulator_combat_counters.bloodied/red_skull_active -> canonical combat_counters -> strict native Player.bloodied/redSkullActive; both required boolean fields, active Skull requires actual ownership",
            "transitions": "fresh preBattlePrep initializes bloodied from HP; positive HP loss crosses into bloodied before revival; heal (including zero after Bloom) clears an existing latch only above half. Max-HP growth precedes heal and must not recompute the previous latch from its changed threshold",
            "impact": "healthy inactive Skull at 51/100 followed by Feed+3 to 54/103 must not subtract3 Strength or consume Artifact; Bloom can leave HP-derived ratio different from the actual latch until later damage/heal",
            "battle_start_healing": "BloodVial and boss Pantograph enqueue TOP HealAction, before RedSkull's BOTTOM start action reads current isBloodied. Native initialization must use its post-heal latch, not GC entry HP: 50/100 -> BloodVial52/100 must not gain3 Strength. Bloom-blocked healing keeps the latch and valid Skull gain; no relic-order/queued-heal redesign is required (BRIDGE150)",
            "battle_start_pending": "RedSkull's conditional gain remains a BOTTOM action in actual atBattleStart relic order, not an eager init mutation. Toolbox's PreDraw selection precedes it; initialize_selection_root restores actual Strength/isActive and preserves this queued continuation. On execution, grant3 only when inactive and bloodied, then mark active and refresh hand powers. Do not patch missing Strength at the root or double-grant an already active Skull (BRIDGE163)",
            "consumers": "raw/canonical diff, complete and initial-selection strict roots, atomic restore/readback, native copies and non-derived NOSL future keys retain both flags; partial editor keeps existing values, never guesses unknown actual flags",
            "compatibility": "paired Java40/Python/native required for real roots; transition metadata only, frozen COMBAT_COUNTER_KEYS and compact scalar columns/OBS20/ACTION8/layout9/finalTRAIN47 unchanged. Ordinary HP-derived latch search keys unchanged",
            "audit": "BRIDGE148 source review and necessary native/Java builds, no offline behavior/model tests; special Feed/Bloom/Skull combination still requires natural real coverage",
        },
        "queued_monster_power_lifecycle_contract": {
            "java": "ApplyPowerAction.update rejects dying/halfDead/escaping targets before power hooks, Artifact consumption or application",
            "native": "Actions.BuffEnemy and BattleContext.debuffEnemy recheck isTargetable at execution, not only at card use",
            "impact": "Bash/Uppercut lethal damage must not reapply debuffs after Darkling/AwakenedOne cleared them; canceled applications trigger neither Sadistic nor ChampionBelt",
            "revival": "Darkling Heal -> clear halfDead -> Regrow/PhilosopherStone buffs remains valid; low-level buff/addDebuff used by initialization and restore is unchanged",
            "compatibility": "same physical slots, power IDs, snapshot/search fields and OBS/ACTION layout; no PAD substitution or weight change",
            "audit": "BRIDGE82 source reviewed; lethal/halfDead and queued-buff combinations not yet naturally covered with this build",
        },
        "combat_start_relic_power_contracts": {
            "java": "AbstractPlayer.applyStartOfCombatLogic iterates actual relic order; AbstractRoom drains atPreBattle actions before calling it",
            "MutagenicStrength": "TOP Strength3 then TOP Flex3; Flex resolves first unless another relic prepends an action; native LOSE_STRENGTH is blocked by Artifact normally",
            "ClockworkSouvenir": "TOP Artifact1, not BOTTOM; if acquired after Mutagenic it blocks Flex, if acquired before it resolves after Flex",
            "GremlinMask": "BOTTOM player Weak1 with isSourceMonster=false; Clockwork may block it unless Artifact was consumed by an earlier debuff",
            "Red Mask": "BOTTOM enemy Weak1 with isSourceMonster=false; it expires after the first round rather than using monster-source justApplied",
            "native": "BattleContext.initRelics preserves this hook order via existing action queue; atBattleStart list capacity10 including conditional BOTTOM RedSkull; preBattle Snecko confusion unchanged",
            "compatibility": "same relic/power IDs and existing counters/justApplied bits in snapshot/search; no OBS/ACTION/checkpoint change",
            "audit": "BRIDGE84 source correction; real relic-order combinations not yet verified",
        },
        "sim_to_canonical": SIM_TO_CANONICAL,
        "monster_moves_by_real_id": REAL_MONSTER_MOVE_TO_CANONICAL,
        "monster_move_canonical_to_real": dict(CS._MONSTER_CANON2REAL),
        "selection_sources": SELECTION_SOURCE_TO_CONTEXT,
        "generated_selection_parameters": DISCOVERY_SELECTION_PARAMETERS,
        "multi_selection_cap_alignment": {
            "source": "Elixir.use -> ExhaustAction(false,true,true), Purity.use -> bounded ExhaustAction; native ExhaustMany (BRIDGE109)",
            "comparison": "after exact candidate physical identity/state validation, MULTI compares min(pick_max,candidate_count); SINGLE/GRID retains exact requested pick_max",
            "guards": "context, zone, pick_min, optionality, selected instances, candidate order/handles and cards_in_hand remain exact; differing effective cap still rejects",
            "compatibility": "Java 99 unbounded sentinel normalizes to actual hand while native Elixir retains ten; compare equivalent legal subsets without rewriting native observations, checkpoint layouts or supplying PAD candidates",
        },
        "selection_instance_transport_contract": {
            "sources": "ForethoughtAction, ExhaustAction, GamblingChipAction, Attack/SkillFromDeckToHandAction, BetterDiscardPileToHandAction and ExhumeAction; native corresponding Actions/BattleContext choices, RealCombatSearch._sync_selection_candidates and RealGameAdapter command_for_joint_action/_continue_joint_multi",
            "identity": "pending choices bind only known actual UUIDs to persistent native UIDs and complete card state. Single-choice candidate permutations map physical handles explicitly; MULTI requires identical source order/handles and rejects a permutation. Never bind an unknown temporary UUID by its displayed index or card name",
            "subset_order": "one frozen native subset is transported as Forethought ascending source slots, exhaust/gamble descending slots. Each real CHOOSE must remove exactly the requested hand instance and append it to selected; unchanged frames only refresh, never repeat CHOOSE or request another model decision before CONFIRM",
            "cost_boundaries": "plain deck/discard retrieval does not run generated-hand callbacks; full-hand Liquid Memories leaves selected source slots and costs intact. Its explicit zero-turn cost preserves base/X costs. Exhume retains its explicit Corruption discount; generated/returned-copy hand arrivals use the separate BRIDGE146 rule",
            "copy_boundaries": "makeStatEquivalentCopy preserves upgrades, base/turn costs, growth and freeToPlayOnce but resets retain/exhaust to the fresh upgraded card's traits. makeSameInstanceOf alone keeps the original UUID; fresh-copy UID assignment occurs once at creation, Stasis same-UUID return does not allocate a new one",
            "audit": "source/caller re-review after BRIDGE146; deployment is evidenced by the full4992861/fca13933 frozen214 runtime, not by running new offline cases. Prior real HAND_SELECT/GRID coverage does not prove every optional/overflow/copy combination",
        },
        "card_cost_reset_boundaries": {
            "Soul draw/discard arrival": "Soul.update calls clearPowers -> resetAttributes for DRAW_PILE/DISCARD_PILE, restoring costForTurn to the actual current base cost. Shared CardManager fresh/move/insert endpoints and empty-draw shuffle shortcut apply this reset to the moved copy, not just at turn end. InfernalBlade/LiquidMemories/temporary Enlightenment discounts cannot leak into same-turn recycling. Keep permanent cost modifications, freeToPlayOnce, retain, cached base damage, UID, exhaust and RNG; do not reset EXHAUST, plain hand retrieval or a complete real snapshot merely because it has a turn/base difference (BRIDGE157)",
            "turn_end": "AbstractRoom.endTurn resets attributes in draw/discard/hand, not exhaust; native resets those same piles' turn costs to current base. resetAttributes does not clear freeToPlayOnce, so do not erase a legitimate Forethought discount merely because the turn ended",
            "confusion_vs_oil": "ConfusionPower.onCardDraw clears freeToPlayOnce after eligible nonnegative-base-cost sampling. RandomizeHandCostAction used by Snecko Oil does not clear it. Both preserve an existing turn-only discount when the sampled cost equals current base cost",
            "single_warcry": "PutOnDeckAction's mandatory one-card path still calls getRandomCard(cardRandomRng); it is not a zero-RNG shortcut merely because the only outcome is known. Native's discarded bounded draw consumes the same one RNG transition/counter; no behavioral change inferred from its unused bound",
            "audit": "original turn-end/Confusion/Oil/PutOnDeck rules remain source-reviewed; BRIDGE157 separately confirms and repairs Soul destination cost reset through all shared native incoming paths. No offline behavior test run; source/build is not natural same-turn recycle coverage",
        },
        "master_deck_event_mutation_contract": {
            "sources": "given Duplicator, LivingWall, Designer and DrugDealer; native Deck.removeImpl/removeSelected/removeAllMatching/upgrade/obtain and GameContext.selectScreenTransform; EventEffectAck single-obtain/Designer/Augmenter receipts",
            "deletion": "multi-remove sorts physical deck indices descending for safe compaction, not selected-card identities for transformation. Removal shifts surviving bottle indices, clears a removed bottle target and applies Parasite max-HP loss; only actual onMasterDeckChange paths refresh DuVu cache",
            "transform": "preserve selected instance order for transform/obtain processing separately from descending deletion. Java queues independent delayed ShowCardAndObtainEffect; a closed GRID is not sufficient acknowledgement. Preserve exact surviving UUID/state map, new physical instances, actual source/context and explicit obtain-pending flag",
            "eligibility": "ordinary LivingWall/Designer remove-transform uses actual nonbottled purgeable candidates; DrugDealer's two-card grid deliberately uses purgeable cards including bottled ones. Duplicator exposes the full master deck for a nonremoving copy; never reuse a single shared candidate filter across these sources",
            "copy_upgrade": "Duplicator makes a stat-equivalent new instance preserving ID/misc/upgrade level, but clears all three bottle flags on the copy; original stays untouched. Eggs apply only their actual acquisition upgrade rule. Ordinary upgrades do not refresh DuVu, SearingBlow remains upgradeable; unchanged/repeated ordinary upgrades must not decrement eligible count again",
            "excluded_false_positives": "Deck.cardTypeCounts excludes nonremovable curses for purge-oriented hasCurse, while getCurseCount scans all actual curses for DuVu; do not substitute one for the other. Uncalled transformRandomCards stub is not a runtime action. FaceTrader Touch genuinely credits gold before NORMAL damage, unlike WorldOfGoop; do not globally reorder all gold/damage events",
            "evidence": "source and existing consumer review only, no new execution defect established or offline behavior/model test run; not proof that every private queue or rare combination has real coverage",
        },
        "event_options": event_options,
        "structured_event_contracts": sorted(event_contract.STRUCTURED_CONTRACTS),
        "event_progress_contracts": {
            event: {"source": "com.megacrit.cardcrawl.events." + source, "phases": sorted(phases)}
            for event, (source, phases) in event_contract.EVENT_PROGRESS_CONTRACTS.items()
        },
        "staged_event_effect_receipts": event_contract.STAGED_EFFECTS,
        "event_damage_terminal_receipts": {
            "effects_by_event": event_contract.EVENT_DAMAGE_TERMINAL_EFFECTS,
            "receipt": "ready GAME_OVER with matching explicit seed/act/floor, prior positive HP, current HP0 and victory=false; releases before delayed obtain/reward UI waits; revival and zero deltas still require normal receipts (BRIDGE92)",
        },
        "single_card_obtain_event_receipts": {
            event: {"effect": effect, "source": "com.megacrit.cardcrawl.events." + source,
                    "context": context,
                    "completion": "requested master UUID; transform removes only that card, duplicate preserves it; final dialogue and no pending obtain; one new UUID or curse blocked by exactly one Omamori charge (BRIDGE86)"}
            for event, (effect, source, context) in event_contract.SINGLE_OBTAIN_EVENTS.items()
        },
        "grid_confirmation_contract": {
            "source": "GridCardSelectScreen.update: single upgrade/transform/purge clears selectedCards until CONFIRM",
            "confirmation_card_uuid": "v22 exports actual hoveredCard UUID only while confirmScreenUp for these flags; explicit null otherwise; never upgradePreviewCard",
            "consumer": "shared real_grid_confirmation_card verifies unique original candidate; event receipt matches requested UUID; adapter confirms once and waits for close",
            "compatibility": "selected_cards remains the actual committed list, not synthetic; multi-select unchanged; no new policy action/observation layout",
            "audit": "BRIDGE79 natural F12 Bonfire exposed old dead wait; same source applies to Blacksmith/Wing/Wheel and other single previews",
            "full-group obtain": "v34 actual is_just_for_confirming -> OBTAIN_CONFIRM; openConfirmationGrid receives entire target group, selected empty/no hovered UUID. openConfirmationGrid/callOnOpen do NOT reset private numCards: retain its actual nonnegative integer, possibly a stale prior pick count such as1, without treating it as a selection requirement or replacing it with0. Reset any_number/upgrade/transform/purge flags remain strictly false, explicit context/confirmation/UUIDs and command inventory remain checked; preserve all real instances, confirm once and await same group closing/obtain queues (BRIDGE131/155)",
            "full-group callers": "CallingBell, PandorasBox, NeowEvent; given-source inventory also contains Endless GrotesqueTrophy outside the current evaluation scope; no synthetic selected_cards or model action",
            "not a policy selection": "validated OBTAIN_CONFIRM has select_descriptor=None, no semantic actions/commands; multi-selection signature must not reinterpret it as a pick. Policy observation is rejected until transport settles, never fabricated",
        },
        "campaign_terminal_contract": {
            "source": "in_game=true/ready_for_command=true/GAME_OVER; actual integer seed/score/floor/act and boolean victory required",
            "transport": "unready terminal waits; PROCEED must be advertised; only ready explicit in_game=false menu advances the campaign, not error/missing frames or interrupted runs",
            "record": "one actual result per requested seed, exact real seed match, no missing-field defaults or metadata overwrites; configured numeric seed aliases cannot repeat (BRIDGE132)",
        },
        "campaign_search_accounting": {
            "source": "each NativeStrategy Decision.diagnostics.search_simulations, including forced END with explicit0; missing/negative/noninteger is an error",
            "record": "search_simulations sums reported calls, search_simulation_budget is decisions*configured limit, label strategy-diagnostics-v1; prefix/retry work survives transport rollback, resets on a new run (BRIDGE133)",
            "history": "older unlabeled search_simulations was a nominal budget product, not measured work; no result rewriting or inference from omitted internal-prefix logs; policy/search settings unchanged",
        },
        "initial_combat_selection_root": {
            "entry": "only fresh combat attachment, actual turn1/cards_played0/cards_discarded0/no selected instances; exact GamblingChipAction+owned GamblingChip GAMBLE or ChooseOneColorless+owned Toolbox DISCOVERY count1/free=false; native task must already agree (BRIDGE136)",
            "restore": "initialize_selection_root atomically restores strict full actual public player/monster/card-instance/relic-order state while retaining pending CardSelectInfo, queued continuation and RNG; GAMBLE pick cap follows actual hand; no nonce search for a matching hand or positional UUID binding",
            "consumers": "normal and later pending selections retain existing strict UID/instance/order/mask checks; new native method capability required before START, no checkpoint OBS20/ACTION8/layout9 or rollout/search budget change",
        },
        "event_progress_transport_fields": {
            "per-card obtain gold/healing": "CeramicFish.onObtainCard calls gainGold9 separately for every actually obtained card; BloodyIdol.onGainGold heals5 per call, Ectoplasm suppresses both gain and heal, Bloom blocks heal. Native batch Deck.obtain keeps each credit rather than one9*N callback (Vampires5Bites, Ghosts3/5Apparitions, WindingHalls2Madness); blocked Omamori curses do not count. Augmenter JAX and two-card transform receipts include capped5 per actual acquired card, not unchanged HP or one batch heal. EventRoom is not COMBAT, so no MagicFlower multiplier; no forced selection/relaxed resource receipt/model change (BRIDGE151)",
            "campfire Toke": "Peace Pipe requires at least one actual purgeable nonbottled master card in both native legality paths. Toke retains REMOVE selection until the selected card is removed; only then does the existing continuation return to map. Existing for_purge/REMOVE physical-instance bridge contract and frozen campfire actions are unchanged (BRIDGE142)",
            "Nloth": "v39 exact shrines.Nloth screenNum0/1; public choice1/choice2 actual owned relic records, fixed donor0/1/leave2 without localized-name lookup. Complete ordered native relic state uses actual counters; same Gift already owned grants/increments Circlet and keeps donor, as vanilla does (BRIDGE140)",
            "Nloth receipt": "same actual room/run and result screen; exact relic identities/counters with only chosen donor removed (unless duplicate Gift), full surviving master instances with only chosen bottle flag cleared or first Necronomicurse removed; no unrelated resources, no second command before gift settles",
            "relic unequip": "GameContext.loseRelic shared by actual losses; bottle flags/transformable count cleared only for an existing target, Necronomicon removes first curse directly without onMasterDeckChange. Energy/draw are already derived from ownership at battle init; no replaying onEquip while restoring snapshots",
            "Du-Vu Doll cache": "RelicInstance.data is actual onEquip/onMasterDeckChange cache, not an unconditional derived count. Ordinary obtain/remove refresh it, direct Necronomicon removal and upgrades preserve it; raw restoration/export and battle strength keep this actual counter even when deck curse count differs. Falling removal/WeMeetAgain card donation refresh receipts, Designer pure upgrades preserve. No new model field/layout (BRIDGE140)",
            "Falling": "v38 exact beyond.Falling INTRO/CHOICE/RESULT; only public CHOICE exports actual private skillCard/powerCard/attackCard via full master serializer, preserving nonbottled UUID/misc and raw disabled slots. Fixed semantic skill0/power1/attack2 maps compressed CHOOSE; no eligible types is explicit leave3 -> CHOOSE0, no localized-name inference or INTRO target leak (BRIDGE139)",
            "Falling receipt": "matching actual seed/act/floor and RESULT; complete surviving master UUID/state map equals original minus exactly the requested instance, including same-ID/same-upgrade copies with different misc. Empty-candidate leave preserves deck. No unrelated HP/maxHP/gold/relic/potion changes; model tensors and frozen event semantics unchanged",
            "NoteForYourself": "v37 exact shrines.NoteForYourself INTRO/CHOOSE/COMPLETE; nullable actual obtain_card/stored_card plus card_select_pending, INTRO offer stays null. Same patched full card serializer, no default IronWave; fixed take0/leave1 and actual REMOVE GRID after acquisition includes the new card",
            "NoteForYourself receipt": "same public offer UUID is acquired before storage; exact selected saveCard UUID/state then disappears from the complete post-acquisition master deck, even when storing the newly obtained card. Require actual completion/no pending callbacks, preserve Omamori charges (direct insertion), egg/Fish/Darkstone/DuVu/Parasite effects; no fabricated gain/remove based on net deck count (BRIDGE135/138)",
            "Designer": "v36 exact shrines.Designer INTRO/MAIN/DONE and current pending_selection; public MAIN service kinds/costs/counts/candidate UUIDs map frozen semantics0..5 to actual compressed CHOOSE, never expose INTRO rolls or future random upgrade/transform targets (BRIDGE135)",
            "Designer receipt": "preserve complete actual eligible master UUIDs through source-specific upgrade/purge/two-transform GRID flags, previews and ordered picks; wait pending_selection NONE and all obtains, then exact selected losses/upgrade count/replacement conservation/cost before final transport; HP_LOSS permits actual Fairy/Tail terminal or revival, no net-loss assumption",
            "WeMeetAgain": "v35 exact shrines.WeMeetAgain INTRO/COMPLETE, nullable given_relic only after COMPLETE; actual initial four options carry kind and exposed card UUID/potion physical slot/gold amount; compressed choice_index maps to frozen semantic0/1/2/3, never positional fallback (BRIDGE134)",
            "WeMeetAgain exporter readiness": "ChoiceType.EVENT is the actual internal enum, not a string; enrich only while GameStateListener is waiting and CHOOSE available; partial loading has no policy decision, actionable target inventory still strict",
            "WeMeetAgain receipt": "selected physical donor/cost only, no unrelated deck losses/gains or relic changes; wait complete page, gift owned, exact surviving resources and known screenless onEquip effects (War Paint, fruit maxHP/heal with Bloom, Old Coin/Ecto/BloodyIdol, Potion Belt, stacked Circlet); excludes bottles/Whetstone; no hidden INTRO prize or new model token",
            "WeMeetAgain native offers": "shuffle physical nonbasic/noncurse master indices and return the selected index unchanged; never fold equal Card values to the first copy because bottle membership is an independent physical property. Potion offers preserve physical slots and use the actual counted miscRng.randomLong, including a single held potion; empty offers consume no shuffle seed (BRIDGE141)",
            "counted shuffle seeds": "stock WeMeetAgain potion/TinyHouse upgrade/five fresh relic pools use randomLong, not raw nextLong. Native preserves the same underlying draws and shuffle order but increments the real call counter; saved-counter restore must not reconstruct a state five draws behind fresh pool initialization (BRIDGE141)",
            "simulator_event_progress": "actual screen/screenNum and source; null outside the explicitly audited event classes",
            "simulator_match_state": "v31 actual GremlinMatchGame phase/attempts/current face-up and remaining physical slots; stock getOrderedCards CHOOSE index mapping, explicit null for other events; no hidden identities or remembered faces",
            "Match and Keep fixed transport": "human-authorized first_two_match_slots only, bypasses policy; acknowledge first physical flip before second, then exact attempt decrement and either same remaining slots or removal of the two sent slots; wait obtain effects/final COMPLETE before actual exit; no model/layout/native change (BRIDGE12)",
            "simulator_obtain_pending": "actual unfinished ShowCardAndObtainEffect in all four game effect queues; never a guessed delay",
            "rewards_taken": "DeadAdventurer.numRewards distinguishes a successful NOTHING result from a stale frame",
            "startSpin/buttonPressed/purgeResult": "actual GremlinWheelGame execution flags; hidden random result is not exported",
            "Bonfire": "BONFIRE_SPIRITS precedes generic for_purge; native opens selection directly, Java intro/offer are transport",
            "automatic_revival_potions": "Skull/Scrap/Goop/Woman damage receipts preserve ordered capacity: unchanged or first FairyPotion -> same empty slot, latter forbidden by Bloom; no net-HP-loss requirement (BRIDGE83)",
            "ScrapOoze failure": "buttonEffect applies damage synchronously before displayed damage+1/chance+10; actual offer change acknowledges failure despite Fairy/Tail net healing, with shared automatic-revival inventory check (BRIDGE80/83)",
            "World of Goop": "Gather Gold applies synchronous NORMAL damage11 then gold75 (Ectoplasm/BloodyIdol respected); final Leave and actual gold delta acknowledge completion, including automatic Fairy consumption; Leave It has no HP/potion effect",
            "Woman in Blue": "buy1/2/3 costs20/30/40 clamped to wallet then opens exact potion rewards; A15 leave applies HP_LOSS ceil(float(maxHP)*.05f), A0 no damage; actual Take N Damage/受到N点伤害 text and final Leave/real death, not net HP loss, acknowledge damaging leave",
            "KnowingSkull": "four displayed HP prices >=6, leave=6; only chosen reward price increments by1 after HP_LOSS and reward, leave requires final dialogue; actual delayed card obtain awaited; real deck UUIDs, uncommon card count/Fish9 gold/Ectoplasm90-or0 gold and ordered potion slots checked (BRIDGE81)",
            "KnowingSkull potions": "damage may consume first Fairy before reward; then first empty physical slot receives reward, Sozu blocks before random generation, full inventory without Sozu still rolls; unchanged full/Sozu or same-potion refill is acknowledged by exact price progression, never a fabricated slot",
        },
        "dynamic_event_kinds": DYNAMIC_EVENT_OPTION_KINDS,
        "dynamic_event_kind_aliases": DYNAMIC_EVENT_OPTION_KIND_ALIASES,
        "commands": sorted(REAL_COMMAND_IDS),
        "reward_types": dict(REWARD_CHOICE_TYPES),
        "chest_reward_link_contracts": {
            "Cursed Key": {
                "java_source": "CursedKey.onChestOpen / ShowCardAndObtainEffect.update / CeramicFish.onObtainCard",
                "effect": "chest rewards can open before curse acquisition and Fish9 gold; wait for explicit simulator_obtain_pending=false before any next policy/deferred action or receipt baseline (BRIDGE111)",
                "guards": "shared obtain wait applies in and out of combat, keeps pause-adjusted ten-second deadline across same-room frames, exempts GAME_OVER; gold receipt remains exact, no fabricated balance or card",
            },
            "Matryoshka": {
                "java_source": "Matryoshka.onChestOpen",
                "native_source": "GameContext::openTreasureRoomChest",
                "effect": "append extra relic before base chest relic; consume one remaining charge",
            },
            "SAPPHIRE_KEY": {
                "java_source": "AbstractChest.open / RewardItem.relicLink",
                "native_source": "Rewards::sapphireKey / executeRewardsAction",
                "effect": "mutually exclusive with last/base chest relic, not first/Matryoshka relic",
            },
            "NlothsMask": {
                "java_source": "NlothsMask.onChestOpenAfter / AbstractRoom.removeOneRelicFromRewards",
                "native_source": "GameContext::openTreasureRoomChest",
                "effect": "consume charge after generation; remove first relic and only its linked key",
            },
        },
        "run_keys": RUN_KEY_FIELDS,
        "real_room_classes": REAL_ROOM_CLASS_IDS,
        "reward_room_exit_contracts": REWARD_ROOM_EXIT_CONTRACTS,
        "chest_reward_screen_contracts": CHEST_REWARD_SCREEN_CONTRACTS,
        "native_combat_root_fields": NATIVE_GLOBAL_FIELDS,
        "potion_capacity_rules": POTION_CAPACITY_RULES,
        "compact_player_scalar_fields": {
            name: {"index": index, "source_and_consumer": meaning}
            for index, (name, meaning) in enumerate(COMPACT_PLAYER_SCALAR_FIELDS.items())
        },
        "live_relic_state_fields": LIVE_RELIC_STATE_FIELDS,
        "persistent_relic_counter_sentinels": RELIC_COUNTER_SENTINELS,
        "combat_relic_counter_sentinels": COMBAT_RELIC_COUNTER_SENTINELS,
        "relic_counter_contracts": relic_counters,
        "combat_counter_fields": {key: key for key in CS._REAL_COMBAT_COUNTER_KEYS},
        "native_relic_snapshot_fields": {"id": "required stable identity", "counter": "required native integer; no Java sentinel/default"},
        "fresh_card_misc": FRESH_CARD_MISC,
        "native_fresh_card_misc": {
            native: {"canonical": canonical, "initial_misc": FRESH_CARD_MISC.get(canonical, 0)}
            for native, canonical in SIM_TO_CANONICAL['card'].items()
        },
        "real_card_wire_fields": REAL_CARD_WIRE_FIELDS,
        "card_state_field_contracts": CARD_STATE_FIELD_CONTRACTS,
        "card_special_data_fields": CARD_SPECIAL_DATA_FIELDS,
        "card_state_transport_fields": CARD_STATE_TRANSPORT_FIELDS,
        "combat_selection_transport_fields": COMBAT_SELECTION_TRANSPORT_FIELDS,
        "shop_transport_fields": {
            "shop_slot": "cards: read-only ShopScreen.init/purchaseCard object tracking, fixed 0..6; relics/potions: actual StoreRelic/StorePotion.slot, fixed 0..2",
            "is_purchased": "actual StoreRelic/StorePotion.isPurchased; retained until and after UI-list removal by read-only references",
            "sold_card_slots": "only slots whose original card was actually removed without Courier replacement",
            "sold_relic_slots": "only initialized StoreRelic instances whose actual isPurchased became true",
            "sold_potion_slots": "only initialized StorePotion instances whose actual isPurchased became true",
            "coverage": "every physical slot must contain a typed item or explicit sale evidence; unknown/missing slots are never fabricated as PAD",
            "observed_empty_encoding": "legacy shop tensor PAD/-1 (or card EMPTY/0/-1) represents only the verified sold-slot absence above. It is not an identity fallback or permission to fill an unobserved slot; retain frozen model geometry while rejecting missing sale evidence",
            "policy": "same fixed shelf slots in observation and SHOP_BUY; choice_list ordinals remain transport-only handles",
            "advertised_vs_legal": "CommunicationMod ChoiceScreenUtils.getAvailableShopItems advertises potions by actual price<=gold even with Sozu/full slots. Match all advertised affordable identities and verify complete coverage first, then exclude only potion purchase policy actions when obtaining is impossible. Do not drop real buttons before identity matching; duplicate same-ID offers retain real shelf order and exact prices, cross-ID ambiguity still rejects (BRIDGE149)",
            "purchase_settlement": "StoreRelic.purchaseRelic uses instantObtain and applies Membership discount/SmilingMask price and preview hooks synchronously; do not infer a missing relic acquisition from cosmetic travel animation alone. Actual selection-opening relics still require their source-specific GRID. StorePotion obtains successfully before charging; Sozu/full belt cannot spend gold. Native openedScreen continuations preserve actual selection rather than forcing shop/map early",
            "delayed_card_and_retry": "shop card ShowCardAndObtainEffect is asynchronous; global explicit simulator_obtain_pending barrier covers all screens before further policy/native-root decisions. Shared mutating-command stale-frame guard also covers shop and dialogue commands, not only combat; never resend a purchase from an identical old frame. RewardCommandAck separately checks the requested card ID/up count, potion/relic count (Circlet counter), exact gold balance or addressed screen exit; cosmetic/gold-label changes are not SKIP/RETURN/PROCEED receipts",
            "audit_boundary": "source/consumer review of Shop, StoreRelic/StorePotion, read-only shelf hooks, adapter and shared acknowledgement barriers; no new effectful mismatch established or offline test executed, not proof of every rare shop/relic combination",
            "purge_available": "actual ShopScreen flag; false maps to native sold service price -1, not the stale actualPurgeCost",
        },
        "native_card_queue_fields": NATIVE_CARD_QUEUE_FIELDS,
        "native_card_instance_fields": NATIVE_CARD_INSTANCE_FIELDS,
        "native_card_snapshot_fields": NATIVE_CARD_SNAPSHOT_FIELDS,
        "native_master_deck_snapshot_fields": {'id': 'native CardId spelling', 'upgrades': 'explicit count', 'misc': 'explicit current native value', 'bottle': 'NONE/ATTACK/SKILL/POWER; physical slot kept'},
        "search_boundary_relic_fields": {"Pocketwatch": "pocketwatch_cards_at_boundary"},
        "search_action_semantics": {
            "producer": "NoslCombatTrainer.publicActionKey / hashActionCard -> both tabular and neural search edge remapping",
            "card_instance": "ID, specialData, base/turn cost, independent upgraded/freeToPlayOnce/retain/exhaust state and persistent RitualDagger/GeneticAlgorithm identity; MindBlast also preserves actual cachedBaseDamage. Single-card non-MindBlast keys keep their existing inputs",
            "multi_selection": "Hash the complete single-card semantics for every selected hand instance, sorted with multiplicity. Ordinary and upgraded same-ID/same-cost cards, free-once versus temporary cost0, independent exhaust and unequal cached damage must not alias. Missing selected hand cards reject, never synthesize an invalid-card/default signature",
            "remapping": "A selected edge may move to a different physical index only through an equal complete semantic key. If no legal counterpart exists, both searches raise contextual errors before executing/backing up any substitute; no first-legal fallback or synthetic terminal value (BRIDGE161)",
            "result_adapter": "Both direct and neural Python consumers require an explicit uint32 action_bits mapping to a live legal policy ID; the direct binding additionally reports action_valid=true. Missing/unmapped/masked actions raise, never become first legal/END or native=True with a lost complete-action target. Complete-action search rejects unfinished external target codec state. The native target checker is mandatory and its errors are not converted to no-target. Stochastic acting keeps its visit sampling but validates both the original native result and sampled complete action (BRIDGE162)",
            "compatibility": "Keep NOSL public composition/randomness, budgets, weight tensors, legal-action enumeration and equivalent-play grouping unchanged. Autoregressive pending selections already hash ordered hand/prefix state and retain that protection; ordinary current-checkpoint integration is not direct coverage of legacy non-AR subset remapping",
        },
        "post_enemy_turn_settlement": {
            "producer": "BattleContext.enterPostEnemyTurnBoundary / applyEndOfRoundPowers / Monster REGEN -> queued HealMonster",
            "contract": "Drain deterministic round-end actions and any card callbacks before marking postEnemyTurnBoundary or projecting next-turn cleanup. Preserve real terminal/selection states, FIFO/TOP reactions and the existing battle execution budget. Do not run another monster phase, player start relic/power hooks, draw or recharge, silently drop pending healing, or replace it with guessed HP",
            "consumers": "NOSL explicit endTurnAsLeaf, RuleTurnPlanner horizon, execute_to_enemy_boundary and both preview_end_turn APIs -> existing raw/canonical/packed HP and boundary marker, cloning and public state key; no new observation/action fields or weights",
            "compatibility": "Final TRAIN47 real configuration records end_turn_as_leaf=false; repair this shared supported boundary path without enabling it, changing search budgets, or claiming current ordinary Full-Gumbel naturally exercises it (BRIDGE160)",
            "audit": "Source/consumer review and necessary native build only for this specific boundary branch; ordinary current-checkpoint real integration is a compatibility check, not dynamic coverage of the disabled leaf mode",
        },
        "neow_bonus_keywords": NEOW_BONUS_IDS,
        "neow_drawback_keywords": NEOW_DRAWBACK_IDS,
        "neow_implicit_drawbacks": NEOW_IMPLICIT_DRAWBACK_IDS,
        "neow_screen_contracts": NEOW_SCREEN_CONTRACTS,
        "neow_transport_fields": {
            "screen": "simulator_neow_screen", "offers": "simulator_neow_options",
            "choice_index": "choice_index", "bonus": "bonus_type", "drawback": "drawback_type",
            "cached_hp_unit": "hp_unit", "absent_offer_marker": "drawback_id=0 only in unoffered slots",
        },
        "map_transport_fields": {
            "next_nodes": "all legal next-room records; up to seven columns, never truncated",
            "next_nodes_order": "CHOOSE UI index; independent of physical x column",
            "x": "MAP_NODE action column 0..6; unique within the candidate row",
            "y": "row 0..14; all next_nodes must be in one row",
            "symbol": "exact ROOM_ALIASES room identity; no absent-room fallback",
            "children": "game_state.map complete physical DAG edges; required, never invented by Wing Boots",
            "current_node": "actual current coordinate/room only; Java first node is (0,-1); edges are looked up in full map",
            "boss_available": "Act1..3 uses current column/final-row DAG slot; Act4 uses the actual (3,3) Boss leaf",
            "simulator_current_node": "v12 actual room coordinate/class across all screens; Java boss button lands at (-1,15)",
            "simulator_map_leaf_nodes": "v12 actual nonempty no-edge map nodes omitted by stock export; no guessed Boss record",
            "terminal_map_nodes": "typed TrueVictoryRoom display leaf retained outside frozen actionable room tokens; not PAD",
            "map_full_rooms": "105 physical slots, room absent only at genuine empty coordinates",
            "map_full_features": "x/y/current/choice/reachable plus all seven outgoing columns",
            "map_node_rooms": "frozen three-token prefix preview; not the complete action list",
            "map_node_x": "same preview order/scaling; wide choices require lossless full-map coverage",
        },
        "intent_components": REAL_INTENT_COMPONENTS,
        "room_phases": sorted(REAL_ROOM_PHASES),
        "action_phases": sorted(REAL_ACTION_PHASES),
        "card_types": sorted(REAL_CARD_TYPE_IDS),
        "card_rarities": sorted(REAL_CARD_RARITY_IDS),
        "card_targets_require_enemy": REAL_CARD_TARGET_IDS,
        "structural_powers": STRUCTURAL_POWER_FIELDS,
        "hit_reaction_contracts": HIT_REACTION_CONTRACTS,
        "card_damage_timing_contracts": CARD_DAMAGE_TIMING_CONTRACTS,
        "combat_victory_action_contracts": COMBAT_VICTORY_ACTION_CONTRACTS,
        "monster_death_contracts": MONSTER_DEATH_CONTRACTS,
        "monster_lifecycle_fields": MONSTER_LIFECYCLE_FIELDS,
        "combat_settlement_transport": {
            "source": "real COMBAT/NONE with every physical monster is_dying or is_escaping, exactly MonsterGroup.areMonstersBasicallyDead; never HP0 or is_gone (which includes half-dead revivers)",
            "action": "await_combat_settlement refresh only, same-room pause-adjusted 10-second deadline; shared UiPauseClock excludes exact SETTINGS/MASTER_DECK_VIEW inspection time from all ACKs, not real selection screens. Other actual is_screen_up NONE overlays are not timed settlement frames and still forbid ordinary combat policy. No empty encounter attachment, policy END, synthetic victory, corpse removal or PAD; explicit empty roster is also only a wait (BRIDGE143)",
            "boundary": "actual HAND_SELECT/GRID/CARD_REWARD remain actionable; actual reward/room/terminal frames end the wait, immutable ready GAME_OVER alone records outcome; model layout/search nonce/budget unchanged",
        },
        "summon_slot_transport": {
            "source": "actual GremlinLeader.gremlins / TheCollector.enemySlots / Reptomancer.daggers object references",
            "Reptomancer capacity": "Java canSpawn counts other non-dying monsters <=3, so native total including leader <=4 permits summon; takeTurn fills up to 1/A18 2 actually empty dagger slots, including a single free slot at A18, preserving reference order and per-created-monster RNG (BRIDGE112)",
            "native_order": {"GremlinLeader": ["gremlins[2]", "gremlins[0]", "gremlins[1]", "leader"],
                             "TheCollector": ["enemySlots[2]", "enemySlots[1]", "leader"],
                             "Reptomancer": ["daggers[3]", "daggers[1]", "leader", "daggers[2]", "daggers[0]"]},
            "Collector references": "v33/BRIDGE128: empty map before initial summon is genuinely vacant; otherwise require exactly non-null keys1/2, including replacement references; never invent key0 or discard a living unreferenced TorchHead",
            "wire": "combat_state.simulator_summon_slots: leader/slots/pending; actual empty reference=null, not-yet-inserted object=-1 and pending=true; non-summon encounter=null",
            "history": "unreferenced minions require is_dead/gone/hp0/not-half_dead/no-stasis; preserved in full raw records and _retired_monsters, explicitly excluded only from finite search projection",
            "restore": "only seven audited minion species can replace a strict root slot without RNG; every public field and explicit Wizard charge is restored",
        },
        "wizard_charge_transport": {
            "java": "GremlinWizard.currentCharge in 0..3; non-wizard explicit null",
            "canonical": "wizard_charge in real/native full state and diff",
            "native": "Monster.miscInfo, already part of search key; no new policy tensor",
            "timing": "increment on charge, reset on every Blast including A17",
        },
        "monster_move_state_fields": MONSTER_MOVE_STATE_CONTRACTS,
        "native_power_amount_storage": {
            "monster": "all dedicated stackable amounts and uniquePower1 use Java-sized int storage, not int8/int16; Transient Shifting can accumulate Shackled134 in one turn (BRIDGE115)",
            "Shackled": "GainStrengthPower constructor/stack uses +/-999 clamp; same bound as Strength, no 127 truncation; strict restore still checks exact readback",
            "TheBomb": "Java-sized per-instance damage/turns; three frozen int damage totals are derived, allowing totals above127 without merging independent hits (BRIDGE115/116)",
            "consumers": "native default copies and NOSL hashValue consume full fields; full wire snapshots/compact scalars remain numeric, requiring rebuilt native ABI rather than a checkpoint migration",
        },
        "bomb_instance_transport": {
            "wire": "v29 combat_state.simulator_bombs is an explicit ordered list of id/turns/damage, including []; each unique ID/countdown must match the actual ordered player.powers",
            "native": "Player.bombs retains each turns/damage instance; each expiring bomb queues its own THORNS-equivalent DamageAllEnemy, preserving Intangible hit count; no decomposition guessed from aggregate totals",
            "consumers": "canonical player.bombs, strict restore, diff, native copies and nonempty-bomb search keys retain the ordered instances; three old counter/compact/model slots remain derived damage sums with unchanged OBS20/ACTION8/layout9",
        },
        "monster_move_state_transport": {
            "wire": "v28 simulator_move_state; required exact species-specific dictionary or explicit null for other species; canonical/native move_state and diff retain every field",
            "native": "miscInfo: bool for TimeEater/RedSlaver/BronzeOrb, 0..6 Spiker thornsCount, 0..4 Automaton numTurns, Champ low2 forgeTimes/bit2 thresholdReached/upper bits numTurns, Book actual stabCount, Louse actual biteDamage, Darkling actual nipDmg, Guardian actual dmgThreshold, GiantHead count -6..5; Hexaghost low3 orb count/bit3 upgraded/bit4 activated and separate hexaghostDividerDamage int; NOSL hash and clone retain all values",
            "timing": "usedHaste and usedStasis set at intent selection, usedEntangle at execution even when Artifact blocks, Spiker increments on Spike; Champ counter increments per getMove and resets at Taunt; Automaton post-beam Stun/Boost does not increment numTurns",
            "book_intent": "restore actual stabCount before checking visible hits; MultiStab must match, SingleStab stays one hit without resetting the hidden count; increment on selected MultiStab and also on A18+ SingleStab, reject native integer overflow (BRIDGE101)",
            "random_base_damage": "Louse biteDamage and Darkling nipDmg survive all intents; strict restoration precedes visible attack damage checks, conflicts rejected; Darkling construction already chooses A2 9..13, never add another 2 (BRIDGE102)",
            "guardian_threshold": "actual dmgThreshold survives offensive/defensive roots; increment 10 upon entering Defensive Mode, use unchanged at Twin Slam return; remaining ModeShift power is distinct from next-cycle threshold, integer overflow rejected (BRIDGE102)",
            "giant_head_count": "count starts at 5; initial getMove decrements, A18+ preBattle decrements once more; each subsequent getMove decrements to -6 floor, ItIsTime selects when previous count <=1 and damage is startingDeathDmg -5*count; native execution and visible intent use the same restored field (BRIDGE103)",
            "scope": "thirteen audited species (BRIDGE98/101/102/103/105); no policy tensor/action ID change, no inferred counters from powers/history or claim that all BRIDGE37 private state is complete",
        },
        "hexaghost_cycle_and_burn": {
            "source": "Hexaghost takeTurn/changeState/getMove and unique.BurnIncreaseAction (BRIDGE105)",
            "cycle": "Activate sets Divider immediately and queues orb count 6 without an RNG draw; Divider/Inferno queue DeactivateAllOrbs then RollMove; Sear/Tackle/Inflame queue ActivateOrb then RollMove",
            "history": "ActivateOrb reaching 6 sets Inferno, then RollMove sets Inferno again; retain both actual history entries",
            "burn": "BurnIncrease upgrades only discard/draw Burn and creates three Burn+ in discard with normal unique IDs; hand/exhaust untouched; Sear reads actual burnUpgraded, never global turn",
            "state": "strict species fields restored before visible intent validation; power clearing cannot erase orb count; dedicated Divider value and miscInfo both in NOSL key and clone, no tensor changes",
        },
        "collector_ascension_effects": {
            "source": "TheCollector constructor and takeTurn -> native Monster::takeTurn (BRIDGE103)",
            "block": "Buff grants 15 below A9, 18 at A9..18, 23 at A19+; do not share the strength 4/19 tier index",
            "mega_debuff": "Weak, Vulnerable and Frail each 5 at A19+, otherwise each 3; preserve source=true and existing queue order",
            "scope": "effect amounts only; no action numbering, current intent, model tensor or checkpoint change",
        },
        "boss_buff_and_debuff_order": {
            "champ": "Champ constructor strAmt 2/3/4 at A0/4/19; Gloat grants strAmt, Anger grants 3*strAmt =6/9/12 (BRIDGE104)",
            "time_eater_ripple": "TimeEater.takeTurn grants 20 block, then Vulnerable1, Weak1, A19+ Frail1 in that order; Artifact must block the first eligible actual debuff, not the swapped status (BRIDGE104)",
            "scope": "effect values/order only; existing source=true and status transport retained, no policy tensor/action ID change",
        },
        "nemesis_scythe_and_burn": {
            "source": "Nemesis.getMove/takeTurn and AbstractMonster.setMove/lastMove (BRIDGE110)",
            "cooldown": "Scythe selection sets2; each getMove decrements before checking. Stock one-roll-per-turn behavior forbids only an immediately preceding Scythe, not either of two preceding moves; Scythe-other-Scythe is valid",
            "burn": "Debuff adds three Burn below A18, five at A18+; the separate triple-attack damage tier remains A3",
            "transport": "existing current intent and actual ordered move history suffice for this stock cooldown; retain exact IDs, RNG boolean evaluation order and policy layouts, no fabricated private state",
        },
        "spire_growth_move_selection": {
            "source": "SpireGrowth.getMove -> native Monster::getMoveForRoll (BRIDGE108)",
            "priority": "A17+ available Constrict first; otherwise roll<50 and not two Quick Tackles; then available Constrict regardless of roll; then Smash unless two consecutive Smashes; otherwise Quick Tackle",
            "available": "player has no Constricted and previous move is not Constrict; never replace the post-repeat fallback with roll>=50",
            "scope": "future move choice only; existing current intent/history, RNG draw and observation/action/checkpoint layouts unchanged",
        },
        "slaver_move_selection": {
            "java": "SlaverBlue.getMove / SlaverRed.getMove; native Monster::getMoveForRoll (BRIDGE99)",
            "repeat_limit": "A17+ Rake/Scrape cannot follow the same move; below A17 disallow only after two consecutive same moves, not an OR of the two permission checks",
            "red_stab": "first move is Stab; after usedEntangle, roll >=55 and not lastTwoMoves(Stab); before entangle, roll >=75 selects Entangle",
            "blue_stab": "roll >=40 and not lastTwoMoves(Stab); otherwise apply the Rake repeat limit",
            "scope": "future move selection only, retaining current intent/history, explicit usedEntangle, RNG draws, and policy tensor/action layouts",
        },
        "city_monster_move_selection": {
            "snake_plant": "SnakePlant.getMove -> A17+ Spore is barred if either of the last two moves was Spore, not only when both were; below A17 only lastMove bars it",
            "shelled_parasite": "ShelledParasite.getMove -> repeat Fell rerolls once in 20..99; compare that new value with 60, never the old value below 20",
            "mystic": "Healer.getMove -> sum maxHP-HP over all actual monsters except dying/escaping; heal at >=21 (A17+) or >=16, unless the last two moves were Heal",
            "scope": "BRIDGE100 native future move rules; same current intents, effect amounts, RNG draws, and policy tensor/action layouts",
        },
        "writhing_mass_implant_transport": {
            "java": "WrithingMass.usedMegaDebuff -> simulator_writhing_mass_used_implant; required boolean for this species, explicit null otherwise (v23)",
            "canonical": "writhing_mass_used_implant in real/native full state, strict root and diff; never infer from recent moves or an existing Parasite",
            "native": "Monster.miscInfo named states: IMPLANT_NOT_USED=0, IMPLANT_PENDING=1, IMPLANT_SETTLED=2; both used states prohibit another Implant and are distinguished by the existing NOSL key/clone",
            "master_deck": "real root first loads the actual master deck, then restores used=true as SETTLED; exitBattle only obtains Parasite/consumes Omamori for a newly simulated PENDING effect, never again for a real settled effect",
            "obtain_receipt": "wait for explicit simulator_obtain_pending=false before combat/selection decisions or native root restoration; 10-second pause-adjusted transport deadline, no automatic card choice, GAME_OVER remains terminal (BRIDGE96)",
            "obtain_relics": "unblocked simulated Implant applies Darkstone and CeramicFish gainGold(9) to combat player before exit; gainGold preserves Ectoplasm/BloodyIdol/Bloom/Flower behavior, and exit's player gold/HP overwrite avoids duplicate Deck.obtain benefits (BRIDGE96)",
            "scope": "normal settled combat root; no tensor/action layout change, no claim that arbitrary pending Java actions or every monster-private field are restored (BRIDGE95/28/37)",
        },
        "monster_intent_damage_contract": {
            "java_source": "AbstractMonster.getIntentDmg / intentDmg",
            "wire": "move_adjusted_damage",
            "native_restore": "strict monster adjusted_damage (required)",
            "native_inactive_field": "Monster::inactiveIntentDamage",
            "observation_and_key": "Monster::getIntentDamage; dict, packed and search hash",
            "active_rule": "derive from actual powers/target; never override attack calculation",
            "inactive_rule": "preserve explicit cached display after powers disappear; keep entity and lifecycle",
            "reset": "setMove or fresh construct/spawn; active restored roots clear inactive cache",
        },
        "rejected_real_ids": {
            "cards": UNSUPPORTED_REAL_CARD_IDS,
            "relics": UNSUPPORTED_REAL_RELIC_IDS,
            "powers": UNSUPPORTED_REAL_POWER_IDS,
        },
        "empty_real_potion_ids": sorted(EMPTY_REAL_POTION_IDS),
        "transport_only_events": TRANSPORT_ONLY_REAL_EVENT_IDS,
    }


def _targets(table):
    for value in table.values():
        if isinstance(value, dict):
            yield from value.values()
        else:
            yield value


@lru_cache(maxsize=1)
def validate_codec_contract() -> dict[str, int]:
    """Raise on an incomplete, colliding, PAD-producing, or stale word table."""
    canonical_sets = {
        "card": set(CARD_VOCAB) - {EMPTY},
        "relic": set(V.RELIC_VOCAB) - {"PAD"},
        "potion": set(V.POTION_VOCAB) - {"PAD"},
        "monster": set(V.MONSTER_VOCAB) - {"PAD"},
        "move": set(V.MOVE_VOCAB) - {"PAD"},
        "player_power": set(V.PLAYER_BUFF_VOCAB),
        "monster_power": set(V.MONSTER_BUFF_VOCAB),
        "event": set(V.EVENT_VOCAB) - {"PAD"},
        "boss": set(BOSS_CANONICAL_IDS),
        "room": set(V.ROOM_VOCAB) - {"NONE", "INVALID"},
        "screen": set(V.PHASE_VOCAB),
        "select_context": set(V.SELECT_CONTEXT_VOCAB) - {"PAD"},
        "character": set(V.CHARACTER_VOCAB) - {"INVALID"},
    }

    for kind, table in REAL_TO_CANONICAL.items():
        targets = set(_targets(table))
        unknown = targets - canonical_sets[kind]
        if unknown:
            raise RuntimeError(f"{kind} real codec targets unknown tokens: {sorted(unknown)!r}")
        missing = canonical_sets[kind] - targets
        if kind != "screen" and missing:
            raise RuntimeError(f"{kind} real codec omits canonical tokens: {sorted(missing)!r}")
        if any(value in {"PAD", EMPTY, "NONE", "INVALID"} for value in targets):
            raise RuntimeError(f"{kind} real codec maps live input to a sentinel")
    if set(SCREEN_PHASES) != set(REAL_SCREEN_TYPE_IDS):
        raise RuntimeError(
            "CommunicationMod screen word table is incomplete: "
            f"missing={sorted(set(REAL_SCREEN_TYPE_IDS) - set(SCREEN_PHASES))!r} "
            f"extra={sorted(set(SCREEN_PHASES) - set(REAL_SCREEN_TYPE_IDS))!r}"
        )
    if not REAL_COMMAND_IDS or any(
        not isinstance(command, str) or command != command.upper()
        for command in REAL_COMMAND_IDS
    ):
        raise RuntimeError("CommunicationMod command word table is malformed")

    # Every stock Java identity has one declared disposition: accepted and
    # mapped, structurally empty, or rejected with an explicit reason.  An
    # exception may never shadow a supported spelling because that would make
    # behavior depend on which consumer checked the value first.
    exceptional = {
        "card": set(UNSUPPORTED_REAL_CARD_IDS),
        "relic": set(UNSUPPORTED_REAL_RELIC_IDS),
        "potion": set(EMPTY_REAL_POTION_IDS),
        "player_power": set(UNSUPPORTED_REAL_POWER_IDS),
        "monster_power": set(UNSUPPORTED_REAL_POWER_IDS),
    }
    for kind, rejected in exceptional.items():
        overlap = set(REAL_TO_CANONICAL[kind]) & rejected
        if overlap:
            raise RuntimeError(
                f"{kind} identities are both supported and exceptional: "
                f"{sorted(overlap)!r}"
            )
    structural_overlap = (
        set(STRUCTURAL_POWER_FIELDS) & set(UNSUPPORTED_REAL_POWER_IDS)
    )
    if structural_overlap:
        raise RuntimeError(
            "real powers are both structurally represented and unsupported: "
            f"{sorted(structural_overlap)!r}"
        )
    transport_event_overlap = (
        set(REAL_TO_CANONICAL["event"]) & set(TRANSPORT_ONLY_REAL_EVENT_IDS)
    )
    if transport_event_overlap:
        raise RuntimeError(
            "real event IDs are both policy events and transport-only: "
            f"{sorted(transport_event_overlap)!r}"
        )
    if set(REWARD_CHOICE_TYPES.values()) != set(REAL_REWARD_TYPE_IDS):
        raise RuntimeError(
            "reward choice word table does not exactly cover RewardItem types"
        )
    if len(REWARD_CHOICE_TYPES) != len(set(REWARD_CHOICE_TYPES)):
        raise RuntimeError("reward choice word table contains duplicate labels")
    if set(RUN_KEY_FIELDS) != {"ruby", "emerald", "sapphire"}:
        raise RuntimeError("run-key word table is incomplete")
    if set(NEOW_BONUS_IDS.values()) != set(range(19)) or set(NEOW_DRAWBACK_IDS.values()) != set(range(1, 6)):
        raise RuntimeError("Neow keyword tables omit or collide native IDs")
    if NEOW_IMPLICIT_DRAWBACK_IDS != {"BOSS_RELIC": 6}:
        raise RuntimeError("Neow starting-relic loss must be an explicit bonus implication")
    for relic, field in LIVE_RELIC_STATE_FIELDS.items():
        if relic not in canonical_sets["relic"] or field not in CS._REAL_COMBAT_COUNTER_KEYS:
            raise RuntimeError(f"live relic state word table names unknown field: {relic!r}/{field!r}")
    for relic, field in LIVE_RELIC_COUNTER_FIELDS.items():
        if relic not in canonical_sets["relic"] or field not in (*CS._REAL_COMBAT_COUNTER_KEYS, "$turn"):
            raise RuntimeError(f"live relic counter table names unknown field: {relic!r}/{field!r}")
    if (set(REAL_RELIC_COUNTER_USAGE) | set(DERIVED_RELIC_COUNTER_SOURCES)) - canonical_sets["relic"]:
        raise RuntimeError("relic counter source inventory names unknown relics")
    if set(FRESH_RELIC_INITIAL_DATA) - canonical_sets["relic"] or any(
        type(value) is not int or value <= 0 for value in FRESH_RELIC_INITIAL_DATA.values()
    ):
        raise RuntimeError("fresh relic initial-data word table is malformed")
    if set(FRESH_CARD_MISC) - canonical_sets['card'] or any(
        type(value) is not int or value <= 0 for value in FRESH_CARD_MISC.values()
    ):
        raise RuntimeError('fresh card misc word table is malformed')
    classified_counters = (set(RELIC_COUNTER_SENTINELS) | set(RELIC_COUNTER_DOMAINS)
                           | set(PHASE_RELIC_COUNTER_DOMAINS) | {"CeramicFish"})
    if classified_counters != set(REAL_RELIC_COUNTER_USAGE):
        raise RuntimeError("Java counter users and numeric disposition word tables differ")
    if (set(LIVE_RELIC_COUNTER_MODULI) | set(LIVE_RELIC_COUNTER_STOP_TURNS)) - set(LIVE_RELIC_COUNTER_FIELDS):
        raise RuntimeError("live relic transforms name unknown runtime fields")
    for table in (RELIC_COUNTER_SENTINELS, COMBAT_RELIC_COUNTER_SENTINELS):
        for relic, values in table.items():
            if relic not in canonical_sets["relic"] or not values:
                raise RuntimeError(f"relic counter sentinel word table is malformed: {relic!r}")
    boss_keys = [key for keys in REAL_ACT_BOSS_KEYS.values() for key in keys]
    if (len(boss_keys) != len(set(boss_keys))
            or any(key not in BOSS_ALIASES for key in boss_keys)
            or {BOSS_ALIASES[key] for key in boss_keys} != BOSS_CANONICAL_IDS):
        raise RuntimeError("stock dungeon boss encounter keys are incompletely mapped")
    for semantic, fields in RUN_KEY_FIELDS.items():
        if set(fields) != {"real", "sim"} or not all(fields.values()):
            raise RuntimeError(f"run-key word table is malformed for {semantic!r}")
    for event, kinds in DYNAMIC_EVENT_OPTION_KINDS.items():
        if event not in canonical_sets["event"]:
            raise RuntimeError(f"dynamic event word table names unknown event {event!r}")
        if not kinds or len(kinds) != len(set(kinds.values())):
            raise RuntimeError(f"dynamic event word table is empty/colliding for {event!r}")
        if any(type(index) is not int or not 0 <= index < 8 for index in kinds.values()):
            raise RuntimeError(f"dynamic event word table has invalid index for {event!r}")
    for event, aliases in DYNAMIC_EVENT_OPTION_KIND_ALIASES.items():
        kinds = DYNAMIC_EVENT_OPTION_KINDS.get(event, {})
        if any(alias in kinds or target not in kinds for alias, target in aliases.items()):
            raise RuntimeError(f"dynamic event aliases are malformed for {event!r}")
    # Import lazily to keep the low-level identity module independent from the
    # event effect verifier.  Every canonical event has exactly one declared
    # static, structured/dynamic, or Neow option contract.
    from bridge import event_contract
    contract_events = event_contract.canonical_contract_events()
    expected_events = canonical_sets["event"]
    if contract_events != expected_events:
        raise RuntimeError(
            "event option word table is incomplete: "
            f"missing={sorted(expected_events - contract_events)!r} "
            f"extra={sorted(contract_events - expected_events)!r}"
        )

    if len(SIM_TO_CANONICAL["card"]) != len(CARD_VOCAB) - 1:
        raise RuntimeError("sim card codec contains a spelling collision")
    for native, canonical in SIM_TO_CANONICAL["card"].items():
        if CS._sim_card_id(native) != canonical:
            raise RuntimeError(
                f"sim card codec is not reversible: {native!r} -> {canonical!r}"
            )

    unknown_native_cards = (
        set(NATIVE_COMBAT_CARD_ENUM_IDS) - set(SIM_TO_CANONICAL["card"])
    )
    if unknown_native_cards:
        raise RuntimeError(
            "native combat capability table names unknown CardId values: "
            f"{sorted(unknown_native_cards)!r}"
        )
    ironclad_native = {
        CS.real_to_sim_card_enum(card_id) for card_id in IRONCLAD_CARDS
    }
    missing_ironclad_rules = ironclad_native - set(NATIVE_COMBAT_CARD_ENUM_IDS)
    if missing_ironclad_rules:
        raise RuntimeError(
            "native combat capability table omits Ironclad cards: "
            f"{sorted(missing_ironclad_rules)!r}"
        )

    for kind, table in SIM_TO_CANONICAL.items():
        allowed = canonical_sets[kind]
        unknown = set(_targets(table)) - allowed
        if unknown:
            raise RuntimeError(f"{kind} sim codec targets unknown tokens: {sorted(unknown)!r}")

    real_monsters = set(CS._MONSTER_SIM2REAL.values())
    if set(REAL_MONSTER_MOVE_TO_CANONICAL) != real_monsters:
        raise RuntimeError("real monster move tables do not cover every simulator monster")
    for monster, moves in REAL_MONSTER_MOVE_TO_CANONICAL.items():
        if not moves:
            raise RuntimeError(f"real monster has an empty move table: {monster!r}")
        unknown = set(moves.values()) - canonical_sets["move"]
        if unknown:
            raise RuntimeError(
                f"real monster {monster!r} maps unknown moves: {sorted(unknown)!r}"
            )

    selection_targets = set(COMBAT_SELECTION_SOURCES.values())
    selection_targets.update(_targets(DYNAMIC_COMBAT_SELECTION_SOURCES))
    selection_targets.update(OVERWORLD_SELECTION_FLAGS.values())
    selection_targets.update(OVERWORLD_SELECTION_RELICS.values())
    selection_targets.update(OVERWORLD_SELECTION_EVENTS.values())
    if selection_targets != canonical_sets["select_context"]:
        raise RuntimeError(
            "selection source table and SELECT_CONTEXT_VOCAB differ: "
            f"missing={sorted(canonical_sets['select_context'] - selection_targets)!r} "
            f"extra={sorted(selection_targets - canonical_sets['select_context'])!r}"
        )
    supported_sources = set(COMBAT_SELECTION_SOURCES) | set(DYNAMIC_COMBAT_SELECTION_SOURCES)
    overlap = supported_sources & set(UNSUPPORTED_SELECTION_SOURCES)
    if overlap:
        raise RuntimeError(f"selection sources are both supported and unsupported: {sorted(overlap)!r}")

    if set(CS._POWER_DROP) != set(STRUCTURAL_POWER_FIELDS):
        raise RuntimeError("a dropped power has no documented structural representation")

    native_player = {native_power_enum(power) for power in V.PLAYER_BUFF_VOCAB}
    if native_player != set(NATIVE_PLAYER_STATUS_IDS):
        raise RuntimeError(
            "player power vocabulary and native PlayerStatus differ: "
            f"missing={sorted(set(NATIVE_PLAYER_STATUS_IDS) - native_player)!r} "
            f"extra={sorted(native_player - set(NATIVE_PLAYER_STATUS_IDS))!r}"
        )
    native_monster = {native_power_enum(power) for power in V.MONSTER_BUFF_VOCAB}
    # ASLEEP is transported by explicit Lagavulin gameplay state, not the shared
    # SLEEP/STUN move token, and is intentionally absent
    # from the ordinary buff vector; every other native status must be present.
    expected_monster = set(NATIVE_MONSTER_STATUS_IDS) - {"ASLEEP"}
    if native_monster != expected_monster:
        raise RuntimeError(
            "monster power vocabulary and native MonsterStatus differ: "
            f"missing={sorted(expected_monster - native_monster)!r} "
            f"extra={sorted(native_monster - expected_monster)!r}"
        )
    if len(NATIVE_PLAYER_STATUS_IDS) != len(set(NATIVE_PLAYER_STATUS_IDS)):
        raise RuntimeError("native player status word table contains duplicates")
    if len(NATIVE_MONSTER_STATUS_IDS) != len(set(NATIVE_MONSTER_STATUS_IDS)):
        raise RuntimeError("native monster status word table contains duplicates")

    return {
        "real_tokens": sum(len(table) for table in REAL_TO_CANONICAL.values()),
        "sim_tokens": sum(len(table) for table in SIM_TO_CANONICAL.values()),
        "monster_moves": sum(len(table) for table in REAL_MONSTER_MOVE_TO_CANONICAL.values()),
        "selection_sources": (
            len(COMBAT_SELECTION_SOURCES)
            + len(DYNAMIC_COMBAT_SELECTION_SOURCES)
            + len(OVERWORLD_SELECTION_FLAGS)
            + len(OVERWORLD_SELECTION_RELICS)
            + len(OVERWORLD_SELECTION_EVENTS)
            + len(UNSUPPORTED_SELECTION_SOURCES)
        ),
        "native_player_statuses": len(NATIVE_PLAYER_STATUS_IDS),
        "native_monster_statuses": len(NATIVE_MONSTER_STATUS_IDS),
        "native_combat_cards": len(NATIVE_COMBAT_CARD_ENUM_IDS),
        "reward_types": len(REAL_REWARD_TYPE_IDS),
        "screen_types": len(REAL_SCREEN_TYPE_IDS),
        "command_keywords": len(REAL_COMMAND_IDS),
        "intent_keywords": len(REAL_INTENT_COMPONENTS),
        "room_phase_keywords": len(REAL_ROOM_PHASES),
        "action_phase_keywords": len(REAL_ACTION_PHASES),
        "card_type_keywords": len(REAL_CARD_TYPE_IDS),
        "card_rarity_keywords": len(REAL_CARD_RARITY_IDS),
        "card_target_keywords": len(REAL_CARD_TARGET_IDS),
        "run_key_fields": len(RUN_KEY_FIELDS),
        "native_combat_root_fields": len(NATIVE_GLOBAL_FIELDS),
        "real_room_classes": len(REAL_ROOM_CLASS_IDS),
        "compact_player_scalar_fields": len(COMPACT_PLAYER_SCALAR_FIELDS),
        "real_card_wire_fields": len(REAL_CARD_WIRE_FIELDS),
        "card_state_field_contracts": len(CARD_STATE_FIELD_CONTRACTS),
        "card_state_transport_fields": len(CARD_STATE_TRANSPORT_FIELDS),
        "combat_selection_transport_fields": len(COMBAT_SELECTION_TRANSPORT_FIELDS),
        "native_card_queue_fields": len(NATIVE_CARD_QUEUE_FIELDS),
        "hit_reaction_contracts": len(HIT_REACTION_CONTRACTS),
        "card_damage_timing_contracts": len(CARD_DAMAGE_TIMING_CONTRACTS),
        "combat_victory_action_contracts": len(COMBAT_VICTORY_ACTION_CONTRACTS),
        "monster_death_contracts": len(MONSTER_DEATH_CONTRACTS),
        "monster_lifecycle_fields": len(MONSTER_LIFECYCLE_FIELDS),
        "card_special_data_fields": len(CARD_SPECIAL_DATA_FIELDS),
        "relic_counter_contracts": len(canonical_sets["relic"]),
        "java_relic_counter_users": len(REAL_RELIC_COUNTER_USAGE),
        "combat_counter_fields": len(CS._REAL_COMBAT_COUNTER_KEYS),
        "live_relic_counter_fields": len(LIVE_RELIC_COUNTER_FIELDS),
        "neow_bonus_keywords": len(NEOW_BONUS_IDS),
        "neow_drawback_keywords": len(NEOW_DRAWBACK_IDS),
        "neow_screen_keywords": len(NEOW_SCREEN_CONTRACTS),
        "dynamic_event_kinds": sum(len(kinds) for kinds in DYNAMIC_EVENT_OPTION_KINDS.values()),
        "event_option_contracts": len(contract_events),
        "explicit_unsupported_real_ids": (
            len(UNSUPPORTED_REAL_CARD_IDS)
            + len(UNSUPPORTED_REAL_RELIC_IDS)
            + len(UNSUPPORTED_REAL_POWER_IDS)
            + len(TRANSPORT_ONLY_REAL_EVENT_IDS)
        ),
    }
