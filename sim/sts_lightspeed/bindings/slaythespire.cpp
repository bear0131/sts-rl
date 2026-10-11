//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"

#include "slaythespire.h"

// added for combat bindings (diff testing / RL)
#include <memory>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include "sim/search/CombatChoices.h"
#include "sim/search/GameAction.h"
#include "sim/search/Nosl.h"
#include "sim/search/NoslCombatTrainer.h"
#include "sim/search/NeuralSearchBatch.h"
#include "sim/search/RuleTurnPlanner.h"
#include "neural-observation.h"
#include <chrono>
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/Cards.h"
#include "constants/MonsterEncounters.h"


using namespace sts;

// ===== combat-state export + action helpers (added for diff testing / RL) =====
// The C++ side only emits structured RAW data keyed by the sim's own enum-name
// strings; mapping those to the real game's ids is done in Python (schemas/),
// symmetric with the real-game normalizer. Keeps the binding minimal.
namespace {
namespace py = pybind11;

void setRulePlayerStatusByName(BattleContext &bc, const std::string &name, int amount);
void setRuleMonsterStatusByName(BattleContext &bc, int target, const std::string &name, int amount);
sts::CardId cardIdFromEnum(const std::string &name);

Potion potionFromNameOrId(const std::string &value) {
    const auto count = sizeof(potionNames) / sizeof(potionNames[0]);
    static_assert(count == sizeof(potionIds) / sizeof(potionIds[0]));
    for (std::size_t i = 1; i < count; ++i) {
        if (value == potionNames[i] || value == potionIds[i]) {
            return static_cast<Potion>(i);
        }
    }
    throw std::invalid_argument("unknown custom potion name or id: " + value);
}

template <typename Pile>
py::list cardPileToList(const Pile &pile) {
    py::list out;
    for (const auto &c : pile) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(c.getId())], c.getUpgradeCount()));
    }
    return out;
}

py::dict combatSelectInfo(const sts::BattleContext &bc);

int playerPowerObservation(const sts::Player &p, PlayerStatus status) {
    // OBS18: presence powers live in status bits, not necessarily statusMap.
    // Keep getStatusRuntime unchanged: search keys and game rules use it too.
    switch (status) {
        case PlayerStatus::CONFUSED: case PlayerStatus::HEX:
        case PlayerStatus::DRAW_REDUCTION: case PlayerStatus::BARRICADE:
        case PlayerStatus::BLASPHEMER: case PlayerStatus::CORRUPTION:
        case PlayerStatus::ELECTRO: case PlayerStatus::SURROUNDED:
        case PlayerStatus::MASTER_REALITY: case PlayerStatus::PEN_NIB:
        case PlayerStatus::WRATH_NEXT_TURN:
            return p.hasStatusRuntime(status) ? 1 : 0;
        default:
            return p.getStatusRuntime(status);
    }
}

py::list playerPowersToList(const sts::Player &p) {
    py::list out;
    for (int i = 1; i <= static_cast<int>(PlayerStatus::THE_BOMB); ++i) {
        const int amount = playerPowerObservation(p, static_cast<PlayerStatus>(i));
        if (amount != 0) out.append(py::make_tuple(playerStatusStrings[i], amount));
    }
    return out;
}

py::list monsterPowersToList(const sts::Monster &m) {
    py::list out;
    for (int s = 0; s < static_cast<int>(sts::MonsterStatus::INVALID); ++s) {
        const auto status = static_cast<sts::MonsterStatus>(s);
        const int v = m.getStatusInternal(status);
        // Time Eater's Time Warp is a presence-bearing counter.  It is active
        // at zero before the first card, and dropping that zero removes the
        // trigger that advances the counter after cards are played.
        const bool zeroTimeWarp = status == sts::MonsterStatus::TIME_WARP
            && m.hasStatusInternal(status);
        if (v != 0 || zeroTimeWarp) {
            out.append(py::make_tuple(sts::enemyStatusStrings[s], v));
        }
    }
    return out;
}

py::dict rawCombatState(const sts::BattleContext &bc, bool includeJointActions = false) {
    py::dict d;
    d["turn"] = bc.turn;
    // Search-only marker: true only after the complete enemy phase and its
    // deterministic end-of-round cleanup have finished.
    d["post_enemy_turn_boundary"] = bc.postEnemyTurnBoundary;
    d["outcome"] = static_cast<int>(bc.outcome);
    d["input_state"] = static_cast<int>(bc.inputState);

    py::dict player;
    player["hp"] = bc.player.curHp;
    player["max_hp"] = bc.player.maxHp;
    player["block"] = bc.player.block;
    player["energy"] = bc.player.energy;
    player["gold"] = bc.player.gold;
    py::dict counters;
    counters["centennial_puzzle_active"] =
        bc.player.hasRelicRuntime(sts::RelicId::CENTENNIAL_PUZZLE);
    counters["happy_flower"] = bc.player.happyFlowerCounter;
    counters["incense_burner"] = bc.player.incenseBurnerCounter;
    counters["ink_bottle"] = bc.player.inkBottleCounter;
    counters["inserter"] = bc.player.inserterCounter;
    counters["lizard_tail_active"] =
        bc.player.hasRelicRuntime(sts::RelicId::LIZARD_TAIL);
    counters["nunchaku"] = bc.player.nunchakuCounter;
    counters["pen_nib"] = bc.player.penNibCounter;
    counters["sundial"] = bc.player.sundialCounter;
    counters["necronomicon_used"] = bc.player.haveUsedNecronomiconThisTurn;
    counters["cards_played"] = bc.player.cardsPlayedThisTurn;
    counters["attacks_played"] = bc.player.attacksPlayedThisTurn;
    counters["skills_played"] = bc.player.skillsPlayedThisTurn;
    counters["orange_attack"] = bc.player.orangePelletsCardTypesPlayed.test(0);
    counters["orange_skill"] = bc.player.orangePelletsCardTypesPlayed.test(1);
    counters["orange_power"] = bc.player.orangePelletsCardTypesPlayed.test(2);
    counters["combust_hp_loss"] = bc.player.combustHpLoss;
    counters["deva_form_energy_per_turn"] = bc.player.devaFormEnergyPerTurn;
    counters["echo_form_cards_doubled"] = bc.player.echoFormCardsDoubled;
    counters["panache_counter"] = bc.player.panacheCounter;
    counters["cards_discarded"] = bc.player.cardsDiscardedThisTurn;
    counters["times_damaged"] = bc.player.timesDamagedThisCombat;
    counters["bomb_1"] = bc.player.bomb1;
    counters["bomb_2"] = bc.player.bomb2;
    counters["bomb_3"] = bc.player.bomb3;
    counters["last_attack_unblocked_damage"] = bc.player.lastAttackUnblockedDamage;
    player["combat_counters"] = counters;
    player["powers"] = playerPowersToList(bc.player);
    player["energy_per_turn"] = static_cast<int>(bc.player.energyPerTurn);
    d["player"] = player;

    py::list monsters;
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &m = bc.monsters.arr[i];
        py::dict md;
        md["id"] = sts::monsterIdStrings[static_cast<int>(m.id)];
        md["hp"] = m.curHp;
        md["max_hp"] = m.maxHp;
        md["block"] = m.block;
        md["gone"] = m.isDeadOrEscaped();
        md["half_dead"] = m.halfDead;
        // Java represents Awakened One phase one with UnawakenedPower.  The
        // simulator already stores the same phase in miscInfo (false for phase
        // one, true after rebirth); expose it so a real settled frame can
        // strictly synchronize the hidden rule state without adding a fake
        // monster status.
        md["unawakened"] =
            m.id == sts::MonsterId::AWAKENED_ONE && !m.miscInfo;
        // Java's CompulsivePower is the persistent presence marker for
        // Writhing Mass's Reactive behavior. The numeric REACTIVE storage is
        // also a transient queued-hit counter, so expose presence separately
        // instead of converting a settled root into a pending reroll.
        md["reactive"] =
            m.id == sts::MonsterId::WRITHING_MASS
            && m.hasStatus<sts::MonsterStatus::REACTIVE>();
        // current intent: the move the monster has rolled for this turn. setMove
        // shifts the previous move to moveHistory[1] and stores the new one in
        // moveHistory[0], so [0] is the *current* intent (the obs MOVE_VOCAB id).
        const auto curMove = m.moveHistory[0];
        const auto previousMove = m.moveHistory[1];
        md["move_id"] = static_cast<int>(curMove);
        md["move"] = sts::monsterMoveStrings[static_cast<int>(curMove)];
        md["last_move_id"] = static_cast<int>(previousMove);
        md["last_move"] = sts::monsterMoveStrings[static_cast<int>(previousMove)];
        const auto dmg = m.getMoveBaseDamage(bc);
        md["base_damage"] = dmg.damage;
        // Use the simulator's actual damage pipeline for the displayed intent.
        // This includes Strength, Weak/Paper Krane, player Vulnerable/Odd
        // Mushroom, Surrounded, Wrath, Intangible, and the game's rounding.
        md["adjusted_damage"] = dmg.damage > 0
            ? m.calculateDamageToPlayer(bc, dmg.damage)
            : 0;
        md["hits"] = dmg.attackCount;
        md["powers"] = monsterPowersToList(m);
        monsters.append(md);
    }
    d["monsters"] = monsters;

    py::list hand;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &c = bc.cards.hand[i];
        const int currentCost = c.freeToPlayOnce ? 0 : c.costForTurn;
        hand.append(py::make_tuple(
            sts::cardEnumStrings[static_cast<int>(c.getId())],
            c.getUpgradeCount(),
            currentCost
        ));
    }
    d["hand"] = hand;
    d["draw"] = cardPileToList(bc.cards.drawPile);
    d["discard"] = cardPileToList(bc.cards.discardPile);
    d["exhaust"] = cardPileToList(bc.cards.exhaustPile);
    py::list potions;
    for (int i = 0; i < bc.potionCapacity; ++i) {
        potions.append(std::string(sts::getPotionName(bc.potions[i])));
    }
    d["potions"] = potions;
    d["select"] = combatSelectInfo(bc);
    if (includeJointActions) d["joint_action_ids"] = sts::search::jointActionIds(bc);
    return d;
}

// Native compact leaf transport.  The old binding converted every hypothetical
// BattleContext into a deeply nested Python dict.  Neural search only needs the
// numeric combat tensors, so keep the hot path as two flat POD buffers and let
// Python perform one vectorized vocab remap per batch.  The layout is versioned
// in env/core/run_env.py and intentionally contains native enum ids, not model
// vocab ids; this keeps the C++ simulator independent of Python's canonical
// naming aliases.
namespace compact_leaf {
constexpr int kMaxMonsters = 5;
constexpr int kHandSlots = 10;
// Keep this in lockstep with encode.COMBAT_COUNTER_KEYS.  The first version
// of the compact transport only carried the original 19 simulator counters;
// the observation contract now has 24 entries (including Panache, discard /
// damage counters, Bomb timers, and last unblocked damage).
constexpr int kCombatCounters = 24;
constexpr int kPotionSlots = 5;
constexpr int kPlayerStatus =
    static_cast<int>(sizeof(playerStatusStrings) /
                     sizeof(playerStatusStrings[0])) - 1;
constexpr int kMonsterStatus =
    static_cast<int>(sizeof(sts::enemyStatusStrings) /
                     sizeof(sts::enemyStatusStrings[0]));
constexpr int kCardIds =
    static_cast<int>(sizeof(sts::cardEnumStrings) /
                     sizeof(sts::cardEnumStrings[0]));
constexpr int kNativeMonsters =
    static_cast<int>(sizeof(sts::monsterIdStrings) /
                     sizeof(sts::monsterIdStrings[0]));
constexpr int kNativeMoves =
    static_cast<int>(sizeof(sts::monsterMoveStrings) /
                     sizeof(sts::monsterMoveStrings[0]));
constexpr int kNativePotions =
    static_cast<int>(sizeof(sts::potionNames) /
                     sizeof(sts::potionNames[0]));
// The language observation has one slot per supported card ID. The Python
// adapter checks this against its observation schema before enabling it.
constexpr int kLanguageSlots = 80;
// Categorical layout: hand cards, fixed monster slots/moves/intent classes,
// potion slots, and the card-select task.
constexpr int kCatSize = kHandSlots + kMaxMonsters + kMaxMonsters +
                         kMaxMonsters + kPotionSlots + 1 + 96 * 2 + 1;

// Float layout: player scalars, player statuses, counters, boundary bit,
// monster scalars/statuses, hand upgrade/cost, three native pile histograms,
// and card-select metadata.
constexpr int kFloatSize = 6 + kPlayerStatus + kCombatCounters + 1 +
                           kMaxMonsters * 6 + kMaxMonsters * kMonsterStatus +
                           kHandSlots * 2 + kCardIds * 3 + 4 + 96 * 2;

struct Row {
    std::array<std::int32_t, kCatSize> categorical{};
    std::array<float, kFloatSize> scalars{};
};

inline int intentCategoryFromName(const std::string &name) {
    auto has = [&name](const char *token) {
        return name.find(token) != std::string::npos;
    };
    if (has("STUNNED")) return 7;
    if (has("SLEEP") || has("ASLEEP") || has("DORMANT")) return 6;
    if (has("ESCAPE") || has("SMOKE_BOMB")) return 8;
    if (has("MEGA_DEBUFF") || has("HEX") || has("DEBILITATE")) return 5;
    if (has("DEBUFF") || has("ENTANGLE") || has("SPORES") ||
        has("ENFEEBL") || has("CONSTRICT") || has("SPIT_WEB") ||
        has("GLARE") || has("MOCK") || has("TAUNT") || has("SCREAM") ||
        has("WITHER") || has("PERPLEXING")) return 4;
    if (has("DEFEND") || has("HARDEN") || has("PROTECT") ||
        has("FORTIFY") || has("DEFENSIVE") || has("STANCE")) return 2;
    if (has("BUFF") || has("BELLOW") || has("INCANTATION") ||
        has("RITUAL") || has("GROW") || has("BOOST") || has("CHARG") ||
        has("RALLY") || has("ENCOURAGE") || has("PREPARING") ||
        has("ACTIVATE") || has("REGROW") || has("REINCARNATE") ||
        has("REBIRTH") || has("SPLIT") || has("SPAWN") || has("SUMMON") ||
        has("HEAL") || has("HASTE") || has("ANGER") || has("GLOAT") ||
        has("FURY") || has("CIRCLE_OF_POWER") ||
        has("SQUARE_OF_PROTECTION") || has("VENT_STEAM") ||
        has("COUNT") || has("IT_IS_TIME") || has("ROAR") ||
        has("DROOL") || has("FLY") || has("CAW") || has("IMPLANT") ||
        has("ECHO")) return 3;
    return 0;
}

inline int intentCategory(const sts::Monster &monster, int baseDamage) {
    if (baseDamage > 0) return 1;
    const int moveId = static_cast<int>(monster.moveHistory[0]);
    if (moveId <= 0 || moveId >= kNativeMoves) return 0;
    static const auto categories = [] {
        std::array<int, kNativeMoves> result{};
        for (int id = 1; id < kNativeMoves; ++id) {
            result[id] = intentCategoryFromName(sts::monsterMoveStrings[id]);
        }
        return result;
    }();
    return categories[moveId];
}

inline int monsterStatusValue(const sts::Monster &monster, int status) {
    // Avoid the simulator's print-only getStatusInternal() helper in this hot
    // transport.  Its generic status lookup intentionally has no canonical
    // vocabulary contract; the explicit fields below mirror the same storage
    // used by the combat rules and preserve the status bit index.
    const auto value = static_cast<sts::MonsterStatus>(status);
    if (value == sts::MonsterStatus::STRENGTH) {
        return monster.strength;
    }
    if (status < 0 || status >= 64 ||
        !(monster.statusBits & (1ULL << status))) {
        return 0;
    }
    switch (value) {
        case sts::MonsterStatus::ARTIFACT: return monster.artifact;
        case sts::MonsterStatus::BLOCK_RETURN: return monster.blockReturn;
        case sts::MonsterStatus::CHOKED: return monster.choked;
        case sts::MonsterStatus::CORPSE_EXPLOSION: return monster.corpseExplosion;
        case sts::MonsterStatus::LOCK_ON: return monster.lockOn;
        case sts::MonsterStatus::MARK: return monster.mark;
        case sts::MonsterStatus::METALLICIZE: return monster.metallicize;
        case sts::MonsterStatus::PLATED_ARMOR: return monster.platedArmor;
        case sts::MonsterStatus::POISON: return monster.poison;
        case sts::MonsterStatus::REGEN: return monster.regen;
        case sts::MonsterStatus::SHACKLED: return monster.shackled;
        case sts::MonsterStatus::VULNERABLE: return monster.vulnerable;
        case sts::MonsterStatus::WEAK: return monster.weak;
        case sts::MonsterStatus::ANGRY:
        case sts::MonsterStatus::BEAT_OF_DEATH:
        case sts::MonsterStatus::CURIOSITY:
        case sts::MonsterStatus::CURL_UP:
        case sts::MonsterStatus::ENRAGE:
        case sts::MonsterStatus::FADING:
        case sts::MonsterStatus::FLIGHT:
        case sts::MonsterStatus::GENERIC_STRENGTH_UP:
        case sts::MonsterStatus::INTANGIBLE:
        case sts::MonsterStatus::MALLEABLE:
        case sts::MonsterStatus::MODE_SHIFT:
        case sts::MonsterStatus::RITUAL:
        case sts::MonsterStatus::SLOW:
        case sts::MonsterStatus::SPORE_CLOUD:
        case sts::MonsterStatus::THIEVERY:
        case sts::MonsterStatus::THORNS:
        case sts::MonsterStatus::TIME_WARP:
            return monster.uniquePower0;
        case sts::MonsterStatus::INVINCIBLE:
        case sts::MonsterStatus::REACTIVE:
        case sts::MonsterStatus::SHARP_HIDE:
            return monster.uniquePower1;
        default:
            return 1;
    }
}

inline Row pack(const sts::BattleContext &bc) {
    Row row;
    std::array<int, kMaxMonsters> baseDamages{};
    std::array<int, kMaxMonsters> attackCounts{};
    int c = 0;
    for (int i = 0; i < bc.cards.cardsInHand && i < kHandSlots; ++i) {
        row.categorical[c + i] = static_cast<int>(bc.cards.hand[i].getId());
    }
    c += kHandSlots;
    for (int i = 0; i < bc.monsters.monsterCount && i < kMaxMonsters;
         ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (monster.id == sts::MonsterId::INVALID) continue;
        row.categorical[c + i] = static_cast<int>(monster.id);
        row.categorical[c + kMaxMonsters + i] =
            static_cast<int>(monster.moveHistory[0]);
    }
    c += 2 * kMaxMonsters;
    for (int i = 0; i < bc.monsters.monsterCount && i < kMaxMonsters;
         ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (monster.id == sts::MonsterId::INVALID) continue;
        const auto damage = monster.getMoveBaseDamage(bc);
        baseDamages[i] = std::max(0, damage.damage);
        attackCounts[i] = damage.attackCount;
        row.categorical[c + i] = intentCategory(
            monster, baseDamages[i]);
    }
    c += kMaxMonsters;
    for (int i = 0; i < bc.potionCapacity && i < kPotionSlots; ++i) {
        row.categorical[c + i] = static_cast<int>(bc.potions[i]);
    }
    c += kPotionSlots;
    const bool selecting = bc.inputState == InputState::CARD_SELECT;
    row.categorical[c++] = selecting ? static_cast<int>(bc.cardSelectInfo.cardSelectTask) : 0;
    const auto choices = sts::search::combatChoices(bc);
    for (std::size_t i = 0; i < choices.cards.size(); ++i) {
        row.categorical[c + i] = static_cast<int>(choices.cards[i].id);
        row.categorical[c + 96 + i] = choices.cards[i].nativeIndex + 1;
    }
    row.categorical[c + 192] = choices.zone;

    int f = 0;
    row.scalars[f++] = static_cast<float>(bc.player.curHp);
    row.scalars[f++] = static_cast<float>(bc.player.maxHp);
    row.scalars[f++] = static_cast<float>(bc.player.block);
    row.scalars[f++] = static_cast<float>(bc.player.energy);
    row.scalars[f++] = static_cast<float>(bc.turn);
    row.scalars[f++] = static_cast<float>(bc.player.gold);
    for (int status = 1; status <= kPlayerStatus; ++status) {
        row.scalars[f++] = static_cast<float>(playerPowerObservation(bc.player,
            static_cast<PlayerStatus>(status)));
    }
    // Keep this order synchronized with encode.COMBAT_COUNTER_KEYS.
    const float counters[kCombatCounters] = {
        static_cast<float>(bc.player.happyFlowerCounter),
        static_cast<float>(bc.player.incenseBurnerCounter),
        static_cast<float>(bc.player.inkBottleCounter),
        static_cast<float>(bc.player.inserterCounter),
        static_cast<float>(bc.player.nunchakuCounter),
        static_cast<float>(bc.player.penNibCounter),
        static_cast<float>(bc.player.sundialCounter),
        static_cast<float>(bc.player.haveUsedNecronomiconThisTurn),
        static_cast<float>(bc.player.cardsPlayedThisTurn),
        static_cast<float>(bc.player.attacksPlayedThisTurn),
        static_cast<float>(bc.player.skillsPlayedThisTurn),
        static_cast<float>(bc.player.orangePelletsCardTypesPlayed.test(0)),
        static_cast<float>(bc.player.orangePelletsCardTypesPlayed.test(1)),
        static_cast<float>(bc.player.orangePelletsCardTypesPlayed.test(2)),
        static_cast<float>(bc.player.combustHpLoss),
        static_cast<float>(bc.player.devaFormEnergyPerTurn),
        static_cast<float>(bc.player.echoFormCardsDoubled),
        static_cast<float>(bc.player.panacheCounter),
        static_cast<float>(bc.player.cardsDiscardedThisTurn),
        static_cast<float>(bc.player.timesDamagedThisCombat),
        static_cast<float>(bc.player.bomb1),
        static_cast<float>(bc.player.bomb2),
        static_cast<float>(bc.player.bomb3),
        static_cast<float>(bc.player.lastAttackUnblockedDamage),
    };
    for (int i = 0; i < kCombatCounters; ++i) row.scalars[f++] = counters[i];
    row.scalars[f++] = bc.postEnemyTurnBoundary ? 1.0f : 0.0f;

    // Keep the packed layout identical to encode._empty_combat(): all
    // monsters' six scalar fields are contiguous, followed by all monster
    // status vectors.  The Python decoder relies on these two rectangular
    // blocks when reshaping a batch; interleaving status fields per monster
    // silently corrupts multi-monster leaves.
    for (int i = 0; i < bc.monsters.monsterCount && i < kMaxMonsters;
         ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (monster.id == sts::MonsterId::INVALID) { f += 6; continue; }
        const int baseDamage = baseDamages[i];
        const int adjusted = baseDamage > 0
            ? monster.calculateDamageToPlayer(bc, baseDamage) : 0;
        row.scalars[f++] = monster.isDeadOrEscaped() ? 0.0f : 1.0f;
        row.scalars[f++] = static_cast<float>(monster.curHp);
        row.scalars[f++] = static_cast<float>(monster.maxHp);
        row.scalars[f++] = static_cast<float>(monster.block);
        row.scalars[f++] = static_cast<float>(adjusted);
        row.scalars[f++] = static_cast<float>(baseDamage > 0 ? attackCounts[i] : 0);
    }
    f += 6 * (kMaxMonsters - std::min(kMaxMonsters, bc.monsters.monsterCount));
    for (int i = 0; i < bc.monsters.monsterCount && i < kMaxMonsters;
         ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (monster.id == sts::MonsterId::INVALID) { f += kMonsterStatus; continue; }
        for (int status = 0; status < kMonsterStatus; ++status) {
            row.scalars[f++] = static_cast<float>(monsterStatusValue(monster, status));
        }
    }
    f += kMonsterStatus * (kMaxMonsters - std::min(kMaxMonsters, bc.monsters.monsterCount));
    for (int i = 0; i < bc.cards.cardsInHand && i < kHandSlots; ++i) {
        const auto &card = bc.cards.hand[i];
        row.scalars[f++] = static_cast<float>(card.getUpgradeCount());
        row.scalars[f++] = static_cast<float>(card.freeToPlayOnce ? 0 : card.costForTurn);
    }
    f += 2 * (kHandSlots - std::min(kHandSlots, bc.cards.cardsInHand));
    auto countPile = [&](const auto &pile) {
        const int base = f;
        for (const auto &card : pile) {
            const int id = static_cast<int>(card.getId());
            if (id >= 0 && id < kCardIds) row.scalars[base + id] += 1.0f;
        }
        f += kCardIds;
    };
    countPile(bc.cards.drawPile);
    countPile(bc.cards.discardPile);
    countPile(bc.cards.exhaustPile);
    row.scalars[f++] = selecting ? static_cast<float>(bc.cardSelectInfo.pickCount) : 0.0f;
    row.scalars[f++] = selecting && (bc.cardSelectInfo.canPickZero ||
        bc.cardSelectInfo.cardSelectTask == CardSelectTask::EXHAUST_MANY ||
        bc.cardSelectInfo.cardSelectTask == CardSelectTask::GAMBLE) ? 1.0f : 0.0f;
    row.scalars[f++] = selecting && bc.cardSelectInfo.canPickAnyNumber ? 1.0f : 0.0f;
    const auto task = bc.cardSelectInfo.cardSelectTask;
    row.scalars[f++] = selecting && bc.cardSelectInfo.isMultiSelection() ? 1.0f : 0.0f;
    for (std::size_t i = 0; i < choices.cards.size(); ++i) {
        row.scalars[f + i] = static_cast<float>(choices.cards[i].upgrades);
        row.scalars[f + 96 + i] = static_cast<float>(choices.cards[i].cost);
    }
    return row;
}

template <typename Pile>
inline void packLanguagePile(const Pile &pile, const std::int32_t *cardIndex,
                             int *ids, float *features) {
    // Group in first-occurrence order, as encode._encode_card_sequence does.
    // A card's native ID is converted to its canonical model ID before
    // grouping, so aliases cannot occupy separate observation slots.
    std::unordered_map<int, int> positions;
    for (const auto &card : pile) {
        const int native = static_cast<int>(card.getId());
        if (native < 0 || native >= kCardIds) continue;
        const int id = cardIndex[native];
        if (id < 0) continue;
        auto [it, inserted] = positions.emplace(id, static_cast<int>(positions.size()));
        const int slot = it->second;
        if (slot >= kLanguageSlots) {
            throw std::runtime_error("language pile capacity exceeded");
        }
        if (inserted) ids[slot] = id;
        features[slot * 3] += 1.0f;
        const int upgrades = card.getUpgradeCount();
        features[slot * 3 + 1] += upgrades > 0 ? 1.0f : 0.0f;
        features[slot * 3 + 2] += static_cast<float>(upgrades);
    }
}
} // namespace compact_leaf

// Apply a deliberately small, data-only combat override.  This is used by the
// dashboard's custom-state editor; it never advances the simulator or touches
// RNG streams.  Card piles are rebuilt with normal CardInstance bookkeeping so
// subsequent draws/discards remain authoritative.
void applyCustomCombatState(sts::BattleContext &bc, const py::dict &spec) {
    if (spec.contains("potions")) {
        const auto names = spec["potions"].cast<std::vector<std::string>>();
        if (names.size() > static_cast<std::size_t>(bc.potionCapacity))
            throw std::invalid_argument("custom potions exceed potion capacity");
        auto potions = bc.potions;
        std::fill(potions.begin(), potions.end(), Potion::EMPTY_POTION_SLOT);
        int count = 0;
        for (std::size_t slot = 0; slot < names.size(); ++slot) {
            potions[slot] = potionFromNameOrId(names[slot]);
            if (potions[slot] != Potion::EMPTY_POTION_SLOT) ++count;
        }
        bc.potions = potions;
        bc.potionCount = count;
    }
    auto getInt = [](const py::dict &d, const char *key, int current) {
        return d.contains(key) ? d[key].cast<int>() : current;
    };
    if (spec.contains("player")) {
        auto p = spec["player"].cast<py::dict>();
        bc.player.maxHp = std::max(1, getInt(p, "max_hp", bc.player.maxHp));
        bc.player.curHp = std::clamp(
            getInt(p, "hp", bc.player.curHp), 0, bc.player.maxHp);
        bc.player.block = std::max(0, getInt(p, "block", bc.player.block));
        bc.player.energy = std::max(0, getInt(p, "energy", bc.player.energy));
        bc.player.energyPerTurn = static_cast<std::int8_t>(
            std::clamp(getInt(p, "energy_per_turn", bc.player.energyPerTurn), 0, 127));
        bc.player.gold = std::max(0, getInt(p, "gold", bc.player.gold));
        if (p.contains("counters")) {
            auto c = p["counters"].cast<py::dict>();
            bc.player.happyFlowerCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "happy_flower", bc.player.happyFlowerCounter), 0, 255));
            bc.player.incenseBurnerCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "incense_burner", bc.player.incenseBurnerCounter), 0, 255));
            bc.player.inkBottleCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "ink_bottle", bc.player.inkBottleCounter), 0, 255));
            bc.player.inserterCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "inserter", bc.player.inserterCounter), 0, 255));
            bc.player.nunchakuCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "nunchaku", bc.player.nunchakuCounter), 0, 255));
            bc.player.penNibCounter = static_cast<std::int8_t>(
                std::clamp(getInt(c, "pen_nib", bc.player.penNibCounter), 0, 255));
            bc.player.cardsDiscardedThisTurn = static_cast<std::int16_t>(
                std::clamp(getInt(c, "cards_discarded", bc.player.cardsDiscardedThisTurn), 0, 32767));
            bc.player.timesDamagedThisCombat = static_cast<std::int16_t>(
                std::clamp(getInt(c, "times_damaged", bc.player.timesDamagedThisCombat), 0, 32767));
        }
        if (p.contains("powers")) {
            static constexpr const char *kStatuses[] = {
                "STRENGTH", "DEXTERITY", "WEAK", "FRAIL", "VULNERABLE",
                "ARTIFACT", "DOUBLE_DAMAGE", "INTANGIBLE", "BARRICADE",
                "CORRUPTION", "CONFUSED", "HEX", "DRAW_REDUCTION",
                "BLASPHEMER", "ELECTRO", "SURROUNDED", "MASTER_REALITY",
                "PEN_NIB", "WRATH_NEXT_TURN", "DOUBLE_TAP", "DUPLICATION", "BRUTALITY",
                "COMBUST", "DARK_EMBRACE", "DEMON_FORM", "EVOLVE",
                "FEEL_NO_PAIN", "FIRE_BREATHING", "FLAME_BARRIER",
                "JUGGERNAUT", "METALLICIZE", "RAGE", "RUPTURE", "THORNS",
                "VIGOR",
            };
            for (const auto *name : kStatuses) setRulePlayerStatusByName(bc, name, 0);
            for (auto item : p["powers"].cast<py::list>()) {
                auto pair = item.cast<py::tuple>();
                const auto name = pair[0].cast<std::string>();
                const int amount = pair[1].cast<int>();
                setRulePlayerStatusByName(bc, name, amount);
            }
        }
    }
    if (spec.contains("monsters")) {
        auto mons = spec["monsters"].cast<py::list>();
        const int n = std::min<int>(static_cast<int>(mons.size()), bc.monsters.monsterCount);
        for (int i = 0; i < n; ++i) {
            auto m = mons[i].cast<py::dict>();
            auto &monster = bc.monsters.arr[i];
            if (m.contains("gone") && m["gone"].cast<bool>()) {
                monster.curHp = 0;
                monster.block = 0;
                continue;
            }
            if (m.contains("id")) {
                const auto requestedId = sts::SimHelpers::getMonsterIdForString(
                    m["id"].cast<std::string>());
                if (requestedId == sts::MonsterId::INVALID) {
                    throw std::invalid_argument("unknown custom monster id");
                }
                if (monster.id != requestedId) {
                    monster.construct(bc, requestedId, i);
                }
            }
            monster.maxHp = std::max(1, getInt(m, "max_hp", monster.maxHp));
            monster.curHp = std::clamp(
                getInt(m, "hp", monster.curHp), 0, monster.maxHp);
            monster.block = std::max(0, getInt(m, "block", monster.block));
            if (m.contains("unawakened")) {
                if (monster.id != sts::MonsterId::AWAKENED_ONE
                    && m["unawakened"].cast<bool>()) {
                    throw std::invalid_argument(
                        "unawakened phase is only valid for Awakened One");
                }
                if (monster.id == sts::MonsterId::AWAKENED_ONE) {
                    monster.miscInfo = m["unawakened"].cast<bool>() ? 0 : 1;
                }
            }
            if (m.contains("move_history")) {
                const auto moves = m["move_history"].cast<std::vector<std::string>>();
                if (moves.size() > 2) {
                    throw std::invalid_argument("custom monster move history exceeds capacity");
                }
                for (std::size_t history = 0; history < moves.size(); ++history) {
                    monster.moveHistory[history] = sts::SimHelpers::getMonsterMoveForString(
                        moves[history]);
                }
            } else if (m.contains("move")) {
                const auto move = sts::SimHelpers::getMonsterMoveForString(
                    m["move"].cast<std::string>());
                if (move != sts::MonsterMoveId::INVALID) monster.moveHistory[0] = move;
            }
            if (m.contains("base_damage") || m.contains("hits")) {
                const auto current = monster.getMoveBaseDamage(bc);
                const int requestedDamage = getInt(m, "base_damage", current.damage);
                const int requestedHits = getInt(m, "hits", current.attackCount);
                // Several monsters roll an encounter-local value that Java keeps
                // outside the public move id.  CommunicationMod exposes the
                // realized intent damage/hit count, so restore the equivalent
                // native hidden field before search.  Fixed-damage moves must
                // already agree; accepting arbitrary overrides would make the
                // simulator transition disagree with the displayed root intent.
                switch (monster.moveHistory[0]) {
                    case sts::MonsterMoveId::GREEN_LOUSE_BITE:
                    case sts::MonsterMoveId::RED_LOUSE_BITE:
                    case sts::MonsterMoveId::HEXAGHOST_DIVIDER:
                        monster.miscInfo = requestedDamage;
                        break;
                    case sts::MonsterMoveId::DARKLING_NIP:
                        monster.miscInfo = requestedDamage - (bc.ascension >= 2 ? 2 : 0);
                        break;
                    case sts::MonsterMoveId::BOOK_OF_STABBING_MULTI_STAB:
                        monster.miscInfo = requestedHits;
                        break;
                    default:
                        break;
                }
                const auto synchronized = monster.getMoveBaseDamage(bc);
                if (synchronized.damage != requestedDamage
                    || synchronized.attackCount != requestedHits) {
                    throw std::invalid_argument(
                        "custom monster move damage does not match native move semantics");
                }
            }
            if (m.contains("powers")) {
                static constexpr const char *kStatuses[] = {
                    "ARTIFACT", "BLOCK_RETURN", "CHOKED", "CORPSE_EXPLOSION",
                    "LOCK_ON", "MARK", "METALLICIZE", "PLATED_ARMOR",
                    "POISON", "REGEN", "SHACKLED", "STRENGTH",
                    "VULNERABLE", "WEAK", "ANGRY", "BEAT_OF_DEATH",
                    "CURIOSITY", "CURL_UP", "ENRAGE", "FADING", "FLIGHT",
                    "GENERIC_STRENGTH_UP", "INTANGIBLE", "MALLEABLE",
                    "MODE_SHIFT", "RITUAL", "SLOW", "SPORE_CLOUD",
                    "THIEVERY", "THORNS", "INVINCIBLE",
                    "REACTIVE", "SHARP_HIDE", "ASLEEP", "BARRICADE",
                    "MINION", "MINION_LEADER", "PAINFUL_STABS", "REGROW",
                    "SHIFTING", "STASIS",
                };
                // The ordinary setter deliberately treats TIME_WARP=0 as an
                // active counter.  Clear its presence explicitly before the
                // incoming real-game power list is applied.
                monster.setStatus<sts::MonsterStatus::TIME_WARP>(0);
                monster.setHasStatus<sts::MonsterStatus::TIME_WARP>(false);
                for (const auto *name : kStatuses) {
                    setRuleMonsterStatusByName(bc, i, name, 0);
                }
                for (auto item : m["powers"].cast<py::list>()) {
                    auto pair = item.cast<py::tuple>();
                    const auto name = pair[0].cast<std::string>();
                    const int amount = pair[1].cast<int>();
                    setRuleMonsterStatusByName(bc, i, name, amount);
                }
            }
            if (m.contains("reactive")) {
                const bool reactive = m["reactive"].cast<bool>();
                if (reactive && monster.id != sts::MonsterId::WRITHING_MASS) {
                    throw std::invalid_argument(
                        "reactive root state is only valid for Writhing Mass");
                }
                monster.setStatus<sts::MonsterStatus::REACTIVE>(0);
                monster.setHasStatus<sts::MonsterStatus::REACTIVE>(reactive);
            }
        }
        bc.monsters.monstersAlive = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            if (!bc.monsters.arr[i].isDeadOrEscaped()) ++bc.monsters.monstersAlive;
        }
    }
    int nextUid = 0;
    auto applyVectorPile = [&](const char *key, auto &pile) {
        if (!spec.contains(key)) return;
        pile.clear();
        for (auto item : spec[key].cast<py::list>()) {
            std::string name;
            int up = 0;
            if (py::isinstance<py::str>(item)) name = item.cast<std::string>();
            else {
                auto pair = item.cast<py::tuple>();
                name = pair[0].cast<std::string>();
                up = pair.size() > 1 ? pair[1].cast<int>() : 0;
            }
            const auto id = sts::SimHelpers::getCardIdForString(name);
            if (id == sts::CardId::INVALID) throw std::invalid_argument("unknown card: " + name);
            sts::CardInstance card(id, up > 0);
            for (int k = 1; k < up; ++k) card.upgrade();
            card.uniqueId = static_cast<std::int16_t>(nextUid++);
            pile.push_back(card);
        }
    };
    if (spec.contains("hand")) {
        bc.cards.cardsInHand = 0;
        for (auto item : spec["hand"].cast<py::list>()) {
            std::string name;
            int up = 0;
            std::optional<int> cost;
            if (py::isinstance<py::str>(item)) name = item.cast<std::string>();
            else {
                auto pair = item.cast<py::tuple>();
                name = pair[0].cast<std::string>();
                up = pair.size() > 1 ? pair[1].cast<int>() : 0;
                if (pair.size() > 2 && !pair[2].is_none()) cost = pair[2].cast<int>();
            }
            auto id = sts::SimHelpers::getCardIdForString(name);
            if (id == sts::CardId::INVALID) throw std::invalid_argument("unknown card: " + name);
            sts::CardInstance card(id, up > 0);
            for (int k = 1; k < up; ++k) card.upgrade();
            if (cost.has_value()) {
                card.costForTurn = static_cast<std::int8_t>(
                    std::clamp(*cost, -1, 127));
                card.freeToPlayOnce = false;
            }
            card.uniqueId = static_cast<std::int16_t>(nextUid++);
            if (bc.cards.cardsInHand >= sts::CardManager::MAX_HAND_SIZE) break;
            bc.cards.hand[bc.cards.cardsInHand++] = card;
        }
    }
    if (spec.contains("draw")) applyVectorPile("draw", bc.cards.drawPile);
    else applyVectorPile("draw_pile", bc.cards.drawPile);
    if (spec.contains("discard")) applyVectorPile("discard", bc.cards.discardPile);
    else applyVectorPile("discard_pile", bc.cards.discardPile);
    if (spec.contains("exhaust")) applyVectorPile("exhaust", bc.cards.exhaustPile);
    else applyVectorPile("exhaust_pile", bc.cards.exhaustPile);
    bc.cards.recomputeDerivedState();
    bc.cards.nextUniqueCardId = std::max(bc.cards.nextUniqueCardId, nextUid);
    if (spec.contains("turn")) bc.turn = std::max(0, spec["turn"].cast<int>());
    bc.inputState = sts::InputState::PLAYER_NORMAL;
    bc.outcome = sts::Outcome::UNDECIDED;
    bc.isBattleOver = false;
    bc.endTurnQueued = false;
    bc.turnHasEnded = false;
    bc.stopAfterEnemyTurn = false;
    bc.postEnemyTurnBoundary = false;
    bc.monsterTurnIdx = bc.monsters.monsterCount;
}

std::vector<sts::search::Action> legalCombatActions(const sts::BattleContext &bc) {
    using SA = sts::search::Action;
    using AT = sts::search::ActionType;
    std::vector<SA> out;
    if (bc.outcome != sts::Outcome::UNDECIDED) {
        return out;
    }
    switch (bc.inputState) {
        case sts::InputState::PLAYER_NORMAL:
            if (bc.isCardPlayAllowed()) {
                for (int h = 0; h < bc.cards.cardsInHand; ++h) {
                    const auto &c = bc.cards.hand[h];
                    if (!c.canUseOnAnyTarget(bc)) {
                        continue;
                    }
                    if (c.requiresTarget()) {
                        for (int t = 0; t < bc.monsters.monsterCount; ++t) {
                            if (bc.monsters.arr[t].isTargetable()) {
                                out.push_back(SA(AT::CARD, h, t));
                            }
                        }
                    } else {
                        out.push_back(SA(AT::CARD, h));
                    }
                }
            }
            out.push_back(SA(AT::END_TURN));
            break;
        case sts::InputState::CARD_SELECT:
            for (const auto &a : SA::enumerateCardSelectActions(bc)) {
                out.push_back(a);
            }
            break;
        default:
            break;
    }
    return out;
}

// Return the fixed combat-only action mask without crossing the Python
// boundary once per Action. The Python environment's 71-id combat layout is
// intentionally mirrored here. Consume the same legalCombatActions result as
// the binding below so this optimization cannot silently change the existing
// Action -> id convention (including its default target sentinel behavior).
py::array_t<bool> legalCombatMask(const sts::BattleContext &bc) {
    constexpr int HAND_SLOTS = 10;
    constexpr int MAX_MONSTERS = 5;
    constexpr int SELECT_BASE = 1 + HAND_SLOTS * MAX_MONSTERS + HAND_SLOTS;
    constexpr int N_ACTIONS = SELECT_BASE + HAND_SLOTS;
    py::array_t<bool> out(static_cast<py::ssize_t>(N_ACTIONS));
    auto mask = out.mutable_unchecked<1>();
    for (int i = 0; i < N_ACTIONS; ++i) {
        mask(i) = false;
    }

    if (bc.outcome != sts::Outcome::UNDECIDED) {
        return out;
    }

    for (const auto &a : legalCombatActions(bc)) {
        switch (a.getActionType()) {
            case sts::search::ActionType::END_TURN:
                mask(0) = true;
                break;
            case sts::search::ActionType::CARD: {
                const int source = a.getSourceIdx();
                if (source < 0 || source >= HAND_SLOTS) {
                    break;
                }
                int target = a.getTargetIdx();
                if (target < 0 || target >= MAX_MONSTERS) {
                    target = -1;
                }
                const int id = target >= 0
                    ? 1 + source * MAX_MONSTERS + target
                    : 1 + HAND_SLOTS * MAX_MONSTERS + source;
                mask(id) = true;
                break;
            }
            case sts::search::ActionType::SINGLE_CARD_SELECT: {
                const int source = a.getSourceIdx();
                if (source >= 0 && source < HAND_SLOTS) {
                    mask(SELECT_BASE + source) = true;
                }
                break;
            }
            case sts::search::ActionType::MULTI_CARD_SELECT:
                // The thin layer represents the simulator's single forced
                // multi-select action as SELECT slot 0.
                mask(SELECT_BASE) = true;
                break;
            default:
                break;
        }
    }
    return out;
}

// Authoritative legal IN-COMBAT potion actions, from the sim's own per-Action
// validator (search::Action::isValidAction -> isValidPotionAction). We don't
// hand-roll potion legality: we probe each held slot with the exact POTION
// Action the env would later execute and keep only the ones the sim accepts.
// Returns a list of {slot, requires_target, targets} dicts:
//   * requires_target == false -> drink in one step (drink_potion(slot)); targets
//     is empty.
//   * requires_target == true  -> the potion needs a monster target (Fire/Fear/
//     Poison/Weak); targets lists the monster indices the sim accepts (each probed
//     with the real POTION Action, so it already excludes dead/untargetable foes).
//     The thin layer runs its PICK_TARGET substep over exactly these, then calls
//     drink_potion(slot, target). potionRequiresTarget() (Potions.h) is the sim's
//     own notion of targeting, so we surface it rather than re-deriving it.
// Discard-only potions (Fairy etc., or a no-effect drink) are intentionally NOT
// offered: in combat the only valid drink is what isValidPotionAction accepts.
py::list legalCombatPotions(const sts::BattleContext &bc) {
    using SA = sts::search::Action;
    using AT = sts::search::ActionType;
    py::list out;
    if (bc.outcome != sts::Outcome::UNDECIDED ||
        bc.inputState != sts::InputState::PLAYER_NORMAL) {
        return out;
    }
    for (int slot = 0; slot < bc.potionCapacity; ++slot) {
        const auto p = bc.potions[slot];
        if (p == sts::Potion::INVALID || p == sts::Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        const bool needsTarget = sts::potionRequiresTarget(p);
        py::list targets;
        if (needsTarget) {
            // a targeted potion (Fire/Fear/Poison/Weak): the env enumerates one
            // action per accepted monster, mirroring the targeted card-play path,
            // so we list exactly the monster indices the sim's validator accepts.
            for (int t = 0; t < bc.monsters.monsterCount; ++t) {
                if (SA(AT::POTION, slot, t).isValidAction(bc)) {
                    targets.append(t);
                }
            }
            if (py::len(targets) == 0) continue;   // no legal target -> not usable
        } else if (!SA(AT::POTION, slot, 0).isValidAction(bc)) {
            // untargeted but the sim rejects drinking it (e.g. Fairy in Battle,
            // or a potion with no in-combat drink effect) -> not offered.
            continue;
        }
        py::dict d;
        d["slot"] = slot;
        d["requires_target"] = needsTarget;
        d["targets"] = targets;
        out.append(d);
    }
    return out;
}

// Authoritative legal OUT-OF-COMBAT (overworld) potion actions, from the sim's
// own GameAction potion validator (isValidPotionAction). Only a handful of
// potions do anything outside combat (Blood/Entropic Brew/Fruit Juice drink;
// the rest are discard-only and we do NOT surface a discard verb to the agent).
// A targeted potion never applies out of combat (no monsters), so every
// overworld entry is single-step (requires_target is always false here). We
// construct the GameAction with the potion bit set (0x80000000) so isValidAction
// routes to isValidPotionAction; the discard bit (0x40000000) is left clear so we
// probe the DRINK path specifically.
py::list legalOverworldPotions(const sts::GameContext &gc) {
    py::list out;
    if (gc.outcome != sts::GameOutcome::UNDECIDED) {
        return out;
    }
    for (int slot = 0; slot < gc.potionCapacity; ++slot) {
        const auto p = gc.potions[slot];
        if (p == sts::Potion::INVALID || p == sts::Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        // bits: potion-action flag | idx1=slot. discard bit clear == DRINK probe.
        sts::search::GameAction a(0x80000000U | (slot & 0xFF));
        if (!a.isValidAction(gc)) {
            continue;
        }
        py::dict d;
        d["slot"] = slot;
        d["requires_target"] = false;   // no targets exist out of combat
        d["targets"] = py::list();
        out.append(d);
    }
    return out;
}

std::unique_ptr<sts::BattleContext> createBattle(const sts::GameContext &gc, const std::string &encounterName) {
    auto enc = sts::MonsterEncounter::INVALID;
    constexpr int n = sizeof(sts::monsterEncounterStrings) / sizeof(sts::monsterEncounterStrings[0]);
    for (int i = 0; i < n; ++i) {
        if (encounterName == sts::monsterEncounterStrings[i]) {
            enc = static_cast<sts::MonsterEncounter>(i);
            break;
        }
    }
    auto bc = std::make_unique<sts::BattleContext>();
    bc->init(gc, enc);
    return bc;
}

std::vector<std::string> encounterNames() {
    std::vector<std::string> v;
    constexpr int n = sizeof(sts::monsterEncounterStrings) / sizeof(sts::monsterEncounterStrings[0]);
    for (int i = 1; i < n; ++i) {  // skip INVALID
        v.emplace_back(sts::monsterEncounterStrings[i]);
    }
    return v;
}

// ===== out-of-combat introspection (added for full-run replay / diff) =====
// Symmetric with the combat side: emit raw data keyed by the sim's own enum
// strings (cards) or the game's stable ids (relics), mapped in Python.

py::list deckIds(const sts::GameContext &gc) {
    py::list out;
    for (const auto &c : gc.deck.cards) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(c.getId())], c.getUpgraded()));
    }
    return out;
}

// sts::relicIds[] holds the game's stable ids (e.g. "Art of War"), matching the
// real CommunicationMod relic `id` field -- not relicNames[] (display strings).
py::list gcRelicIds(const sts::GameContext &gc) {
    py::list out;
    for (const auto &r : gc.relics.relics) {
        out.append(std::string(sts::relicIds[static_cast<int>(r.id)]));
    }
    return out;
}

py::list gcRelics(const sts::GameContext &gc) {
    py::list out;
    for (const auto &r : gc.relics.relics) {
        py::dict item;
        item["id"] = std::string(sts::relicIds[static_cast<int>(r.id)]);
        item["counter"] = r.data;
        out.append(item);
    }
    return out;
}

std::vector<sts::search::Action> legalRuleCombatActions(const sts::BattleContext &bc) {
    using SA = sts::search::Action;
    using AT = sts::search::ActionType;
    if (bc.inputState != sts::InputState::CARD_SELECT
        || !bc.cardSelectInfo.isMultiSelection()) {
        return legalCombatActions(bc);
    }

    std::vector<SA> out;
    const int combinationCount = 1 << bc.cards.cardsInHand;
    for (int selectedBits = 0; selectedBits < combinationCount; ++selectedBits) {
        int selectedCount = 0;
        for (int bits = selectedBits; bits; bits >>= 1) {
            selectedCount += bits & 1;
        }
        if (bc.cardSelectInfo.cardSelectTask == sts::CardSelectTask::EXHAUST_MANY
            && selectedCount > bc.cardSelectInfo.pickCount) {
            continue;
        }
        out.emplace_back(AT::MULTI_CARD_SELECT, selectedBits);
    }
    return out;
}

template <PlayerStatus status>
void setRulePlayerStatus(Player &player, int amount) {
    player.setStatusValueNoChecks<status>(amount);
    player.setHasStatus<status>(amount != 0);
    if constexpr (status == PlayerStatus::COMBUST) {
        // Combust stores its HP payment outside statusMap. A directly injected
        // power must initialize both pieces just as playing the card does.
        player.combustHpLoss = amount != 0 ? 1 : 0;
    }
    if constexpr (
        status == PlayerStatus::DOUBLE_DAMAGE
        || status == PlayerStatus::FRAIL
        || status == PlayerStatus::INTANGIBLE
        || status == PlayerStatus::VULNERABLE
        || status == PlayerStatus::WEAK
    ) {
        player.setJustApplied<status>(false);
    }
}

void setRulePlayerStatusByName(
    BattleContext &bc, const std::string &name, int amount
) {
    if (amount < 0 && name != "STRENGTH" && name != "DEXTERITY") {
        throw std::invalid_argument("player status amount must be non-negative");
    }
#define SET_PLAYER_STATUS(value) \
    if (name == #value) { \
        setRulePlayerStatus<PlayerStatus::value>(bc.player, amount); \
        return; \
    }
    SET_PLAYER_STATUS(STRENGTH)
    SET_PLAYER_STATUS(DEXTERITY)
    SET_PLAYER_STATUS(WEAK)
    SET_PLAYER_STATUS(FRAIL)
    SET_PLAYER_STATUS(VULNERABLE)
    SET_PLAYER_STATUS(ARTIFACT)
    SET_PLAYER_STATUS(DOUBLE_DAMAGE)
    SET_PLAYER_STATUS(INTANGIBLE)
    SET_PLAYER_STATUS(BARRICADE)
    SET_PLAYER_STATUS(CORRUPTION)
    SET_PLAYER_STATUS(CONFUSED)
    SET_PLAYER_STATUS(HEX)
    SET_PLAYER_STATUS(DRAW_REDUCTION)
    SET_PLAYER_STATUS(BLASPHEMER)
    SET_PLAYER_STATUS(ELECTRO)
    SET_PLAYER_STATUS(SURROUNDED)
    SET_PLAYER_STATUS(MASTER_REALITY)
    SET_PLAYER_STATUS(PEN_NIB)
    SET_PLAYER_STATUS(WRATH_NEXT_TURN)
    SET_PLAYER_STATUS(NO_DRAW)
    SET_PLAYER_STATUS(LOSE_STRENGTH)
    SET_PLAYER_STATUS(DOUBLE_TAP)
    SET_PLAYER_STATUS(DUPLICATION)
    SET_PLAYER_STATUS(BRUTALITY)
    SET_PLAYER_STATUS(COMBUST)
    SET_PLAYER_STATUS(DARK_EMBRACE)
    SET_PLAYER_STATUS(DEMON_FORM)
    SET_PLAYER_STATUS(EVOLVE)
    SET_PLAYER_STATUS(FEEL_NO_PAIN)
    SET_PLAYER_STATUS(FIRE_BREATHING)
    SET_PLAYER_STATUS(FLAME_BARRIER)
    SET_PLAYER_STATUS(JUGGERNAUT)
    SET_PLAYER_STATUS(METALLICIZE)
    SET_PLAYER_STATUS(RAGE)
    SET_PLAYER_STATUS(REGEN)
    SET_PLAYER_STATUS(RUPTURE)
    SET_PLAYER_STATUS(THORNS)
    SET_PLAYER_STATUS(VIGOR)
#undef SET_PLAYER_STATUS
    throw std::invalid_argument("unsupported player status for rule data: " + name);
}

template <MonsterStatus status>
void setRuleMonsterStatus(Monster &monster, int amount) {
    monster.setStatus<status>(amount);
    if constexpr (status == MonsterStatus::TIME_WARP) {
        // Zero is the initial public clock value, not absence of the power.
        monster.setHasStatus<status>(true);
    } else {
        monster.setHasStatus<status>(amount != 0);
    }
    if constexpr (
        status == MonsterStatus::VULNERABLE
        || status == MonsterStatus::WEAK
        || status == MonsterStatus::RITUAL
    ) {
        monster.setJustApplied<status>(false);
    }
}

void setRuleMonsterStatusByName(
    BattleContext &bc, int target, const std::string &name, int amount
) {
    if (target < 0 || target >= bc.monsters.monsterCount) {
        throw std::out_of_range("monster target is out of range");
    }
    if (amount < 0 && name != "STRENGTH") {
        throw std::invalid_argument("monster status amount must be non-negative");
    }
    auto &monster = bc.monsters.arr[target];
#define SET_MONSTER_STATUS(value) \
    if (name == #value) { \
        setRuleMonsterStatus<MonsterStatus::value>(monster, amount); \
        return; \
    }
    SET_MONSTER_STATUS(STRENGTH)
    SET_MONSTER_STATUS(VULNERABLE)
    SET_MONSTER_STATUS(WEAK)
    SET_MONSTER_STATUS(ARTIFACT)
    SET_MONSTER_STATUS(METALLICIZE)
    SET_MONSTER_STATUS(PLATED_ARMOR)
    SET_MONSTER_STATUS(POISON)
    SET_MONSTER_STATUS(BLOCK_RETURN)
    SET_MONSTER_STATUS(CHOKED)
    SET_MONSTER_STATUS(CORPSE_EXPLOSION)
    SET_MONSTER_STATUS(LOCK_ON)
    SET_MONSTER_STATUS(MARK)
    SET_MONSTER_STATUS(REGEN)
    SET_MONSTER_STATUS(SHACKLED)
    SET_MONSTER_STATUS(ANGRY)
    SET_MONSTER_STATUS(BEAT_OF_DEATH)
    SET_MONSTER_STATUS(CURIOSITY)
    SET_MONSTER_STATUS(CURL_UP)
    SET_MONSTER_STATUS(ENRAGE)
    SET_MONSTER_STATUS(FADING)
    SET_MONSTER_STATUS(FLIGHT)
    SET_MONSTER_STATUS(GENERIC_STRENGTH_UP)
    SET_MONSTER_STATUS(INTANGIBLE)
    SET_MONSTER_STATUS(MALLEABLE)
    SET_MONSTER_STATUS(MODE_SHIFT)
    SET_MONSTER_STATUS(RITUAL)
    SET_MONSTER_STATUS(SLOW)
    SET_MONSTER_STATUS(SPORE_CLOUD)
    SET_MONSTER_STATUS(THIEVERY)
    SET_MONSTER_STATUS(THORNS)
    SET_MONSTER_STATUS(TIME_WARP)
    SET_MONSTER_STATUS(INVINCIBLE)
    SET_MONSTER_STATUS(REACTIVE)
    SET_MONSTER_STATUS(SHARP_HIDE)
    SET_MONSTER_STATUS(ASLEEP)
    SET_MONSTER_STATUS(BARRICADE)
    SET_MONSTER_STATUS(MINION)
    SET_MONSTER_STATUS(MINION_LEADER)
    SET_MONSTER_STATUS(PAINFUL_STABS)
    SET_MONSTER_STATUS(REGROW)
    SET_MONSTER_STATUS(SHIFTING)
    SET_MONSTER_STATUS(STASIS)
#undef SET_MONSTER_STATUS
    throw std::invalid_argument("unsupported monster status for rule data: " + name);
}

// the cards offered on a CARD_SELECT (grid) screen, in the sim's order
py::list selectCards(const sts::GameContext &gc) {
    py::list out;
    for (const auto &sc : gc.info.toSelectCards) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(sc.card.getId())], sc.card.getUpgraded()));
    }
    return out;
}

// reverse lookups: real-side strings -> sim enums (for force-syncing gc state
// to a real-run snapshot before a combat, decoupled from out-of-combat RNG).
sts::CardId cardIdFromEnum(const std::string &name) {
    constexpr int n = sizeof(sts::cardEnumStrings) / sizeof(sts::cardEnumStrings[0]);
    for (int i = 0; i < n; ++i) {
        if (name == sts::cardEnumStrings[i]) return static_cast<sts::CardId>(i);
    }
    throw std::runtime_error("unknown card enum: " + name);
}

sts::RelicId relicIdFromString(const std::string &name) {
    constexpr int n = sizeof(sts::relicIds) / sizeof(sts::relicIds[0]);
    for (int i = 0; i < n; ++i) {
        if (name == sts::relicIds[i]) return static_cast<sts::RelicId>(i);
    }
    throw std::runtime_error("unknown relic id: " + name);
}

void setDeck(sts::GameContext &gc, const std::vector<std::pair<std::string, int>> &cards) {
    gc.deck.cards.clear();
    gc.deck.cardTypeCounts = {0, 0, 0, 0};
    gc.deck.upgradeableCount = 0;
    gc.deck.transformableCount = 0;
    for (const auto &[name, up] : cards) {
        sts::Card card(cardIdFromEnum(name));
        for (int upgrade = 0; upgrade < up; ++upgrade) {
            card.upgrade();
        }
        gc.deck.obtainRaw(card);
    }
}

void setRelics(sts::GameContext &gc, const std::vector<std::string> &ids) {
    gc.relics.relics.clear();
    gc.relics.relicBits0 = gc.relics.relicBits1 = gc.relics.relicBits2 = 0;
    for (const auto &id : ids) {
        const auto relicId = relicIdFromString(id);
        const int data = relicId == sts::RelicId::LIZARD_TAIL ? 1 : 0;
        gc.relics.add(sts::RelicInstance{relicId, data});
    }
}

void setRelicState(sts::GameContext &gc, const py::list &items) {
    gc.relics.relics.clear();
    gc.relics.relicBits0 = gc.relics.relicBits1 = gc.relics.relicBits2 = 0;
    for (auto item : items) {
        std::string id;
        int data = 0;
        if (py::isinstance<py::str>(item)) id = item.cast<std::string>();
        else {
            auto d = item.cast<py::dict>();
            id = d["id"].cast<std::string>();
            if (d.contains("counter")) data = d["counter"].cast<int>();
        }
        gc.relics.add(sts::RelicInstance{relicIdFromString(id), data});
    }
}

// The sim's OWN out-of-combat action enumerator (search::GameAction) already
// solves legality for every overworld screen -- events (incl. phase-dependent
// ones like Golden Idol / Cursed Tome), campfires (relic/key-gated), shop,
// rewards, card-select -- with the same relic/gold/deck awareness the search
// agent uses. We expose just the option *indices* the model needs per screen,
// keyed the same way the thick layer already dispatches (chooseEventOption(i),
// chooseCampfireOption(i)). This is the out-of-combat analog of the bound
// combat `legalCombatActions`, and the authoritative fix for the event stub.

// valid event-option indices for the CURRENT event (getIdx1 of each action).
py::list legalEventOptions(const sts::GameContext &gc) {
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        out.append(a.getIdx1());
    }
    return out;
}

// Public, action-aligned details shown on event buttons whose exact sacrifice
// or service is selected when the event is created.  These values are already
// displayed by ConsoleSimulator; exporting them here keeps Python policies
// from having to guess an entity from the option index.  Other events return
// an empty list because their stable event id + option index is sufficient.
py::list eventOptionDetails(const sts::GameContext &gc) {
    py::list out;
    if (gc.screenState != sts::ScreenState::EVENT_SCREEN) {
        return out;
    }

    const auto cardTuple = [](const sts::Card &card) {
        return py::make_tuple(
            sts::cardEnumStrings[static_cast<int>(card.getId())],
            card.getUpgraded()
        );
    };
    const bool unfavorable = gc.ascension >= 15;
    for (const auto &action : sts::search::GameAction::getAllActionsInState(gc)) {
        const int idx = action.getIdx1();
        py::dict detail;
        detail["index"] = idx;
        switch (gc.curEvent) {
            case sts::Event::FALLING: {
                const int deckIdx = idx == 0 ? gc.info.skillCardDeckIdx
                                  : idx == 1 ? gc.info.powerCardDeckIdx
                                  : idx == 2 ? gc.info.attackCardDeckIdx : -1;
                if (idx == 3) {
                    detail["kind"] = "leave";
                } else if (deckIdx >= 0 && deckIdx < gc.deck.size()) {
                    detail["kind"] = "lose_card";
                    detail["card"] = cardTuple(gc.deck.cards[deckIdx]);
                } else {
                    throw std::runtime_error("Falling option has no displayed card");
                }
                break;
            }
            case sts::Event::NLOTH: {
                const int relicIdx = idx == 0 ? gc.info.relicIdx0
                                   : idx == 1 ? gc.info.relicIdx1 : -1;
                if (idx == 2) {
                    detail["kind"] = "leave";
                } else if (relicIdx >= 0 && relicIdx < gc.relics.size()) {
                    detail["kind"] = "lose_relic";
                    detail["relic"] = std::string(sts::relicIds[
                        static_cast<int>(gc.relics.relics[relicIdx].id)]);
                } else {
                    throw std::runtime_error("N'loth option has no displayed relic");
                }
                break;
            }
            case sts::Event::WE_MEET_AGAIN:
                if (idx == 0 && gc.info.potionIdx >= 0 &&
                        gc.info.potionIdx < gc.potionCapacity) {
                    detail["kind"] = "lose_potion";
                    detail["potion"] = std::string(
                        sts::getPotionName(gc.potions[gc.info.potionIdx]));
                } else if (idx == 1 && gc.info.gold >= 0) {
                    detail["kind"] = "lose_gold";
                    detail["gold"] = gc.info.gold;
                } else if (idx == 2 && gc.info.cardIdx >= 0 &&
                        gc.info.cardIdx < gc.deck.size()) {
                    detail["kind"] = "lose_card";
                    detail["card"] = cardTuple(gc.deck.cards[gc.info.cardIdx]);
                } else if (idx == 3) {
                    detail["kind"] = "leave";
                } else {
                    throw std::runtime_error("We Meet Again option is missing displayed details");
                }
                break;
            case sts::Event::NOTE_FOR_YOURSELF:
                if (idx == 0) {
                    detail["kind"] = "gain_card_then_remove";
                    detail["card"] = cardTuple(gc.noteForYourselfCard);
                } else if (idx == 1) {
                    detail["kind"] = "leave";
                } else {
                    throw std::runtime_error("Note For Yourself has an invalid option");
                }
                break;
            case sts::Event::DESIGNER_IN_SPIRE:
                if (idx == 0) {
                    detail["kind"] = "upgrade_card";
                    detail["gold"] = unfavorable ? 50 : 40;
                } else if (idx == 1) {
                    detail["kind"] = "upgrade_random_cards";
                    detail["gold"] = unfavorable ? 50 : 40;
                    detail["count"] = 2;
                } else if (idx == 2) {
                    detail["kind"] = "remove_card";
                    detail["gold"] = unfavorable ? 75 : 60;
                } else if (idx == 3) {
                    detail["kind"] = "transform_random_cards";
                    detail["gold"] = unfavorable ? 75 : 60;
                    detail["count"] = 2;
                } else if (idx == 4) {
                    detail["kind"] = "remove_card_upgrade_random";
                    detail["gold"] = unfavorable ? 110 : 90;
                    detail["count"] = 1;
                } else if (idx == 5) {
                    detail["kind"] = "lose_hp";
                    detail["hp"] = unfavorable ? 5 : 3;
                } else {
                    throw std::runtime_error("Designer has an invalid option");
                }
                break;
            default:
                continue;
        }
        out.append(detail);
    }
    return out;
}

// valid campfire-option indices (0=rest,1=smith,2=recall,3=lift,4=toke,5=dig,
// 6=skip) for the current REST_ROOM, gated by relics/keys exactly as the sim's
// own chooseCampfireOption expects.
py::list legalCampfireOptions(const sts::GameContext &gc) {
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        out.append(a.getIdx1());
    }
    return out;
}

// what is claimable on the current combat-reward screen
py::dict rewardInfo(const sts::GameContext &gc) {
    const auto &r = gc.info.rewardsContainer;
    py::dict d;
    py::list gold, relics, potions;
    for (int i = 0; i < r.goldRewardCount; ++i) gold.append(r.gold[i]);
    for (int i = 0; i < r.relicCount; ++i) relics.append(std::string(sts::relicIds[static_cast<int>(r.relics[i])]));
    for (int i = 0; i < r.potionCount; ++i) potions.append(std::string(sts::getPotionName(r.potions[i])));
    d["gold"] = gold;
    d["relics"] = relics;
    d["potions"] = potions;
    d["card_count"] = r.cardRewardCount;
    return d;
}

// The current shop's shelf, read POSITIONALLY (slot index is the buy key the
// shop_buy_* methods expect): cards[7], relics[3], potions[3]. A price of -1
// marks a sold/empty slot (Shop::cardPrice etc. return -1 after a purchase).
// Ids match the rest of the bindings: cardEnumStrings / relicIds / getPotionName.
py::dict shopContents(const sts::GameContext &gc) {
    const auto &s = gc.info.shop;
    py::dict d;
    py::list cards, relics, potions;
    for (int i = 0; i < 7; ++i) {
        py::dict c;
        c["id"] = std::string(sts::cardEnumStrings[static_cast<int>(s.cards[i].getId())]);
        c["upgrades"] = s.cards[i].getUpgraded();
        c["price"] = s.cardPrice(i);
        cards.append(c);
    }
    for (int i = 0; i < 3; ++i) {
        py::dict r;
        r["id"] = std::string(sts::relicIds[static_cast<int>(s.relics[i])]);
        r["price"] = s.relicPrice(i);
        relics.append(r);
    }
    for (int i = 0; i < 3; ++i) {
        py::dict p;
        p["id"] = std::string(sts::getPotionName(s.potions[i]));
        p["price"] = s.potionPrice(i);
        potions.append(p);
    }
    d["cards"] = cards;
    d["relics"] = relics;
    d["potions"] = potions;
    d["remove_cost"] = s.removeCost;
    return d;
}

// Authoritative legal shop options, from the sim's own getAllShopActions (gold-
// and sold-out-gated). Unlike legalEventOptions, idx1 COLLIDES across kinds
// (card#0 / relic#0 / potion#0 all have idx1==0), so we return (kind, idx)
// pairs. kind is a stable string keyed to shop_buy_*/shop_remove/shop_proceed:
// "card"/"relic"/"potion"/"remove"/"skip". "skip" always present.
py::list legalShopOptions(const sts::GameContext &gc) {
    using RAT = sts::search::GameAction::RewardsActionType;
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        const char *kind;
        switch (a.getRewardsActionType()) {
            case RAT::CARD:        kind = "card";   break;
            case RAT::RELIC:       kind = "relic";  break;
            case RAT::POTION:      kind = "potion"; break;
            case RAT::CARD_REMOVE: kind = "remove"; break;
            case RAT::SKIP:        kind = "skip";   break;
            default:               continue;
        }
        out.append(py::make_tuple(kind, a.getIdx1()));
    }
    return out;
}

// The OVERWORLD card-select (grid) screen's semantics + progress, for the obs
// select_context / pick_remaining / selected fields (obs_design §4.1). Read off
// gc.info (ScreenStateInfo): selectScreenType says WHAT this grid is for
// (TRANSFORM/UPGRADE/REMOVE/...), toSelectCount is how many must be picked, and
// haveSelectedCards are the ones already picked THIS screen (the sim removes a
// picked card from toSelectCards and pushes it here, so already-picked cards are
// NOT in get_select_cards() anymore -- we surface their ids here instead).
const char *selectScreenTypeStr(sts::CardSelectScreenType t) {
    using T = sts::CardSelectScreenType;
    switch (t) {
        case T::TRANSFORM:         return "TRANSFORM";
        case T::TRANSFORM_UPGRADE: return "TRANSFORM_UPGRADE";
        case T::UPGRADE:           return "UPGRADE";
        case T::REMOVE:            return "REMOVE";
        case T::DUPLICATE:         return "DUPLICATE";
        case T::OBTAIN:            return "OBTAIN";
        case T::BOTTLE:            return "BOTTLE";
        case T::BONFIRE_SPIRITS:   return "BONFIRE_SPIRITS";
        case T::INVALID:
        default:                   return "INVALID";
    }
}

py::dict selectScreenInfo(const sts::GameContext &gc) {
    const auto &info = gc.info;
    py::dict d;
    d["context"] = std::string(selectScreenTypeStr(info.selectScreenType));
    d["pick_count"] = info.toSelectCount;
    d["have_selected"] = static_cast<int>(info.haveSelectedCards.size());
    py::list sel;
    for (const auto &sc : info.haveSelectedCards) {
        sel.append(py::make_tuple(
            sts::cardEnumStrings[static_cast<int>(sc.card.getId())],
            sc.card.getUpgraded()
        ));
    }
    d["selected"] = sel;
    return d;
}

// The COMBAT card-select sub-screen's semantics + progress (bc.cardSelectInfo),
// for the same obs fields while in combat (Armaments/Headbutt/Gamble/...).
//
// Two structurally different combat card-selects exist:
//   * SINGLE (Armaments/Headbutt/...) -- one pick resolves the screen in one
//     action; the thin layer maps each candidate to a PICK_CARD id.
//   * MULTI  (Exhaust Many / Gamble)  -- the sim resolves the WHOLE screen with
//     one Action whose low 10 bits are a hand-index bitset (getSelectedIdxs() ->
//     chooseExhaustCards/chooseGambleCards). The thin layer therefore TOGGLES
//     hand indices across env steps and CONFIRMs once, building that bitset.
// `multi` lets the thin layer pick which machine to drive. For a MULTI screen we
// also expose `cards_in_hand` (the toggleable indices 0..n-1) and `pick_max`
// (the cap: Exhaust Many = pickCount limit; Gamble = the whole hand). pickCount
// is only meaningful for MULTI -- SINGLE tasks leave it 0 (see CardSelectInfo).
py::dict combatSelectInfo(const sts::BattleContext &bc) {
    const auto &csi = bc.cardSelectInfo;
    const auto task = csi.cardSelectTask;
    const bool multi = csi.isMultiSelection();
    py::dict d;
    d["context"] = std::string(
        sts::cardSelectTaskStrings[static_cast<int>(task)]);
    d["pick_count"] = csi.pickCount;
    d["can_pick_zero"] = csi.canPickZero;
    d["can_pick_any"] = csi.canPickAnyNumber;
    d["multi"] = multi;
    d["cards_in_hand"] = bc.cards.cardsInHand;
    // pick_max: hand indices a MULTI screen may toggle. Exhaust Many caps at its
    // limit (pickCount); Gamble has no limit beyond the hand. 0 for SINGLE.
    int pickMax = 0;
    if (task == sts::CardSelectTask::EXHAUST_MANY) {
        pickMax = csi.pickCount;
    } else if (multi) {
        pickMax = bc.cards.cardsInHand;
    }
    d["pick_max"] = pickMax;
    // Public choices on generated-card screens, in the action's source order.
    // Never read this array off-screen: it can contain uninitialized ids.
    py::list generated;
    if (bc.inputState == sts::InputState::CARD_SELECT &&
        (task == sts::CardSelectTask::CODEX || task == sts::CardSelectTask::DISCOVERY)) {
        for (auto id : csi.cards) {
            generated.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(id)], 0));
        }
    }
    d["generated_cards"] = generated;
    const auto choices = sts::search::combatChoices(bc);
    const char *zones[] = {"none", "hand", "draw", "discard", "exhaust", "generated", "grid"};
    d["zone"] = zones[choices.zone];
    py::list candidates;
    for (const auto &choice : choices.cards) {
        py::dict item;
        item["native_index"] = choice.nativeIndex;
        item["skip"] = choice.skip;
        item["card"] = py::make_tuple(sts::cardEnumStrings[static_cast<int>(choice.id)],
                                     choice.upgrades, choice.cost);
        candidates.append(item);
    }
    d["candidates"] = candidates;
    return d;
}

void setGeneratedCombatSelectionCards(
    sts::BattleContext &bc, const std::vector<std::string> &names
) {
    if (bc.inputState != sts::InputState::CARD_SELECT ||
        (bc.cardSelectInfo.cardSelectTask != sts::CardSelectTask::CODEX &&
         bc.cardSelectInfo.cardSelectTask != sts::CardSelectTask::DISCOVERY)) {
        throw std::invalid_argument(
            "generated combat choices require a live CODEX or DISCOVERY screen");
    }
    if (names.size() != bc.cardSelectInfo.cards.size()) {
        throw std::invalid_argument("generated combat choices require exactly three cards");
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
        bc.cardSelectInfo.cards[i] = cardIdFromEnum(names[i]);
    }
}

std::uint32_t remapDrawSelection(const sts::BattleContext &sampled,
                                const sts::BattleContext &source,
                                std::uint32_t actionBits) {
    using namespace sts;
    using namespace sts::search;
    const search::Action original(actionBits);
    const auto task = source.cardSelectInfo.cardSelectTask;
    if (source.outcome != Outcome::UNDECIDED || sampled.outcome != Outcome::UNDECIDED ||
        source.inputState != InputState::CARD_SELECT ||
        sampled.inputState != InputState::CARD_SELECT ||
        sampled.cardSelectInfo.cardSelectTask != task ||
        (task != CardSelectTask::SECRET_TECHNIQUE &&
         task != CardSelectTask::SECRET_WEAPON && task != CardSelectTask::SEEK) ||
        original.getActionType() != ActionType::SINGLE_CARD_SELECT ||
        !original.isValidAction(source)) {
        throw std::invalid_argument("draw-selection remap requires matching live draw-select screens and a legal source action");
    }
    const auto &card = source.cards.drawPile[original.getSelectIdx()];
    const auto uid = card.getUniqueId();
    const auto sameIdentity = [uid](const CardInstance &other) {
        return other.getUniqueId() == uid;
    };
    if (uid < 0 || std::count_if(source.cards.drawPile.begin(), source.cards.drawPile.end(), sameIdentity) != 1 ||
        std::count_if(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), sameIdentity) != 1) {
        throw std::invalid_argument("draw-selection remap requires one matching card entity in each pile");
    }
    const auto found = std::find_if(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), sameIdentity);
    if (card.id != found->id || card.specialData != found->specialData ||
        card.upgraded != found->upgraded || card.cost != found->cost ||
        card.costForTurn != found->costForTurn || card.freeToPlayOnce != found->freeToPlayOnce ||
        card.retain != found->retain) {
        throw std::invalid_argument("draw-selection entity changed after sampling");
    }
    const search::Action mapped(ActionType::SINGLE_CARD_SELECT,
                        static_cast<int>(found - sampled.cards.drawPile.begin()));
    if (!mapped.isValidAction(sampled)) {
        throw std::invalid_argument("remapped draw selection is not legal in sampled state");
    }
    return mapped.bits;
}

pybind11::dict packedNeuralLeaves(
    const std::vector<sts::search::NoslNeuralSearchSession::LeafRequest> &requests,
    bool done, pybind11::object language_card_index) {
    // Keep the simulator-to-Python boundary to contiguous numeric
    // arrays. Do not call rawCombatState() here: that
    // recursively allocates Python dicts/lists for every leaf.
    const auto count = static_cast<pybind11::ssize_t>(requests.size());
    std::size_t maxLegalActions = 0;
    for (const auto &request : requests) {
        maxLegalActions = std::max(
            maxLegalActions, request.legalActions.size());
    }
    const auto legalWidth =
        static_cast<pybind11::ssize_t>(maxLegalActions);
    pybind11::array_t<std::int32_t> categorical(
        std::vector<pybind11::ssize_t>{count, compact_leaf::kCatSize});
    pybind11::array_t<float> scalars(
        std::vector<pybind11::ssize_t>{count, compact_leaf::kFloatSize});
    const bool language = !language_card_index.is_none();
    pybind11::array_t<std::int32_t> language_index;
    if (language) {
        language_index = pybind11::array_t<std::int32_t,
            pybind11::array::c_style | pybind11::array::forcecast>(
                language_card_index);
        if (language_index.ndim() != 1 ||
            language_index.shape(0) != compact_leaf::kCardIds) {
            throw std::invalid_argument("language card index shape mismatch");
        }
    }
    pybind11::array_t<std::int32_t> language_ids;
    pybind11::array_t<float> language_features;
    if (language) {
        language_ids = pybind11::array_t<std::int32_t>(
            std::vector<pybind11::ssize_t>{count, 2, compact_leaf::kLanguageSlots});
        language_features = pybind11::array_t<float>(
            std::vector<pybind11::ssize_t>{count, 2, compact_leaf::kLanguageSlots, 3});
        std::fill(language_ids.mutable_data(),
                  language_ids.mutable_data() + language_ids.size(), 0);
        std::fill(language_features.mutable_data(),
                  language_features.mutable_data() + language_features.size(), 0.0f);
    }
    auto cat = categorical.mutable_unchecked<2>();
    auto scalar = scalars.mutable_unchecked<2>();
    pybind11::list request_ids;
    pybind11::array_t<std::uint32_t> legal_actions(
        std::vector<pybind11::ssize_t>{count, legalWidth});
    pybind11::array_t<std::uint32_t> legal_action_counts(
        std::vector<pybind11::ssize_t>{count});
    if (legal_actions.size() > 0) {
        std::fill(
            legal_actions.mutable_data(),
            legal_actions.mutable_data() + legal_actions.size(),
            std::uint32_t{0});
    }
    pybind11::array_t<std::int32_t> legal_policy_ids(
        std::vector<pybind11::ssize_t>{count, legalWidth});
    std::fill(legal_policy_ids.mutable_data(), legal_policy_ids.mutable_data() + legal_policy_ids.size(), -1);
    auto policy_ids = legal_policy_ids.mutable_unchecked<2>();
    auto legal = legal_actions.mutable_unchecked<2>();
    auto legal_count = legal_action_counts.mutable_unchecked<1>();
    pybind11::list needs_policy;
    pybind11::list selection_prefix;
    for (pybind11::ssize_t row = 0; row < count; ++row) {
        const auto &request = requests[static_cast<std::size_t>(row)];
        const auto packed = compact_leaf::pack(*request.state);
        for (int col = 0; col < compact_leaf::kCatSize; ++col) {
            cat(row, col) = packed.categorical[static_cast<std::size_t>(col)];
        }
        for (int col = 0; col < compact_leaf::kFloatSize; ++col) {
            scalar(row, col) = packed.scalars[static_cast<std::size_t>(col)];
        }
        if (language) {
            auto *ids = language_ids.mutable_data() +
                row * 2 * compact_leaf::kLanguageSlots;
            auto *features = language_features.mutable_data() +
                row * 2 * compact_leaf::kLanguageSlots * 3;
            compact_leaf::packLanguagePile(request.state->cards.drawPile,
                language_index.data(), ids, features);
            compact_leaf::packLanguagePile(request.state->cards.discardPile,
                language_index.data(), ids + compact_leaf::kLanguageSlots,
                features + compact_leaf::kLanguageSlots * 3);
        }
        request_ids.append(request.requestId);
        const auto nLegal = request.legalActions.size();
        legal_count(row) = static_cast<std::uint32_t>(nLegal);
        for (std::size_t col = 0; col < nLegal; ++col) {
            legal(row, static_cast<pybind11::ssize_t>(col)) =
                request.legalActions[col].bits;
            policy_ids(row, static_cast<pybind11::ssize_t>(col)) =
                sts::search::jointActionId(request.legalActions[col], *request.state)
                    - (request.autoregressivePolicy ? sts::search::kJointEnd : 0);
        }
        needs_policy.append(request.needsPolicy);
        selection_prefix.append(request.selectionPrefix);
    }
    pybind11::dict out;
    out["request_ids"] = request_ids;
    out["categorical"] = categorical;
    out["scalars"] = scalars;
    if (language) {
        out["language_ids"] = language_ids;
        out["language_features"] = language_features;
    }
    out["legal_action_bits"] = legal_actions;
    out["legal_action_counts"] = legal_action_counts;
    out["legal_policy_ids"] = legal_policy_ids;
    out["needs_policy"] = needs_policy;
    out["selection_prefix"] = selection_prefix;
    out["done"] = done;
    return out;
}

std::vector<search::Action> completeRuleCombatActions(const BattleContext &bc) {
    auto out=legalRuleCombatActions(bc);
    if (bc.outcome==Outcome::UNDECIDED && bc.inputState==InputState::PLAYER_NORMAL) {
        for (int slot=0;slot<bc.potionCapacity;++slot) {
            const auto potion=bc.potions[slot];
            if (potion==Potion::INVALID || potion==Potion::EMPTY_POTION_SLOT) continue;
            const int targets=potionRequiresTarget(potion) ? bc.monsters.monsterCount:1;
            for (int target=0;target<targets;++target) {
                search::Action a(search::ActionType::POTION,slot,target);
                if (a.isValidAction(bc)) out.push_back(a);
            }
        }
    }
    std::vector<search::Action> unique;
    for (auto a:out) if (std::find(unique.begin(),unique.end(),a)==unique.end()) unique.push_back(a);
    return unique;
}

py::dict ruleTurnResult(const search::ruleturn::Result &r,int worlds) {
    const auto &s=r.stats;
    py::dict d,values,outcomes;
    for (const auto &root:r.roots) {
        const auto key=std::to_string(root.bits);
        values[py::str(key)]=root.mean;
        py::dict o;
        o["deaths"]=root.deaths; o["wins"]=root.wins;
        o["live"]=worlds-root.deaths-root.wins; o["unresolved"]=root.unresolved;
        o["boundary_live"]=worlds-root.deaths-root.wins-root.unresolved;
        o["heuristic_mean"]=root.mean; o["winning_steps_total"]=root.winningSteps;
        o["winning_steps_mean"]=root.wins ? double(root.winningSteps)/root.wins:0.;
        outcomes[py::str(key)]=o;
    }
    d["native_transitions"]=s.nativeTransitions; d["expanded_nodes"]=s.expandedNodes;
    d["worlds_completed"]=worlds; d["budget_exhausted"]=s.budgetExhausted;
    d["effective_node_budget"]=r.budget; d["effective_total_budget"]=r.budget*worlds;
    d["root_actions"]=r.roots.size(); d["fallback_leaves"]=s.fallbackLeaves;
    d["second_turn_entries"]=s.secondTurnEntries; d["completed_horizon_leaves"]=s.completedHorizonLeaves;
    d["max_action_depth"]=s.maxDepth; d["root_values"]=values; d["root_outcomes"]=outcomes;
    d["search_seconds"]=r.seconds; d["search_decisions"]=1;
    py::dict out; out["action_bits"]=r.action; out["diagnostics"]=d; return out;
}

} // namespace

PYBIND11_MODULE(slaythespire, m) {
    m.def("boss_opening_route_version", []() { return 1; });
    bindNeuralObservation(m);
    pybind11::class_<sts::search::NeuralSearchBatch>(m, "NeuralSearchBatch")
        .def(pybind11::init<int>(), pybind11::arg("threads") = 1)
        .def("rule_turn_plan", [](sts::search::NeuralSearchBatch &batch,
                std::vector<const BattleContext *> battles,
                const std::vector<std::vector<std::uint64_t>> &nonces,
                const std::vector<std::vector<std::string>> &relics,
                int worlds,int nodes,int beam,int depth,int turns,const std::string &profile) {
            auto ownership=batch.acquire();
            if (worlds<1 || nodes<1 || beam<1 || depth<1 || (turns!=1 && turns!=2) ||
                (profile!="legacy-v8" && profile!="time-clock-v9") ||
                battles.size()!=nonces.size() || battles.size()!=relics.size())
                throw std::invalid_argument("invalid native rule-search batch/config");
            std::vector<std::unique_ptr<BattleContext>> copies;
            std::vector<bool> outputRelics;
            const std::unordered_set<std::string> outputs={"mercuryhourglass","bronzescales","charonsashes",
                "letteropener","stonecalendar","deadbranch","nilryscodex","tingsha","toolbox"};
            for (std::size_t i=0;i<battles.size();++i) {
                if (!battles[i] || nonces[i].size()!=static_cast<std::size_t>(worlds))
                    throw std::invalid_argument("one nonce per world and a nonnull battle are required");
                copies.push_back(std::make_unique<BattleContext>(*battles[i]));
                bool output=false;
                for (const auto &relic:relics[i]) {
                    std::string key;
                    for (unsigned char c:relic) if (std::isalnum(c)) key+=char(std::tolower(c));
                    output|=outputs.count(key)>0;
                }
                outputRelics.push_back(output);
            }
            const sts::search::ruleturn::Planner planner{
                {worlds,nodes,beam,depth,turns,profile=="time-clock-v9"},
                completeRuleCombatActions,remapDrawSelection,playerPowerObservation};
            std::vector<sts::search::ruleturn::Result> results(battles.size());
            {
                pybind11::gil_scoped_release release;
                batch.runIndependent(results.size(),[&](std::size_t i){
                    results[i]=planner.choose(*copies[i],nonces[i],outputRelics[i]);
                });
            }
            pybind11::list out;
            for (const auto &r:results) out.append(ruleTurnResult(r,worlds));
            return out;
        },pybind11::arg("battles"),pybind11::arg("nonces"),pybind11::arg("relics"),
          pybind11::arg("worlds"),pybind11::arg("nodes"),pybind11::arg("beam"),
          pybind11::arg("depth"),pybind11::arg("turns"),pybind11::arg("profile"))
        .def("begin", [](sts::search::NeuralSearchBatch &batch,
                          std::vector<std::shared_ptr<sts::search::NoslNeuralSearchSession>> sessions,
                          pybind11::object language_card_index) {
            auto ownership = batch.acquire();
            const auto started = std::chrono::steady_clock::now();
            {
                pybind11::gil_scoped_release release;
                batch.begin(std::move(sessions));
            }
            const auto selected = std::chrono::steady_clock::now();
            auto result = packedNeuralLeaves(batch.requests(), batch.requests().empty(), language_card_index);
            result["owners"] = batch.owners();
            result["native_seconds"] = std::chrono::duration<double>(selected-started).count();
            result["pack_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-selected).count();
            return result;
        }, pybind11::arg("sessions"), pybind11::arg("language_card_index") = pybind11::none())
        .def("advance", [](sts::search::NeuralSearchBatch &batch,
                            pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> values,
                            pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> logits,
                            pybind11::object language_card_index) {
            auto ownership = batch.acquire();
            if (values.ndim() != 1 || logits.ndim() != 2 || values.shape(0) != logits.shape(0))
                throw std::invalid_argument("native search predictions require values[N], logits[N,K]");
            const auto rows = static_cast<std::size_t>(values.shape(0));
            const auto width = static_cast<std::size_t>(logits.shape(1));
            const auto *v = values.data();
            const auto *l = logits.data();
            const auto started = std::chrono::steady_clock::now();
            {
                pybind11::gil_scoped_release release;
                batch.advance(v, l, rows, width);
            }
            const auto selected = std::chrono::steady_clock::now();
            auto result = packedNeuralLeaves(batch.requests(), batch.requests().empty(), language_card_index);
            result["owners"] = batch.owners();
            result["native_seconds"] = std::chrono::duration<double>(selected-started).count();
            result["pack_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-selected).count();
            return result;
        }, pybind11::arg("values"), pybind11::arg("logits"),
           pybind11::arg("language_card_index") = pybind11::none())
        .def("clear", &sts::search::NeuralSearchBatch::clear)
        .def_property_readonly("threads", &sts::search::NeuralSearchBatch::threadCount);
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");
    m.def("nosl_compact_layout", []() {
        // Names are exported once so the Python adapter can build native-id
        // -> canonical-vocab LUTs without asking C++ to serialize each leaf.
        pybind11::dict out;
        pybind11::list player_status;
        for (int i = 1; i <= compact_leaf::kPlayerStatus; ++i) {
            player_status.append(playerStatusStrings[i]);
        }
        pybind11::list monster_status;
        for (int i = 0; i < compact_leaf::kMonsterStatus; ++i) {
            monster_status.append(sts::enemyStatusStrings[i]);
        }
        pybind11::list monsters;
        for (int i = 0; i < compact_leaf::kNativeMonsters; ++i) {
            monsters.append(sts::monsterIdStrings[i]);
        }
        pybind11::list potions;
        for (int i = 0; i < compact_leaf::kNativePotions; ++i) {
            potions.append(sts::potionNames[i]);
        }
        pybind11::list cards;
        for (int i = 0; i < compact_leaf::kCardIds; ++i) {
            cards.append(sts::cardEnumStrings[i]);
        }
        out["player_status_names"] = player_status;
        out["monster_status_names"] = monster_status;
        out["monster_names"] = monsters;
        out["potion_names"] = potions;
        out["card_names"] = cards;
        out["player_status_count"] = compact_leaf::kPlayerStatus;
        out["monster_status_count"] = compact_leaf::kMonsterStatus;
        out["card_count"] = compact_leaf::kCardIds;
        out["version"] = 3;
        out["observation_version"] = 19;
        out["language_slots"] = compact_leaf::kLanguageSlots;
        out["candidate_capacity"] = 96;
        out["candidate_overflow_contract"] = "overflow-equivalent-v1";
        out["joint_policy_size"] = sts::search::kJointSize;
        out["autoregressive_policy_size"] = sts::search::kAutoregressiveSize;
        out["categorical_width"] = compact_leaf::kCatSize;
        out["scalar_width"] = compact_leaf::kFloatSize;
        return out;
    }, "return the versioned native NOSL compact-observation layout");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout);

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def("choose_neow_option",
             [](GameContext &gc, int idx) { gc.chooseNeowOption(gc.info.neowRewards[idx]); },
             "apply the Neow reward option at the given index (0-3)")
        .def("get_neow_options", [](const GameContext &gc) {
            pybind11::list out;
            for (std::size_t idx = 0; idx < gc.info.neowRewards.size(); ++idx) {
                const auto &option = gc.info.neowRewards[idx];
                pybind11::dict item;
                item["index"] = idx;
                item["bonus_id"] = static_cast<int>(option.r);
                item["drawback_id"] = static_cast<int>(option.d);
                item["bonus"] = Neow::bonusStrings[static_cast<int>(option.r)];
                item["drawback"] = Neow::drawbackStrings[static_cast<int>(option.d)];
                out.append(item);
            }
            return out;
        }, "return the four generated Neow rewards and their drawbacks")
        .def("transition_to_map_node",
             [](GameContext &gc, int x) { gc.transitionToMapNode(x); },
             "advance to the map node at column x on the next row (enters that room)")
        .def("_dbg_set_card_rng", [](GameContext &gc, int n) { gc.cardRng = sts::Random(gc.seed, n); })
        .def("_dbg_card_reward", [](GameContext &gc) {
            auto cr = gc.createCardReward(sts::Room::MONSTER);
            std::vector<std::pair<std::string,int>> out;
            for (auto &c : cr) out.emplace_back(sts::cardEnumStrings[(int)c.getId()], c.getUpgraded());
            return out;
        })
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        // ----- out-of-combat replay/diff (added for full-run replay) -----
        .def("deck_ids", &deckIds, "deck as [(sim_card_enum, upgrade_count), ...]")
        .def("relic_ids", &gcRelicIds, "relics as [stable_id_string, ...]")
        .def("relic_state", &gcRelics, "relics as [{id, counter}, ...]")
        .def("set_deck", &setDeck, "replace the deck with [(sim_card_enum, upgrade_count), ...]")
        .def("set_relics", &setRelics, "replace the relics with [stable_id_string, ...]")
        .def("set_relic_state", &setRelicState, "replace relics with [{id, counter}, ...]")
        .def("set_potion_state", [](GameContext &gc, const std::vector<std::string> &names) {
            if (names.size() > static_cast<std::size_t>(gc.potionCapacity)) {
                throw std::invalid_argument("custom potions exceed potion capacity");
            }
            std::fill(gc.potions.begin(), gc.potions.end(), Potion::EMPTY_POTION_SLOT);
            gc.potionCount = 0;
            for (std::size_t slot = 0; slot < names.size(); ++slot) {
                gc.potions[slot] = potionFromNameOrId(names[slot]);
                if (gc.potions[slot] != Potion::EMPTY_POTION_SLOT) ++gc.potionCount;
            }
        }, "replace held potion slots by simulator name or stable game id without advancing the run")
        .def_property_readonly("screen_state_name",
               [](const GameContext &gc) { return std::string(pybind11::str(pybind11::cast(gc.screenState))); })
        .def_property_readonly("cur_event_name",
               [](const GameContext &gc) { return std::string(sts::eventIdStrings[static_cast<int>(gc.curEvent)]); })
        .def_property_readonly("burning_elite_x", [](const GameContext &gc) { return gc.map ? gc.map->burningEliteX : -1; })
        .def_property_readonly("burning_elite_y", [](const GameContext &gc) { return gc.map ? gc.map->burningEliteY : -1; })
        // decision dispatch (mirrors GameContext's own screen state machine)
        .def("choose_event_option", [](GameContext &gc, int i) { gc.chooseEventOption(i); })
        .def("choose_campfire_option", [](GameContext &gc, int i) { gc.chooseCampfireOption(i); })
        .def("legal_event_options", &legalEventOptions,
             "valid option indices for the current event (sim-authoritative, phase/relic aware)")
        .def("event_option_details", &eventOptionDetails,
             "public action-aligned details for event buttons with dynamic displayed candidates")
        .def("legal_campfire_options", &legalCampfireOptions,
             "valid campfire option indices (0=rest,1=smith,2=recall,3=lift,4=toke,5=dig,6=skip)")
        .def("choose_select_card_option", [](GameContext &gc, int i) { gc.chooseSelectCardScreenOption(i); })
        .def("choose_treasure_room", [](GameContext &gc, bool open) { gc.chooseTreasureRoomOption(open); })
        .def("choose_boss_relic", [](GameContext &gc, int i) { gc.chooseBossRelic(i); })
        .def("get_select_cards", &selectCards, "cards offered on the current CARD_SELECT/grid screen")
        .def("select_screen_info", &selectScreenInfo,
             "overworld card-select screen semantics+progress: {context (CardSelectScreenType "
             "name), pick_count, have_selected, selected:[card ids already picked this screen]}")
        // combat-reward screen
        .def("get_reward_info", &rewardInfo, "claimable items on the current REWARDS screen")
        .def("claim_gold", [](GameContext &gc, int idx) { auto &r = gc.info.rewardsContainer; gc.obtainGold(r.gold[idx]); r.removeGoldReward(idx); })
        .def("claim_relic", [](GameContext &gc, int idx) {
            auto &r = gc.info.rewardsContainer;
            // Some relics open a screen when obtained (e.g. Bottled Tornado's
            // "pick a Power to bottle"). That screen's confirm calls
            // regainControl(), which would otherwise run -- and consume -- the
            // combat's return-to-map action, leaving it null and crashing the
            // later PROCEED. Redirect the screen's regainControl back to the
            // REWARDS screen (more rewards may remain) and restore the original
            // action so the final PROCEED still returns to the map.
            auto saved = gc.regainControlAction;
            gc.regainControlAction = [saved](GameContext &g) {
                g.screenState = ScreenState::REWARDS;
                g.regainControlAction = saved;
            };
            bool opened = gc.obtainRelic(r.relics[idx]);
            r.removeRelicReward(idx);
            if (!opened) gc.regainControlAction = saved;
        })
        .def("claim_potion", [](GameContext &gc, int idx) { auto &r = gc.info.rewardsContainer; gc.obtainPotion(r.potions[idx]); r.removePotionReward(idx); })
        .def("discard_potion", [](GameContext &gc, int slot) {
            if (slot >= 0 && slot < gc.potionCapacity && gc.potions[slot] != sts::Potion::EMPTY_POTION_SLOT) {
                gc.potions[slot] = sts::Potion::EMPTY_POTION_SLOT;
                --gc.potionCount;
            }
        }, "discard the held potion in slot idx (0-based), out of combat")
        .def("drink_potion", [](GameContext &gc, int slot) { gc.drinkPotionAtIdx(slot); },
             pybind11::arg("idx"),
             "drink the held potion in slot idx (0-based) OUT OF COMBAT (Blood/Entropic "
             "Brew/Fruit Juice etc.); use BattleContext.drink_potion in combat")
        .def("legal_overworld_potions", &legalOverworldPotions,
             "authoritative legal out-of-combat potion DRINKS as [{slot, requires_target}]; "
             "requires_target is always false out of combat (no targets). sim-gated by "
             "GameAction::isValidPotionAction (only a few potions drink outside combat)")
        .def_property_readonly("overworld_potions", [](const GameContext &gc) {
            pybind11::list lst;
            for (int i = 0; i < gc.potionCapacity; ++i) lst.append(std::string(sts::getPotionName(gc.potions[i])));
            return lst;
        }, "held potion slots out of combat")
        .def("reward_proceed", [](GameContext &gc) { gc.regainControl(); })
        // shop
        .def("shop_buy_card", [](GameContext &gc, int idx) { gc.info.shop.buyCard(gc, idx); })
        .def("shop_buy_relic", [](GameContext &gc, int idx) { gc.info.shop.buyRelic(gc, idx); })
        .def("shop_buy_potion", [](GameContext &gc, int idx) { gc.info.shop.buyPotion(gc, idx); })
        .def("shop_remove", [](GameContext &gc) { gc.info.shop.buyCardRemove(gc); })
        .def("shop_proceed", [](GameContext &gc) { gc.screenState = ScreenState::MAP_SCREEN; })
        .def("shop_contents", &shopContents,
             "the current shop shelf, positional: {cards:[{id,upgrades,price}x7], "
             "relics:[{id,price}x3], potions:[{id,price}x3], remove_cost}. price -1 == sold/empty")
        .def("legal_shop_options", &legalShopOptions,
             "authoritative legal shop actions as (kind, idx) pairs; kind in "
             "{card,relic,potion,remove,skip} (sim-gated by gold/sold-out)")
        // obs globals: character + ascension are construction-time, boss_relic_ids
        // are the 3 choices offered on a BOSS_RELIC_REWARDS screen (info.bossRelics).
        .def_property_readonly("character",
            [](const GameContext &gc) {
                switch (gc.cc) {
                    case CharacterClass::IRONCLAD: return "IRONCLAD";
                    case CharacterClass::SILENT:   return "SILENT";
                    case CharacterClass::DEFECT:   return "DEFECT";
                    case CharacterClass::WATCHER:  return "WATCHER";
                    default:                       return "INVALID";
                }
            }, "the player's character class as a stable enum-name string")
        .def_readwrite("ascension", &GameContext::ascension)
        .def_readwrite("skip_battles", &GameContext::skipBattles)
        .def("boss_relic_ids", [](const GameContext &gc) {
            pybind11::list out;
            for (int i = 0; i < 3; ++i)
                out.append(std::string(sts::relicIds[static_cast<int>(gc.info.bossRelics[i])]));
            return out;
        }, "the 3 boss relics offered on the current boss-relic screen, as stable ids")
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
//        .def_readwrite("cur_event", &GameContext::curEvent) // todo standardize event names
        .def_readwrite("boss", &GameContext::boss)
        .def_readwrite("second_boss", &GameContext::secondBoss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard);

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    // ===== combat bindings (added for diff testing / RL) =====
    pybind11::enum_<search::ActionType>(m, "ActionType")
        .value("CARD", search::ActionType::CARD)
        .value("POTION", search::ActionType::POTION)
        .value("SINGLE_CARD_SELECT", search::ActionType::SINGLE_CARD_SELECT)
        .value("MULTI_CARD_SELECT", search::ActionType::MULTI_CARD_SELECT)
        .value("END_TURN", search::ActionType::END_TURN);

    pybind11::class_<search::Action>(m, "Action")
        .def(pybind11::init<search::ActionType>())
        .def(pybind11::init<search::ActionType, int>())
        .def(pybind11::init<search::ActionType, int, int>())
        .def(pybind11::init<std::uint32_t>())
        .def_property_readonly("action_type", &search::Action::getActionType)
        .def_property_readonly("source_idx", &search::Action::getSourceIdx)
        .def_property_readonly("target_idx", &search::Action::getTargetIdx)
        .def_property_readonly("selected_indices", [](const search::Action &a) {
            const auto selected = a.getSelectedIdxs();
            return std::vector<int>(selected.begin(), selected.end());
        })
        .def("__repr__", [](const search::Action &a) {
            std::ostringstream oss;
            oss << "<Action type=" << static_cast<int>(a.getActionType())
                << " src=" << a.getSourceIdx() << " tgt=" << a.getTargetIdx() << ">";
            return oss.str();
        });

    // Incremental neural NOSL/AlphaZero search.  The C++ session keeps all
    // hidden-state clones and tree statistics; Python only exchanges public
    // leaf states and batched policy/value predictions.
    pybind11::class_<search::NoslNeuralSearchSession,
                     std::shared_ptr<search::NoslNeuralSearchSession>>(
        m, "NoslNeuralSearchSession")
        .def("next_batch", [](search::NoslNeuralSearchSession &session,
                               int max_leaves) {
            pybind11::list out;
            for (const auto &request : session.nextBatch(max_leaves)) {
                pybind11::dict item;
                item["request_id"] = request.requestId;
                auto state = rawCombatState(*request.state, !request.autoregressivePolicy);
                if (request.autoregressivePolicy) {
                    std::vector<int> ids;
                    for (const auto &action : request.legalActions)
                        ids.push_back(sts::search::jointActionId(action, *request.state) - sts::search::kJointEnd);
                    state["joint_action_ids"] = ids;
                    state["selection_prefix"] = request.selectionPrefix;
                }
                item["state"] = state;
                std::vector<std::uint32_t> legal;
                legal.reserve(request.legalActions.size());
                for (const auto &action : request.legalActions) {
                    legal.push_back(action.bits);
                }
                item["legal_action_bits"] = legal;
                item["needs_policy"] = request.needsPolicy;
                out.append(item);
            }
            return out;
         }, pybind11::arg("max_leaves") = 16,
            "select a batch of NOSL leaf states using virtual visits")
        .def("next_batch_packed", [](search::NoslNeuralSearchSession &session,
                                      int max_leaves,
                                      pybind11::object language_card_index) {
            const auto requests = session.nextBatch(max_leaves);
            return packedNeuralLeaves(requests, requests.empty() && session.done(), language_card_index);
        }, pybind11::arg("max_leaves") = 16,
           pybind11::arg("language_card_index") = pybind11::none(),
           "select leaves into contiguous native categorical/scalar arrays")
        .def("backprop", [](search::NoslNeuralSearchSession &session,
                            std::uint64_t request_id,
                            double value,
                            pybind11::array_t<double,
                                pybind11::array::c_style |
                                pybind11::array::forcecast> policy_logits) {
            std::vector<double> logits;
            auto view = policy_logits.unchecked<1>();
            logits.reserve(static_cast<std::size_t>(view.shape(0)));
            for (pybind11::ssize_t i = 0; i < view.shape(0); ++i) {
                logits.push_back(view(i));
            }
            session.backprop(request_id, value, logits);
        }, pybind11::arg("request_id"), pybind11::arg("value"),
           pybind11::arg("policy_logits"),
           "install leaf policy/value and backpropagate one simulation")
        .def("backprop_batch",
             [](search::NoslNeuralSearchSession &session,
                const std::vector<std::uint64_t> &request_ids,
                const std::vector<double> &values,
                const std::vector<std::vector<double>> &policy_logits) {
                 session.backpropBatch(request_ids, values, policy_logits);
             },
             pybind11::arg("request_ids"),
             pybind11::arg("values"),
             pybind11::arg("policy_logits"),
             "install and backpropagate a complete leaf batch")
        .def("backprop_batch_array",
             [](search::NoslNeuralSearchSession &session,
                pybind11::array_t<std::uint64_t,
                    pybind11::array::c_style | pybind11::array::forcecast>
                    request_ids,
                pybind11::array_t<double,
                    pybind11::array::c_style | pybind11::array::forcecast>
                    values,
                pybind11::array_t<double,
                    pybind11::array::c_style | pybind11::array::forcecast>
                    policy_logits) {
                 const auto ids = request_ids.unchecked<1>();
                 const auto vals = values.unchecked<1>();
                 const auto logits = policy_logits.unchecked<2>();
                 const auto count = ids.shape(0);
                 if (vals.shape(0) != count || logits.shape(0) != count) {
                     throw std::invalid_argument(
                         "NOSL backprop array batch lengths differ");
                 }
                 const auto width = logits.shape(1);
                 session.backpropBatchArray(
                     request_ids.data(0), vals.data(0), logits.data(0, 0),
                     static_cast<std::size_t>(count),
                     static_cast<std::size_t>(width));
             },
             pybind11::arg("request_ids"),
             pybind11::arg("values"),
             pybind11::arg("policy_logits"),
             "install a contiguous 2-D policy/value leaf batch")
        .def("done", &search::NoslNeuralSearchSession::done)
        .def("simulations_completed",
             &search::NoslNeuralSearchSession::simulationsCompleted)
        .def("result", [](const search::NoslNeuralSearchSession &session) {
            const auto decision = session.result();
            pybind11::dict result;
            result["action_bits"] = decision.action.bits;
            result["total_root_visits"] = decision.totalRootVisits;
            result["root_value"] = decision.rootValue;
            result["root_visits"] = decision.rootVisits;
            result["root_policy"] = decision.rootPolicy;
            result["selected_target"] = decision.selectedTarget;
            result["selected_action_targeted"] = decision.selectedActionTargeted;
            result["simulations"] = decision.totalRootVisits;
            return result;
        })
        .def("tree_snapshot", [](const search::NoslNeuralSearchSession &session,
                                  pybind11::sequence path,
                                  int max_children) {
            std::vector<int> indexes;
            indexes.reserve(static_cast<std::size_t>(pybind11::len(path)));
            for (const auto item : path) {
                indexes.push_back(pybind11::cast<int>(item));
            }
            const auto snapshot = session.treeSnapshot(indexes, max_children);
            pybind11::dict result;
            result["valid"] = snapshot.valid;
            result["visits"] = snapshot.visits;
            result["value_visits"] = snapshot.valueVisits;
            result["value"] = snapshot.value;
            result["sample_value"] = snapshot.sampleValue;
            result["network_value_valid"] = snapshot.networkValueValid;
            result["network_value"] = snapshot.networkValue;
            pybind11::list children;
            for (const auto &edge : snapshot.children) {
                pybind11::dict item;
                item["edge_index"] = edge.edgeIndex;
                item["action_bits"] = edge.actionBits;
                item["visits"] = edge.visits;
                item["child_visits"] = edge.childVisits;
                item["child_states"] = edge.childStates;
                item["value_visits"] = edge.valueVisits;
                item["prior"] = edge.prior;
                item["value"] = edge.value;
                item["sample_q"] = edge.sampleQ;
                item["raw_q"] = edge.rawQ;
                item["normalized_q"] = edge.normalizedQ;
                item["completed_q"] = edge.completedQ;
                item["gumbel"] = edge.gumbel;
                item["gumbel_score"] = edge.gumbelScore;
                item["completed_score"] = edge.completedScore;
                item["successor_model_value_valid"] =
                    edge.successorModelValueValid;
                item["successor_model_value"] = edge.successorModelValue;
                item["successor_model_value_count"] =
                    edge.successorModelValueCount;
                item["candidate"] = edge.candidate;
                item["active_candidate"] = edge.activeCandidate;
                item["random_split"] = edge.randomSplit;
                item["expandable"] = edge.expandable;
                item["targeted"] = edge.targeted;
                item["label"] = edge.label;
                children.append(item);
            }
            result["children"] = children;
            return result;
        }, pybind11::arg("path") = std::vector<int>{},
           pybind11::arg("max_children") = 64,
           "return lazy outgoing edge statistics for a root-to-node path");

    pybind11::class_<BattleContext>(m, "BattleContext")
        .def("nosl_future_key", &search::NoslCombatTrainer::stateKey,
             "NOSL public future fingerprint, ignoring RNG and interchangeable card positions")
        .def("equivalent_play_slots", &search::Action::equivalentPlaySlots,
             "representative hand slots for equal complete card plays; no state or RNG changes")
        .def("clone", [](const BattleContext &bc) {
            auto copy = std::make_unique<BattleContext>(bc);
#ifdef STS_ENABLE_RULE_TRACE
            copy->atomicStepObserver = {};
#endif
            return copy;
        }, "return an independent deep copy of the complete combat state")
        .def("remap_draw_selection_from", &remapDrawSelection,
             pybind11::arg("source"), pybind11::arg("action_bits"),
             "map a public draw selection to this sampled clone by stable card identity; returns legal complete action bits without executing")
        .def("execute_to_enemy_boundary", [](BattleContext &bc,
                                             std::uint32_t actionBits) {
            const search::Action action(actionBits);
            if (bc.outcome != Outcome::UNDECIDED || bc.postEnemyTurnBoundary ||
                !action.isValidAction(bc)) {
                throw std::invalid_argument("boundary execution requires a legal live action before the enemy boundary");
            }
            bc.stopAfterEnemyTurn = true;
            action.execute(bc);
            return rawCombatState(bc);
        }, pybind11::arg("action_bits"),
           "mutate one legal action and pump until input, terminal, or the post-enemy boundary; preserve the boundary stop across pending selections")
        .def("preview_end_turn_result", [](const BattleContext &bc,
                                           std::uint32_t actionBits) {
            const search::Action action(actionBits);
            if (bc.outcome != Outcome::UNDECIDED ||
                action.getActionType() != search::ActionType::END_TURN ||
                !action.isValidAction(bc)) {
                throw std::invalid_argument("end-turn preview requires a legal live END_TURN action");
            }
            BattleContext boundary(bc);
#ifdef STS_ENABLE_RULE_TRACE
            boundary.atomicStepObserver = {};
#endif
            boundary.stopAfterEnemyTurn = true;
            action.execute(boundary);
            return rawCombatState(boundary);
        }, pybind11::arg("action_bits"),
           "execute END once on a clone and return pending-choice, post-enemy-boundary, or terminal raw state; never mutates the source")
        .def("preview_end_turn_boundary", [](const BattleContext &bc,
                                               std::uint32_t actionBits) {
            const search::Action action(actionBits);
            if (action.getActionType() != search::ActionType::END_TURN ||
                !action.isValidAction(bc)) {
                return pybind11::object(pybind11::none());
            }
            BattleContext boundary(bc);
#ifdef STS_ENABLE_RULE_TRACE
            boundary.atomicStepObserver = {};
#endif
            boundary.stopAfterEnemyTurn = true;
            action.execute(boundary);
            if (!boundary.postEnemyTurnBoundary) {
                // A terminal outcome (normally player death) is valued by its
                // objective HP directly and has no non-terminal boundary leaf.
                return pybind11::object(pybind11::none());
            }
            return pybind11::object(rawCombatState(boundary));
        }, pybind11::arg("action_bits"),
           "preview the stable post-enemy-turn value boundary for End Turn")
        .def("is_targeted_action", [](const BattleContext &bc,
                                       std::uint32_t action_bits) {
            const search::Action action(action_bits);
            switch (action.getActionType()) {
                case search::ActionType::CARD:
                    return action.getSourceIdx() >= 0 &&
                           action.getSourceIdx() < bc.cards.cardsInHand &&
                           bc.cards.hand[action.getSourceIdx()].requiresTarget();
                case search::ActionType::POTION:
                    return action.getSourceIdx() >= 0 &&
                           action.getSourceIdx() < bc.potionCapacity &&
                           potionRequiresTarget(bc.potions[action.getSourceIdx()]);
                default:
                    return false;
            }
        }, pybind11::arg("action_bits"),
           "whether a complete action uses an enemy target in this state")
        .def("nosl_redeterminize", &sts::search::noslRedeterminize,
             pybind11::arg("nonce"),
             "replace future RNG streams and resample hidden draw-pile order")
        .def("nosl_native_search", [](const BattleContext &bc,
                                      int simulations,
                                      int depth,
                                      double c_puct,
                                      std::uint64_t nonce,
                                      int boss_transition_value_mode) {
            sts::search::NoslCombatSearchConfig config;
            config.simulations = simulations;
            config.maxDepth = depth;
            config.cPuct = c_puct;
            config.seed = nonce;
            config.bossTransitionValueMode = boss_transition_value_mode;
            sts::search::NoslCombatTrainer trainer(config);
            sts::search::NoslCombatSearchStats stats;
            const auto decision = trainer.search(bc, nonce, &stats);

            pybind11::dict result;
            result["action_bits"] = decision.action.bits;
            result["action_valid"] = decision.action.isValidAction(bc);
            result["root_value"] = decision.rootValue;
            result["total_root_visits"] = decision.totalRootVisits;
            result["root_visits"] = decision.rootVisits;
            result["selected_target"] = decision.selectedTarget;
            result["selected_action_targeted"] = decision.selectedActionTargeted;
            result["simulations"] = stats.simulations;
            result["transitions"] = stats.transitions;
            result["expanded_nodes"] = stats.expandedNodes;
            return result;
        }, pybind11::arg("simulations") = 128,
           pybind11::arg("depth") = 3,
           pybind11::arg("c_puct") = 1.5,
           pybind11::arg("nonce") = 1,
           pybind11::arg("boss_transition_value_mode") = 0,
           "run a native NOSL combat search from this public state; each "
           "simulation clones and independently redeterminizes the root")
        .def("nosl_neural_search_begin",
             [](const BattleContext &bc,
                pybind11::array_t<double,
                    pybind11::array::c_style |
                    pybind11::array::forcecast> root_policy_logits,
                int simulations,
                int depth,
                double c_puct,
                std::uint64_t nonce,
                double dirichlet_alpha,
                double dirichlet_epsilon,
                double root_value,
                double prior_temperature,
                bool end_turn_as_leaf,
                int root_search_mode,
                int gumbel_candidates,
                double gumbel_scale,
                double gumbel_value_scale,
                double gumbel_max_visit_init,
                bool gumbel_rescale_q,
                bool gumbel_use_mixed_value,
                double gumbel_interior_uniformity,
                int gumbel_interior_min_visits,
                double gumbel_interior_coverage_fraction,
                int gumbel_interior_value_mode,
                double gumbel_interior_max_mix_visits,
                bool paper_full_gumbel,
                bool fixed_worldline,
                int boss_transition_value_mode, int selection_prefix,
                bool model_victory_terminal, bool merge_equivalent_actions,
                double boss_win_weight) {
                std::vector<double> logits;
                auto view = root_policy_logits.unchecked<1>();
                logits.reserve(static_cast<std::size_t>(view.shape(0)));
                for (pybind11::ssize_t i = 0; i < view.shape(0); ++i) {
                    logits.push_back(view(i));
                }
                sts::search::NoslCombatSearchConfig config;
                config.simulations = simulations;
                config.maxDepth = depth;
                config.cPuct = c_puct;
                config.priorTemperature = prior_temperature;
                config.autoregressivePolicy = logits.size() == sts::search::kAutoregressiveSize;
                config.selectionPrefix = selection_prefix;
                if (!config.autoregressivePolicy && selection_prefix != 0)
                    throw std::invalid_argument("selection prefix requires autoregressive policy");
                config.jointPolicy = config.autoregressivePolicy || logits.size() == sts::search::kJointSize;
                if (config.jointPolicy && boss_transition_value_mode != 0)
                    throw std::invalid_argument("joint-v2 requires combat-end-hp-v1");
                config.endTurnAsLeaf = end_turn_as_leaf;
                config.dirichletAlpha = dirichlet_alpha;
                config.dirichletEpsilon = dirichlet_epsilon;
                config.rootSearchMode = root_search_mode;
                config.gumbelCandidates = gumbel_candidates;
                config.gumbelScale = gumbel_scale;
                config.gumbelValueScale = gumbel_value_scale;
                config.gumbelMaxVisitInit = gumbel_max_visit_init;
                config.gumbelRescaleQ = gumbel_rescale_q;
                config.gumbelUseMixedValue = gumbel_use_mixed_value;
                config.gumbelInteriorUniformity = gumbel_interior_uniformity;
                config.gumbelInteriorMinVisits = gumbel_interior_min_visits;
                config.gumbelInteriorCoverageFraction =
                    gumbel_interior_coverage_fraction;
                config.gumbelInteriorValueMode = gumbel_interior_value_mode;
                config.gumbelInteriorMaxMixVisits = gumbel_interior_max_mix_visits;
                config.paperFullGumbel = paper_full_gumbel;
                config.fixedWorldline = fixed_worldline;
                config.bossTransitionValueMode = boss_transition_value_mode;
                config.modelVictoryTerminal = model_victory_terminal;
                config.mergeEquivalentActions = merge_equivalent_actions;
                config.bossWinWeight = boss_win_weight;
                config.seed = nonce;
                return std::make_shared<sts::search::NoslNeuralSearchSession>(
                    bc, logits, config, nonce, root_value);
             },
             pybind11::arg("root_policy_logits"),
             pybind11::arg("simulations") = 128,
             pybind11::arg("depth") = 3,
             pybind11::arg("c_puct") = 1.5,
             pybind11::arg("nonce") = 1,
             pybind11::arg("dirichlet_alpha") = 0.0,
             pybind11::arg("dirichlet_epsilon") = 0.0,
             pybind11::arg("root_value") = 0.0,
             pybind11::arg("prior_temperature") = 1.0,
             pybind11::arg("end_turn_as_leaf") = false,
             pybind11::arg("root_search_mode") = 0,
             pybind11::arg("gumbel_candidates") = 0,
             pybind11::arg("gumbel_scale") = 1.0,
             pybind11::arg("gumbel_value_scale") = 0.1,
             pybind11::arg("gumbel_max_visit_init") = 50.0,
             pybind11::arg("gumbel_rescale_q") = true,
             pybind11::arg("gumbel_use_mixed_value") = true,
             pybind11::arg("gumbel_interior_uniformity") = 0.0,
             pybind11::arg("gumbel_interior_min_visits") = 1,
             pybind11::arg("gumbel_interior_coverage_fraction") = 0.0,
             pybind11::arg("gumbel_interior_value_mode") = 2,
             pybind11::arg("gumbel_interior_max_mix_visits") = 64.0,
             pybind11::arg("paper_full_gumbel") = false,
             pybind11::arg("fixed_worldline") = false,
             pybind11::arg("boss_transition_value_mode") = 0,
             pybind11::arg("selection_prefix") = 0,
             pybind11::arg("model_victory_terminal") = false,
             pybind11::arg("merge_equivalent_actions") = true,
             pybind11::arg("boss_win_weight") = 0.0,
             "start a neural-guided C++ NOSL AlphaZero session")
        .def("projected_exit_hp", [](const BattleContext &bc, const GameContext &gc) {
            GameContext projected = gc;
            projected.curHp = bc.player.curHp;
            projected.maxHp = bc.player.maxHp;
            bc.updateRelicsOnExit(projected);
            return projected.curHp;
        }, "HP after applying combat-exit relic healing to a temporary GameContext")
        .def("to_dict", [](const BattleContext &bc) { return rawCombatState(bc, false); }, "export the combat state as a raw dict (sim-side, normalized to canonical in Python)")
        .def("player_power_flags", [](const BattleContext &bc) {
            // Display metadata only. Existing checkpoints consume the numeric
            // powers/packed observation contract, which is intentionally unchanged.
            pybind11::list flags;
            for (auto status : {PlayerStatus::CONFUSED, PlayerStatus::BARRICADE,
                    PlayerStatus::BLASPHEMER, PlayerStatus::CORRUPTION,
                    PlayerStatus::ELECTRO, PlayerStatus::SURROUNDED,
                    PlayerStatus::MASTER_REALITY, PlayerStatus::PEN_NIB,
                    PlayerStatus::WRATH_NEXT_TURN}) {
                if (bc.player.hasStatusRuntime(status)) {
                    flags.append(playerStatusStrings[static_cast<int>(status)]);
                }
            }
            return flags;
        }, "active boolean player powers for display, independent of numeric observations")
        .def("compact_leaf_arrays", [](const BattleContext &bc) {
            // Diagnostic endpoint for packed-vs-legacy contract tests. The
            // search hot path uses next_batch_packed(), which does not expose
            // this per-state call.
            const auto packed = compact_leaf::pack(bc);
            pybind11::array_t<std::int32_t> categorical(
                std::vector<pybind11::ssize_t>{compact_leaf::kCatSize});
            pybind11::array_t<float> scalars(
                std::vector<pybind11::ssize_t>{compact_leaf::kFloatSize});
            auto cat = categorical.mutable_unchecked<1>();
            auto scalar = scalars.mutable_unchecked<1>();
            for (int i = 0; i < compact_leaf::kCatSize; ++i) {
                cat(i) = packed.categorical[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < compact_leaf::kFloatSize; ++i) {
                scalar(i) = packed.scalars[static_cast<std::size_t>(i)];
            }
            pybind11::dict out;
            out["categorical"] = categorical;
            out["scalars"] = scalars;
            return out;
        }, "export the current state in the native compact leaf layout")
        .def("autoregressive_policy_actions", [](const BattleContext &bc, int selected) {
            pybind11::list out;
            for (const auto &action : sts::search::autoregressiveActions(bc, selected)) {
                const int id = sts::search::jointActionId(action, bc) - sts::search::kJointEnd;
                if (id >= 0) out.append(pybind11::make_tuple(id, action.bits));
            }
            return out;
        }, pybind11::arg("selected") = 0, "legal next-card/STOP actions without enumerating subsets")
        .def("joint_policy_actions", [](const BattleContext &bc) {
            pybind11::list out;
            for (const auto &action : sts::search::Action::enumerateCombatActions(bc)) {
                const int id = sts::search::jointActionId(action, bc);
                if (id >= 0) out.append(pybind11::make_tuple(id, action.bits));
            }
            return out;
        }, "complete joint-v2 policy IDs paired with current native action bits")
        .def("apply_custom_state", &applyCustomCombatState,
             pybind11::arg("state"),
             "apply a data-only custom combat state without advancing RNG or simulation")
        .def("legal_actions", &legalCombatActions, "list the legal actions in the current state")
        .def("legal_combat_mask", &legalCombatMask,
             "return the fixed 71-id combat mask, constructed in C++")
        .def("rule_actions", &legalRuleCombatActions,
             "legal complete actions for rule collection, including multi-card subsets")
        .def("legal_potions", &legalCombatPotions,
             "authoritative legal in-combat potion uses as [{slot, requires_target, targets}]; "
             "requires_target true == needs a monster target (Fire/Fear/Poison/Weak) and targets "
             "lists the accepted monster idxs (drink via drink_potion(slot, target)); false == "
             "drink in one step (targets empty). sim-gated by search::Action::isValidPotionAction")
        .def("select_info", &combatSelectInfo,
             "combat card-select sub-screen semantics+progress: {context (CardSelectTask name), "
             "pick_count, can_pick_zero, can_pick_any}")
        .def("set_generated_selection_cards", &setGeneratedCombatSelectionCards,
             pybind11::arg("cards"),
             "replace the three public CODEX/DISCOVERY candidates without advancing the battle")
        .def("execute", [](BattleContext &bc, const search::Action &a) { a.execute(bc); }, "apply an action and pump to the next input state")
#ifdef STS_ENABLE_RULE_TRACE
        .def("exhaust_hand_card_with_hint", [](BattleContext &bc, int cardIndex, int indexHint) {
            if (cardIndex < 0 || cardIndex >= bc.cards.cardsInHand) {
                throw std::out_of_range("card index is out of range");
            }
            const auto uniqueId = bc.cards.hand[cardIndex].uniqueId;
            bc.exhaustSpecificCardInHand(indexHint, uniqueId);
        }, pybind11::arg("card_index"), pybind11::arg("index_hint"),
           "rule fixture: exhaust a card by identity with a possibly stale index hint")
        .def("set_player_strength", [](BattleContext &bc, int amount) {
            if (amount < 0) {
                throw std::invalid_argument("strength must be non-negative");
           }
           bc.player.setStatusValueNoChecks<PlayerStatus::STRENGTH>(amount);
        }, pybind11::arg("amount"),
           "set player Strength for controlled rule-model experiments")
        .def("set_player_block", [](BattleContext &bc, int amount) {
            if (amount < 0) {
                throw std::invalid_argument("player block must be non-negative");
            }
            bc.player.block = amount;
        }, pybind11::arg("amount"),
           "set player Block for controlled rule-model experiments")
        .def("set_player_energy", [](BattleContext &bc, int amount) {
            if (amount < 0) {
                throw std::invalid_argument("player energy must be non-negative");
            }
            bc.player.energy = amount;
        }, pybind11::arg("amount"),
           "set player Energy for controlled rule-model experiments")
        .def("set_player_status", &setRulePlayerStatusByName,
             pybind11::arg("status"), pybind11::arg("amount"),
             "set a supported player status by enum name for rule data")
        .def("set_monster_block", [](BattleContext &bc, int target, int amount) {
            if (target < 0 || target >= bc.monsters.monsterCount) {
                throw std::out_of_range("monster target is out of range");
            }
            if (amount < 0) {
                throw std::invalid_argument("monster block must be non-negative");
            }
            bc.monsters.arr[target].block = amount;
        }, pybind11::arg("target"), pybind11::arg("amount"),
           "set monster Block for controlled rule-model experiments")
        .def("set_monster_vulnerable", [](BattleContext &bc, int target, int amount) {
            if (target < 0 || target >= bc.monsters.monsterCount) {
                throw std::out_of_range("monster target is out of range");
            }
            if (amount < 0) {
                throw std::invalid_argument("monster Vulnerable must be non-negative");
            }
            auto &monster = bc.monsters.arr[target];
            monster.setStatus<MonsterStatus::VULNERABLE>(amount);
            monster.setHasStatus<MonsterStatus::VULNERABLE>(amount > 0);
        }, pybind11::arg("target"), pybind11::arg("amount"),
           "set monster Vulnerable for controlled rule-model experiments")
        .def("set_monster_status", &setRuleMonsterStatusByName,
             pybind11::arg("target"), pybind11::arg("status"),
             pybind11::arg("amount"),
             "set a supported monster status by enum name for rule data")
        .def("execute_with_trace", [](BattleContext &bc, const search::Action &a) {
            pybind11::dict result;
            result["before"] = rawCombatState(bc);
            pybind11::list steps;
            bc.atomicStepObserver = [&steps](const BattleContext &step) {
                steps.append(rawCombatState(step));
            };
            try {
                a.execute(bc);
            } catch (...) {
                bc.atomicStepObserver = {};
                throw;
            }
            bc.atomicStepObserver = {};
            pybind11::dict after = rawCombatState(bc);
            steps.append(after);
            result["steps"] = steps;
            result["after"] = after;
            return result;
        }, "apply an action and return state snapshots after each atomic simulator step")
#endif
        .def_property_readonly("turn", [](const BattleContext &bc) { return bc.turn; })
        .def_property_readonly("outcome", [](const BattleContext &bc) { return static_cast<int>(bc.outcome); })
        .def_property_readonly("input_state", [](const BattleContext &bc) { return static_cast<int>(bc.inputState); })
        .def_property_readonly("is_battle_over", [](const BattleContext &bc) { return bc.outcome != Outcome::UNDECIDED; })
        .def("exit_battle", [](const BattleContext &bc, GameContext &gc) { bc.exitBattle(gc); },
             "write combat results back into the GameContext (hp/gold/deck/rng) and open the reward screen on victory")
        .def("drink_potion", [](BattleContext &bc, int idx, int target) { bc.drinkPotion(idx, target); },
             pybind11::arg("idx"), pybind11::arg("target") = 0, "use the potion in slot idx (0-based) on target monster idx")
        .def("discard_potion", [](BattleContext &bc, int idx) { bc.discardPotion(idx); }, "discard the potion in slot idx (0-based)")
        .def_property_readonly("potions", [](const BattleContext &bc) {
            pybind11::list lst;
            for (int i = 0; i < bc.potionCapacity; ++i) lst.append(std::string(sts::getPotionName(bc.potions[i])));
            return lst;
        }, "combat potion slots");

    m.def("create_battle", &createBattle, "construct a combat from a GameContext + encounter name (forces the encounter)");
    m.def("encounter_names", &encounterNames, "list valid encounter names for create_battle");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")

