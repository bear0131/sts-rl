#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "combat/BattleContext.h"
#include "sim/search/CombatChoices.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/Cards.h"
namespace compact_leaf {
using namespace sts;
inline int playerPowerObservation(const sts::Player &p, PlayerStatus status) {
    // OBS18: presence powers live in status bits, not necessarily statusMap.
    // Keep getStatusRuntime unchanged: search keys and game rules use it too.
    switch (status) {
        case PlayerStatus::CONFUSED: case PlayerStatus::HEX:
        case PlayerStatus::BARRICADE:
        case PlayerStatus::BLASPHEMER: case PlayerStatus::CORRUPTION:
        case PlayerStatus::ELECTRO: case PlayerStatus::SURROUNDED:
        case PlayerStatus::MASTER_REALITY: case PlayerStatus::PEN_NIB:
        case PlayerStatus::WRATH_NEXT_TURN: case PlayerStatus::NO_DRAW:
            return p.hasStatusRuntime(status) ? 1 : 0;
        default:
            return p.getStatusRuntime(status);
    }
}

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
constexpr int kGridPickSlots = sts::CardSelectInfo::GRID_PICK_CAPACITY;
// Categorical layout: hand cards, fixed monster slots/moves/intent classes,
// potion slots, and the card-select task.
constexpr int kCatSize = kHandSlots + kMaxMonsters + kMaxMonsters +
                         kMaxMonsters + kPotionSlots + 1 + 96 * 2 + 1 + kGridPickSlots;

// Float layout: player scalars, player statuses, counters, boundary bit,
// monster scalars/statuses, hand upgrade/cost, three native pile histograms,
// and card-select metadata.
constexpr int kFloatSize = 6 + kPlayerStatus + kCombatCounters + 1 +
                           kMaxMonsters * 6 + kMaxMonsters * kMonsterStatus +
                           kHandSlots * 2 + kCardIds * 3 + 4 + 96 * 2 + kGridPickSlots + 4;

struct Row {
    std::array<std::int32_t, kCatSize> categorical{};
    std::array<float, kFloatSize> scalars{};
    int selectionCount = 0;
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

// The caller supplies zeroed categorical/scalar storage. CPU Row and the
// cooperative GPU writer share this exact observation contract.
template <typename Output>
inline void packInto(const sts::BattleContext &bc, Output &row) {
    if (bc.cards.cardsInHand < 0 || bc.cards.cardsInHand > kHandSlots) {
        throw std::invalid_argument("combat hand exceeds compact observation capacity");
    }
    if (bc.monsters.monsterCount < 0 || bc.monsters.monsterCount > kMaxMonsters) {
        throw std::invalid_argument("monster slots exceed compact observation capacity");
    }
    if (bc.potionCapacity < 0 || bc.potionCapacity > kPotionSlots) {
        throw std::invalid_argument("potion slots exceed compact observation capacity");
    }
    std::array<int, kMaxMonsters> baseDamages{};
    std::array<int, kMaxMonsters> attackCounts{};
    int c = 0;
    for (int i = 0; i < bc.cards.cardsInHand && i < kHandSlots; ++i) {
        const int card = static_cast<int>(bc.cards.hand[i].getId());
        if (card <= 0 || card >= kCardIds) {
            throw std::invalid_argument("live hand contains an invalid card identity");
        }
        row.categorical[c + i] = card;
    }
    c += kHandSlots;
    for (int i = 0; i < bc.monsters.monsterCount && i < kMaxMonsters;
         ++i) {
        const auto &monster = bc.monsters.arr[i];
        const int monsterId = static_cast<int>(monster.id);
        if (monster.id == sts::MonsterId::INVALID) continue;
        if (monsterId < 0 || monsterId >= kNativeMonsters) {
            throw std::invalid_argument("monster slot contains an invalid identity");
        }
        const int move = static_cast<int>(monster.moveHistory[0]);
        if (move < 0 || move >= kNativeMoves) {
            throw std::invalid_argument("monster slot contains an invalid move identity");
        }
        row.categorical[c + i] = monsterId;
        row.categorical[c + kMaxMonsters + i] = move;
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
        if (bc.potions[i] == sts::Potion::INVALID) {
            throw std::invalid_argument(
                "physical potion slot contains INVALID instead of EMPTY_POTION_SLOT");
        }
        const int potion = static_cast<int>(bc.potions[i]);
        if (potion < 0 || potion >= kNativePotions) {
            throw std::invalid_argument("physical potion slot has an invalid identity");
        }
        row.categorical[c + i] = potion;
    }
    c += kPotionSlots;
    const bool selecting = bc.inputState == InputState::CARD_SELECT;
    row.categorical[c++] = selecting ? static_cast<int>(bc.cardSelectInfo.cardSelectTask) : 0;
    const auto choices = sts::search::combatChoices(bc);
    row.selectionCount = static_cast<int>(choices.cards.size());
    if (choices.cards.size() > 96 || choices.zone < 0 || choices.zone > 6) {
        throw std::invalid_argument("combat selection exceeds compact observation contract");
    }
    for (std::size_t i = 0; i < choices.cards.size(); ++i) {
        const int card = static_cast<int>(choices.cards[i].id);
        if ((!choices.cards[i].skip && (card <= 0 || card >= kCardIds)) ||
            (choices.cards[i].skip && card != 0)) {
            throw std::invalid_argument("combat selection contains an invalid card identity");
        }
        if (choices.cards[i].nativeIndex < 0 ||
            choices.cards[i].nativeIndex >= 65536) {
            throw std::invalid_argument("combat selection handle is outside observation range");
        }
        row.categorical[c + i] = card;
        row.categorical[c + 96 + i] = choices.cards[i].nativeIndex + 1;
    }
    row.categorical[c + 192] = choices.zone;
    const auto selectedCards = sts::search::combatSelectedCards(bc);
    for (std::size_t i = 0; i < selectedCards.size(); ++i)
        row.categorical[c + 193 + i] = static_cast<int>(selectedCards[i].id);

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
        static_cast<float>(bc.player.bombDamage(1)),
        static_cast<float>(bc.player.bombDamage(2)),
        static_cast<float>(bc.player.bombDamage(3)),
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
        const int adjusted = monster.getIntentDamage(bc, baseDamage);
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
            if (id <= 0 || id >= kCardIds) {
                throw std::invalid_argument("combat pile contains an invalid card identity");
            }
            row.scalars[base + id] += 1.0f;
        }
        f += kCardIds;
    };
    countPile(bc.cards.drawPile);
    countPile(bc.cards.discardPile);
    countPile(bc.cards.exhaustPile);
    // GRID prefixes remain in the discard pile until the mandatory last pick.
    row.scalars[f++] = selecting
        ? static_cast<float>(bc.cardSelectInfo.pickCount - bc.cardSelectInfo.selectedCount)
        : 0.0f;
    row.scalars[f++] = selecting && bc.cardSelectInfo.canPickZero ? 1.0f : 0.0f;
    row.scalars[f++] = selecting && bc.cardSelectInfo.canPickAnyNumber ? 1.0f : 0.0f;
    row.scalars[f++] = selecting && bc.cardSelectInfo.isMultiSelection() ? 1.0f : 0.0f;
    for (std::size_t i = 0; i < choices.cards.size(); ++i) {
        row.scalars[f + i] = static_cast<float>(choices.cards[i].upgrades);
        row.scalars[f + 96 + i] = static_cast<float>(choices.cards[i].cost);
    }
    for (std::size_t i = 0; i < selectedCards.size(); ++i)
        row.scalars[f + 192 + i] = static_cast<float>(selectedCards[i].getUpgradeCount());
    row.scalars[kFloatSize - 4] = bc.player.hasRelicRuntime(sts::RelicId::CENTENNIAL_PUZZLE) ? 1.0f : 0.0f;
    row.scalars[kFloatSize - 3] = bc.player.hasRelicRuntime(sts::RelicId::LIZARD_TAIL) ? 1.0f : 0.0f;
    row.scalars[kFloatSize - 2] = bc.postEnemyTurnBoundary ? bc.pocketwatchCardsAtBoundary : bc.player.cardsPlayedThisTurn;
    row.scalars[kFloatSize - 1] = bc.player.neowLamentRemaining;
}

inline Row pack(const sts::BattleContext &bc) {
    Row row;
    packInto(bc, row);
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
        if (native <= 0 || native >= kCardIds) {
            throw std::invalid_argument("language pile contains an invalid card identity");
        }
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
