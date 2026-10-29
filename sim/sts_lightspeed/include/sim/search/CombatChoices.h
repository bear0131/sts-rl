#pragma once

#include "sim/search/Action.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <tuple>

namespace sts::search {

// Public candidate order is independent of hidden draw order. Native indices
// are opaque execution handles; callers must never embed them as features.
struct CombatChoice {
    int nativeIndex;
    CardId id;
    int upgrades;
    int cost;
    bool skip;
};
struct CombatChoices {
    int zone = 0; // none, hand, draw, discard, exhaust, generated, grid
    std::vector<CombatChoice> cards;
};

inline CardInstance combatChoiceCard(const BattleContext &bc, int zone, int index) {
    switch (zone) {
        case 1: return bc.cards.hand[index];
        case 2: return bc.cards.drawPile[index];
        case 3: return bc.cards.discardPile[index];
        case 4: return bc.cards.exhaustPile[index];
        case 5: return bc.makeCardCopy(bc.cardSelectInfo.cards[index]);
        default: throw std::invalid_argument("combat choice has no card zone");
    }
}

inline std::vector<CardInstance> combatSelectedCards(const BattleContext &bc) {
    std::vector<CardInstance> result;
    if (bc.inputState != InputState::CARD_SELECT) return result;
    const auto &info = bc.cardSelectInfo;
    if (info.selectedCount < 0 || info.selectedCount > CardSelectInfo::GRID_PICK_CAPACITY ||
        (info.selectedCount && (info.cardSelectTask != CardSelectTask::LIQUID_MEMORIES_POTION ||
            info.selectedCount >= info.pickCount || info.pickCount > CardSelectInfo::GRID_PICK_CAPACITY)))
        throw std::logic_error("invalid pending combat GRID selection");
    for (int i = 0; i < info.selectedCount; ++i) {
        const auto uid = info.selectedUids[i];
        if (uid < 0 || std::count(info.selectedUids.begin(), info.selectedUids.begin() + i, uid) ||
            std::count_if(bc.cards.discardPile.begin(), bc.cards.discardPile.end(),
                [uid](const CardInstance &card) { return card.uniqueId == uid; }) != 1)
            throw std::logic_error("pending combat GRID identity is missing or duplicated");
        const auto it = std::find_if(bc.cards.discardPile.begin(), bc.cards.discardPile.end(),
            [uid](const CardInstance &card) { return card.uniqueId == uid; });
        if (it == bc.cards.discardPile.end())
            throw std::logic_error("pending combat GRID card is absent from source pile");
        result.push_back(*it);
    }
    return result;
}

inline CombatChoices combatChoices(const BattleContext &bc) {
    CombatChoices result;
    if (bc.inputState != InputState::CARD_SELECT) return result;
    const auto task = bc.cardSelectInfo.cardSelectTask;
    switch (task) {
        case CardSelectTask::SECRET_TECHNIQUE:
        case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK: result.zone = 2; break;
        case CardSelectTask::HEADBUTT:
        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
        case CardSelectTask::MEDITATE: result.zone = 3; break;
        case CardSelectTask::EXHUME: result.zone = 4; break;
        case CardSelectTask::CODEX:
        case CardSelectTask::DISCOVERY: result.zone = 5; break;
        case CardSelectTask::ARMAMENTS:
        case CardSelectTask::DUAL_WIELD:
        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::GAMBLE:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY: result.zone = 1; break;
        default: throw std::runtime_error("unsupported combat selection context");
    }
    auto cardAt = [&](int index) -> CardInstance {
        return combatChoiceCard(bc, result.zone, index);
    };
    auto append = [&](int index) {
        if (task == CardSelectTask::CODEX && index == 3) {
            result.cards.push_back({index, CardId::INVALID, 0, 0, true});
            return;
        }
        const CardInstance card = cardAt(index);
        result.cards.push_back({index, card.getId(), card.getUpgradeCount(),
                               card.freeToPlayOnce ? 0 : card.costForTurn, false});
    };
    if (bc.cardSelectInfo.isMultiSelection()) {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) append(i);
    } else {
        for (const auto &action : Action::enumerateCardSelectActions(bc)) {
            if (action.getActionType() == ActionType::SINGLE_CARD_SELECT)
                append(action.getSelectIdx());
        }
    }
    if (result.zone == 2) {
        std::stable_sort(result.cards.begin(), result.cards.end(),
            [](const CombatChoice &a, const CombatChoice &b) {
                return std::tie(a.id, a.upgrades, a.cost) <
                       std::tie(b.id, b.upgrades, b.cost);
            });
    }
    if (result.cards.size() > 96) {
        // overflow-equivalent-v1: battle piles can grow beyond the permanent
        // deck capacity (e.g. Repulsor Dazed + Exhume). Preserve all previously
        // encodable states exactly; compact only the formerly rejected states.
        // Keep one executable handle for each effect, without truncating unique
        // choices or treating different runtime costs as the same card.
        std::vector<CombatChoice> unique;
        for (const auto &choice : result.cards) {
            const auto card = cardAt(choice.nativeIndex);
            if (std::none_of(unique.begin(), unique.end(), [&](const CombatChoice &other) {
                return Action::equivalentCards(card, cardAt(other.nativeIndex));
            })) unique.push_back(choice);
        }
        if (unique.size() > 96) {
            std::ostringstream detail;
            detail << "combat candidate capacity 96 exceeded after equivalence: "
                   << unique.size() << " distinct of " << result.cards.size()
                   << "; task=" << cardSelectTaskStrings[static_cast<int>(task)]
                   << "; floor=" << bc.floorNum << '\n' << bc;
            throw std::runtime_error(detail.str());
        }
        result.cards = std::move(unique);
    }
    return result;
}

constexpr int kJointEnd = 181;
constexpr int kJointPlay = 182;
constexpr int kJointPotion = 242;
constexpr int kJointPick = 272;
constexpr int kJointMulti = 368;
constexpr int kJointSize = 1392;
constexpr int kAutoregressiveStop = kJointMulti + 10;
constexpr int kAutoregressiveSize = kAutoregressiveStop + 1 - kJointEnd;
// Policy-only steps. They must never be passed to Action::execute directly.
constexpr ActionType kSelectNext = static_cast<ActionType>(5);
constexpr ActionType kSelectStop = static_cast<ActionType>(6);

inline bool isMultiSelection(const BattleContext &bc) {
    return bc.outcome == Outcome::UNDECIDED && bc.inputState == InputState::CARD_SELECT &&
        bc.cardSelectInfo.isMultiSelection();
}

inline std::vector<Action> neuralCombatActions(const BattleContext &bc) {
    auto actions = Action::enumerateCombatActions(bc);
    if (bc.inputState == InputState::CARD_SELECT &&
        !bc.cardSelectInfo.isMultiSelection() && actions.size() > 96) {
        // The observation exposes one real handle per equivalent overflow
        // choice. Use those same handles below the search root: the omitted
        // aliases have no policy slot and must not enter a leaf's legal mask.
        // Keep the native order and every action in already encodable states.
        const auto choices = combatChoices(bc);
        actions.erase(std::remove_if(actions.begin(), actions.end(), [&](const Action &action) {
            return std::none_of(choices.cards.begin(), choices.cards.end(), [&](const CombatChoice &choice) {
                return choice.nativeIndex == action.getSelectIdx();
            });
        }), actions.end());
    }
    return actions;
}

// Full atomic actions encodable by the policy contract, before multi prefixes.
// Potion discard is deliberately absent from native-v1/autoregressive-v2.
inline std::vector<Action> completePolicyCombatActions(const BattleContext &bc) {
    auto actions = neuralCombatActions(bc);
    actions.erase(std::remove_if(actions.begin(), actions.end(), [](const Action &action) {
        // Packed -1 is the unsigned 13-bit sentinel 8191.
        return action.getActionType() == ActionType::POTION && action.getTargetIdx() > 5;
    }), actions.end());
    return actions;
}

inline std::vector<Action> autoregressiveActions(const BattleContext &bc, int selected) {
    if (!isMultiSelection(bc)) {
        if (selected != 0) throw std::invalid_argument("selection prefix outside multi screen");
        return completePolicyCombatActions(bc);
    }
    if (selected < 0 || selected >= (1 << bc.cards.cardsInHand))
        throw std::invalid_argument("invalid selection prefix");
    std::vector<Action> result;
    // Each set has exactly one increasing-index path. Native legality remains
    // authoritative; no subset enumeration is needed for current multi tasks.
    int last = -1;
    for (int i = 0; i < bc.cards.cardsInHand; ++i)
        if (selected & (1 << i)) last = i;
    for (int i = last + 1; i < bc.cards.cardsInHand; ++i)
        if (Action(ActionType::MULTI_CARD_SELECT, selected | (1 << i)).isValidAction(bc))
            result.emplace_back(kSelectNext, i);
    if (Action(ActionType::MULTI_CARD_SELECT, selected).isValidAction(bc))
        result.emplace_back(kSelectStop);
    if (result.empty()) throw std::invalid_argument("selection prefix has no completion");
    return result;
}

inline bool validAutoregressiveAction(const Action &action, const BattleContext &bc, int selected) {
    if (!isMultiSelection(bc))
        return selected == 0 && action.getActionType() != ActionType::MULTI_CARD_SELECT &&
            static_cast<int>(action.getActionType()) < 5 && action.isValidAction(bc);
    const auto actions = autoregressiveActions(bc, selected);
    return std::any_of(actions.begin(), actions.end(), [&](const Action &a) { return a.bits == action.bits; });
}

inline bool executeAutoregressiveAction(const Action &action, BattleContext &bc, int &selected) {
    if (!validAutoregressiveAction(action, bc, selected))
        throw std::invalid_argument("illegal autoregressive action");
    if (action.getActionType() == kSelectNext) {
        selected |= 1 << action.getSourceIdx();
        return false; // no simulator transition, RNG consumption or depth
    }
    if (action.getActionType() == kSelectStop)
        Action(ActionType::MULTI_CARD_SELECT, selected).execute(bc);
    else
        action.execute(bc);
    selected = 0;
    return true;
}

inline int jointActionId(const Action &action, const BattleContext &bc) {
    const int source = action.getSourceIdx();
    const int target = action.getTargetIdx();
    if (action.getActionType() == kSelectNext) return source < 10 ? kJointMulti + source : -1;
    if (action.getActionType() == kSelectStop) return kAutoregressiveStop;
    switch (action.getActionType()) {
        case ActionType::END_TURN: return kJointEnd;
        case ActionType::CARD:
            if (source < 0 || source >= bc.cards.cardsInHand || target >= 5) return -1;
            return kJointPlay + source * 6 + (bc.cards.hand[source].requiresTarget() ? target : 5);
        case ActionType::POTION:
            if (source < 0 || source >= bc.potionCapacity || target >= 5) return -1;
            return kJointPotion + source * 6 + (potionRequiresTarget(bc.potions[source]) ? target : 5);
        case ActionType::SINGLE_CARD_SELECT: {
            const auto choices = combatChoices(bc);
            for (std::size_t i = 0; i < choices.cards.size(); ++i)
                if (choices.cards[i].nativeIndex == action.getSelectIdx()) return kJointPick + static_cast<int>(i);
            return -1;
        }
        case ActionType::MULTI_CARD_SELECT:
            return source < 1024 ? kJointMulti + source : -1;
        default: return -1;
    }
}

inline std::vector<int> jointActionIds(const BattleContext &bc) {
    std::vector<int> result;
    for (const auto &action : Action::enumerateCombatActions(bc)) {
        const int id = jointActionId(action, bc);
        if (id >= 0) result.push_back(id);
    }
    return result;
}
} // namespace sts::search
