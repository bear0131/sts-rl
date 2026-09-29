#pragma once

#include "sim/search/Action.h"
#include <algorithm>
#include <stdexcept>
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
    auto append = [&](int index) {
        if (task == CardSelectTask::CODEX && index == 3) {
            result.cards.push_back({index, CardId::INVALID, 0, 0, true});
            return;
        }
        const CardInstance card = result.zone == 1 ? bc.cards.hand[index] :
            result.zone == 2 ? bc.cards.drawPile[index] :
            result.zone == 3 ? bc.cards.discardPile[index] :
            result.zone == 4 ? bc.cards.exhaustPile[index] :
            CardInstance(bc.cardSelectInfo.cards[index]);
        result.cards.push_back({index, card.getId(), card.getUpgradeCount(),
                               card.freeToPlayOnce ? 0 : card.costForTurn, false});
    };
    if (task == CardSelectTask::EXHAUST_MANY || task == CardSelectTask::GAMBLE) {
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
    if (result.cards.size() > 96)
        throw std::runtime_error("combat candidate capacity 96 exceeded");
    return result;
}
} // namespace sts::search
