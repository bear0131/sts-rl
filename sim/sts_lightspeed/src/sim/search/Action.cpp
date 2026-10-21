//
// Created by keega on 9/17/2021.
//

#include "sim/search/Action.h"

#include <functional>
#include <algorithm>
#include <stdexcept>

using namespace sts;

bool search::Action::equivalentCards(const CardInstance &a, const CardInstance &b) {
    // These cards write back to a particular permanent deck instance.
    if (a.id == CardId::RITUAL_DAGGER || a.id == CardId::GENETIC_ALGORITHM)
        return false;
    return a.id == b.id && a.upgraded == b.upgraded &&
        a.specialData == b.specialData && a.cachedBaseDamage == b.cachedBaseDamage && a.cost == b.cost &&
        a.costForTurn == b.costForTurn &&
          a.freeToPlayOnce == b.freeToPlayOnce && a.retain == b.retain &&
          a.exhaust == b.exhaust;
}

std::array<int, 10> search::Action::equivalentPlaySlots(const BattleContext &bc) {
    std::array<int, 10> result{};
    for (int i = 0; i < 10; ++i) {
        result[i] = i;
        if (i >= bc.cards.cardsInHand || bc.inputState != InputState::PLAYER_NORMAL)
            continue;
        const auto &a = bc.cards.hand[i];
        for (int j = 0; j < i; ++j) {
            const auto &b = bc.cards.hand[j];
            if (equivalentCards(a, b)) {
                result[i] = result[j];
                break;
            }
        }
    }
    return result;
}

search::Action::Action(std::uint32_t bits) : bits(bits) {}
search::Action::Action(search::ActionType actionType) : bits(static_cast<int>(actionType) << 29) {}
search::Action::Action(search::ActionType actionType, int idx1) : bits((static_cast<int>(actionType) << 29) | (idx1 & 0xFFFF)) {}
search::Action::Action(search::ActionType actionType, int idx1, int idx2) : bits((static_cast<int>(actionType) << 29) | ((idx2 & 0x1FFF) << 16) | (idx1 & 0xFFFF)) {}
//
//search::Action::Action(search::ActionType actionType) : Action(actionType, 0, 0) {}
//
//search::Action::Action(search::ActionType actionType, int idx1) : Action(actionType, idx1, 0) {}
//
//search::Action::Action(search::ActionType actionType, int idx1, int idx2) : actionType(actionType), idx1(idx1),
//                                                                            idx2(idx2) {}

bool search::Action::operator==(const search::Action &rhs) const {
    return bits == rhs.bits;
//    return actionType == rhs.actionType && idx1 == rhs.idx1 && idx2 == rhs.idx2;
}

bool search::Action::operator!=(const search::Action &rhs) const {
    return !(rhs == *this);
}

search::ActionType search::Action::getActionType() const {
    return static_cast<ActionType>((bits >> 29) & 0xF);
//    return actionType;
}

int search::Action::getSourceIdx() const {
//    return idx1;
    return bits & 0xFFFF;
}

int search::Action::getTargetIdx() const {
//    return idx2;
    return (bits >> 16) & 0x1FFF;
}

int search::Action::getSelectIdx() const {
//    return idx1;
    return bits & 0xFFFF;
}

fixed_list<int, 10> search::Action::getSelectedIdxs() const {
    fixed_list<int,10> ret;
    int i = 0;
    int bitsCopy = bits & 0x3FF;
    while (bitsCopy) {
        if (bitsCopy & 0x1) {
            ret.push_back(i);
        }
        bitsCopy >>= 1;
        ++i;
    }
    return ret;
}

bool isValidPotionAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::PLAYER_NORMAL) {
        return false;
    }

    if (a.getSourceIdx() < 0 || a.getSourceIdx() >= bc.potionCapacity ||
            a.getSourceIdx() >= static_cast<int>(bc.potions.size())) {
        return false;
    }

    const auto p = bc.potions[a.getSourceIdx()];
    if (p == Potion::INVALID || p == Potion::EMPTY_POTION_SLOT) {
        return false;
    }

    if (a.getTargetIdx() > 5) {
        // discard action
        return true;

    } else {

        if (!bc.canDrinkPotion(a.getSourceIdx())) {
            return false;
        }

        if (!potionRequiresTarget(p)) {
            return true;
        }

        if (a.getTargetIdx() < 0 || a.getTargetIdx() >= bc.monsters.monsterCount ||
                a.getTargetIdx() >= 5) {
            return false;
        }

        return bc.monsters.arr[a.getTargetIdx()].isTargetable();
    }


}

bool isValidCardAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::PLAYER_NORMAL) {
        return false;
    }

    if (!bc.isCardPlayAllowed()) {
        return false;
    }

    if (a.getSourceIdx() < 0 || a.getSourceIdx() >= bc.cards.cardsInHand) {
        return false;
    }

    const auto &c = bc.cards.hand[a.getSourceIdx()];

    if (a.getTargetIdx() < 0 || a.getTargetIdx() >= 5) {
        return false;
    }

    // Queue admission is deliberately looser than a new player target choice.
    if (c.requiresTarget() && (a.getTargetIdx() >= bc.monsters.monsterCount ||
            !bc.monsters.arr[a.getTargetIdx()].isTargetable())) {
        return false;
    }

    return c.canUse(bc, a.getTargetIdx(), false);
}

bool isValidSingleCardSelectAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::CARD_SELECT || bc.cardSelectInfo.isMultiSelection()) {
        return false;
    }

    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::CODEX:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < 4;

        case CardSelectTask::DISCOVERY:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < 3;

        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::HEADBUTT:
        case CardSelectTask::MEDITATE:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.discardPile.size();

        case CardSelectTask::LIQUID_MEMORIES_POTION:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.discardPile.size() &&
                !bc.cardSelectInfo.hasSelectedUid(bc.cards.discardPile[a.getSelectIdx()].uniqueId);

        case CardSelectTask::EXHUME:
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.exhaustPile.size()) {
                return false;
            }
            return bc.cards.exhaustPile[a.getSelectIdx()].id != sts::CardId::EXHUME;

        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.cardsInHand;

        case CardSelectTask::ARMAMENTS:
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.cardsInHand) {
                return false;
            }
            return bc.cards.hand[a.getSelectIdx()].canUpgrade();

        case CardSelectTask::DUAL_WIELD: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.cardsInHand) {
                return false;
            }
            const auto &c = bc.cards.hand[a.getSelectIdx()];
            return c.getType() == sts::CardType::POWER || c.getType() == sts::CardType::ATTACK;
        }

        case CardSelectTask::SECRET_TECHNIQUE: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.drawPile.size()) {
                return false;
            }
            return bc.cards.drawPile[a.getSelectIdx()].getType() == CardType::SKILL;
        }

        case CardSelectTask::SECRET_WEAPON: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.drawPile.size()) {
                return false;
            }
            return bc.cards.drawPile[a.getSelectIdx()].getType() == CardType::ATTACK;
        }

        case CardSelectTask::SEEK:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.drawPile.size();

        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::GAMBLE:
        default:
            return false;
    }
}

bool isValidMultiCardSelectAction(const BattleContext &bc, const search::Action &a) { // this can be sped up using compiler intrinsics
    if (bc.inputState != InputState::CARD_SELECT || !bc.cardSelectInfo.isMultiSelection())
        return false;
    switch (bc.cardSelectInfo.cardSelectTask) {
        case sts::CardSelectTask::EXHAUST_MANY: {
            const auto selected = a.getSelectedIdxs();
            if (selected.size() > bc.cardSelectInfo.pickCount) {
                return false;
            }
            for (auto x : selected) {
                if (x >= bc.cards.cardsInHand) {
                    return false;
                }
            }
            return true;
        }

        case sts::CardSelectTask::FORETHOUGHT:
        case sts::CardSelectTask::GAMBLE: {
            const auto selected = a.getSelectedIdxs();
            for (auto x : selected) {
                if (x >= bc.cards.cardsInHand) {
                    return false;
                }
            }
            return true;
        }

        default:
            return false;
    }
}

bool search::Action::isValidAction(const BattleContext &bc) const {
    if (bc.outcome != Outcome::UNDECIDED) {
        return false;
    }

    switch (getActionType()) {
        case ActionType::CARD:
            return isValidCardAction(bc, *this);

        case ActionType::POTION:
            return isValidPotionAction(bc, *this);

        case ActionType::SINGLE_CARD_SELECT:
            return isValidSingleCardSelectAction(bc, *this);

        case ActionType::MULTI_CARD_SELECT:
            return isValidMultiCardSelectAction(bc, *this);

        case ActionType::END_TURN:
            return bc.inputState == InputState::PLAYER_NORMAL;

        default:
            return false;
    }
}

std::vector<search::Action> search::Action::enumerateCombatActions(const BattleContext &bc) {
    std::vector<search::Action> actions;
    if (bc.outcome != Outcome::UNDECIDED) {
        return actions;
    }

    switch (bc.inputState) {
        case InputState::PLAYER_NORMAL: {
            if (bc.isCardPlayAllowed()) {
                for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
                    const auto &card = bc.cards.hand[handIdx];
                    if (!card.canUseOnAnyTarget(bc)) {
                        continue;
                    }
                    if (card.requiresTarget()) {
                        for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                            if (bc.monsters.arr[targetIdx].isTargetable()) {
                                actions.emplace_back(ActionType::CARD, handIdx, targetIdx);
                            }
                        }
                    } else {
                        actions.emplace_back(ActionType::CARD, handIdx);
                    }
                }
            }

            const bool hasValidTarget = bc.monsters.getTargetableCount() > 0;
            for (int potionIdx = 0; potionIdx < bc.potionCapacity; ++potionIdx) {
                const auto potion = bc.potions[potionIdx];
                if (potion == Potion::INVALID || potion == Potion::EMPTY_POTION_SLOT) {
                    continue;
                }
                if (!bc.canDrinkPotion(potionIdx) ||
                    (potionRequiresTarget(potion) && !hasValidTarget)) {
                    actions.emplace_back(ActionType::POTION, potionIdx, -1);
                } else if (!potionRequiresTarget(potion)) {
                    actions.emplace_back(ActionType::POTION, potionIdx);
                } else {
                    for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                        if (bc.monsters.arr[targetIdx].isTargetable()) {
                            actions.emplace_back(ActionType::POTION, potionIdx, targetIdx);
                        }
                    }
                }
            }
            actions.emplace_back(ActionType::END_TURN);
            break;
        }

        case InputState::CARD_SELECT:
            actions = enumerateCardSelectActions(bc);
            break;

        default:
            break;
    }

    return actions;
}

std::ostream& printSingleCardSelectDescHelper(std::ostream &os, const BattleContext &bc, const search::Action &a) {
    // action is known to be valid here
    const auto task = bc.cardSelectInfo.cardSelectTask;
    const int index = a.getSelectIdx();
    os << "{ " << cardSelectTaskStrings[static_cast<int>(task)]
       << " (" << index << ") ";

    // The index is interpreted in the source collection for the prompt, not
    // necessarily in the hand.  Include the card itself in the description so
    // lazy search-tree nodes remain understandable after actions such as
    // Armaments reorder and upgrade the hand.
    const CardInstance *card = nullptr;
    switch (task) {
        case CardSelectTask::ARMAMENTS:
        case CardSelectTask::DUAL_WIELD:
        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            if (index >= 0 && index < bc.cards.cardsInHand) {
                card = &bc.cards.hand[index];
            }
            break;
        case CardSelectTask::EXHUME:
            if (index >= 0 && index < static_cast<int>(bc.cards.exhaustPile.size())) {
                card = &bc.cards.exhaustPile[index];
            }
            break;
        case CardSelectTask::HEADBUTT:
        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
        case CardSelectTask::MEDITATE:
            if (index >= 0 && index < static_cast<int>(bc.cards.discardPile.size())) {
                card = &bc.cards.discardPile[index];
            }
            break;
        case CardSelectTask::SECRET_TECHNIQUE:
        case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK:
            if (index >= 0 && index < static_cast<int>(bc.cards.drawPile.size())) {
                card = &bc.cards.drawPile[index];
            }
            break;
        case CardSelectTask::CODEX:
        case CardSelectTask::DISCOVERY: {
            const auto &ids = bc.cardSelectInfo.cards;
            if (index >= 0 && index < static_cast<int>(ids.size())) {
                static thread_local CardInstance option;
                option = CardInstance(ids[index]);
                card = &option;
            }
            break;
        }
        default:
            break;
    }
    if (card != nullptr) {
        card->printSimpleDesc(os);
    }
    return os << " }";
}

std::ostream& printMultiCardSelectDescHelper(std::ostream &os, const BattleContext &bc, const search::Action &a) {
    // action is known to be valid here
    os << "{ " << cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)];

    const auto selected = a.getSelectedIdxs();
    if (selected.empty()) {
        return os << " none";
    }

    for (int i = 0; i < selected.size(); ++i) {
        os << " (" << a.getSelectIdx() << ") ";
        const auto &c = bc.cards.hand[selected[i]];
        c.printSimpleDesc(os);
        if (i+1 < selected.size()) {
            os << ',';
        }
    }

    return os << " }";
}

std::ostream& search::Action::printDesc(std::ostream &os, const BattleContext &bc) const {
    if (!isValidAction(bc)) {
        return os << "{ INVALID ACTION }";
    }

    switch (getActionType()) {
        case ActionType::CARD: {
            const auto &c = bc.cards.hand[getSourceIdx()];
            os << "{ use card (" << getSourceIdx() << ") " << c;
            if (c.requiresTarget()) {
                const auto &m = bc.monsters.arr[getTargetIdx()];
                os << " -> (" << getTargetIdx() << ") " << m.getName();
            }
            return os << " }";
        }

        case ActionType::POTION: {
            const auto &p = bc.potions[getSourceIdx()];
            os << "{ ";
            if (getTargetIdx() == -1) {
                os << "discard potion ";
            } else {
                os << "drink potion ";
            }

            os << "(" << getSourceIdx() << ") " << getPotionName(p);

            if (potionRequiresTarget(p)) {
                const auto &m = bc.monsters.arr[getTargetIdx()];
                os << " -> (" << getTargetIdx() << ") " << m.getName();
            }
            return os << " }";
        }

        case ActionType::SINGLE_CARD_SELECT:
            return printSingleCardSelectDescHelper(os, bc, *this);

        case ActionType::MULTI_CARD_SELECT:
            return printMultiCardSelectDescHelper(os, bc, *this);

        case ActionType::END_TURN:
            return os << "{ end turn }";

        default:
            return os;
    }
}

bool executeSingleCardSelectActionHelper(BattleContext &bc, search::Action a) {
    // assumed to be valid action here

    const auto idx = a.getSelectIdx();

    switch (bc.cardSelectInfo.cardSelectTask) {

        case CardSelectTask::ARMAMENTS:
            bc.chooseArmamentsCard(idx);
            break;

        case CardSelectTask::CODEX:
            if (idx != 3) { // idx 3 means skip
                bc.chooseCodexCard(bc.cardSelectInfo.codexCards()[idx]);
            }
            break;

        case CardSelectTask::DISCOVERY:
            bc.chooseDiscoveryCard(bc.cardSelectInfo.discovery_Cards()[idx]);
            break;

        case CardSelectTask::DUAL_WIELD:
            bc.chooseDualWieldCard(idx);
            break;

        case CardSelectTask::EXHAUST_ONE:
            bc.chooseExhaustOneCard(idx);
            break;

        case CardSelectTask::EXHUME:
            bc.chooseExhumeCard(idx);
            break;

        case CardSelectTask::FORETHOUGHT:
            bc.chooseForethoughtCard(idx);
            break;

        case CardSelectTask::HEADBUTT:
            bc.chooseHeadbuttCard(idx);
            break;

        case CardSelectTask::HOLOGRAM:
            bc.chooseDiscardToHandCard(idx, false);
            break;

        case CardSelectTask::LIQUID_MEMORIES_POTION:
            return bc.chooseLiquidMemoriesCard(idx);

        case CardSelectTask::SECRET_TECHNIQUE:
            bc.chooseDrawToHandCards(&idx, 1);
            break;

        case CardSelectTask::SECRET_WEAPON:
            bc.chooseDrawToHandCards(&idx, 1);
            break;

        case CardSelectTask::WARCRY:
            bc.chooseWarcryCard(idx);
            break;

        case CardSelectTask::MEDITATE:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::SEEK:
            // not implemented
        case CardSelectTask::INVALID:
        default:
            break;
    }
    return true;
}

void executeMultiCardSelectActionHelper(BattleContext &bc, search::Action a) {
    // assumed to be valid action here
    switch (bc.cardSelectInfo.cardSelectTask) {

        case sts::CardSelectTask::FORETHOUGHT:
            bc.chooseForethoughtCards(a.getSelectedIdxs());
            break;

        case sts::CardSelectTask::EXHAUST_MANY:
            bc.chooseExhaustCards(a.getSelectedIdxs());
            break;

        case sts::CardSelectTask::GAMBLE:
            bc.chooseGambleCards(a.getSelectedIdxs());
            break;

        default:
            break;
    }
}

void search::Action::execute(BattleContext &bc) const {
#ifdef sts_asserts
    if (!isValidAction(bc)) {
        std::cerr << bc.seed << " " << static_cast<int>(getActionType()) << " " << getSourceIdx() << " " << getTargetIdx() << std::endl;
        std::cerr <<  cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)] << std::endl;
        std::cerr << bc << std::endl;
        assert(false);
    }
#endif

    switch (getActionType()) {
        case ActionType::CARD: {
            const CardQueueItem item(bc.cards.hand[getSourceIdx()], getTargetIdx(), bc.player.energy);
            bc.addToBotCard(item);
            break;
        }

        case ActionType::POTION: {
            if (getTargetIdx() > 5) {
                bc.discardPotion(getSourceIdx());
            } else {
                bc.drinkPotion(getSourceIdx(), getTargetIdx());
            }
            break;
        }

        case ActionType::SINGLE_CARD_SELECT:
            // Partial GRID choices do not execute the waiting action queue.
            if (!executeSingleCardSelectActionHelper(bc, *this)) return;
            break;

        case ActionType::MULTI_CARD_SELECT:
            executeMultiCardSelectActionHelper(bc, *this);
            break;

        case ActionType::END_TURN:
            bc.endTurn();
            break;

        default:
            break;
    }

    bc.inputState = InputState::EXECUTING_ACTIONS;
    bc.executeActions();
}

template <typename ForwardIt>
void setupCardOptionsHelper(std::vector<search::Action> &actions, const ForwardIt begin, const ForwardIt end, const std::function<bool(const CardInstance &)> &p= nullptr) {
    for (int i = 0; begin+i != end; ++i) {
        const auto &c = begin[i];
        if (!p || (p(c))) {
            actions.emplace_back(search::ActionType::SINGLE_CARD_SELECT, i);
        }
    }
}

std::vector<search::Action> search::Action::enumerateCardSelectActions(const BattleContext &bc) {
    std::vector<search::Action> actions;

    if (bc.cardSelectInfo.isMultiSelection()) {
        for (int subset = 0; subset < (1 << bc.cards.cardsInHand); ++subset) {
            const Action candidate(ActionType::MULTI_CARD_SELECT, subset);
            if (candidate.isValidAction(bc)) actions.push_back(candidate);
        }
        return actions;
    }

    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::ARMAMENTS:
            setupCardOptionsHelper( actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) { return c.canUpgrade(); });
            break;

        case CardSelectTask::CODEX:
            for (int i = 0; i < 4; ++i) { // i -> 3 action means skip
                actions.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DISCOVERY:
            for (int i = 0; i < 3; ++i) {
                actions.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DUAL_WIELD:
            setupCardOptionsHelper( actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) {
                                        return c.getType() == CardType::POWER || c.getType() == CardType::ATTACK;
                                    });
            break;

        case CardSelectTask::EXHUME:
            setupCardOptionsHelper(actions, bc.cards.exhaustPile.begin(), bc.cards.exhaustPile.end(),
                                   [](const auto &c) { return c.getId() != CardId::EXHUME; });
            break;

        case CardSelectTask::EXHAUST_ONE:
            setupCardOptionsHelper(actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::WARCRY:
            setupCardOptionsHelper(actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::HEADBUTT:
            setupCardOptionsHelper(actions, bc.cards.discardPile.begin(), bc.cards.discardPile.end());
            break;

        case CardSelectTask::LIQUID_MEMORIES_POTION:
            for (int i = 0; i < bc.cards.discardPile.size(); ++i) {
                const Action action(ActionType::SINGLE_CARD_SELECT, i);
                if (action.isValidAction(bc)) actions.push_back(action);
            }
            break;

        case CardSelectTask::SECRET_TECHNIQUE:
            setupCardOptionsHelper(actions, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                   [] (const CardInstance &c) {
                                       return c.getType() == CardType::SKILL;
                                   });
            break;

        case CardSelectTask::SECRET_WEAPON:
            setupCardOptionsHelper(actions, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                   [] (const CardInstance &c) {
                                       return c.getType() == CardType::ATTACK;
                                   });
            break;

        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::GAMBLE:
            for (int subset = 0; subset < (1 << bc.cards.cardsInHand); ++subset) {
                const Action candidate(ActionType::MULTI_CARD_SELECT, subset);
                if (candidate.isValidAction(bc)) actions.push_back(candidate);
            }
            break;

        default:
#ifdef sts_asserts
            assert(false);
#endif
            break;
    }
    return actions;
}








std::uint32_t search::Action::remapDrawSelection(const sts::BattleContext &sampled,
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
        card.cachedBaseDamage != found->cachedBaseDamage ||
        card.upgraded != found->upgraded || card.cost != found->cost ||
        card.costForTurn != found->costForTurn || card.freeToPlayOnce != found->freeToPlayOnce ||
        card.retain != found->retain || card.exhaust != found->exhaust) {
        throw std::invalid_argument("draw-selection entity changed after sampling");
    }
    const search::Action mapped(ActionType::SINGLE_CARD_SELECT,
                        static_cast<int>(found - sampled.cards.drawPile.begin()));
    if (!mapped.isValidAction(sampled)) {
        throw std::invalid_argument("remapped draw selection is not legal in sampled state");
    }
    return mapped.bits;
}
