//
// Created by keega on 9/17/2021.
//

#ifndef STS_LIGHTSPEED_ACTION_H
#define STS_LIGHTSPEED_ACTION_H

#include "sts_common.h"
#include "data_structure/fixed_list.h"

#include "combat/BattleContext.h"


#include <iostream>


namespace sts {
    class BattleContext;
}


namespace sts::search {

    enum class ActionType {
        CARD=0,
        POTION,
        SINGLE_CARD_SELECT,
        MULTI_CARD_SELECT,
        END_TURN,
    };

    // couldn't make a union work in only 32 bits
    struct Action {
//        ActionType actionType;
//        int idx1;
//        int idx2;
        std::uint32_t bits = -1;

        Action() = default;
        Action(std::uint32_t bits);
        Action(ActionType actionType);
        Action(ActionType actionType, int idx1);
        Action(ActionType actionType, int idx1, int idx2);

        bool operator==(const Action &rhs) const;
        bool operator!=(const Action &rhs) const;

        [[nodiscard]] ActionType getActionType() const;
        [[nodiscard]] int getSourceIdx() const; // for card/potion actions
        [[nodiscard]] int getTargetIdx() const; // for card/potion actions

        [[nodiscard]] int getSelectIdx() const;   // for single card select action
        [[nodiscard]] fixed_list<int,10> getSelectedIdxs() const;         // for multi card select actions

        [[nodiscard]] bool isValidAction(const sts::BattleContext &bc) const;
        std::ostream& printDesc(std::ostream &os, const sts::BattleContext &bc) const;
        void execute(BattleContext &bc) const;


        // Enumerate the simulator-authoritative actions available in a combat
        // decision state. This is shared by native search and bindings so the
        // C++ trainer does not need to reconstruct the action layout itself.
        static std::vector<Action> enumerateCombatActions(const BattleContext &bc);
        // Append a live PLAYER_NORMAL turn in native order. Search omits potion discards.
        static void appendNormalActions(const BattleContext &bc, std::vector<Action> &actions, bool allowDiscard);
        static std::vector<Action> enumerateCardSelectActions(const BattleContext &bc);
        static std::uint32_t remapDrawSelection(const BattleContext &sampled,
                                               const BattleContext &source,
                                               std::uint32_t actionBits);
        static bool equivalentCards(const CardInstance &a, const CardInstance &b);
        // Same play effect, preserving all runtime card fields. Selection
        // prefixes are positional decisions and are deliberately not grouped.
        static std::array<int, 10> equivalentPlaySlots(const BattleContext &bc);
    };

}


#endif //STS_LIGHTSPEED_ACTION_H
