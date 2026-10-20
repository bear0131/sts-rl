//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_CARDINSTANCE_H
#define STS_LIGHTSPEED_CARDINSTANCE_H

#include <iostream>
#include <vector>
#include <string>
#include <array>

#include "constants/Cards.h"

namespace sts {

    class BattleContext;
    class Card;

    struct CardInstance  {
        // Persistent instance state; transported by the versioned card snapshot.
        CardId id = CardId::INVALID;
        std::int16_t uniqueId = -1;
        std::int16_t specialData = 0;
        // MindBlast.applyPowers captures this per object. It is transition
        // metadata, not misc/specialData or a new frozen-model feature.
        std::int32_t cachedBaseDamage = 0;
        std::int8_t cost = 0;
        std::int8_t costForTurn = 0;

        // todo dont need all of these, also put in bitset
        bool upgraded = false;
        bool freeToPlayOnce = false;
        bool retain = false;
        bool exhaust = false; // Actual instance property, not the one-use queue flag.

        CardInstance() = default;
        CardInstance(CardId id, bool upgraded=false);
        CardInstance(const Card &card);

        // New Java stat-equivalent instance, not a move/restore of this card.
        [[nodiscard]] CardInstance makeStatEquivalentCopy() const;
        void applyPowers(const BattleContext &bc);

        [[nodiscard]] CardId getId() const;
        [[nodiscard]] CardType getType() const;
        [[nodiscard]] const char *getName() const;
        [[nodiscard]] std::int16_t getUniqueId() const;
        [[nodiscard]] bool isUpgraded() const;
        [[nodiscard]] int getUpgradeCount() const;
        [[nodiscard]] bool canUpgrade() const;

        [[nodiscard]] bool isEthereal() const;
        [[nodiscard]] bool isStrikeCard() const;
        [[nodiscard]] bool doesExhaust() const;
        [[nodiscard]] bool hasSelfRetain() const;
        [[nodiscard]] bool requiresTarget() const;
        [[nodiscard]] bool isXCost() const;
        [[nodiscard]] bool isBloodCard() const;
        [[nodiscard]] bool usesSpecialData() const;

        // *****  *****
        void upgradeBaseCost(int newBaseCost); // Blood for Blood when upgraded
        void updateCost(int amount);
//        void modifyCostForCombat(int amount);

        void setCostForCombat(int cost);
        void setCostForTurn(int newCost);
        void setUniqueId(int uniqueId);
        void upgrade();
        void tookDamage(); // only call this on "Masterful Stab" and "Blood for Blood"

        std::ostream& printSimpleDesc(std::ostream &o) const;

        void triggerOnExhaust(BattleContext &bc);
        void triggerOnManualDiscard(BattleContext &bc);
        void triggerWhenDrawn(BattleContext &bc, int myHandIdx);


        [[nodiscard]] bool isFreeToPlay(const BattleContext &bc) const;

        [[nodiscard]] bool canUseOnAnyTarget(const BattleContext &bc) const; // not for use in critical path
        [[nodiscard]] bool canUse(const BattleContext &bc, int target, bool inAutoplay) const;
    };

    std::ostream& operator <<(std::ostream &os, const CardInstance &c);

    typedef CardInstance CI;


}


#endif //STS_LIGHTSPEED_CARDINSTANCE_H
