//
// Created by gamerpuppy on 8/24/2021.
//

#ifndef STS_LIGHTSPEED_CARDSELECTINFO_H
#define STS_LIGHTSPEED_CARDSELECTINFO_H

#include <array>
#include <cstdint>

#include "constants/Cards.h"

namespace sts {

    enum class CardSelectTask {
        INVALID=0,
        ARMAMENTS,
        CODEX,
        DISCOVERY,
        DUAL_WIELD,
        EXHAUST_ONE,
        EXHAUST_MANY,
        EXHUME,
        FORETHOUGHT,
        GAMBLE,
        HEADBUTT,
        HOLOGRAM,
        LIQUID_MEMORIES_POTION,
        MEDITATE,
        NIGHTMARE,
        RECYCLE,
        SECRET_TECHNIQUE,
        SECRET_WEAPON,
        SEEK,
        SETUP,
        WARCRY,
    };

    static constexpr const char* cardSelectTaskStrings[] = {
            "INVALID",
            "ARMAMENTS",
            "CODEX",
            "DISCOVERY",
            "DUAL_WIELD",
            "EXHAUST_ONE",
            "EXHAUST_MANY",
            "EXHUME",
            "FORETHOUGHT",
            "GAMBLE",
            "HEADBUTT",
            "HOLOGRAM",
            "LIQUID_MEMORIES_POTION",
            "MEDITATE",
            "NIGHTMARE",
            "RECYCLE",
            "SECRET_TECHNIQUE",
            "SECRET_WEAPON",
            "SEEK",
            "SETUP",
            "WARCRY",
    };

    struct CardSelectInfo {
        std::array<CardId, 3> cards{};
        bool canPickZero = false;
        bool canPickAnyNumber = false;
        int pickCount = 0;
        int data0 = 0;

        // Mandatory GRID picks stay in the source pile until all are chosen.
        // Store physical identities so a cloned/search-shuffled pile is safe.
        static constexpr int GRID_PICK_CAPACITY = 2;
        std::array<std::int16_t, GRID_PICK_CAPACITY> selectedUids{};
        int selectedCount = 0;

        bool hasSelectedUid(int uid) const {
            for (int i = 0; i < selectedCount; ++i)
                if (selectedUids[i] == uid) return true;
            return false;
        }

        CardSelectTask cardSelectTask = CardSelectTask::INVALID;

        bool isMultiSelection() const {
            return cardSelectTask == CardSelectTask::EXHAUST_MANY ||
                cardSelectTask == CardSelectTask::GAMBLE ||
                (cardSelectTask == CardSelectTask::FORETHOUGHT && canPickAnyNumber);
        }

        std::array<CardId, 3>& discovery_Cards() { return cards; }
        int& discovery_CopyCount() { return data0; }
        int& dualWield_CopyCount() { return data0; }
        std::array<CardId, 3>& codexCards() { return cards; }
    };

}

#endif //STS_LIGHTSPEED_CARDSELECTINFO_H
