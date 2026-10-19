//
// Created by gamerpuppy on 6/24/2021.
//

#ifndef STS_LIGHTSPEED_RELICCONTAINER_H
#define STS_LIGHTSPEED_RELICCONTAINER_H

#include <vector>
#include <cstdint>
#include <stdexcept>

#include "sts_common.h"
#include "constants/Relics.h"

namespace sts {

    struct RelicInstance {
        RelicId id;
        int data;

        // Native persistence uses nonnegative charges/flags, not Java's
        // display sentinels. Keep save import distinct from native snapshots.
        static int fromJavaCounter(RelicId id, int value) {
            switch (id) {
                case RelicId::ANCIENT_TEA_SET:
                    if (value == -1) return 0;
                    if (value == -2) return 1;
                    break;
                case RelicId::LIZARD_TAIL:
                case RelicId::MAW_BANK:
                    if (value == -1) return 1;
                    if (value == -2) return 0;
                    break;
                case RelicId::MATRYOSHKA:
                    if (value == -2) return 0;
                    if (value >= 1 && value <= 2) return value;
                    break;
                case RelicId::NLOTHS_HUNGRY_FACE:
                    if (value == -2) return 0;
                    if (value == 1) return 1;
                    break;
                case RelicId::NEOWS_LAMENT:
                case RelicId::WING_BOOTS:
                    if (value == -2) return 0;
                    if (value >= 0 && value <= 3) return value;
                    break;
                case RelicId::OMAMORI:
                    if (value >= 0 && value <= 2) return value;
                    break;
                default:
                    return value;
            }
            throw std::invalid_argument("Java relic counter outside its declared sentinel/charge domain");
        }
    };

    class GameContext;

    struct RelicContainer {

        std::vector<RelicInstance> relics;

        std::uint64_t relicBits0 = 0;
        std::uint64_t relicBits1 = 0;
        std::uint64_t relicBits2 = 0;


        // private
        void setHasRelic(RelicId r, bool value);

        // public
        void add(RelicInstance r);
        void remove(RelicId r);
        void replaceRelic(RelicId o, RelicId r);

        int size() const;
        [[nodiscard]] bool has(RelicId r) const;
        int getRelicValue(RelicId r) const;
        int& getRelicValueRef(RelicId r);

    };


}



#endif //STS_LIGHTSPEED_RELICCONTAINER_H
