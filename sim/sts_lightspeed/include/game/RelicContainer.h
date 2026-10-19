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
            const auto bounded = [](int input, int minimum, int maximum) {
                if (input < minimum || input > maximum)
                    throw std::invalid_argument("Java relic counter outside its numeric domain");
                return input == -1 ? 0 : input;
            };
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
                case RelicId::CIRCLET: return bounded(value, 1, 2147483647);
                case RelicId::DU_VU_DOLL: return bounded(value, 0, 2147483647);
                case RelicId::GIRYA: return bounded(value, 0, 3);
                case RelicId::HAPPY_FLOWER: return bounded(value, 0, 2);
                case RelicId::INCENSE_BURNER: return bounded(value, 0, 5);
                case RelicId::INK_BOTTLE:
                case RelicId::NUNCHAKU:
                case RelicId::PEN_NIB: return bounded(value, 0, 9);
                case RelicId::INSERTER: return bounded(value, 0, 1);
                case RelicId::SUNDIAL: return bounded(value, 0, 2);
                case RelicId::TINY_CHEST: return bounded(value, 0, 3);
                case RelicId::CAPTAINS_WHEEL:
                case RelicId::KUNAI:
                case RelicId::LETTER_OPENER:
                case RelicId::ORNAMENTAL_FAN:
                case RelicId::SHURIKEN: return bounded(value, -1, 2);
                case RelicId::HORN_CLEAT: return bounded(value, -1, 1);
                case RelicId::POCKETWATCH:
                case RelicId::STONE_CALENDAR: return bounded(value, -1, 2147483647);
                case RelicId::VELVET_CHOKER: return bounded(value, -1, 6);
                default:
                    if (value == -1) return 0; // explicit inherited display sentinel
                    break;
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
