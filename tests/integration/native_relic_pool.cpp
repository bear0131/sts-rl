// RELICPOOL01: run each case in a separate process so native faults are reported.
#include "game/GameContext.h"
#include <iostream>
#include <stdexcept>
#include <string>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace sts;

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<RelicId> &pool(GameContext &gc, RelicTier tier) {
    switch (tier) {
        case RelicTier::COMMON: return gc.commonRelicPool;
        case RelicTier::UNCOMMON: return gc.uncommonRelicPool;
        case RelicTier::RARE: return gc.rareRelicPool;
        case RelicTier::BOSS: return gc.bossRelicPool;
        case RelicTier::SHOP: return gc.shopRelicPool;
        default: throw std::invalid_argument("no pool for this tier");
    }
}

int main(int argc, char **argv) {
#if defined(__unix__) || defined(__APPLE__)
    const rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
#endif
    if (argc != 4) return 2;
    try {
        const std::string mode(argv[1]);
        const auto tier = static_cast<RelicTier>(std::stoi(argv[2]));
        const bool front = std::stoi(argv[3]) != 0;
        GameContext gc(CharacterClass::IRONCLAD, 42, 0);
        for (const auto t : {RelicTier::COMMON, RelicTier::UNCOMMON, RelicTier::RARE,
                             RelicTier::BOSS, RelicTier::SHOP}) pool(gc, t).clear();
        std::cout << mode << " tier=" << static_cast<int>(tier) << " front=" << front << std::endl;
        if (mode == "nonempty") {
            // These two unrestricted relics distinguish ordering and consumption.
            pool(gc, tier) = {RelicId::ANCHOR, RelicId::STRAWBERRY};
            require(gc.returnRandomRelic(tier, false, front) ==
                    (front ? RelicId::ANCHOR : RelicId::STRAWBERRY), "wrong pool end");
            require(pool(gc, tier) == std::vector<RelicId>{front ? RelicId::STRAWBERRY : RelicId::ANCHOR},
                    "must consume exactly one entry");
        } else if (mode == "fallback") {
            const auto next = tier == RelicTier::UNCOMMON ? RelicTier::RARE : RelicTier::UNCOMMON;
            pool(gc, next) = {RelicId::ANCHOR, RelicId::STRAWBERRY};
            // Preserve the existing cross-tier call's default front selection.
            require(gc.returnRandomRelic(tier, false, front) == RelicId::ANCHOR, "wrong tier fallback");
            require(pool(gc, next) == std::vector<RelicId>{RelicId::STRAWBERRY}, "fallback over-consumed pool");
        } else if (mode == "shop_fallback") {
            const auto next = tier == RelicTier::UNCOMMON ? RelicTier::RARE : RelicTier::UNCOMMON;
            pool(gc, next) = {RelicId::OLD_COIN, RelicId::ANCHOR};
            require(gc.returnRandomRelic(tier, true, front) == RelicId::ANCHOR,
                    "fallback must preserve shop-only spawn restrictions");
            require(pool(gc, next).empty(), "filtered fallback did not consume both entries");
        } else if (mode == "exhausted") {
            for (int i = 0; i < 3; ++i)
                require(gc.returnRandomRelic(tier, false, front) == RelicId::CIRCLET,
                        "empty pool must terminate in ordinary Circlet, including vanilla bosses");
        } else if (mode == "filtered") {
            gc.deck = Deck(); // Bottled Flame cannot spawn without an eligible attack.
            pool(gc, tier) = {RelicId::BOTTLED_FLAME};
            require(gc.returnRandomRelic(tier, false, front) == RelicId::CIRCLET,
                    "filtered last entry must terminate through fallback");
            require(pool(gc, tier).empty(), "rejected entry not consumed");
        } else if (mode == "retry") {
            gc.deck = Deck();
            pool(gc, tier) = {RelicId::BOTTLED_FLAME, RelicId::ANCHOR, RelicId::STRAWBERRY};
            require(gc.returnRandomRelic(tier, false, true) == RelicId::STRAWBERRY,
                    "same-tier rejected front entry must retain existing back retry");
            require(pool(gc, tier) == std::vector<RelicId>{RelicId::ANCHOR}, "retry over-consumed pool");
        } else if (mode == "invalid") {
            bool rejected = false;
            try { gc.returnRandomRelic(tier, false, front); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "unsupported tier must be rejected");
        } else {
            throw std::invalid_argument("unknown case");
        }
        std::cout << "PASS" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
