// Internal GameContext contracts, isolated by pytest so native faults are reported.
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


static bool sameRng(const Random &a, const Random &b) {
    return a.counter == b.counter && a.seed0 == b.seed0 && a.seed1 == b.seed1;
}

static void requireSameRewardState(const GameContext &a, const GameContext &b) {
    require(sameRng(a.cardRng, b.cardRng) && sameRng(a.potionRng, b.potionRng), "reward RNG differs");
    require(a.cardRarityFactor == b.cardRarityFactor && a.potionChance == b.potionChance,
            "reward probabilities differ");
    require(a.screenState == b.screenState, "reward screen differs");
    const auto &x = a.info.rewardsContainer;
    const auto &y = b.info.rewardsContainer;
    require(x.goldRewardCount == y.goldRewardCount && x.gold == y.gold, "gold reward differs");
    require(x.relicCount == y.relicCount && x.relics == y.relics, "relic reward differs");
    require(x.potionCount == y.potionCount && x.potions == y.potions, "potion reward differs");
    require(x.emeraldKey == y.emeraldKey && x.sapphireKey == y.sapphireKey, "key reward differs");
    require(x.cardRewardCount == y.cardRewardCount, "card reward count differs");
    for (int i = 0; i < x.cardRewardCount; ++i) {
        require(x.cardRewards[i].size() == y.cardRewards[i].size(), "card offer size differs");
        for (int j = 0; j < x.cardRewards[i].size(); ++j)
            require(x.cardRewards[i][j] == y.cardRewards[i][j], "card reward identity/value differs");
    }
}

static void prepareEvent(GameContext &gc, int variant) {
    const Event events[] = {Event::DEAD_ADVENTURER, Event::DEAD_ADVENTURER,
                           Event::MINDBLOOM, Event::HYPNOTIZING_COLORED_MUSHROOMS,
                           Event::MASKED_BANDITS};
    require(variant >= 0 && variant < 5, "unknown event fixture");
    gc.act = variant == 2 ? 3 : (variant == 4 ? 2 : 1);
    gc.floorNum = gc.act == 3 ? 40 : (gc.act == 2 ? 20 : 10);
    gc.curHp = 60;
    gc.gold = 500;
    gc.curRoom = Room::EVENT;
    gc.curEvent = events[variant];
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    gc.setupEvent();
    if (variant == 0) gc.info.phase = 3; // guaranteed fight with no remaining relic
    if (variant == 1) {
        gc.info.phase = 2;
        gc.info.rewards[2] = 2; // last unclaimed reward is a relic
        // Fix an RNG state whose next roll triggers combat at either ascension.
        bool found = false;
        auto candidate = gc.miscRng;
        for (int i = 0; i < 100; ++i) {
            const auto before = candidate;
            if (candidate.random(99) < 25) { gc.miscRng = before; found = true; break; }
        }
        require(found, "could not establish fixed encounter RNG");
    }
    gc.chooseEventOption(variant == 4 ? 1 : 0);
    require(gc.screenState == ScreenState::BATTLE, "fixture did not create pending battle reward");
}

static void checkEventCallback(int variant, int ascension) {
    auto root = std::make_unique<GameContext>(CharacterClass::IRONCLAD, 200207, ascension);
    prepareEvent(*root, variant);
    const auto before = *root;
    auto child = *root;
    auto sibling = *root;
    auto detached = *root;
    GameContext direct(CharacterClass::IRONCLAD, 200207, ascension);
    prepareEvent(direct, variant);
    direct.regainControl();
    child.regainControl();
    requireSameRewardState(*root, before);
    require(child.cardRng.counter > before.cardRng.counter, "child card RNG did not advance");
    require(child.screenState == ScreenState::REWARDS, "child did not receive rewards");
    requireSameRewardState(child, direct);
    sibling.regainControl();
    requireSameRewardState(child, sibling);
    // A captured origin must not affect either live or detached descendants.
    root->cardRng.randomLong();
    root->potionRng.randomLong();
    root.reset();
    detached.regainControl();
    requireSameRewardState(detached, direct);
    detached.regainControl();
    require(detached.screenState == ScreenState::MAP_SCREEN, "reward callback did not hand back to map");
}

static void checkBossCallback(int act, int ascension) {
    require(act == 1 || act == 2, "invalid boss act");
    auto root = std::make_unique<GameContext>(CharacterClass::IRONCLAD, 200207, ascension);
    root->act = act;
    root->enterBossTreasureRoom();
    auto child = *root;
    auto detached = *root;
    root->act = act == 1 ? 2 : 1;
    child.regainControl();
    require(child.act == act + 1 && child.screenState == ScreenState::MAP_SCREEN,
            "boss callback read origin act instead of child's act");
    require(root->screenState == ScreenState::BOSS_RELIC_REWARDS, "boss callback mutated origin screen");
    root.reset();
    detached.regainControl();
    require(detached.act == act + 1 && detached.screenState == child.screenState,
            "detached boss callback used destroyed origin");
    require(sameRng(detached.miscRng, child.miscRng), "detached act transition differs");
}

int main(int argc, char **argv) {
#if defined(__unix__) || defined(__APPLE__)
    const rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
#endif
    if (argc != 4) return 2;
    try {
        const std::string mode(argv[1]);
        if (mode == "event" || mode == "boss") {
            if (mode == "event") checkEventCallback(std::stoi(argv[2]), std::stoi(argv[3]));
            else checkBossCallback(std::stoi(argv[2]), std::stoi(argv[3]));
            std::cout << "PASS" << std::endl;
            return 0;
        }
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
