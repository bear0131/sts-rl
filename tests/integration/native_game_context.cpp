// Internal GameContext contracts, isolated by pytest so native faults are reported.
#include "game/GameContext.h"
#include "sim/search/GameAction.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
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


static void checkReward(int variant, int ascension) {
    using A = search::GameAction;
    using T = A::RewardsActionType;
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    Rewards r;
    if (variant == 0) {
        r.addGold(13); r.addGold(29); gc.openCombatRewardScreen(r);
        std::vector<int> indices;
        for (const auto a : A::getAllActionsInState(gc))
            if (a.getRewardsActionType() == T::GOLD) indices.push_back(a.getIdx1());
        require(indices == std::vector<int>{0, 1}, "gold rewards lost their indices");
        const int before = gc.gold;
        A(T::GOLD, 1).execute(gc); A(T::GOLD, 0).execute(gc);
        require(gc.gold == before + 42 && gc.info.rewardsContainer.goldRewardCount == 0,
                "gold reward consumed wrong entry");
    } else if (variant == 1) {
        gc.obtainRelic(RelicId::SINGING_BOWL);
        for (auto id : {CardId::ANGER, CardId::BASH}) {
            CardReward cards; cards.push_back(Card(id)); r.addCardReward(cards);
        }
        r.addGold(19); gc.openCombatRewardScreen(r);
        std::vector<int> skips;
        for (const auto a : A::getAllActionsInState(gc))
            if (a.getRewardsActionType() == T::CARD && a.getIdx2() == 5) skips.push_back(a.getIdx1());
        require(skips == std::vector<int>{0, 1}, "per-group card skip missing");
        const auto hp = gc.maxHp; const auto size = gc.deck.size();
        A(T::CARD, 1, 5).execute(gc);
        require(gc.maxHp == hp + 2 && gc.deck.size() == size, "bowl skip effect wrong");
        require(gc.info.rewardsContainer.cardRewardCount == 1 &&
                gc.info.rewardsContainer.cardRewards[0][0].id == CardId::ANGER &&
                gc.info.rewardsContainer.gold[0] == 19, "skip consumed unrelated rewards");
    } else if (variant == 2) {
        for (int mode = 0; mode < 3; ++mode) {
            auto child = gc;
            if (mode == 1) child.obtainRelic(RelicId::SOZU);
            if (mode == 2) for (int i = 0; i < child.potionCapacity; ++i) child.obtainPotion(Potion::BLOOD_POTION);
            Rewards offer; offer.addPotion(Potion::FIRE_POTION); child.openCombatRewardScreen(offer);
            const bool expected = mode == 0;
            require(A(T::POTION, 0).isValidAction(child) == expected, "potion validator ignores capacity/Sozu");
            bool listed = false;
            for (const auto a : A::getAllActionsInState(child)) {
                require(a.isValidAction(child), "reward enumerator emitted invalid action");
                if (a.getRewardsActionType() == T::POTION) listed = true;
            }
            require(listed == expected, "potion enumeration disagrees with capacity/Sozu");
        }
    } else if (variant == 3) {
        gc.curEvent = Event::BIG_FISH; gc.setupEvent(); gc.maxHp = 80; gc.curHp = 20;
        for (bool discard : {false, true}) {
            auto child = gc;
            child.obtainPotion(Potion::BLOOD_POTION);
            A action(static_cast<std::uint32_t>(discard ? 0xc0000000U : 0x80000000U));
            require(action.isValidAction(child), "invalid potion fixture");
            action.execute(child);
            require(child.screenState == ScreenState::EVENT_SCREEN &&
                    child.curHp == (discard ? 20 : 36) && child.potionCount == 0,
                    "potion action also executed event option");
        }
    } else if (variant == 4) {
        const CardId ids[] = {CardId::ANGER, CardId::BASH, CardId::CLEAVE, CardId::POMMEL_STRIKE, CardId::INFLAME};
        for (auto id : ids) { CardReward cards; cards.push_back(Card(id)); r.addCardReward(cards); }
        r.removeCardReward(2);
        require(r.cardRewardCount == 4, "card reward count not reduced");
        for (int i = 0; i < 4; ++i) require(r.cardRewards[i][0].id == ids[i < 2 ? i : i + 1], "card group shift wrong");
        while (r.cardRewardCount) r.removeCardReward(r.cardRewardCount - 1);
        require(r.cardRewardCount == 0, "last card group not removed");
    } else if (variant == 5) {
        gc.deck = Deck(); gc.deck.obtainRaw(Card(CardId::ANGER));
        r.addGold(19); r.addRelic(RelicId::BOTTLED_FLAME); r.sapphireKey = true;
        gc.openCombatRewardScreen(r);
        A(T::RELIC, 0).execute(gc);
        require(gc.screenState == ScreenState::CARD_SELECT, "bottle did not open selection");
        auto child = gc;
        child.chooseSelectCardScreenOption(0);
        require(child.screenState == ScreenState::REWARDS, "bottle selection skipped remaining rewards");
        require(gc.screenState == ScreenState::CARD_SELECT, "bottle selection mutated origin");
        require(!child.info.rewardsContainer.sapphireKey && child.info.rewardsContainer.gold[0] == 19,
                "bottle lost gold or retained linked sapphire key");
        A(T::GOLD, 0).execute(child); A(T::SKIP).execute(child);
        require(child.screenState == ScreenState::MAP_SCREEN, "original proceed callback was lost");
    } else if (variant == 6) {
        r.addRelic(RelicId::ANCHOR);
        require(r.getTotalCount() == 1, "relic-only reward counted as empty");
    } else if (variant == 7) {
        r.addGold(19); r.addRelic(RelicId::CAULDRON); r.addRelic(RelicId::ANCHOR);
        r.sapphireKey = true;
        gc.openCombatRewardScreen(r);
        A(T::RELIC, 0).execute(gc);
        require(gc.info.rewardsContainer.relicCount == 0 && gc.info.rewardsContainer.potionCount == 5,
                "nested relic reward corrupted replacement reward container");
        const int capacity = gc.potionCapacity;
        for (int i = 0; i < capacity; ++i) {
            require(A(T::POTION, 0).isValidAction(gc), "nested potion missing");
            A(T::POTION, 0).execute(gc);
        }
        A(T::SKIP).execute(gc);
        require(gc.screenState == ScreenState::REWARDS && gc.info.rewardsContainer.gold[0] == 19 &&
                gc.info.rewardsContainer.relicCount == 1 && gc.info.rewardsContainer.relics[0] == RelicId::ANCHOR &&
                gc.info.rewardsContainer.sapphireKey, "nested screen lost parent rewards");
        A(T::RELIC, 0).execute(gc);
        require(!gc.info.rewardsContainer.sapphireKey, "parent sapphire link was lost");
        A(T::SKIP).execute(gc);
        require(gc.screenState == ScreenState::MAP_SCREEN, "nested reward return lost continuation");
    } else { throw std::invalid_argument("unknown reward fixture"); }
}

static void checkShopEgg(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.screenState = ScreenState::SHOP_ROOM;
    gc.gold = 10000;
    auto &s = gc.info.shop;
    if (variant == 0) {
        auto plain = gc;
        plain.info.shop.setupCards(plain);
        for (auto egg : {RelicId::MOLTEN_EGG, RelicId::TOXIC_EGG, RelicId::FROZEN_EGG}) gc.obtainRelic(egg);
        s.setupCards(gc);
        for (int i = 0; i < 7; ++i) {
            require(s.cards[i].getId() == plain.info.shop.cards[i].getId(), "egg changed generated card identity");
            require(s.cardPrice(i) == plain.info.shop.cardPrice(i), "egg changed card price");
            require(s.cards[i].getUpgraded() == 1, "egg missing from initial shelf preview");
        }
        require(sameRng(gc.cardRng, plain.cardRng) && sameRng(gc.merchantRng, plain.merchantRng), "egg preview changed RNG");
    } else if (variant >= 1 && variant <= 3) {
        const RelicId egg = variant == 1 ? RelicId::MOLTEN_EGG :
                            variant == 2 ? RelicId::TOXIC_EGG : RelicId::FROZEN_EGG;
        const CardType type = variant == 1 ? CardType::ATTACK :
                              variant == 2 ? CardType::SKILL : CardType::POWER;
        for (bool courier : {false, true}) {
            auto child = gc;
            if (courier) child.obtainRelic(RelicId::THE_COURIER);
            child.commonRelicPool = {RelicId::ANCHOR};
            child.uncommonRelicPool = {RelicId::KUNAI};
            child.rareRelicPool = {RelicId::GINGER};
            auto &shop = child.info.shop;
            const CardId ids[] = {CardId::SEARING_BLOW, CardId::ANGER, CardId::SHRUG_IT_OFF,
                CardId::DEFEND_RED, CardId::INFLAME, CardId::FLASH_OF_STEEL, CardId::APOTHEOSIS};
            for (int i = 0; i < 7; ++i) {
                shop.cards[i] = Card(ids[i]);
                shop.cardPrice(i) = 50;
            }
            shop.relics[0] = egg;
            shop.relicPrice(0) = 200;
            shop.buyRelic(child, 0);
            require(child.hasRelic(egg) && child.gold == 9800, "egg purchase not completed");
            if (courier) require(!isEggRelic(shop.relics[0]), "fixture must restock a non-egg relic");
            for (int i = 0; i < 7; ++i) {
                const int expected = shop.cards[i].getType() == type ? 1 : 0;
                require(shop.cards[i].getUpgraded() == expected, "purchased egg did not refresh matching shelf cards");
                require(shop.cardPrice(i) == 50, "egg purchase changed card prices");
            }
            shop.buyCard(child, 0);
            require(child.deck.cards[child.deck.size() - 1].getUpgraded() == (type == CardType::ATTACK ? 1 : 0),
                    "buying previewed Searing Blow upgraded it again");
        }
    } else if (variant == 4) {
        gc.obtainRelic(RelicId::MOLTEN_EGG);
        for (int upgrades : {0, 1, 3}) {
            Card card(CardId::SEARING_BLOW);
            for (int i = 0; i < upgrades; ++i) card.upgrade();
            for (int i = 0; i < 3; ++i) card = gc.previewObtainCard(card);
            const int expected = upgrades == 0 ? 1 : upgrades;
            require(card.getUpgraded() == expected, "repeated egg preview added upgrades");
            const int before = gc.deck.getUpgradeableCount();
            gc.deck.obtain(gc, card, 2);
            require(gc.deck.getUpgradeableCount() == before + 2, "Searing Blow remains upgradeable after egg");
            require(gc.deck.cards[gc.deck.size() - 1].getUpgraded() == expected, "obtain added another egg upgrade");
        }
    } else if (variant == 5) {
        for (auto egg : {RelicId::MOLTEN_EGG, RelicId::TOXIC_EGG, RelicId::FROZEN_EGG}) gc.obtainRelic(egg);
        for (auto id : {CardId::SEARING_BLOW, CardId::ANGER, CardId::SHRUG_IT_OFF, CardId::INFLAME}) {
            gc.deck.obtain(gc, Card(id), 2);
            require(gc.deck.cards[gc.deck.size() - 1].getUpgraded() == 1, "direct obtain must upgrade exactly once");
        }
        int actual = 0;
        for (const auto &card : gc.deck.cards) actual += card.canUpgrade();
        require(gc.deck.getUpgradeableCount() == actual, "egg obtain corrupted cached upgradeable count");
    } else { throw std::invalid_argument("unknown shop-egg fixture"); }
}

static void checkShopLegality(int variant, int ascension) {
    using RAT = search::GameAction::RewardsActionType;
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.screenState = ScreenState::SHOP_ROOM;
    gc.gold = 10000;
    gc.info.shop.setup(gc);
    if (variant == 1) gc.obtainRelic(RelicId::SOZU);
    else if (variant == 2) {
        while (gc.potionCount < gc.potionCapacity) gc.obtainPotion(Potion::BLOCK_POTION);
    } else if (variant == 3) gc.gold = 0;
    else if (variant == 4) {
        for (auto &price : gc.info.shop.prices) price = -1;
        gc.info.shop.removeCost = -1;
    }
    const auto actions = search::GameAction::getAllActionsInState(gc);
    for (const auto &action : actions) require(action.isValidAction(gc), "enumerated illegal shop action");
    for (int slot = 0; slot < 8; ++slot) {
        for (auto type : {RAT::CARD, RAT::RELIC, RAT::POTION}) {
            const search::GameAction action(type, slot);
            bool listed = false;
            for (const auto &candidate : actions) {
                if (candidate.getRewardsActionType() == type && candidate.getIdx1() == slot) listed = true;
            }
            require(action.isValidAction(gc) == listed, "shop validator disagrees with enumeration");
            if (type == RAT::POTION) {
                require(action.isValidAction(gc) == (variant == 0 && slot < 3),
                        "potion validator ignores capacity/Sozu/gold/stock");
            }
        }
    }
}

static void checkShopPrice(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.gold = 10000;
    gc.screenState = ScreenState::SHOP_ROOM;
    auto &shop = gc.info.shop;
    if (variant < 4) {
        if (variant & 1) gc.obtainRelic(RelicId::THE_COURIER);
        if (variant & 2) gc.obtainRelic(RelicId::MEMBERSHIP_CARD);
        auto raw = gc;
        raw.info.shop.setupCards(raw);
        raw.info.shop.setupRelics(raw);
        raw.info.shop.setupPotions(raw);
        shop.setup(gc);
        for (int i = 0; i < 13; ++i) {
            int expected = raw.info.shop.prices[i];
            if (ascension >= 16) expected = std::lround(expected * 1.1f);
            if (variant & 1) expected = std::lround(expected * 0.8f);
            if (variant & 2) expected = std::lround(expected * 0.5f);
            require(shop.prices[i] == expected, "initial shelf price violates A16/discount order");
        }
        require(sameRng(raw.merchantRng, gc.merchantRng), "price modifiers consumed RNG");
    } else if (variant == 4) {
        for (int discounts = 0; discounts < 4; ++discounts) {
            auto child = gc;
            if (discounts & 1) child.obtainRelic(RelicId::THE_COURIER);
            if (discounts & 2) child.obtainRelic(RelicId::MEMBERSHIP_CARD);
            for (bool colorless : {false, true}) {
                auto fullPrice = gc;
                child.merchantRng = fullPrice.merchantRng;
                const int p = Shop::getNewCardPrice(child, CardRarity::RARE, colorless);
                // A rare card is 150 +/-10%, colorless adds 20%; restocks ignore A16.
                const float discount = (discounts & 1 ? 0.8f : 1.f) * (discounts & 2 ? 0.5f : 1.f);
                const float color = colorless ? 1.2f : 1.f;
                require(p >= static_cast<int>(135 * color * discount) &&
                        p <= static_cast<int>(165 * color * discount), "restocked card discount/range wrong");
            }
        }
    } else if (variant == 5) {
        gc.obtainRelic(RelicId::THE_COURIER);
        for (bool membership : {false, true}) {
            auto child = gc;
            if (membership) child.obtainRelic(RelicId::MEMBERSHIP_CARD);
            child.commonRelicPool = {RelicId::ANCHOR};
            child.uncommonRelicPool = {RelicId::KUNAI};
            child.rareRelicPool = {RelicId::GINGER};
            auto &s = child.info.shop;
            s.relics[0] = RelicId::STRAWBERRY;
            s.relicPrice(0) = 1; // Deliberately distinguish old shelf price from the replacement.
            s.buyRelic(child, 0);
            require(child.gold == 9999 && child.hasRelic(RelicId::STRAWBERRY), "relic purchase charged wrong item");
            const int base = getRelicBasePrice(s.relics[0]);
            int low = std::lround(std::lround(base * 0.95f) * 0.8f);
            int high = std::lround(std::lround(base * 1.05f) * 0.8f);
            if (membership) { low = std::lround(low * 0.5f); high = std::lround(high * 0.5f); }
            require(s.relicPrice(0) >= low && s.relicPrice(0) <= high, "restocked relic retained previous price");
        }
    } else if (variant == 6) {
        gc.obtainRelic(RelicId::SMILING_MASK);
        for (int discounts = 0; discounts < 4; ++discounts) {
            auto child = gc;
            if (discounts & 1) child.obtainRelic(RelicId::THE_COURIER);
            if (discounts & 2) child.obtainRelic(RelicId::MEMBERSHIP_CARD);
            for (int removed : {0, 3}) {
                child.shopRemoveCount = removed;
                require(Shop::getRemoveCost(child) == 50, "Smiling Mask removal price must remain 50");
            }
        }
    } else if (variant == 7 || variant == 8) {
        gc.obtainRelic(RelicId::SMILING_MASK);
        shop.setup(gc);
        shop.removeCost = variant == 7 ? 50 : -1;
        shop.relics[0] = RelicId::MEMBERSHIP_CARD;
        const int cost = shop.relicPrice(0);
        shop.buyRelic(gc, 0);
        require(gc.gold == 10000 - cost, "membership purchase used discounted purchase price");
        require(shop.removeCost == (variant == 7 ? 50 : -1), "membership changed fixed/used removal price");
    } else if (variant == 9) {
        // ShopScreen.init applies each removal discount to the base purge cost;
        // Membership is last. Item prices still stack both reductions.
        gc.obtainRelic(RelicId::THE_COURIER);
        for (int removed : {0, 1, 3}) {
            auto child = gc;
            child.shopRemoveCount = removed;
            auto &s = child.info.shop;
            s.setup(child);
            s.relics[0] = RelicId::MEMBERSHIP_CARD;
            const int expected = std::lround((75 + removed * 25) * 0.5f);
            s.buyRelic(child, 0);
            require(s.removeCost == expected, "buying membership stacked removal discounts");
            require(Shop::getRemoveCost(child) == expected, "entering shop stacked removal discounts");
        }
    } else { throw std::invalid_argument("unknown shop-price fixture"); }
}

int main(int argc, char **argv) {
#if defined(__unix__) || defined(__APPLE__)
    const rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
#endif
    if (argc != 4) return 2;
    try {
        const std::string mode(argv[1]);
        if (mode == "event" || mode == "boss" || mode == "reward" || mode == "shop-price" || mode == "shop-legality" || mode == "shop-egg") {
            if (mode == "event") checkEventCallback(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "reward") checkReward(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-price") checkShopPrice(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-legality") checkShopLegality(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-egg") checkShopEgg(std::stoi(argv[2]), std::stoi(argv[3]));
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
