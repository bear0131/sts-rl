// Internal GameContext contracts, isolated by pytest so native faults are reported.
#include "game/GameContext.h"
#include "game/Game.h"
#include "combat/BattleContext.h"
#include "sim/search/GameAction.h"
#include "sim/search/NoslCombatTrainer.h"
#include "sim/search/BattleScumSearcher2.h"
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

static void checkDynamicExhaustSearch(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    BattleContext bc;
    bc.init(gc, MonsterEncounter::CULTIST);
    bc.cards.cardsInHand = 2;
    bc.cards.hand[0] = CardInstance(CardId::STRIKE_RED);
    bc.cards.hand[1] = CardInstance(CardId::STRIKE_RED);
    bc.cards.hand[0].uniqueId = 0;
    bc.cards.hand[1].uniqueId = 1;
    bc.cards.drawPile.clear(); bc.cards.discardPile.clear(); bc.cards.exhaustPile.clear();
    CardInstance *changed = &bc.cards.hand[1];
    if (variant > 0 && variant < 4) {
        auto &pile = variant == 1 ? bc.cards.drawPile : (variant == 2 ? bc.cards.discardPile : bc.cards.exhaustPile);
        pile.push_back(CardInstance(CardId::STRIKE_RED));
        pile.back().uniqueId = 2;
        changed = &pile.back();
    }
    bc.cards.recomputeDerivedState();
    if (variant == 4) {
        search::BattleScumSearcher2 searcher(bc);
        search::BattleScumSearcher2::Node same, different;
        searcher.enumerateCardActions(same, bc);
        changed->exhaust = true;
        searcher.enumerateCardActions(different, bc);
        require(same.edges.size() == 1 && different.edges.size() == 2,
                "scum search merged cards with different exhaustion futures");
    } else {
        search::NoslCombatSearchConfig config;
        config.simulations = 1; config.maxDepth = 1;
        search::NoslCombatTrainer trainer(config);
        const auto baseline = trainer.search(bc, 42).stateKey;
        changed->exhaust = true;
        require(trainer.search(bc, 42).stateKey != baseline, "NOSL merged different dynamic exhaustion states");
        changed->exhaust = false;
        require(trainer.search(bc, 42).stateKey == baseline, "restoring printed exhaustion changed state key");
    }
}

static void checkTransformUpgrade(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    const CardId originals[] = {CardId::SEARING_BLOW, CardId::FLASH_OF_STEEL, CardId::PARASITE};
    require(variant >= 0 && variant < 3, "invalid transform color variant");
    const auto original = originals[variant];
    for (int seed = 1; seed <= 64; ++seed) {
        Random normalRng(seed), upgradedRng(seed);
        const auto normal = gc.getTransformedCard(normalRng, original, false);
        const auto upgraded = gc.getTransformedCard(upgradedRng, original, true);
        require(normal.id == upgraded.id && normal.id != original, "autoUpgrade changed transformed identity");
        require(getCardColor(normal.id) == getCardColor(original), "transform changed color pool");
        require(normal.getUpgraded() == 0, "ordinary transform unexpectedly upgraded");
        require(upgraded.getUpgraded() == (normal.canUpgrade() ? 1 : 0), "transform omitted requested upgrade or upgraded a curse");
        require(sameRng(normalRng, upgradedRng), "autoUpgrade changed RNG consumption");
    }
}

static void checkRelicSelection(int variant, int ascension) {
    const bool reward = variant >= 14;
    const bool boss = variant >= 10 && !reward;
    const bool astrolabe = reward ? variant >= 16 : (boss ? variant >= 12 : variant >= 5);
    const int count = variant >= 10 ? (variant % 2 ? 4 : 2) : variant % 5;
    const int limit = astrolabe ? 3 : 2;
    const auto relic = astrolabe ? RelicId::ASTROLABE : RelicId::EMPTY_CAGE;
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.deck = Deck{};
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    gc.deck.obtainRaw(Card(CardId::ASCENDERS_BANE));
    const CardId ids[] = {CardId::ANGER, CardId::PARASITE, CardId::INFLAME, CardId::SHRUG_IT_OFF};
    const RelicId bottles[] = {RelicId::BOTTLED_FLAME, RelicId::INVALID, RelicId::BOTTLED_TORNADO, RelicId::BOTTLED_LIGHTNING};
    for (int i = 0; i < count; ++i) {
        gc.deck.obtainRaw(Card(ids[i]));
        if (bottles[i] != RelicId::INVALID) {
            require(gc.obtainRelic(bottles[i]), "relic selection bottle setup failed");
            gc.chooseSelectCardScreenOption(0);
        }
    }
    gc.obtainRelic(RelicId::CERAMIC_FISH);
    if (boss) {
        gc.enterBossTreasureRoom();
        gc.info.bossRelics[0] = relic;
    } else {
        gc.screenState = ScreenState::REWARDS;
        gc.regainControlAction = [](GameContext &g) {
            require(g.hasRelic(RelicId::EMPTY_CAGE) || g.hasRelic(RelicId::ASTROLABE),
                    "selection resumed before relic acquisition finished");
            g.screenState = ScreenState::REWARDS;
            ++g.info.eventData;
        };
        gc.info.eventData = 0;
    }
    if (reward) {
        Rewards offer;
        offer.addRelic(relic);
        offer.addGold(19);
        gc.openCombatRewardScreen(offer);
    }
    const auto before = gc;
    std::vector<int> chosen;
    if (count <= limit) {
        // onEquip builds a temporary list with addToTop, reversing deck order.
        for (int i = count; i > 0; --i) chosen.push_back(i);
    } else {
        chosen = {count, 1};
        if (astrolabe) chosen.push_back(2);
    }
    auto expected = gc;
    std::vector<Card> generated;
    if (astrolabe) for (int i : chosen)
        generated.push_back(expected.getTransformedCard(expected.miscRng, before.deck.cards[i].id, true));
    const int hpLoss = std::find(chosen.begin(), chosen.end(), 2) != chosen.end() ? 3 : 0;
    const bool opens = count > limit;
    if (boss) gc.chooseBossRelic(0);
    else if (reward) search::GameAction(search::GameAction::RewardsActionType::RELIC, 0).execute(gc);
    else require(gc.obtainRelic(relic) == opens, "relic opened selection with no player choice, or skipped required choice");
    if (opens) {
        require(gc.screenState == ScreenState::CARD_SELECT && gc.info.toSelectCards.size() == count,
                "relic selection excluded bottled cards or included permanent curse");
        require(gc.info.toSelectCount == limit, "wrong relic pick count");
        gc.chooseSelectCardScreenOption(count - 1);
        gc.chooseSelectCardScreenOption(0);
        if (astrolabe) gc.chooseSelectCardScreenOption(0);
    }
    require(gc.hasRelic(relic), "relic was not retained");
    require(gc.screenState == (boss ? ScreenState::MAP_SCREEN : ScreenState::REWARDS), "relic left wrong continuation screen");
    if (boss) require(gc.act == 2, "boss relic did not advance exactly one act");
    else {
        require(gc.info.eventData == (opens && !reward ? 1 : 0), "automatic relic incorrectly invoked continuation");
        require(sameRng(gc.miscRng, expected.miscRng), "relic consumed wrong misc RNG stream");
        require(sameRng(gc.cardRng, before.cardRng) && sameRng(gc.merchantRng, before.merchantRng),
                "relic altered unrelated RNG stream");
    }
    require(gc.maxHp == before.maxHp - hpLoss, "relic lost Parasite removal side effect");
    require(gc.gold == before.gold + 9 * generated.size(), "relic lost obtain effect or acquired a card twice");
    int dst = 0;
    for (int i = 0; i <= count; ++i) {
        if (std::find(chosen.begin(), chosen.end(), i) != chosen.end()) continue;
        require(gc.deck.cards[dst].id == before.deck.cards[i].id, "relic removed an unselected card");
        if (before.deck.isCardBottled(i)) require(gc.deck.isCardBottled(dst), "surviving bottle index was lost");
        ++dst;
    }
    for (const auto &card : generated) {
        require(gc.deck.cards[dst].id == card.id && gc.deck.cards[dst].getUpgraded() == card.getUpgraded(),
                "Astrolabe transform order or upgrade mismatch");
        require(!gc.deck.isCardBottled(dst), "transformed card inherited bottle");
        ++dst;
    }
    require(gc.deck.size() == dst, "relic changed wrong number of cards");
    if (reward) {
        require(gc.info.rewardsContainer.relicCount == 0 && gc.info.rewardsContainer.goldRewardCount == 1,
                "relic lost remaining reward or remained claimable");
        search::GameAction(search::GameAction::RewardsActionType::SKIP).execute(gc);
        require(gc.info.eventData == 1, "relic did not restore original reward continuation");
    }
    for (int i : chosen) if (before.deck.isCardBottled(i))
        require(gc.deck.bottleIdxs[static_cast<int>(before.deck.cards[i].getType())] == -1,
                "removed card retained bottle binding");
}

static void checkAugmenterRewards(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.deck = Deck{};
    gc.curEvent = Event::AUGMENTER;
    gc.screenState = ScreenState::EVENT_SCREEN;
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    gc.gold = 12;
    gc.curHp = gc.maxHp = 80;
    if (variant < 2) {
        if (variant == 1) {
            gc.obtainRelic(RelicId::TOXIC_EGG);
            gc.obtainRelic(RelicId::CERAMIC_FISH);
        }
        gc.chooseEventOption(0);
        require(gc.deck.size() == 1 && gc.deck.cards[0].id == CardId::JAX &&
                gc.deck.cards[0].getUpgraded() == variant, "JAX gain/egg mismatch");
        require(gc.gold == 12 + 9 * variant && gc.screenState == ScreenState::MAP_SCREEN,
                "JAX obtain side effect/continuation mismatch");
    } else if (variant < 5) {
        if (variant >= 3) gc.obtainRelic(RelicId::MUTAGENIC_STRENGTH);
        if (variant == 4) gc.obtainRelic(RelicId::CIRCLET);
        gc.chooseEventOption(2);
        require(gc.hasRelic(RelicId::MUTAGENIC_STRENGTH), "missing Mutagenic Strength");
        require(gc.hasRelic(RelicId::CIRCLET) == (variant >= 3), "duplicate Mutagens must give Circlet");
        if (variant >= 3) require(gc.relics.getRelicValue(RelicId::CIRCLET) == variant - 2,
                                 "duplicate Circlet counter mismatch");
    } else if (variant < 11) {
        const int charges = (variant - 5) % 3;
        const bool bloom = variant >= 8;
        gc.obtainRelic(RelicId::OMAMORI);
        gc.relics.getRelicValueRef(RelicId::OMAMORI) = charges;
        for (auto r : {RelicId::DARKSTONE_PERIAPT, RelicId::CERAMIC_FISH, RelicId::DU_VU_DOLL}) gc.obtainRelic(r);
        if (bloom) gc.obtainRelic(RelicId::MARK_OF_THE_BLOOM);
        for (auto c : {CardId::PARASITE, CardId::PAIN, CardId::ASCENDERS_BANE}) gc.deck.obtainRaw(Card(c));
        gc.chooseEventOption(1);
        gc.chooseSelectCardScreenOption(0);
        gc.chooseSelectCardScreenOption(0);
        const int obtained = 2 - charges;
        require(gc.deck.size() == 1 + obtained && gc.deck.cards[0].id == CardId::ASCENDERS_BANE,
                "Omamori transform count/permanent curse mismatch");
        require(gc.maxHp == 77 + 6 * obtained && gc.curHp == (bloom ? 77 : gc.maxHp),
                "Darkstone/Parasite/Bloom transform HP mismatch");
        require(gc.gold == 12 + 9 * obtained && gc.relics.getRelicValue(RelicId::OMAMORI) == 0,
                "blocked curses must not trigger gain effects");
        require(gc.deck.getCountMatching([](const Card &c) { return c.getType() == CardType::CURSE; }) == 1 + obtained, "physical curse count mismatch");
    } else if (variant == 11) {
        gc.curHp = gc.maxHp = 2;
        gc.deck.obtainRaw(Card(CardId::PARASITE));
        gc.deck.remove(gc, 0);
        require(gc.maxHp == 1 && gc.curHp == 1, "Parasite must not lower maximum HP below one");
    } else if (variant == 12) {
        gc.obtainRelic(RelicId::OMAMORI);
        gc.obtainRelic(RelicId::DARKSTONE_PERIAPT);
        gc.obtainRelic(RelicId::CERAMIC_FISH);
        gc.deck.obtain(gc, Card(CardId::PAIN), 3);
        require(gc.deck.size() == 1 && gc.maxHp == 86 && gc.curHp == 86 && gc.gold == 21,
                "bulk curse obtain must apply effects only to the unblocked card");
    } else if (variant >= 13 && variant <= 15) {
        gc.curHp = 30;
        gc.obtainRelic(RelicId::DARKSTONE_PERIAPT);
        if (variant == 14) gc.obtainRelic(RelicId::MARK_OF_THE_BLOOM);
        gc.deck.obtain(gc, Card(variant == 15 ? CardId::BASH : CardId::PAIN), 2);
        require(gc.maxHp == (variant == 15 ? 80 : 92), "max HP must count only obtained curses");
        require(gc.curHp == (variant == 13 ? 42 : 30), "curse gain heals by six, not to full");
    } else if (variant == 16) {
        gc.curHp = gc.maxHp = 1;
        gc.deck.obtainRaw(Card(CardId::PARASITE));
        gc.deck.remove(gc, 0);
        require(gc.maxHp == 1 && gc.curHp == 1, "minimum HP must survive another Parasite removal");
    } else throw std::invalid_argument("unknown Augmenter reward fixture");
}

static void checkBottleTransform(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.deck = Deck{};
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    for (auto id : {CardId::ANGER, CardId::SHRUG_IT_OFF, CardId::INFLAME, CardId::ASCENDERS_BANE}) gc.deck.obtainRaw(Card(id));
    for (auto relic : {RelicId::BOTTLED_FLAME, RelicId::BOTTLED_LIGHTNING, RelicId::BOTTLED_TORNADO}) {
        require(gc.obtainRelic(relic), "bottle fixture did not open selection");
        require(gc.info.toSelectCards.size() == 1, "bottle fixture target was ambiguous");
        gc.chooseSelectCardScreenOption(0);
    }
    require(gc.deck.getTransformableCount() == 0, "bottle fixture must have no ordinary removable cards");
    gc.curEvent = Event::AUGMENTER;
    gc.screenState = ScreenState::EVENT_SCREEN;
    if (variant == 0 || variant == 1) {
        if (variant == 0) require(search::GameAction(1).isValidAction(gc), "Augmenter excludes otherwise transformable bottled cards");
        gc.chooseEventOption(1);
        require(gc.info.toSelectCards.size() == 3, "Augmenter must offer all three bottled cards but not permanent curse");
        if (variant == 0) return;
        gc.chooseSelectCardScreenOption(2); // Power, then attack; original deck indices must survive.
        gc.chooseSelectCardScreenOption(0);
        require(gc.screenState == ScreenState::MAP_SCREEN && gc.deck.size() == 4, "Augmenter did not complete exactly two transforms");
        require(gc.deck.cards[0].id == CardId::SHRUG_IT_OFF && gc.deck.cards[1].id == CardId::ASCENDERS_BANE,
                "transform removed an unselected card");
        require(gc.deck.bottleIdxs[0] == -1 && gc.deck.bottleIdxs[1] == 0 && gc.deck.bottleIdxs[2] == -1,
                "transform did not clear selected bottles and shift surviving bottle");
        require(gc.deck.getTransformableCount() == 2, "transformable count lost bottle removal/shift");
        for (int i = 0; i < 6; ++i) gc.deck.obtainRaw(Card(CardId::DEFEND_RED));
        for (int seed = 1; seed <= 8; ++seed) {
            gc.seed = seed;
            auto withoutEmptyBottles = gc;
            withoutEmptyBottles.relics.remove(RelicId::BOTTLED_FLAME);
            withoutEmptyBottles.relics.remove(RelicId::BOTTLED_TORNADO);
            BattleContext withRelics, withoutRelics;
            withRelics.init(gc, MonsterEncounter::CULTIST);
            withoutRelics.init(withoutEmptyBottles, MonsterEncounter::CULTIST);
            require(withRelics.cards.cardsInHand == withoutRelics.cards.cardsInHand, "empty bottle changed opening hand size");
            bool survivingBottleDrawn = false;
            for (int i = 0; i < withRelics.cards.cardsInHand; ++i) {
                const auto uid = withRelics.cards.hand[i].getUniqueId();
                survivingBottleDrawn |= uid == 0;
                require(uid == withoutRelics.cards.hand[i].getUniqueId(), "empty bottle changed opening draw identities");
            }
            require(survivingBottleDrawn, "surviving bottle lost opening hand effect");
        }
    } else if (variant == 2 || variant == 3) {
        gc.openCardSelectScreen(variant == 2 ? CardSelectScreenType::TRANSFORM : CardSelectScreenType::REMOVE, 1);
        require(gc.info.toSelectCards.empty(), "ordinary transform/remove must exclude bottles and permanent curses");
    } else if (variant == 4) {
        gc.deck.obtainRaw(Card(CardId::STRIKE_RED));
        gc.deck.obtainRaw(Card(CardId::DEFEND_RED));
        require(gc.obtainRelic(RelicId::ASTROLABE), "Astrolabe should open five-candidate selection");
        require(gc.info.toSelectCards.size() == 5, "Astrolabe must include bottled cards");
        gc.chooseSelectCardScreenOption(0);
        gc.chooseSelectCardScreenOption(0);
        gc.chooseSelectCardScreenOption(0);
        require(!gc.deck.anyCardBottled() && gc.deck.cards[0].id == CardId::ASCENDERS_BANE,
                "Astrolabe failed to clear selected bottles or preserve permanent curse");
        for (int i = 3; i < gc.deck.size(); ++i) require(gc.deck.cards[i].getUpgraded() == 1, "Astrolabe did not upgrade transformed cards");
    } else if (variant == 5) {
        gc.deck.remove(gc, 2);
        gc.deck.remove(gc, 1);
        require(!search::GameAction(1).isValidAction(gc), "Augmenter offered two transforms with only one eligible card");
        gc.deck.obtainRaw(Card(CardId::DEFEND_RED));
        require(search::GameAction(1).isValidAction(gc), "Augmenter ignored bottled plus ordinary eligible pair");
    } else { throw std::invalid_argument("unknown bottle-transform fixture"); }
}

static void checkDeckUpgrade(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.deck = Deck{};
    gc.screenState = ScreenState::REST_ROOM;
    auto &deck = gc.deck;
    if (variant == 0) {
        deck.obtainRaw(Card(CardId::SEARING_BLOW));
        for (int i = 1; i <= 8; ++i) {
            deck.upgrade(0);
            require(deck.cards[0].getUpgraded() == i, "Searing Blow lost repeated upgrades");
            require(deck.getUpgradeableCount() == 1, "Searing Blow upgrade removed eligibility");
            require(search::GameAction(1).isValidAction(gc), "smith validator lost Searing Blow");
        }
        deck.remove(gc, 0);
        require(deck.getUpgradeableCount() == 0, "removing upgraded Searing Blow corrupted count");
    } else if (variant == 1) {
        for (auto id : {CardId::SEARING_BLOW, CardId::STRIKE_RED, CardId::DEFEND_RED}) deck.obtainRaw(Card(id));
        Random expected = gc.miscRng;
        expected.randomLong();
        deck.upgradeRandomCards(gc.miscRng, 99);
        require(sameRng(expected, gc.miscRng), "random upgrades changed shuffle seed consumption");
        for (const auto &c : deck.cards) require(c.getUpgraded() == 1, "random upgrade did not upgrade each candidate once");
        require(deck.getUpgradeableCount() == 1, "random upgrades lost remaining Searing Blow");
        for (int i = 2; i <= 6; ++i) {
            deck.upgradeRandomCards(gc.miscRng, 2);
            require(deck.cards[0].getUpgraded() == i && deck.getUpgradeableCount() == 1,
                    "repeated random upgrades corrupted Searing Blow count");
        }
        deck.remove(gc, 0);
        deck.upgradeRandomCards(gc.miscRng, 2);
        require(deck.getUpgradeableCount() == 0, "empty candidate upgrade changed count");
    } else if (variant == 2) {
        deck.obtainRaw(Card(CardId::BASH));
        deck.upgrade(0);
        require(deck.getUpgradeableCount() == 0, "normal card should lose upgrade eligibility");
        deck.upgrade(0);
        require(deck.getUpgradeableCount() == 0, "already upgraded card was counted twice");
        deck.remove(gc, 0);
        require(deck.getUpgradeableCount() == 0, "removing ordinary upgraded card corrupted count");
    } else if (variant == 3 || variant == 4) {
        for (auto id : {CardId::SEARING_BLOW, CardId::STRIKE_RED, CardId::DEFEND_RED}) deck.obtainRaw(Card(id));
        if (variant == 3) gc.obtainRelic(RelicId::WHETSTONE);
        else {
            gc.curEvent = Event::MINDBLOOM;
            gc.screenState = ScreenState::EVENT_SCREEN;
            gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
            gc.chooseEventOption(1);
        }
        require(deck.cards[0].getUpgraded() == 1 && deck.cards[1].getUpgraded() == 1,
                "relic/event did not upgrade attack cards");
        require(deck.cards[2].getUpgraded() == (variant == 4 ? 1 : 0), "relic/event upgraded wrong card types");
        require(deck.getUpgradeableCount() == (variant == 4 ? 1 : 2), "relic/event upgrade corrupted eligibility count");
    } else { throw std::invalid_argument("unknown deck-upgrade fixture"); }
}

static Random rngForRoll(int wanted) {
    for (int seed = 1; seed <= 10000; ++seed) {
        Random rng(seed);
        if (rng.random(99) == wanted) return Random(seed);
    }
    throw std::runtime_error("no bounded RNG fixture found");
}

static void checkShopRestock(int variant, int ascension) {
    if (variant == 4) {
        GameContext base(CharacterClass::IRONCLAD, 42, ascension);
        base.screenState = ScreenState::SHOP_ROOM;
        base.gold = 10000;
        base.obtainRelic(RelicId::THE_COURIER);
        base.cardRng = rngForRoll(46);
        base.cardRarityFactor = 0;
        base.info.shop.cards[4] = Card(CardId::INFLAME);
        base.info.shop.cardPrice(4) = 31;
        auto reference = base;
        reference.info.shop.buyCard(reference, 4);
        bool changedCard = false, changedPrice = false;
        for (int seed = 100; seed < 116; ++seed) {
            auto mathChild = base;
            mathChild.mathUtilRng = Random(seed);
            mathChild.info.shop.buyCard(mathChild, 4);
            require(sameRng(mathChild.cardRng, reference.cardRng) &&
                    sameRng(mathChild.merchantRng, reference.merchantRng), "math RNG perturbation leaked into rarity/price RNG");
            require(mathChild.info.shop.cardPrice(4) == reference.info.shop.cardPrice(4), "math RNG changed price");
            require(mathChild.info.shop.cards[4].getRarity() == CardRarity::UNCOMMON, "math RNG changed rarity");
            changedCard |= mathChild.info.shop.cards[4].getId() != reference.info.shop.cards[4].getId();
            auto priceChild = base;
            priceChild.merchantRng = Random(seed);
            priceChild.info.shop.buyCard(priceChild, 4);
            require(sameRng(priceChild.cardRng, reference.cardRng) &&
                    sameRng(priceChild.mathUtilRng, reference.mathUtilRng), "price RNG perturbation leaked into card RNGs");
            require(priceChild.info.shop.cards[4].getId() == reference.info.shop.cards[4].getId(), "merchant RNG changed card identity");
            changedPrice |= priceChild.info.shop.cardPrice(4) != reference.info.shop.cardPrice(4);
        }
        require(changedCard && changedPrice, "RNG fixture failed to vary the intended outputs");
        return;
    }
    for (int roll : {8, 9, 45, 46}) {
        for (bool memberAndGift : {false, true}) {
            GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
            gc.screenState = ScreenState::SHOP_ROOM;
            gc.gold = 10000;
            gc.obtainRelic(RelicId::THE_COURIER);
            for (auto egg : {RelicId::MOLTEN_EGG, RelicId::TOXIC_EGG, RelicId::FROZEN_EGG}) gc.obtainRelic(egg);
            if (memberAndGift) {
                gc.obtainRelic(RelicId::MEMBERSHIP_CARD);
                gc.obtainRelic(RelicId::NLOTHS_GIFT);
            }
            gc.cardRng = rngForRoll(roll);
            gc.cardRarityFactor = 0;
            gc.mathUtilRng = Random(817);
            auto expectedCard = gc.cardRng;
            auto expectedMath = gc.mathUtilRng;
            auto expectedMerchant = gc.merchantRng;
            auto &shop = gc.info.shop;
            const CardId ids[] = {CardId::ANGER, CardId::SHRUG_IT_OFF, CardId::INFLAME, CardId::FLASH_OF_STEEL};
            const int slot = variant == 3 ? 5 : variant * 2;
            shop.cards[slot] = Card(ids[variant]);
            shop.cardPrice(slot) = 31;
            const auto type = shop.cards[slot].getType();
            CardRarity expectedRarity;
            if (variant == 3) {
                expectedRarity = expectedMerchant.random() < 0.3f ? CardRarity::RARE : CardRarity::UNCOMMON;
                expectedCard.random(ColorlessRarityCardPool::getGroupSize(expectedRarity) - 1);
            } else {
                require(expectedCard.random(99) == roll, "bad rarity RNG fixture");
                expectedRarity = roll < 9 ? CardRarity::RARE : roll < 46 ? CardRarity::UNCOMMON : CardRarity::COMMON;
                if (type == CardType::POWER && expectedRarity == CardRarity::COMMON) expectedRarity = CardRarity::UNCOMMON;
                expectedMath.random(TypeRarityCardPool::getPoolSize(gc.cc, type, expectedRarity) - 1);
            }
            const int basePrice = expectedRarity == CardRarity::RARE ? 150 : expectedRarity == CardRarity::UNCOMMON ? 75 : 50;
            float price = basePrice * expectedMerchant.random(0.9f, 1.1f);
            if (variant == 3) price *= 1.2f;
            price *= 0.8f;
            if (memberAndGift) price *= 0.5f;
            shop.buyCard(gc, slot);
            require(gc.gold == 9969 && gc.deck.cards[gc.deck.size() - 1].getId() == ids[variant], "restock charged or granted wrong original card");
            require(shop.cards[slot].getRarity() == expectedRarity, "restock rarity/fallback differs from shop thresholds");
            if (variant == 3) require(getCardColor(shop.cards[slot].getId()) == CardColor::COLORLESS, "colorless slot became colored");
            else require(shop.cards[slot].getType() == type, "restock changed purchased card type");
            require(shop.cards[slot].getUpgraded() == 1, "restock lost egg preview");
            require(shop.cardPrice(slot) == static_cast<int>(price), "restock price not based on resulting rarity");
            require(sameRng(gc.cardRng, expectedCard), "restock consumed wrong card RNG draws");
            require(sameRng(gc.mathUtilRng, expectedMath), "restock consumed wrong math RNG draws");
            require(sameRng(gc.merchantRng, expectedMerchant), "restock consumed wrong merchant RNG draws");
            require(gc.cardRarityFactor == 0, "restock changed rarity adjustment");
        }
    }
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

static void checkDuVu(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    gc.deck = Deck{};
    gc.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    if (variant % 2 == 0) gc.obtainRelic(RelicId::DU_VU_DOLL);
    gc.deck.obtainRaw(Card(CardId::STRIKE_RED));
    for (auto id : {CardId::ASCENDERS_BANE, CardId::CURSE_OF_THE_BELL, CardId::NECRONOMICURSE})
        gc.deck.obtain(gc, Card(id));
    if (variant % 2 == 1) gc.obtainRelic(RelicId::DU_VU_DOLL);
    auto check = [&](int expected) {
        BattleContext bc;
        bc.init(gc, MonsterEncounter::CULTIST);
        require(bc.player.strength == expected, "Du-Vu strength differs from current curses");
    };
    require(!gc.deck.hasCurse(), "permanent curses became removable event candidates");
    check(3);
    gc.deck.obtain(gc, Card(CardId::PAIN), 2);
    require(gc.deck.hasCurse(), "ordinary curse lost event eligibility");
    check(5);
    gc.deck.remove(gc, gc.deck.size() - 1);
    check(4);
    if (variant >= 2) {
        gc.obtainRelic(RelicId::OMAMORI);
        gc.relics.getRelicValueRef(RelicId::OMAMORI) = 1;
    }
    // Astrolabe transforms Strike and Pain; the three permanent curses stay.
    require(!gc.obtainRelic(RelicId::ASTROLABE), "unexpected selection for two eligible cards");
    check(variant >= 2 ? 3 : 4);
    require(gc.deck.hasCurse() == (variant < 2), "transform changed removable-curse semantics");
}

static void checkWingBoots(int variant, int ascension) {
    GameContext gc(CharacterClass::IRONCLAD, 42, ascension);
    if (variant != 0) {
        gc.obtainRelic(RelicId::WING_BOOTS);
        require(gc.relics.getRelicValue(RelicId::WING_BOOTS) == 3, "Wing Boots must start with three charges");
    }
    const int charges[] = {0, 0, -2, 1, 2, 3, 3, 3, 3};
    if (variant != 0) gc.relics.getRelicValueRef(RelicId::WING_BOOTS) = charges[variant];
    gc.map = std::make_shared<Map>();
    for (int y = 0; y < 15; ++y) for (int x = 0; x < 7; ++x) {
        auto &node = gc.map->getNode(x, y);
        node.x = x; node.y = y;
        if (x == 1 || x == 4) {
            node.room = Room::REST;
            if (y < 14) { node.edges[0] = x; node.edgeCount = 1; }
        }
    }
    gc.screenState = ScreenState::MAP_SCREEN;
    gc.curMapNodeX = 1; gc.curMapNodeY = 0;
    if (variant == 8) {
        gc.act = 3;
        gc.boss = MonsterEncounter::TIME_EATER;
        gc.curEvent = Event::SECRET_PORTAL;
        gc.screenState = ScreenState::EVENT_SCREEN;
        gc.chooseEventOption(0);
        require(gc.curRoom == Room::BOSS && gc.info.encounter == MonsterEncounter::TIME_EATER,
                "Secret Portal was blocked by player map-choice validation");
        require(gc.relics.getRelicValue(RelicId::WING_BOOTS) == 3, "Secret Portal consumed flight");
        return;
    }
    if (variant == 6) {
        gc.curMapNodeX = -1; gc.curMapNodeY = -1;
        search::GameAction(4).execute(gc);
        require(gc.relics.getRelicValue(RelicId::WING_BOOTS) == 3, "first row consumed a charge");
        gc.screenState = ScreenState::MAP_SCREEN;
        gc.curMapNodeY = 14;
        const auto options = search::GameAction::getAllActionsInState(gc);
        require(options.size() == 1 && options[0].getIdx1() == 4, "boss action lost physical current column");
        options[0].execute(gc);
        require(gc.curRoom == Room::BOSS && gc.relics.getRelicValue(RelicId::WING_BOOTS) == 3,
                "boss entry consumed a charge or entered wrong room");
        return;
    }
    if (variant == 7) {
        gc.transitionToAct(4);
        for (int y = -1; y < 3; ++y) {
            gc.curMapNodeX = y < 0 ? -1 : 3; gc.curMapNodeY = y;
            gc.screenState = ScreenState::MAP_SCREEN;
            const auto options = search::GameAction::getAllActionsInState(gc);
            require(options.size() == 1 && options[0].getIdx1() == 3, "Wing Boots invented an Act4 route");
            auto child = gc;
            options[0].execute(child);
            require(child.relics.getRelicValue(RelicId::WING_BOOTS) == 3, "Act4 consumed a charge");
        }
        return;
    }
    for (int remaining = charges[variant]; ; --remaining) {
        const int cross = gc.curMapNodeX == 1 ? 4 : 1;
        require(search::GameAction(cross).isValidAction(gc) == (remaining > 0), "cross-route legality ignores charges");
        require(!search::GameAction(2).isValidAction(gc), "empty map node became legal");
        const auto options = search::GameAction::getAllActionsInState(gc);
        require(options.size() == (remaining > 0 ? 2 : 1), "map enumeration differs from legality");
        auto ordinary = gc;
        search::GameAction(gc.curMapNodeX).execute(ordinary);
        if (variant != 0) require(ordinary.relics.getRelicValue(RelicId::WING_BOOTS) == remaining,
                                 "ordinary edge consumed a charge");
        if (remaining <= 0) break;
        const auto parent = gc;
        auto child = gc;
        search::GameAction(cross).execute(child);
        require(gc.relics.getRelicValue(RelicId::WING_BOOTS) == remaining && sameRng(gc.miscRng, parent.miscRng),
                "branch consumed parent resources");
        require(child.curMapNodeX == cross && child.curMapNodeY == gc.curMapNodeY + 1,
                "cross-route entered the wrong node");
        require(child.relics.getRelicValue(RelicId::WING_BOOTS) == (remaining == 1 ? -2 : remaining - 1),
                "cross-route charge/exhausted sentinel mismatch");
        gc = child;
        gc.screenState = ScreenState::MAP_SCREEN;
        if (remaining == 1) {
            require(!search::GameAction(cross == 1 ? 4 : 1).isValidAction(gc), "exhausted boots still allow jumping");
            break;
        }
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
        if (mode == "wing-boots") {
            checkWingBoots(std::stoi(argv[2]), std::stoi(argv[3]));
            std::cout << "PASS" << std::endl;
            return 0;
        }
        if (mode == "duvu") {
            checkDuVu(std::stoi(argv[2]), std::stoi(argv[3]));
            std::cout << "PASS" << std::endl;
            return 0;
        }
        if (mode == "augmenter-reward") {
            checkAugmenterRewards(std::stoi(argv[2]), std::stoi(argv[3]));
            std::cout << "PASS" << std::endl;
            return 0;
        }
        if (mode == "event" || mode == "boss" || mode == "reward" || mode == "shop-price" || mode == "shop-legality" || mode == "shop-egg" || mode == "shop-restock" || mode == "deck-upgrade" || mode == "bottle-transform" || mode == "relic-selection" || mode == "transform-upgrade" || mode == "dynamic-exhaust-search") {
            if (mode == "event") checkEventCallback(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "reward") checkReward(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-price") checkShopPrice(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-legality") checkShopLegality(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-egg") checkShopEgg(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "shop-restock") checkShopRestock(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "deck-upgrade") checkDeckUpgrade(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "dynamic-exhaust-search") checkDynamicExhaustSearch(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "transform-upgrade") checkTransformUpgrade(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "relic-selection") checkRelicSelection(std::stoi(argv[2]), std::stoi(argv[3]));
            else if (mode == "bottle-transform") checkBottleTransform(std::stoi(argv[2]), std::stoi(argv[3]));
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
