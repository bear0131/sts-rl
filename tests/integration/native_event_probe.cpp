// Bounded diagnostic probe, not a real-game oracle. Link against the frozen
// native build objects; never rebuild/replace the production Python extension.
#include "game/GameContext.h"
#include "game/SaveFile.h"
#include "combat/BattleContext.h"
#include "sim/search/GameAction.h"
#include <iostream>
#include <cstdlib>
#include <deque>
#include <utility>
#include <stdexcept>

int main(int argc, char **argv) {
    using namespace sts;
    if (argc != 4) return 2;
    const int event = std::atoi(argv[1]);
    const int ascension = std::atoi(argv[2]);
    const int variant = std::atoi(argv[3]);
    if (event < 5 || event > static_cast<int>(Event::WORLD_OF_GOOP)) return 2;
    // Synthetic bounded fixtures, not new real-game campaign seeds.
    GameContext root(CharacterClass::IRONCLAD, 200207, ascension);
    root.act = event == static_cast<int>(Event::MINDBLOOM) ? 3 : 2;
    root.floorNum = root.act == 3 ? 40 : 20;
    root.curHp = variant == 1 ? 1 : 60;
    root.gold = variant == 1 ? 0 : 500;
    root.curEvent = static_cast<Event>(event);
    root.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    if (root.curEvent != Event::NEOW) root.setupEvent();
    if (variant == 29) {
        for (const int maximum : {75, 80}) {
            for (const auto drawback : {Neow::Drawback::NONE, Neow::Drawback::TEN_PERCENT_HP_LOSS}) {
                for (const auto bonus : {Neow::Bonus::TEN_PERCENT_HP_BONUS, Neow::Bonus::TWENTY_PERCENT_HP_BONUS}) {
                    auto fixture = root;
                    fixture.relics = RelicContainer();
                    fixture.maxHp = maximum;
                    fixture.curHp = maximum - 5;
                    const int unit = maximum / 10;
                    const int afterCost = maximum - (drawback == Neow::Drawback::TEN_PERCENT_HP_LOSS ? unit : 0);
                    const int gain = unit * (bonus == Neow::Bonus::TWENTY_PERCENT_HP_BONUS ? 2 : 1);
                    fixture.chooseNeowOption({bonus, drawback});
                    if (fixture.maxHp != afterCost + gain || fixture.curHp != std::min(maximum - 5, afterCost) + gain) return 36;
                }
            }
        }
        for (const int hp : {1, 9, 10, 57, 75, 80, 81}) {
            for (const bool rod : {false, true}) {
                auto fixture = root;
                fixture.relics = RelicContainer();
                if (rod) fixture.obtainRelic(RelicId::TUNGSTEN_ROD);
                fixture.maxHp = 100;
                fixture.curHp = hp;
                const int damage = hp / 10 * 3;
                const int expected = hp - (rod && damage > 0 ? damage - 1 : damage);
                fixture.chooseNeowOption({Neow::Bonus::HUNDRED_GOLD, Neow::Drawback::PERCENT_DAMAGE});
                if (fixture.curHp != expected) return 36;
            }
        }
        return 0;
    }
    if (variant == 27) {
        root.relics = RelicContainer();
        root.deck = Deck();
        root.deck.obtain(root, CardId::ASCENDERS_BANE);
        root.deck.obtain(root, CardId::CURSE_OF_THE_BELL);
        root.deck.obtain(root, CardId::PAIN);
        root.obtainRelic(RelicId::DU_VU_DOLL);
        BattleContext three;
        three.init(root, MonsterEncounter::CULTIST);
        if (three.player.getStatus<PlayerStatus::STRENGTH>() != 3) return 34;
        root.deck.remove(root, 2);
        BattleContext two;
        two.init(root, MonsterEncounter::CULTIST);
        if (two.player.getStatus<PlayerStatus::STRENGTH>() != 2) return 34;
        root.obtainRelic(RelicId::OMAMORI);
        root.deck.obtain(root, CardId::DOUBT);
        BattleContext blocked;
        blocked.init(root, MonsterEncounter::CULTIST);
        if (blocked.player.getStatus<PlayerStatus::STRENGTH>() != 2) return 34;
        return 0;
    }
    if (variant == 28) {
        root.relics = RelicContainer();
        for (int copies = 1; copies <= 4; ++copies) {
            root.obtainRelic(RelicId::CIRCLET);
            if (root.relics.size() != 1 || root.relics.getRelicValue(RelicId::CIRCLET) != copies) return 35;
        }
        return 0;
    }
    if (variant == 25) {
        // Tea Set is armed by a rest room until the next combat, even when
        // other noncombat rooms intervene. It is consumed exactly once.
        root.relics = RelicContainer();
        root.obtainRelic(RelicId::ANCIENT_TEA_SET);
        root.relicsOnEnterRoom(Room::REST);
        root.relicsOnEnterRoom(Room::EVENT);
        root.lastRoom = Room::EVENT;
        BattleContext first;
        first.init(root, MonsterEncounter::CULTIST);
        if (first.player.energy != 5) return 32;
        first.updateRelicsOnExit(root);
        if (root.relics.getRelicValue(RelicId::ANCIENT_TEA_SET) != 0) return 32;
        root.lastRoom = Room::REST; // lastRoom alone must not re-arm it.
        BattleContext second;
        second.init(root, MonsterEncounter::CULTIST);
        if (second.player.energy != 3) return 32;
        root.relicsOnEnterRoom(Room::REST);
        BattleContext third;
        third.init(root, MonsterEncounter::CULTIST);
        if (third.player.energy != 5) return 32;
        return 0;
    }
    if (variant == 26) {
        const RelicId ids[] = {RelicId::MATRYOSHKA, RelicId::NLOTHS_HUNGRY_FACE};
        const int counts[] = {2, 1};
        for (int index = 0; index < 2; ++index) {
            auto fixture = root;
            fixture.relics = RelicContainer();
            fixture.obtainRelic(ids[index]);
            if (fixture.relics.getRelicValue(ids[index]) != counts[index]) return 33;
        }
        SaveFile save;
        save.relics = {RelicId::LIZARD_TAIL, RelicId::MAW_BANK,
            RelicId::ANCIENT_TEA_SET, RelicId::MATRYOSHKA, RelicId::NLOTHS_HUNGRY_FACE,
            RelicId::NEOWS_LAMENT, RelicId::OMAMORI};
        save.relic_counters = {-2, -2, -2, -2, -2, -2, 0};
        auto loaded = root;
        loaded.relics = RelicContainer();
        loaded.initRelicsFromSave(save);
        const int expected[] = {0, 0, 1, 0, 0, 0, 0};
        for (int index = 0; index < 7; ++index)
            if (loaded.relics.getRelicValue(save.relics[index]) != expected[index]) return 33;
        return 0;
    }
    if (variant == 24) {
        // CursedTome.randomBook filters owned books before its single draw,
        // and falls back to Circlet only when all three are already owned.
        const RelicId books[] = {RelicId::NECRONOMICON, RelicId::ENCHIRIDION,
                                 RelicId::NILRYS_CODEX};
        for (int owned = 0; owned < 8; ++owned) {
            for (int seed = 0; seed < 16; ++seed) {
                auto fixture = root;
                fixture.relics = RelicContainer();
                for (int index = 0; index < 3; ++index) {
                    if (owned & (1 << index)) fixture.obtainRelic(books[index]);
                }
                fixture.miscRng = Random(91000 + seed);
                fixture.curEvent = Event::CURSED_TOME;
                fixture.info.eventData = 4;
                fixture.screenState = ScreenState::EVENT_SCREEN;
                const int hp = fixture.curHp;
                fixture.chooseEventOption(5);
                const auto &reward = fixture.info.rewardsContainer;
                if (fixture.screenState != ScreenState::REWARDS || reward.relicCount != 1
                        || fixture.curHp != hp - (ascension >= 15 ? 15 : 10)) return 31;
                const auto chosen = reward.relics[0];
                if (owned == 7) {
                    if (chosen != RelicId::CIRCLET) return 31;
                } else {
                    bool available = false;
                    for (int index = 0; index < 3; ++index) {
                        if (!(owned & (1 << index)) && chosen == books[index]) available = true;
                    }
                    if (!available) return 31;
                }
            }
        }
        std::cout << "cursed_tome_draws_only_unowned_books_or_circlet" << std::endl;
        return 0;
    }
    if (variant == 23) {
        // ShopScreen applies existing eggs to every initial shelf card. Courier
        // restocks a colored purchase with the same card type, reprices relics,
        // and applies the purchased (not replacement) relic to the shelf.
        root.deck = Deck();
        root.obtainRelic(RelicId::THE_COURIER);
        root.obtainRelic(RelicId::MOLTEN_EGG);
        root.info.shop.setupCards(root);
        for (const auto &card : root.info.shop.cards) {
            if (card.getType() == CardType::ATTACK && card.canUpgrade() && !card.isUpgraded()) return 30;
        }

        root.info.shop.cards[0] = Card(CardId::STRIKE_RED);
        root.info.shop.cardPrice(0) = 0;
        root.info.shop.buyCard(root, 0);
        if (root.info.shop.cards[0].getType() != CardType::ATTACK) return 30;

        root.info.shop.relics[0] = RelicId::TOXIC_EGG;
        root.info.shop.relicPrice(0) = 1;
        for (auto &card : root.info.shop.cards) card = Card(CardId::DEFEND_RED);
        root.info.shop.buyRelic(root, 0);
        if (root.info.shop.relicPrice(0) <= 1) return 30;
        for (const auto &card : root.info.shop.cards) {
            if (card.getType() == CardType::SKILL && card.canUpgrade() && !card.isUpgraded()) return 30;
        }

        auto membershipOnly = root;
        membershipOnly.relics = RelicContainer();
        membershipOnly.obtainRelic(RelicId::MEMBERSHIP_CARD);
        auto noDiscount = membershipOnly;
        noDiscount.relics = RelicContainer();
        const int memberPrice = Shop::getNewCardPrice(membershipOnly, CardRarity::RARE, false);
        const int plainPrice = Shop::getNewCardPrice(noDiscount, CardRarity::RARE, false);
        if (memberPrice != static_cast<int>(plainPrice * Shop::MEMBERSHIP_CARD_FACTOR)) return 30;

        auto mask = root;
        mask.relics = RelicContainer();
        mask.info.shop.removeCost = 125;
        mask.info.shop.relics[0] = RelicId::SMILING_MASK;
        mask.info.shop.relicPrice(0) = 0;
        mask.info.shop.buyRelic(mask, 0);
        if (mask.info.shop.removeCost != Shop::SMILING_MASK_PRICE) return 30;
        std::cout << "shop_restock_discount_and_preview_match_java" << std::endl;
        return 0;
    }
    if (variant == 22) {
        Rewards reward;
        reward.addRelic(RelicId::ANCHOR);
        reward.addGold(11);
        reward.addPotion(Potion::BLOOD_POTION);
        CardReward cards;
        cards.push_back(Card(CardId::STRIKE_RED));
        reward.addCardReward(cards);
        reward.sapphireKey = true;
        reward.emeraldKey = true;
        if (reward.getTotalCount() != 6) return 29;
        for (int i = 1; i < 5; ++i) {
            CardReward extra;
            extra.push_back(Card(CardId::DEFEND_RED));
            reward.addCardReward(extra);
        }
        reward.removeCardReward(4);
        if (reward.cardRewardCount != 4) return 29;
        std::cout << "reward_count_includes_relic_and_card_remove_stays_bounded" << std::endl;
        return 0;
    }
    if (variant == 19) {
        Rewards reward;
        CardReward cards;
        cards.push_back(Card(CardId::STRIKE_RED));
        cards.push_back(Card(CardId::BASH));
        reward.addCardReward(cards);
        root.openCombatRewardScreen(reward);
        int skipCardCount = 0;
        for (const auto &action : search::GameAction::getAllActionsInState(root)) {
            if (action.getRewardsActionType() == search::GameAction::RewardsActionType::CARD
                && action.getIdx1() == 0 && action.getIdx2() == 5) ++skipCardCount;
        }
        if (skipCardCount != 1) return 26;
        search::GameAction skipCard(search::GameAction::RewardsActionType::CARD, 0, 5);
        if (!skipCard.isValidAction(root)) return 26;
        const int maxHp = root.maxHp;
        skipCard.execute(root);
        if (root.info.rewardsContainer.cardRewardCount != 0 || root.maxHp != maxHp) return 26;
        std::cout << "card_reward_skip_is_enumerated" << std::endl;
        return 0;
    }
    if (variant == 20) {
        Rewards reward;
        reward.addPotion(Potion::BLOOD_POTION);
        root.openCombatRewardScreen(reward);
        search::GameAction claim(search::GameAction::RewardsActionType::POTION, 0);
        const auto hasPotionClaim = [](const GameContext &gc) {
            for (const auto &action : search::GameAction::getAllActionsInState(gc)) {
                if (action.getRewardsActionType() == search::GameAction::RewardsActionType::POTION) {
                    return true;
                }
            }
            return false;
        };
        if (!claim.isValidAction(root) || !hasPotionClaim(root)) return 27;
        auto full = root;
        full.potions.fill(Potion::BLOOD_POTION);
        full.potionCount = full.potionCapacity;
        if (claim.isValidAction(full) || hasPotionClaim(full)) return 27;
        auto sozu = root;
        sozu.obtainRelic(RelicId::SOZU);
        if (claim.isValidAction(sozu) || hasPotionClaim(sozu)) return 27;
        std::cout << "potion_reward_requires_claimable_slot" << std::endl;
        return 0;
    }
    if (variant == 21) {
        root.screenState = ScreenState::SHOP_ROOM;
        root.gold = 500;
        root.info.shop.potionPrice(0) = 10;
        search::GameAction buy(search::GameAction::RewardsActionType::POTION, 0);
        if (!buy.isValidAction(root)) return 28;
        auto full = root;
        full.potions.fill(Potion::BLOOD_POTION);
        full.potionCount = full.potionCapacity;
        if (buy.isValidAction(full)) return 28;
        auto sozu = root;
        sozu.obtainRelic(RelicId::SOZU);
        if (buy.isValidAction(sozu)) return 28;
        std::cout << "shop_potion_validation_matches_enumerator" << std::endl;
        return 0;
    }
    if (variant == 17) {
        Rewards reward;
        reward.addGold(11);
        reward.addGold(29);
        root.openCombatRewardScreen(reward);
        const auto actions = search::GameAction::getAllActionsInState(root);
        std::vector<int> goldIdxs;
        for (const auto &action : actions) {
            if (action.getRewardsActionType() == search::GameAction::RewardsActionType::GOLD) {
                goldIdxs.push_back(action.getIdx1());
            }
        }
        if (goldIdxs != std::vector<int>{0, 1}) return 24;
        auto child = root;
        const int goldBefore = child.gold;
        search::GameAction(search::GameAction::RewardsActionType::GOLD, 1).execute(child);
        if (child.gold != goldBefore + 29
            || child.info.rewardsContainer.goldRewardCount != 1
            || child.info.rewardsContainer.gold[0] != 11) return 24;
        std::cout << "gold_rewards_preserve_distinct_indices" << std::endl;
        return 0;
    }
    if (variant == 18) {
        root.curEvent = Event::THE_CLERIC;
        root.screenState = ScreenState::EVENT_SCREEN;
        const int purifyCost = ascension >= 15 ? 75 : 50;
        const std::array<std::pair<int, int>, 5> cases {{
            {0, 0b100}, {34, 0b100}, {35, 0b101},
            {purifyCost - 1, 0b101}, {purifyCost, 0b111},
        }};
        for (const auto &[gold, expected] : cases) {
            root.gold = gold;
            const int actual = search::GameAction::getValidEventSelectBits(root);
            if (actual != expected) return 25;
            for (int option = 0; option < 3; ++option) {
                const bool valid = search::GameAction(option).isValidAction(root);
                if (valid != static_cast<bool>(expected & (1 << option))) return 25;
            }
        }
        std::cout << "cleric_options_respect_heal_and_purify_costs" << std::endl;
        return 0;
    }
    if (variant == 15) {
        root.curEvent = Event::BIG_FISH;
        root.setupEvent();
        root.curHp = 20;
        root.maxHp = 80;
        for (bool discard : {false, true}) {
            auto child = root;
            child.potions[0] = Potion::BLOOD_POTION;
            child.potionCount = 1;
            search::GameAction action(static_cast<std::uint32_t>(discard ? 0xc0000000U : 0x80000000U));
            if (!action.isValidAction(child)) return 22;
            action.execute(child);
            if (child.screenState != ScreenState::EVENT_SCREEN
                || child.curHp != (discard ? 20 : 36) || child.potionCount != 0) return 22;
        }
        std::cout << "potion_does_not_execute_event_option" << std::endl;
        return 0;
    }
    if (variant == 16) {
        for (int count : {1, 2}) {
            auto child = root;
            Rewards reward;
            if (count == 2) reward.addRelic(RelicId::ANCHOR);
            reward.addRelic(RelicId::BAG_OF_MARBLES);
            reward.sapphireKey = true;
            child.openCombatRewardScreen(reward);
            search::GameAction(search::GameAction::RewardsActionType::RELIC, count - 1).execute(child);
            if (child.info.rewardsContainer.sapphireKey || child.blueKey) return 23;
            if (count == 2) {
                child = root;
                child.openCombatRewardScreen(reward);
                search::GameAction(search::GameAction::RewardsActionType::RELIC, 0).execute(child);
                if (!child.info.rewardsContainer.sapphireKey) return 23;
            }
        }
        std::cout << "sapphire_relic_exclusion_uses_pre_remove_index" << std::endl;
        return 0;
    }
    if (variant == 14) {
        for (bool bark : {false, true}) {
            auto child = root;
            child.maxHp = 80;
            child.curHp = 1;
            if (bark) child.obtainRelic(RelicId::SACRED_BARK);
            child.potions.fill(Potion::EMPTY_POTION_SLOT);
            child.potions[0] = Potion::FAIRY_POTION;
            child.potionCount = 1;
            child.playerLoseHp(2);
            if (child.curHp != (bark ? 48 : 24) || child.potionCount != 0) return 20;
        }
        root.obtainRelic(RelicId::TUNGSTEN_ROD);
        const int hp = root.curHp;
        root.playerLoseHp(0);
        if (root.curHp != hp) return 21;
        root.playerLoseHp(1);
        if (root.curHp != hp) return 21;
        root.playerLoseHp(2);
        if (root.curHp != hp - 1) return 21;
        std::cout << "fairy_bark_60_percent_zero_damage_no_heal" << std::endl;
        return 0;
    }
    if (variant >= 10 && variant <= 13) {
        root.deck = Deck();
        if (variant == 10) {
            root.deck.obtainRaw(Card(CardId::STRIKE_RED));
            root.deck.obtainRaw(Card(CardId::DEFEND_RED));
            root.deck.obtainRaw(Card(CardId::REGRET));
            if (root.deck.getTransformableCount() != 3
                || root.deck.cardTypeCounts[0] != 1 || root.deck.cardTypeCounts[1] != 1
                || root.deck.cardTypeCounts[3] != 1 || root.deck.getUpgradeableCount() != 2) return 16;
        } else if (variant == 11) {
            root.deck.obtain(root, CardId::STRIKE_RED);
            root.deck.obtain(root, CardId::DEFEND_RED);
            root.deck.obtain(root, CardId::BASH);
            root.deck.obtain(root, CardId::INFLAME);
            fixed_list<SelectScreenCard, 3> selected;
            selected.push_back({root.deck.cards[2], 2});
            selected.push_back({root.deck.cards[0], 0});
            root.deck.removeSelected(root, selected);
            if (root.deck.size() != 2 || root.deck.cards[0].id != CardId::DEFEND_RED
                || root.deck.cards[1].id != CardId::INFLAME) return 17;
        } else if (variant == 12) {
            if (root.deck.anyCardBottled()) return 18;
            root.deck.obtain(root, CardId::REGRET);
            root.deck.obtain(root, CardId::STRIKE_RED);
            root.deck.obtain(root, CardId::DEFEND_RED);
            root.deck.obtain(root, CardId::INFLAME);
            root.deck.bottleCard(1, CardType::ATTACK);
            root.deck.bottleCard(2, CardType::SKILL);
            root.deck.bottleCard(3, CardType::POWER);
            if (!root.deck.anyCardBottled()) return 18;
            root.deck.remove(root, 0);
            if (root.deck.bottleIdxs != std::array<int,3>{0,1,2}
                || root.deck.getTransformableCount() != 0) return 18;
        } else {
            root.deck.obtain(root, CardId::STRIKE_RED);
            root.deck.obtain(root, CardId::DEFEND_RED);
            root.deck.obtain(root, CardId::STRIKE_RED);
            root.deck.remove(root, 2); // leave a stale matching card outside logical size
            root.deck.removeAllMatching(root, [](const Card &c) { return c.isStarterStrike(); });
            if (root.deck.size() != 1 || root.deck.cards[0].id != CardId::DEFEND_RED
                || root.deck.cardTypeCounts[0] != 0) return 19;
            root.deck.removeAllMatching(root, [](const Card &) { return true; });
            root.deck.removeAllMatching(root, [](const Card &) { return true; });
            if (root.deck.size() != 0) return 19;
        }
        std::cout << "deck_invariant_variant=" << variant << std::endl;
        return 0;
    }
    if (variant == 9 && root.curEvent == Event::FALLING) {
        root.deck = Deck();
        root.deck.obtain(root, CardId::STRIKE_RED);
        root.deck.obtain(root, CardId::DEFEND_RED);
        root.deck.obtain(root, CardId::INFLAME);
        root.deck.bottleCard(0, CardType::ATTACK);
        root.deck.bottleCard(1, CardType::SKILL);
        root.deck.bottleCard(2, CardType::POWER);
        const auto before = root.miscRng.counter;
        root.setupEvent();
        if (root.info.attackCardDeckIdx != -1 || root.info.skillCardDeckIdx != -1
            || root.info.powerCardDeckIdx != -1 || root.miscRng.counter != before) return 15;
        root.deck.obtain(root, CardId::REGRET);
        root.deck.obtain(root, CardId::BASH);
        root.setupEvent();
        if (root.info.attackCardDeckIdx != 4 || root.info.skillCardDeckIdx != -1
            || root.info.powerCardDeckIdx != -1 || root.miscRng.counter != before + 1) return 15;
        root.chooseEventOption(2);
        if (root.deck.size() != 4 || !root.deck.isCardBottled(0)
            || !root.deck.isCardBottled(1) || !root.deck.isCardBottled(2)) return 15;
        std::cout << "falling_bottles_excluded_curse_safe" << std::endl;
        return 0;
    }
    if (variant == 8 && (root.curEvent == Event::DEAD_ADVENTURER
                        || root.curEvent == Event::MINDBLOOM
                        || root.curEvent == Event::HYPNOTIZING_COLORED_MUSHROOMS)) {
        if (root.curEvent == Event::DEAD_ADVENTURER) root.info.phase = 3; // guaranteed fight
        root.chooseEventOption(0);
        auto child = root;
        const auto cardBefore = root.cardRng;
        const auto potionBefore = root.potionRng;
        // Execute the already constructed post-combat callback on a search
        // copy. The origin must remain unchanged and copies must replay alike.
        child.regainControl();
        if (root.cardRng.counter != cardBefore.counter
            || root.potionRng.counter != potionBefore.counter) return 14;
        auto sibling = root;
        sibling.regainControl();
        if (child.cardRng.seed0 != sibling.cardRng.seed0
            || child.cardRng.seed1 != sibling.cardRng.seed1
            || child.potionRng.seed0 != sibling.potionRng.seed0
            || child.potionRng.seed1 != sibling.potionRng.seed1
            || child.cardRng.counter <= cardBefore.counter) return 14;
        std::cout << "reward_callback_copy_isolated=" << eventIdStrings[event] << std::endl;
        return 0;
    }
    if (variant == 7 && root.curEvent == Event::DESIGNER_IN_SPIRE) {
        for (int option = 0; option < 6; ++option) {
            auto child = root;
            child.chooseEventOption(option);
            const int cost = option < 2 ? (ascension >= 15 ? 50 : 40)
                           : option < 4 ? (ascension >= 15 ? 75 : 60)
                           : option == 4 ? (ascension >= 15 ? 110 : 90) : 0;
            const int hpLoss = option == 5 ? (ascension >= 15 ? 5 : 3) : 0;
            std::cout << "designer_option=" << option << " cost=" << root.gold - child.gold
                      << " hp_loss=" << root.curHp - child.curHp << std::endl;
            if (root.gold - child.gold != cost || root.curHp - child.curHp != hpLoss) return 13;
            if (option == 3) {
                if (child.info.selectScreenType != CardSelectScreenType::TRANSFORM
                    || child.info.toSelectCount != 2
                    || child.screenState != ScreenState::CARD_SELECT) return 13;
                if (child.deck.size() != root.deck.size()) return 13;
                child.chooseSelectCardScreenOption(0);
                child.chooseSelectCardScreenOption(0);
                if (child.deck.size() != root.deck.size()
                    || child.screenState != ScreenState::MAP_SCREEN) return 13;
            }
            if (option == 4) {
                if (child.info.selectScreenType != CardSelectScreenType::REMOVE
                    || child.screenState != ScreenState::CARD_SELECT) return 13;
                // Copy the pending transaction as search does. Its callback
                // must use the copy's RNG, never the captured original this.
                auto replay = child;
                const auto parentRng = child.miscRng;
                replay.chooseSelectCardScreenOption(0);
                if (child.miscRng.counter != parentRng.counter
                    || replay.miscRng.counter == parentRng.counter
                    || replay.deck.size() != root.deck.size() - 1
                    || replay.screenState != ScreenState::MAP_SCREEN) return 13;
            }
        }
        return 0;
    }
    if (variant == 6 && root.curEvent == Event::THE_MAUSOLEUM) {
        auto leave = root;
        leave.chooseEventOption(1);
        if (leave.relics.size() != root.relics.size() || leave.deck.size() != root.deck.size()) return 12;
        auto opened = root;
        opened.chooseEventOption(0);
        std::cout << "mausoleum_relic_gain=" << opened.relics.size() - root.relics.size()
                  << " expected=1" << std::endl;
        return opened.relics.size() == root.relics.size() + 1 ? 0 : 12;
    }
    if (variant == 5 && root.curEvent == Event::THE_WOMAN_IN_BLUE) {
        for (int option = 0; option < 3; ++option) {
            auto child = root;
            child.chooseEventOption(option);
            const int expected = 20 + 10 * option;
            std::cout << "woman_option=" << option << " gold_spent=" << root.gold - child.gold
                      << " expected=" << expected << std::endl;
            if (root.gold - child.gold != expected) return 11;
        }
        return 0;
    }
    if (variant == 4 && root.curEvent == Event::MATCH_AND_KEEP) {
        // Isolate the dormant transition; this does NOT enable the event in
        // production or claim that the frozen policy can play the minigame.
        root.screenState = ScreenState::EVENT_SCREEN;
        root.info.toSelectCards.clear();
        for (int i = 0; i < 12; ++i)
            root.info.toSelectCards.push_back({Card(CardId::STRIKE_RED), -1});
        root.info.eventData = 5;
        const int before = root.deck.size();
        if (!search::GameAction(0, 1).isValidAction(root)) return 10;
        root.chooseMatchAndKeepCards(0, 1);
        if (root.deck.size() != before + 1 || root.info.eventData != 4) return 10;
        if (search::GameAction(0, 1).isValidAction(root)
            || search::GameAction(2, 2).isValidAction(root)
            || search::GameAction(12, 2).isValidAction(root)) return 10;
        try { root.chooseMatchAndKeepCards(0, 1); return 10; }
        catch (const std::invalid_argument &) {}
        if (root.deck.size() != before + 1 || root.info.eventData != 4) return 10;
        root.info.toSelectCards.clear();
        if (search::GameAction(0, 1).isValidAction(root)) return 10;
        try { root.chooseMatchAndKeepCards(0, 1); return 10; }
        catch (const std::invalid_argument &) {}
        std::cout << "matched_slots_not_reusable; invalid_pairs_atomic" << std::endl;
        return 0;
    }
    // Returning directly to the map is not event coverage. In particular the
    // frozen build disables Match and Keep: its map actions previously made
    // this probe report a misleading successful event traversal.
    if (root.screenState == ScreenState::MAP_SCREEN) {
        std::cerr << "unsupported_or_bypassed_event=" << eventIdStrings[event]
                  << " setup_returned_to_map" << std::endl;
        return 9;
    }
    if (variant == 2 && root.curEvent == Event::SCRAP_OOZE) {
        // Real Java increments damage after each failure. Public event state
        // after two failures must charge base + 2, independent of the next roll.
        root.info.eventData = 2;
        const int before = root.curHp;
        root.chooseEventOption(0);
        const int expected = (ascension >= 15 ? 5 : 3) + 2;
        std::cout << "scrap_damage=" << before - root.curHp << " expected=" << expected << std::endl;
        return before - root.curHp == expected ? 0 : 7;
    }
    if (variant == 3 && root.curEvent == Event::MYSTERIOUS_SPHERE) {
        // Java's Leave branch changes dialogue only, never grants gold.
        const int gold = root.gold, hp = root.curHp;
        root.chooseEventOption(1);
        std::cout << "sphere_leave_gold=" << root.gold - gold << " expected=0" << std::endl;
        return root.gold == gold && root.curHp == hp ? 0 : 8;
    }
    const auto actions = search::GameAction::getAllActionsInState(root);
    std::cout << "event=" << eventIdStrings[event] << " asc=" << ascension
              << " variant=" << variant << " actions=" << actions.size() << std::endl;
    if (actions.empty()) return 3;
    std::deque<std::pair<GameContext, int>> pending;
    pending.emplace_back(root, 0);
    int count = 0, capped = 0;
    while (!pending.empty()) {
        auto node = std::move(pending.front());
        pending.pop_front();
        for (const auto &action : search::GameAction::getAllActionsInState(node.first)) {
        if (count >= 512) { ++capped; pending.clear(); break; }
        ++count;
        auto child = node.first;
        if (!action.isValidAction(child)) return 5;
        std::cout << "before depth=" << node.second << " action=" << action.bits << " hp=" << child.curHp
                  << " gold=" << child.gold << " cards=" << child.deck.size() << std::endl;
        action.execute(child);
        std::cout << "after screen=" << static_cast<int>(child.screenState)
                  << " hp=" << child.curHp << " maxhp=" << child.maxHp
                  << " gold=" << child.gold << " cards=" << child.deck.size()
                  << " outcome=" << static_cast<int>(child.outcome) << std::endl;
        if (child.maxHp <= 0 || child.curHp > child.maxHp || child.gold < 0) return 6;
        // Stop at external map/combat boundaries. Follow event-owned reward
        // and card-selection screens; caps are reported, never called coverage.
        if (child.outcome == GameOutcome::UNDECIDED &&
            child.screenState != ScreenState::MAP_SCREEN &&
            child.screenState != ScreenState::BATTLE) {
            if (node.second < 7) pending.emplace_back(std::move(child), node.second + 1);
            else ++capped;
        }
        }
    }
    std::cout << "transitions=" << count << " capped=" << capped << std::endl;
    return 0;
}
