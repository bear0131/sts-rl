// Bounded diagnostic probe, not a real-game oracle. Link against the frozen
// native build objects; never rebuild/replace the production Python extension.
#include "game/GameContext.h"
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
