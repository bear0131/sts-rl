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
