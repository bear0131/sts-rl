// Bounded diagnostic probe, not a real-game oracle. Link against the frozen
// native build objects; never rebuild/replace the production Python extension.
#include "game/GameContext.h"
#include "sim/search/GameAction.h"
#include <iostream>
#include <cstdlib>

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
    root.curHp = variant ? 1 : 60;
    root.gold = variant ? 0 : 500;
    root.curEvent = static_cast<Event>(event);
    root.regainControlAction = [](GameContext &g) { g.screenState = ScreenState::MAP_SCREEN; };
    if (root.curEvent != Event::NEOW) root.setupEvent();
    const auto actions = search::GameAction::getAllActionsInState(root);
    std::cout << "event=" << eventIdStrings[event] << " asc=" << ascension
              << " variant=" << variant << " actions=" << actions.size() << std::endl;
    if (actions.empty()) return 3;
    int count = 0;
    for (const auto &action : actions) {
        if (++count > 128) return 4;
        auto child = root;
        if (!action.isValidAction(child)) return 5;
        std::cout << "before action=" << action.bits << " hp=" << child.curHp
                  << " gold=" << child.gold << " cards=" << child.deck.size() << std::endl;
        action.execute(child);
        std::cout << "after screen=" << static_cast<int>(child.screenState)
                  << " hp=" << child.curHp << " maxhp=" << child.maxHp
                  << " gold=" << child.gold << " cards=" << child.deck.size()
                  << " outcome=" << static_cast<int>(child.outcome) << std::endl;
        if (child.maxHp <= 0 || child.curHp > child.maxHp || child.gold < 0) return 6;
    }
    return 0;
}
