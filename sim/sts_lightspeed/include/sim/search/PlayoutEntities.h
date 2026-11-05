#pragma once
#include "sim/search/SmallPlayout.h"
#include <array>
#include <vector>

namespace sts { class GameContext; }
namespace sts::search {
// Public combat entities v1. Row order is not a learned feature. Source row
// references are gather indices, never numeric policy inputs. Each monster
// retains its simulator slot, including gaps between living monsters.
enum class PlayoutKind { Battle, Player, Power, Relic, Card, Monster, MonsterPower,
                         Move, Potion, Selection, Counter, Order, Count };
constexpr int PLAYOUT_ENTITY_STRIDE = 512, PLAYOUT_ENTITY_FLOATS = 16;
enum class PlayoutZone { Battle, Player, Hand, HiddenDraw, KnownDraw, Discard,
                         Exhaust, Stasis, Generated, Pending, MasterDeck,
                         OwnedRelic, ActiveRelic, Potion, Monster, Count };
struct PlayoutEntity {
    int id, zone, monster = -1;
    std::array<float, PLAYOUT_ENTITY_FLOATS> numbers{};
};
struct PlayoutEntityObservation {
    std::vector<PlayoutEntity> entities;
    std::vector<std::array<int, SMALL_KEYS>> sources;
    std::vector<int> targets;
};
// GameContext contributes only public inventory, never its future room/relic
// pools, encounter schedule or RNG. Keep this part reusable across a search.
std::vector<PlayoutEntity> playoutInventory(const GameContext &game);
void playoutEntities(const BattleContext &bc, int masterDeckSize,
                     const std::vector<SmallActionFeatures> &actions,
                     PlayoutEntityObservation &observation);
}
