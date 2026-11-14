#pragma once
#include "combat/CardInstance.h"
#include "sim/search/PlayoutEntities.h"
#include <array>
#include <vector>
namespace sts::search {
struct CardZoneSnapshot {
    std::vector<CardInstance> cards;
    std::vector<PlayoutEntity> rows;
    int deck=-1,known=-1;
    std::uint64_t hit=0,miss=0,reused=0,emitted=0;
};
struct CardZoneCache {std::array<CardZoneSnapshot,4> zones;};
void playoutZoneEntities(const BattleContext&,int,const std::vector<SmallActionFeatures>&,PlayoutEntityObservation&,bool,CardZoneCache*);
}
