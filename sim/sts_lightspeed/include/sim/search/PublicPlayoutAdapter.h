#pragma once
#include "sim/search/PlayoutEntities.h"

namespace sts::search {
constexpr int PUBLIC_SKETCH_SIZE = 1024;
using PublicSketch = std::array<float,PUBLIC_SKETCH_SIZE>;
using PublicShift = std::array<float,SMALL_HIDDEN>;

class PublicPlayoutAdapter {
    std::vector<float> columns_;
public:
    explicit PublicPlayoutAdapter(const std::vector<float> &weights);
    PublicShift transform(const PublicSketch &s) const;
};
struct PublicPlayoutRoot {
    std::shared_ptr<const PublicPlayoutAdapter> adapter;
    std::vector<PlayoutEntity> inventory;
    int deckSize;
    bool refreshEachState;
    PublicSketch sketch;
    PublicShift shift;
    PublicSketch observe(const BattleContext &bc) const;
    static std::shared_ptr<const PublicPlayoutRoot> prepare(std::shared_ptr<const PublicPlayoutAdapter>,
        const GameContext &,const BattleContext &,bool);
};
}
