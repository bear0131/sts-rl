#pragma once
#include "sim/search/PlayoutEntities.h"

namespace sts::search {
constexpr int PUBLIC_SKETCH_SIZE = 1024;
using PublicSketch = std::array<float,PUBLIC_SKETCH_SIZE>;
using PublicShift = std::array<float,SMALL_HIDDEN>;

class PublicPlayoutAdapter {
    std::vector<float> columns_;
    std::uint64_t snapshot_;
public:
    explicit PublicPlayoutAdapter(const std::vector<float> &weights);
    void addEntity(PublicShift &sum,const PlayoutEntity &row) const;
    PublicShift transform(const PublicSketch &s) const;
};
enum class ContextRefresh { Root, Turn, State };
struct PublicPlayoutRoot {
    std::shared_ptr<const PublicPlayoutAdapter> adapter;
    std::vector<PlayoutEntity> inventory;
    int deckSize;
    ContextRefresh refresh;
    PublicSketch sketch;
    PublicShift shift,inventorySum{};
    PublicShift liveShift(const BattleContext &bc) const;
    PublicSketch observe(const BattleContext &bc) const;
    static std::shared_ptr<const PublicPlayoutRoot> prepare(std::shared_ptr<const PublicPlayoutAdapter>,
        const GameContext &,const BattleContext &,ContextRefresh);
};
}
