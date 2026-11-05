#include "sim/search/PublicPlayoutAdapter.h"
#include "game/GameContext.h"
#include <algorithm>
#include <cmath>

namespace sts::search {
namespace {
std::uint32_t hash32(std::uint32_t x) {
    x=(x^(x>>16))*0x7feb352du;
    x=(x^(x>>15))*0x846ca68bu;
    return x^(x>>16);
}
void add(PublicSketch &s, std::uint32_t name, float value) {
    const auto h=hash32(name);
    s[h&(PUBLIC_SKETCH_SIZE-1)] += h>>31 ? -value : value;
}
}
PublicPlayoutAdapter::PublicPlayoutAdapter(const std::vector<float> &weights) {
    if(weights.size()!=PUBLIC_SKETCH_SIZE*SMALL_HIDDEN ||
       !std::all_of(weights.begin(),weights.end(),[](float x){return std::isfinite(x);}))
        throw std::invalid_argument("invalid public playout adaptation weights");
    columns_.resize(weights.size());
    for(int i=0;i<SMALL_HIDDEN;++i) for(int j=0;j<PUBLIC_SKETCH_SIZE;++j)
        columns_[j*SMALL_HIDDEN+i]=weights[i*PUBLIC_SKETCH_SIZE+j];
}
PublicShift PublicPlayoutAdapter::transform(const PublicSketch &s) const {
    PublicShift out{};
    for(int j=0;j<PUBLIC_SKETCH_SIZE;++j) if(s[j]!=0.f) {
        const auto *w=columns_.data()+j*SMALL_HIDDEN;
        for(int i=0;i<SMALL_HIDDEN;++i) out[i]+=s[j]*w[i];
    }
    return out;
}
PublicSketch PublicPlayoutRoot::observe(const BattleContext &bc) const {
    PlayoutEntityObservation obs;
    playoutEntities(bc,deckSize,{},obs);
    PublicSketch out{};
    for(int pass=0;pass<3;++pass) {
        auto accumulate=[&](const auto &rows){for(const auto &r:rows){
            const int id=pass==2?r.id/PLAYOUT_ENTITY_STRIDE:r.id;
            const int key=(id*int(PlayoutZone::Count)+r.zone)*6+r.monster+1+
                (pass==2?int(PlayoutKind::Count)*PLAYOUT_ENTITY_STRIDE*int(PlayoutZone::Count)*6:0);
            if(pass==0) add(out,key*17,1.f);
            else for(int i=0;i<PLAYOUT_ENTITY_FLOATS;++i) if(r.numbers[i]!=0.f)
                add(out,key*17+i+1,r.numbers[i]);
        }};
        accumulate(obs.entities);accumulate(inventory);
    }
    const float scale=std::sqrt(float(obs.entities.size()+inventory.size()));
    for(auto &v:out)v/=scale;
    return out;
}
std::shared_ptr<const PublicPlayoutRoot> PublicPlayoutRoot::prepare(
    std::shared_ptr<const PublicPlayoutAdapter> adapter,const GameContext &gc,const BattleContext &bc,bool refresh) {
    auto r=std::make_shared<PublicPlayoutRoot>();
    r->adapter=std::move(adapter);r->inventory=playoutInventory(gc);r->deckSize=gc.deck.size();
    r->refreshEachState=refresh;r->sketch=r->observe(bc);r->shift=r->adapter->transform(r->sketch);
    return r;
}
}
