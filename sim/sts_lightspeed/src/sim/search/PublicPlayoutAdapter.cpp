#include "sim/search/PublicPlayoutAdapter.h"
#include "game/GameContext.h"
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstring>

namespace sts::search {
namespace {
std::atomic<std::uint64_t> nextPublicSnapshot{1};
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
    snapshot_=nextPublicSnapshot.fetch_add(1,std::memory_order_relaxed);
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
// W times the sum of hashed entity features equals the sum of their W
// projections in real arithmetic. Only the floating-point reduction order
// changes. A cheap cache index is never sufficient evidence for a hit.
void PublicPlayoutAdapter::addEntity(PublicShift &sum,const PlayoutEntity &r) const {
    struct Entry {
        std::uint64_t snapshot=0;
        int key=0;
        std::array<float,PLAYOUT_ENTITY_FLOATS> numbers{};
        PublicShift contribution{};
    };
    thread_local std::array<Entry,2048> cache{};
    const int key=(r.id*int(PlayoutZone::Count)+r.zone)*6+r.monster+1;
    std::uint32_t index=key;
    for(int i: {0,1,3,4,8}){
        std::uint32_t v;std::memcpy(&v,&r.numbers[i],sizeof(v));
        index=(index*0x9e3779b1u)^(v^(v>>13));
    }
    Entry &e=cache[hash32(index)&(cache.size()-1)];
    if(e.snapshot!=snapshot_ || e.key!=key || e.numbers!=r.numbers){
        const auto base=hash32(key*17);
        const float *w=columns_.data()+(base&(PUBLIC_SKETCH_SIZE-1))*SMALL_HIDDEN;
        for(int j=0;j<SMALL_HIDDEN;++j)e.contribution[j]=base>>31?-w[j]:w[j];
        const int kind=(r.id/PLAYOUT_ENTITY_STRIDE*int(PlayoutZone::Count)+r.zone)*6+r.monster+1
            +int(PlayoutKind::Count)*PLAYOUT_ENTITY_STRIDE*int(PlayoutZone::Count)*6;
        for(int i=0;i<PLAYOUT_ENTITY_FLOATS;++i) if(r.numbers[i]!=0.f){
            const auto a=hash32(key*17+i+1),b=hash32(kind*17+i+1);
            const float *u=columns_.data()+(a&(PUBLIC_SKETCH_SIZE-1))*SMALL_HIDDEN;
            const float *v=columns_.data()+(b&(PUBLIC_SKETCH_SIZE-1))*SMALL_HIDDEN;
            const float x=a>>31?-r.numbers[i]:r.numbers[i];
            const float y=b>>31?-r.numbers[i]:r.numbers[i];
            for(int j=0;j<SMALL_HIDDEN;++j)e.contribution[j]+=x*u[j]+y*v[j];
        }
        e.numbers=r.numbers;e.key=key;e.snapshot=snapshot_;
    }
    for(int j=0;j<SMALL_HIDDEN;++j)sum[j]+=e.contribution[j];
}
PublicShift PublicPlayoutRoot::liveShift(const BattleContext &bc) const {
    thread_local PlayoutEntityObservation obs;
    playoutEntities(bc,deckSize,{},obs);
    PublicShift out=inventorySum;
    for(const auto &r:obs.entities)adapter->addEntity(out,r);
    const float scale=std::sqrt(float(obs.entities.size()+inventory.size()));
    for(auto &v:out)v/=scale;
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
    std::shared_ptr<const PublicPlayoutAdapter> adapter,const GameContext &gc,const BattleContext &bc,ContextRefresh refresh) {
    auto r=std::make_shared<PublicPlayoutRoot>();
    r->adapter=std::move(adapter);r->inventory=playoutInventory(gc);r->deckSize=gc.deck.size();
    r->refresh=refresh;r->sketch=r->observe(bc);r->shift=r->adapter->transform(r->sketch);
    if(refresh!=ContextRefresh::Root) for(const auto &e:r->inventory)r->adapter->addEntity(r->inventorySum,e);
    return r;
}
}
