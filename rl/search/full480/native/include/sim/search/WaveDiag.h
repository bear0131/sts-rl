#pragma once
#include "zones.h"
#include "sim/search/NeuralSearchBatch.h"
#include "sim/search/PlayoutEntities.h"
#include "game/GameContext.h"
#include <random>
#include <atomic>

#include <cuda.h>
#include <new>
#include <stdexcept>
namespace sts::search {
struct MappedBattleBlock {
    void *host=nullptr;CUdeviceptr device=0;size_t bytes=0;std::vector<unsigned char> constructed;
    explicit MappedBattleBlock(size_t count){
        constructed.resize(count,0);bytes=count*sizeof(BattleContext);
        if(cuMemHostAlloc(&host,bytes,CU_MEMHOSTALLOC_DEVICEMAP)!=CUDA_SUCCESS)
            throw std::runtime_error("mapped BattleContext allocation failed");
        if(cuMemHostGetDevicePointer(&device,host,0)!=CUDA_SUCCESS){
            cuMemFreeHost(host);host=nullptr;
            throw std::runtime_error("mapped BattleContext device address failed");
        }
    }
    BattleContext *get(size_t i){
        if(i>=constructed.size())throw std::logic_error("mapped pool capacity");
        auto *p=reinterpret_cast<BattleContext*>(static_cast<char*>(host)+i*sizeof(BattleContext));
        if(!constructed[i]){new(p)BattleContext();constructed[i]=1;}
        return p;
    }
    ~MappedBattleBlock(){if(host){
        for(size_t i=0;i<constructed.size();++i)if(constructed[i])get(i)->~BattleContext();
        cuMemFreeHost(host);
    }}
    MappedBattleBlock(const MappedBattleBlock&)=delete;
};
struct WaveStateDelete {
    bool mapped=false;
    void operator()(BattleContext *p)const{if(!mapped)delete p;}
};
using WaveStatePtr=std::unique_ptr<BattleContext,WaveStateDelete>;
}

namespace sts::search {
struct WaveRoot {
    BattleContext root;
    std::vector<PlayoutEntity> inventory;
    std::vector<CardInstance> canonical;int keyForInventory=0;int deck; SmallSearchConfig cfg; std::uint64_t nonce;
    std::vector<Action> actions;std::vector<std::uint64_t> semantic;std::vector<int> order;
    WaveRoot(const GameContext&,const BattleContext&,SmallSearchConfig,std::uint64_t);
};
struct SharedStart {
    std::once_flag once;
    std::atomic<int> remaining{0};
    std::unique_ptr<BattleContext> base;
    std::array<double,64> uniforms{};
};
struct WaveLane { int factBound=0;
    int sharedIndex=-1,drawIndex=0;std::uint64_t policySeed=0;
    WaveStatePtr state;
    std::unique_ptr<std::mt19937_64> rng; int task=-1, depth=0;
    SmallFeatures obs; PlayoutEntityObservation entity;
    std::vector<int> symbols, live, queries, qindex;
    std::vector<std::array<float,128>> features;
    std::vector<float> ctx;
    std::vector<std::array<int,10>> src;std::vector<int> target;
    struct TokenCache{std::array<std::vector<PlayoutEntity>,33> priorRows; std::array<std::array<float,128>,33> cachedTokens{};};std::unique_ptr<TokenCache> cached; std::unique_ptr<CardZoneCache> cardCache;int lastRow=-1; double pubSec=0,encodeSec=0; std::uint64_t dirtyTokens=0,allTokens=0; std::vector<int> rawOwner,rawExtra,rawCounts; int chosenNext=-1;double drawNext=0.;int jobForced=0;std::uint64_t jobActions=0;
    std::uint64_t forced=0,steps=0,scored=0; bool ready=false;
};
class WaveDiag {
public:
    std::vector<std::shared_ptr<WaveRoot>> roots;
    bool zoneMode=false,zoneVerify=false,resident=false;bool shortAtoms=false;int shareMode=0;std::vector<int> worldOffsets;
    std::unique_ptr<SharedStart[]> starts;bool counterDraw=false;std::shared_ptr<NeuralSearchBatch> poolOwner;NeuralSearchBatch &pool; std::unique_ptr<MappedBattleBlock> mappedBlock; std::vector<WaveLane> lanes;std::vector<int> workRoot,local,chosen;
    std::vector<int> depths,forceN,outcomes;std::vector<std::uint64_t> actionN;
    std::vector<double> values;std::vector<int> active;std::atomic<int> nextTask{0};bool awaiting=false;
    WaveDiag(std::vector<std::shared_ptr<WaveRoot>>,int,int);
    std::vector<int> pendingPicks;bool aosMode=false;bool gpuChoice=false;bool rawMode=false;std::vector<int> entityOffsets;bool fastCodec=true;bool fullTokens=false;std::vector<int> offset;
    void enableMapped(){
        if(nextTask.load()!=0||awaiting)throw std::logic_error("mapping after lane execution");
        if(!mappedBlock)mappedBlock=std::make_unique<MappedBattleBlock>(8192);
        for(size_t i=0;i<lanes.size();++i)if(!lanes[i].state)
            lanes[i].state=WaveStatePtr(mappedBlock->get(i),WaveStateDelete{true});
    }
    CUdeviceptr mappedAddress(const WaveLane &l)const{
        if(!mappedBlock)throw std::logic_error("mapped encoding without pool");
        return mappedBlock->device+size_t(&l-lanes.data())*sizeof(BattleContext);
    }
    void resetGroups(std::vector<std::shared_ptr<WaveRoot>>,int);void prepareLane(std::size_t);void prepare();void advance(const float*,int,int);void encode(WaveLane&);
};
}
