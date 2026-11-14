#include "/root/autodl-tmp/token-speed-r7-20261105/nine-token-speed/online480/waves/devicebudgetarena141/device_std.h"
#include "src/combat/Actions.cpp"
#include "src/combat/BattleContext.cpp"
#include "src/combat/CardInstance.cpp"
#include "src/combat/CardManager.cpp"
#include "src/combat/CardQueue.cpp"
#include "src/combat/Monster.cpp"
#include "src/combat/MonsterGroup.cpp"
#include "src/combat/MonsterMoveDamage.cpp"
#include "src/combat/MonsterSpecific.cpp"
#include "src/combat/Player.cpp"
#include "src/game/Game.cpp"
#include "src/game/Card.cpp"
#include "src/sim/search/Action.cpp"
#include "src/sim/search/Nosl.cpp"
#include "state_codec.h"
#include "device_observation.h"
#include "small_features.cuh"
#include "src/sim/search/PlayoutEntities.cpp"


#include "bridge_common.h"
#ifdef STS_FULL480_DECK
#include "deck_inputs.cuh"
#define DECK_EXTRA const int*prefixes,const unsigned long long*weights,float*raw,int*cat,int*legal_mask,float*selected,int*remaining,const float*root_inputs,const int*root_relics,const float*root_counters,
#else
#define DECK_EXTRA
#endif

inline unsigned char* packed_address(unsigned char*table,int index){
 const auto address=reinterpret_cast<const unsigned long long*>(table)[index];
 device_state::require(address!=0 && address%alignof(sts::BattleContext)==0);
 return reinterpret_cast<unsigned char*>(address);
}
extern "C" __global__ void packed_layout(unsigned long long*out){
 if(blockIdx.x==0 && threadIdx.x==0){out[0]=1;out[1]=sizeof(sts::BattleContext);out[2]=sizeof(void*);}
}

namespace device_tokens {
unsigned hash(unsigned x){x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;return x^(x>>16);}
void add(float*f,unsigned name,float v){unsigned h=hash(name);f[h%127]+=(h>>31)?-v:v;}
int primary(int id,int zone,int mon,const float*x){
 if(zone==2)return 18+int(rintf(x[8]*40.f))-1;
 if(zone==13)return 13+int(rintf(x[0]*5.f))-1;
 if(zone==14&&id/512==5)return 28+mon;
 return -1;
}
int slot(int id,int zone,int mon,const float*x){
 constexpr int zones[15]={1,2,18,6,7,8,9,10,12,11,5,3,4,13,28};
 device_state::require(zone>=0&&zone<15&&id>=0&&id<6144&&mon>=-1&&mon<5);
 if(zone==1&&id/512==1)return 1;
 if(zone==2||zone==13)return primary(id,zone,mon,x);
 if(zone==14){device_state::require(mon>=0);return 28+mon;}
 return zones[zone];
}
}
constexpr int POOL_CHUNK=64;
constexpr int POOL_CHUNKS=16;
extern "C" __global__ void frame_observe(unsigned char*arena,int*worlds,int*live,int*deck_sizes,
 float*state,unsigned*bits,float*action_numbers,int*keys,int*source_rows,int*targets,
 int*ids,int*zones,int*monsters,float*numbers,int*meta,int*chunks,int*allocator,
 const int*frame_count,int base,int root_count,int action_capacity,int entity_capacity){
 int count=frame_count[0];int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const auto&b=*reinterpret_cast<const sts::BattleContext*>(packed_address(arena,live[base+i]));
 auto legal=device_state::legal_actions(b);int na=legal.size();
 device_state::require(na>0&&na<=2048);
 int start=atomicAdd(allocator,na);meta[i*4]=na;meta[i*4+1]=0;meta[i*4+2]=start;meta[i*4+3]=0;
 atomicMax(allocator+4,na);
 if(start>action_capacity-na){atomicOr(allocator+2,1);return;}
 sts::search::SmallFeatures f{};
 f.actions={bits+start,action_numbers+start*24,keys+start*10,0,na};
 sts::search::featuresForActions(b,legal,f);
 for(int j=0;j<32;++j)state[i*32+j]=f.state[j];
 sts::search::PlayoutEntityObservation o{};
 o.entities={ids,zones,monsters,numbers,0,1024};
 o.entities.pool_chunks=chunks+i*POOL_CHUNKS;o.entities.pool_allocator=allocator+1;
 o.entities.pool_overflow=allocator+2;o.entities.pool_capacity=entity_capacity;
 o.sources={reinterpret_cast<std::array<int,10>*>(source_rows+start*10),0,na};
 o.targets={targets+start,0,na};
 sts::search::playoutEntities(b,deck_sizes[worlds[base+i]%root_count],f.actions,o,false);
 device_state::require(o.sources.size()==na&&o.targets.size()==na);
 meta[i*4+1]=o.entities.size();meta[i*4+3]=!o.entities.pool_failed;
 atomicMax(allocator+3,int(o.entities.size()));
}
extern "C" __global__ void pool_inventory_slots(int*ids,int*zones,int*monsters,float*numbers,int*counts,int*slots,int n,int cap){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int r=0;r<counts[i];++r){int p=i*cap+r;slots[p]=device_tokens::slot(ids[p],zones[p],monsters[p],numbers+p*16);}
}
extern "C" __global__ void frame_layout(
 int*worlds,int*ids,int*zones,int*monsters,float*numbers,int*meta,int*chunks,
 int*inv_ids,int*inv_zones,int*inv_monsters,float*inv_numbers,int*inv_counts,int*inv_slots,
 int*source_rows,int*target_slots,int*symbols,int*counts,int*canonical_sources,int*canonical_targets,
 int*entity_slots,int*extra_slots,int*token_order,int*query_order,int*query_map,int*input_map,int*geometry,
 const int*allocator,const int*frame_count,int base,int root_count,int inventory_cap,int token_cap){
 if(allocator[2])return;int n=frame_count[0];int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int ne=meta[i*4+1],na=meta[i*4],start=meta[i*4+2],root=worlds[base+i]%root_count,ni=inv_counts[root];
 device_state::require(meta[i*4+3]&&ne<=1024&&token_cap==161);
 auto*sy=symbols+i*token_cap;auto*ct=counts+i*token_cap;
 for(int t=0;t<token_cap;++t){sy[t]=0;ct[t]=0;}
 auto entity=[&](int r){return chunks[i*POOL_CHUNKS+r/POOL_CHUNK]+r%POOL_CHUNK;};
 int source_map[1024];
 for(int r=0;r<ne;++r){
  int at=entity(r),id=ids[at],z=zones[at],m=monsters[at];const float*x=numbers+at*16;
  int slot=device_tokens::slot(id,z,m,x),primary=device_tokens::primary(id,z,m,x);
  device_state::require(slot>=0&&slot<33);++ct[slot];entity_slots[at]=slot;extra_slots[at]=-1;
  if(primary>=0)sy[slot]=id+1;source_map[r]=primary;
 }
 for(int r=0;r<ni;++r){
  int at=root*inventory_cap+r,slot=inv_slots[at];++ct[slot];
  if(device_tokens::primary(inv_ids[at],inv_zones[at],inv_monsters[at],inv_numbers+at*16)>=0)sy[slot]=inv_ids[at]+1;
 }
 for(int a=0;a<na;++a)for(int j=0;j<10;++j){
  int r=source_rows[(start+a)*10+j];device_state::require(r>=-1&&r<ne);
  if(r>=0&&source_map[r]<0)source_map[r]=-2;
 }
 int nt=33;
 for(int r=0;r<ne;++r)if(source_map[r]==-2){
  device_state::require(nt<token_cap);int at=entity(r);source_map[r]=nt;extra_slots[at]=nt;
  sy[nt]=ids[at]+1;ct[nt]=1;++nt;
 }
 bool used[161]{},queried[161]{};int compact_rank[161];
 for(int t=0;t<token_cap;++t)used[t]=(t==0||ct[t]>0);queried[0]=true;
 for(int a=0;a<na;++a){
  for(int j=0;j<10;++j){
   int r=source_rows[(start+a)*10+j],t=r>=0?source_map[r]:-1;
   canonical_sources[(start+a)*10+j]=t;if(t>=0){used[t]=true;queried[t]=true;}
  }
  int raw=target_slots[start+a];device_state::require(raw>=-1&&raw<5);int t=raw>=0?28+raw:-1;
  canonical_targets[start+a]=t;if(t>=0){used[t]=true;queried[t]=true;}
 }
 int nused=0,nquery=0;
 for(int t=0;t<token_cap;++t){
  token_order[i*token_cap+t]=-1;query_order[i*token_cap+t]=0;query_map[i*token_cap+t]=-1;
  compact_rank[t]=used[t]?nused++:-1;input_map[i*token_cap+t]=compact_rank[t];
 }
 for(int t=0;t<token_cap;++t){
  if(used[t])token_order[i*token_cap+compact_rank[t]]=t;
  if(queried[t]){device_state::require(compact_rank[t]>=0);query_order[i*token_cap+nquery]=compact_rank[t];query_map[i*token_cap+t]=nquery++;}
 }
 constexpr int token_bins[8]={16,24,32,48,64,96,128,161};
 constexpr int query_bins[9]={1,2,4,8,16,32,64,128,161};
 int tb=0,qb=0;while(tb<7&&token_bins[tb]<nused)++tb;while(qb<8&&query_bins[qb]<nquery)++qb;
 geometry[i*4]=nused;geometry[i*4+1]=nquery;geometry[i*4+2]=ne+ni;geometry[i*4+3]=tb*9+qb;
}

extern "C" __global__ void observer_layout(unsigned long long*x){if(threadIdx.x==0&&blockIdx.x==0){x[0]=sizeof(sts::BattleContext);x[1]=sizeof(sts::Action);}}

namespace sts::search {
void parallel_prefix(const BattleContext &bc,PlayoutEntityObservation &out,int*pot_rows){
    out.entities.clear(); out.sources.clear(); out.targets.clear();
    auto add=[&](PlayoutEntity v){int i=int(out.entities.size());out.entities.push_back(v);return i;};
    const auto &p=bc.player; const auto &c=bc.cards; const auto &s=bc.cardSelectInfo;
    const bool selection=bc.inputState==InputState::CARD_SELECT;
    if(selection && (s.selectedCount<0 || s.selectedCount>CardSelectInfo::GRID_PICK_CAPACITY))
        gpu_fail(__LINE__);
    auto battle=row(K::Battle,int(bc.encounter),Z::Battle);
    battle.numbers={f(bc.ascension)/20,f(bc.floorNum)/60,f(bc.turn)/50,f(bc.inputState)/8,
        f(bc.potionCapacity)/5,f(bc.potionCount)/5,f(bc.monsters.monsterCount)/5,
        f(bc.monsters.monstersAlive)/5,f(bc.postEnemyTurnBoundary),f(bc.pocketwatchCardsAtBoundary)/10,
        f(bc.endTurnQueued),f(bc.turnHasEnded),f(bc.skipMonsterTurn),f(c.cardsInHand)/10,
        f(c.knownDrawTop)/40,f(bc.outcome)/4}; add(battle);
    auto pl=row(K::Player,int(p.cc),Z::Player);
    pl.numbers={f(p.curHp)/200,f(p.maxHp)/200,f(p.block)/100,f(p.energy)/10,f(p.gold)/500,
        f(p.energyPerTurn)/10,f(p.cardDrawPerTurn)/10,f(p.stance)/5,f(p.orbSlots)/10,
        f(p.lastTargetedMonster)/5,f(p.strength)/20,f(p.dexterity)/20,f(p.focus)/20,
        f(p.artifact)/5,f(p.bloodied),f(p.redSkullActive)}; add(pl);
    const int counters[]={p.happyFlowerCounter,p.incenseBurnerCounter,p.inkBottleCounter,p.inserterCounter,
        p.nunchakuCounter,p.penNibCounter,p.sundialCounter,p.neowLamentRemaining,p.haveUsedNecronomiconThisTurn,
        p.combustHpLoss,p.devaFormEnergyPerTurn,p.echoFormCardsDoubled,p.panacheCounter,p.cardsPlayedThisTurn,
        p.attacksPlayedThisTurn,p.skillsPlayedThisTurn,int(p.orangePelletsCardTypesPlayed.to_ulong()),
        p.cardsDiscardedThisTurn,p.lastAttackUnblockedDamage,p.timesDamagedThisCombat,p.feelNoPainBeforeDarkEmbrace};
    for (int i=0;i<int(sizeof(counters)/sizeof(counters[0]));++i) if (counters[i]) {
        auto v=row(K::Counter,i,Z::Player);v.numbers[0]=f(counters[i])/10;add(v);
    }
    std::uint64_t flags[]={p.statusBits0,p.statusBits1};
    for(PS power:{PS::ARTIFACT,PS::DEXTERITY,PS::FOCUS,PS::STRENGTH}){
        const int i=int(power);
        if(p.hasStatusRuntime(power)) flags[i/64]|=1ull<<(i%64);
        else flags[i/64]&=~(1ull<<(i%64));
    }
    auto addPower=[&](int i){
        if(i==0 || i>int(PS::THE_BOMB))return;
        auto v=row(K::Power,i,Z::Player);v.numbers[0]=f(p.getStatusRuntime(PS(i)))/10;
        v.numbers[1]=i<32 && ((p.justAppliedBits>>i)&1u);add(v);
    };
    eachBit(flags[0],0,addPower);eachBit(flags[1],64,addPower);
    for (int i=0;i<int(p.bombs.size());++i) {
        auto v=row(K::Counter,32,Z::Player);v.numbers[0]=f(p.bombs[i].turns)/10;
        v.numbers[1]=f(p.bombs[i].damage)/100;v.numbers[2]=f(i+1)/10;add(v);
    }
    auto addRelic=[&](int i){add(row(K::Relic,i,Z::ActiveRelic));};
    eachBit(p.relicBits0,0,addRelic);eachBit(p.relicBits1,64,addRelic);
    auto order=[&](const auto &items,int group){for(int i=0;i<int(items.size());++i){
        auto v=row(K::Order,int(items[i]),Z::Player);v.numbers[0]=f(group);
        v.numbers[1]=f(i+1)/32;add(v);
    }};
    order(bc.onUseRelicOrder,1);order(bc.onEndTurnRelicOrder,2);order(bc.onPostDrawRelicOrder,3);
    order(p.endTurnPowerOrder,4);
    auto *pot=pot_rows;
    for(int i=0;i<bc.potionCapacity;++i){auto v=row(K::Potion,int(bc.potions[i]),Z::Potion);
        v.numbers[0]=f(i+1)/5;pot[i]=add(v);}
    const bool dome=p.hasRelic<R::RUNIC_DOME>();
    for(int i=0;i<5;++i){
        const auto &m=bc.monsters.arr[i];auto v=row(K::Monster,int(m.id),Z::Monster,i);
        if(m.id!=MonsterId::INVALID){
            v.numbers={f(m.curHp)/200,f(m.maxHp)/200,f(m.block)/100,f(m.strength)/20,
                f(m.isEscapingB),f(m.isDyingB),f(m.isDeadB),f(m.halfDead),f(m.isTargetable()),
                f(m.isDeadOrEscaped()),f(m.isAlive()),f(m.isEscaping()),0,0,0,0};
            // A concealed intent may not leak through escapeNext, displayed
            // inactive damage or its cached RNG-selected base damage.
            if(!dome){v.numbers[12]=f(m.escapeNext);v.numbers[13]=f(m.inactiveIntentDamage)/100;}
        }
        add(v);if(m.id==MonsterId::INVALID)continue;
        eachBit(m.statusBits|(m.strength ? 1ull<<int(MS::STRENGTH):0),0,[&](int j){
            if(j>=int(MS::INVALID))return;
            auto q=row(K::MonsterPower,j,Z::Monster,i);q.numbers[0]=f(m.getStatusInternal(MS(j)))/10;
            if(MS(j)==MS::VULNERABLE)q.numbers[1]=m.wasJustApplied<MS::VULNERABLE>();
            if(MS(j)==MS::WEAK)q.numbers[1]=m.wasJustApplied<MS::WEAK>();
            if(MS(j)==MS::RITUAL)q.numbers[1]=m.wasJustApplied<MS::RITUAL>();
            add(q);
        });
        for(int h=dome?1:0;h<3;++h){auto q=row(K::Move,int(m.moveHistory[h]),Z::Monster,i);
            q.numbers[0]=f(h);if(!h){const auto d=m.getMoveBaseDamage(bc);
                q.numbers[1]=f(d.damage)/100;q.numbers[2]=f(m.getIntentDamage(bc,d.damage))/100;
                q.numbers[3]=f(d.attackCount)/10;}add(q);}
        if(dome)continue; // Engine move-state latches can be set while rolling a concealed intent.
        auto q=row(K::Counter,33,Z::Monster,i);
        switch(m.id){
        case MonsterId::LOOTER:case MonsterId::MUGGER:
            q.numbers[1]=f(m.thiefSlashCount)/10;q.numbers[2]=f(m.thiefGoldAmount)/100;
            [[fallthrough]];
        case MonsterId::LAGAVULIN:case MonsterId::GREMLIN_WIZARD:case MonsterId::WRITHING_MASS:
        case MonsterId::TIME_EATER:case MonsterId::RED_SLAVER:case MonsterId::BRONZE_ORB:
        case MonsterId::SPIKER:case MonsterId::THE_CHAMP:case MonsterId::BRONZE_AUTOMATON:
        case MonsterId::BOOK_OF_STABBING:case MonsterId::THE_GUARDIAN:case MonsterId::GIANT_HEAD:
        case MonsterId::AWAKENED_ONE:case MonsterId::HEXAGHOST:
            q.numbers[0]=f(m.miscInfo)/32;add(q);break;
        default:break; // Never encode concealed Louse bite/Darkling nip rolls or an AI target slot.
        }
    }

}
PlayoutEntity parallel_card(const BattleContext &bc,int masterDeckSize,const CardInstance &x,PlayoutZone zone,int position,int picked=0){
        auto v=row(K::Card,int(x.id),zone);
        v.numbers={f(x.getUpgradeCount())/10,f(x.specialData)/100,f(x.cachedBaseDamage)/100,
            f(x.cost)/5,f(x.costForTurn)/5,f(x.freeToPlayOnce),f(x.retain),f(x.exhaust),
            f(position)/40,f(picked)/2,f(x.uniqueId>=0 && x.uniqueId<masterDeckSize),0,0,0,0,0};
        return v;

}
void parallel_state(const BattleContext &bc,SmallFeatures &f){
    auto &s = f.state;
    s[0] = float(bc.player.curHp) / 200; s[1] = float(bc.player.maxHp) / 200;
    s[2] = float(bc.player.block) / 100; s[3] = float(bc.player.energy) / 10;
    s[4] = float(bc.turn) / 20; s[5] = float(bc.potionCount) / 5;
    s[6] = float(bc.cards.cardsInHand) / 10;
    s[7] = float(bc.cards.drawPile.size()) / 40;
    s[8] = float(bc.cards.discardPile.size()) / 40;
    s[9] = float(bc.cards.exhaustPile.size()) / 40;
    s[10] = float(bc.player.strength) / 20; s[11] = float(bc.player.dexterity) / 20;
    s[12] = float(bc.player.getStatus<PS::WEAK>()) / 5;
    s[13] = float(bc.player.getStatus<PS::VULNERABLE>()) / 5;
    s[14] = float(bc.player.getStatus<PS::FRAIL>()) / 5;
    s[15] = float(bc.player.artifact) / 5;
    for (int i = 0; i < 5; ++i) {
        const auto &m = bc.monsters.arr[i];
        s[16 + 3*i] = float(std::max(0, m.curHp)) / 200;
        s[17 + 3*i] = float(m.block) / 100;
        s[18 + 3*i] = incoming(bc, m) / 50;
    }
    s[31] = bc.inputState == InputState::CARD_SELECT ? 1.f : 0.f;

}
SmallActionFeatures parallel_action(const BattleContext &bc,const Action &action){
const CombatChoices choices{};
        SmallActionFeatures a; a.bits = action.bits;
        const auto kind = action.getActionType();
        a.x[static_cast<int>(kind)] = 1;
        a.x[13] = float(choices.zone) / 7;
        if (bc.inputState == InputState::CARD_SELECT) {
            a.x[14] = float(bc.cardSelectInfo.cardSelectTask) / 64;
            a.x[23] = float(bc.cardSelectInfo.selectedCount) / 10;
        }
        auto append = [&](const CardInstance &c) {
            addCard(a, c);
            a.x[9] += c.freeToPlayOnce;
            a.x[10] += c.doesExhaust(); a.x[11] += c.retain;
            a.x[12] += c.isEthereal();
        };
        if (kind == ActionType::CARD) append(bc.cards.hand[action.getSourceIdx()]);
        else if (kind == ActionType::POTION) {
            const int p = int(bc.potions[action.getSourceIdx()]);
            if (p < POTION_FIRST || p >= POTION_FIRST + POTION_KEYS)
                gpu_fail(__LINE__);
            a.addKey(3 + CARD_KEYS + p - POTION_FIRST);
        } else if (kind == ActionType::SINGLE_CARD_SELECT) {
            if (bc.cardSelectInfo.cardSelectTask == CardSelectTask::CODEX && action.getSelectIdx() == 3)
                a.addKey(1);
            else append(combatChoiceCard(bc, choices.zone, action.getSelectIdx()));
        } else if (kind == ActionType::MULTI_CARD_SELECT) {
            for (int index : action.getSelectedIdxs()) append(bc.cards.hand[index]);
            a.x[15] = float(a.keyCount) / 10;
            if (a.keyCount==0) a.addKey(2);
        } else a.addKey(0);
        const bool targeted =
            (kind == ActionType::CARD && bc.cards.hand[action.getSourceIdx()].requiresTarget()) ||
            (kind == ActionType::POTION && potionRequiresTarget(bc.potions[action.getSourceIdx()]));
        if (targeted) {
            const int target = action.getTargetIdx();
            if (target >= 0 && target < 5) {
                const auto &m = bc.monsters.arr[target];
                a.x[16] = float(m.curHp) / 200; a.x[17] = float(m.maxHp) / 200;
                a.x[18] = float(m.block) / 100; a.x[19] = float(m.strength) / 20;
                a.x[20] = float(m.vulnerable) / 5; a.x[21] = float(m.weak) / 5;
                a.x[22] = float(target + 1) / 5;
            }
        }

return a;
}
}

bool parallel_capable(const sts::BattleContext& b){
 if(b.inputState!=sts::InputState::PLAYER_NORMAL)return false;
 int draw=b.cards.drawPile.size();
 int known=b.player.hasRelic<sts::R::FROZEN_EYE>()?draw:std::clamp(b.cards.knownDrawTop,0,draw);
 return draw-known<=64;
}
extern "C" __global__ void frame_observe_special(unsigned char*arena,int*worlds,int*live,int*deck_sizes,
 float*state,unsigned*bits,float*action_numbers,int*keys,int*source_rows,int*targets,
 int*ids,int*zones,int*monsters,float*numbers,int*meta,int*chunks,int*allocator,
 const int*frame_count,DECK_EXTRA int base,int root_count,int action_capacity,int entity_capacity){
 int count=frame_count[0];int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const auto&b=*reinterpret_cast<const sts::BattleContext*>(packed_address(arena,live[base+i]));
 if(parallel_capable(b))return;
#ifdef STS_FULL480_DECK
 auto deck_actions=sts::search::autoregressiveActions(b,prefixes[live[base+i]]);
 deck_input::prepare(b,deck_actions,i,prefixes[live[base+i]],bits,keys,source_rows,targets,ids,zones,monsters,meta,chunks,allocator,action_capacity,entity_capacity,raw,cat,legal_mask,selected,remaining);
 return;
#endif

 auto legal=device_state::legal_actions(b);int na=legal.size();
 device_state::require(na>0&&na<=2048);
 int start=atomicAdd(allocator,na);meta[i*4]=na;meta[i*4+1]=0;meta[i*4+2]=start;meta[i*4+3]=0;
 atomicMax(allocator+4,na);
 if(start>action_capacity-na){atomicOr(allocator+2,1);return;}
 sts::search::SmallFeatures f{};
 f.actions={bits+start,action_numbers+start*24,keys+start*10,0,na};
 sts::search::featuresForActions(b,legal,f);
 for(int j=0;j<32;++j)state[i*32+j]=f.state[j];
 sts::search::PlayoutEntityObservation o{};
 o.entities={ids,zones,monsters,numbers,0,1024};
 o.entities.pool_chunks=chunks+i*POOL_CHUNKS;o.entities.pool_allocator=allocator+1;
 o.entities.pool_overflow=allocator+2;o.entities.pool_capacity=entity_capacity;
 o.sources={reinterpret_cast<std::array<int,10>*>(source_rows+start*10),0,na};
 o.targets={targets+start,0,na};
 sts::search::playoutEntities(b,deck_sizes[worlds[base+i]%root_count],f.actions,o,false);
 device_state::require(o.sources.size()==na&&o.targets.size()==na);
 meta[i*4+1]=o.entities.size();meta[i*4+3]=!o.entities.pool_failed;
 atomicMax(allocator+3,int(o.entities.size()));
}


namespace sts::search {
struct NormalActionSink {
 unsigned bits[76];int n=0;
 template<class...T>void emplace_back(T...args){device_state::require(n<76);bits[n++]=Action(args...).bits;}
 int size()const{return n;}
 Action operator[](int i)const{device_state::require(i>=0&&i<n);return Action(bits[i]);}
};
NormalActionSink normal_legal(const BattleContext&bc){
 device_state::require(bc.inputState==InputState::PLAYER_NORMAL&&bc.outcome==Outcome::UNDECIDED);
 device_state::require(bc.cards.cardsInHand<=10&&bc.potionCapacity<=5&&bc.monsters.monsterCount<=5);
 NormalActionSink actions;
 if(bc.player.cardsPlayedThisTurn>=device_state::TURN_CARD_LIMIT){actions.emplace_back(ActionType::END_TURN);return actions;}
 constexpr bool allowDiscard=false;

    if (bc.isCardPlayAllowed()) {
        for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
            const auto &card = bc.cards.hand[handIdx];
            if (!card.canUseOnAnyTarget(bc)) {
                continue;
            }
            if (card.requiresTarget()) {
                for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                    if (bc.monsters.arr[targetIdx].isTargetable()) {
                        actions.emplace_back(ActionType::CARD, handIdx, targetIdx);
                    }
                }
            } else {
                actions.emplace_back(ActionType::CARD, handIdx);
            }
        }
    }

    const bool hasValidTarget = bc.monsters.getTargetableCount() > 0;
    for (int potionIdx = 0; potionIdx < bc.potionCapacity; ++potionIdx) {
        const auto potion = bc.potions[potionIdx];
        if (potion == Potion::INVALID || potion == Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        if (!bc.canDrinkPotion(potionIdx) ||
            (potionRequiresTarget(potion) && !hasValidTarget)) {
            if (allowDiscard) actions.emplace_back(ActionType::POTION, potionIdx, -1);
        } else if (!potionRequiresTarget(potion)) {
            actions.emplace_back(ActionType::POTION, potionIdx);
        } else {
            for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                if (bc.monsters.arr[targetIdx].isTargetable()) {
                    actions.emplace_back(ActionType::POTION, potionIdx, targetIdx);
                }
            }
        }
    }
    actions.emplace_back(ActionType::END_TURN);

 return actions;
}
}

extern "C" __global__ void normal_prepare(unsigned char*arena,int*worlds,int*live,int*deck_sizes,
 float*state,unsigned*bits,float*action_numbers,int*keys,int*source_rows,int*targets,
 int*ids,int*zones,int*monsters,float*numbers,int*meta,int*chunks,int*allocator,
 const int*frame_count,DECK_EXTRA int*desc,int base,int root_count,int action_capacity,int entity_capacity){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=frame_count[0])return;
 const auto&b=*reinterpret_cast<const sts::BattleContext*>(packed_address(arena,live[base+i]));
 if(!parallel_capable(b))return;
#ifdef STS_FULL480_DECK
 auto deck_actions=sts::search::normal_legal(b);
 deck_input::prepare(b,deck_actions,i,prefixes[live[base+i]],bits,keys,source_rows,targets,ids,zones,monsters,meta,chunks,allocator,action_capacity,entity_capacity,raw,cat,legal_mask,selected,remaining);
 return;
#endif

 auto legal=sts::search::normal_legal(b);int na=legal.size();device_state::require(na>0&&na<=2048);
 int start=atomicAdd(allocator,na);meta[i*4]=na;meta[i*4+1]=0;meta[i*4+2]=start;meta[i*4+3]=0;
 atomicMax(allocator+4,na);if(start>action_capacity-na){atomicOr(allocator+2,1);return;}
 for(int a=0;a<na;++a)bits[start+a]=legal[a].bits;
 sts::search::PlayoutEntityObservation o{};
 o.entities={ids,zones,monsters,numbers,0,1024};
 o.entities.pool_chunks=chunks+i*POOL_CHUNKS;o.entities.pool_allocator=allocator+1;
 o.entities.pool_overflow=allocator+2;o.entities.pool_capacity=entity_capacity;
 sts::search::parallel_prefix(b,o,desc+i*8+1);
 int head=o.entities.size();desc[i*8]=head;
 const auto&c=b.cards;int draw=c.drawPile.size();
 int known=b.player.hasRelic<sts::R::FROZEN_EYE>()?draw:std::clamp(c.knownDrawTop,0,draw);desc[i*8+6]=known;
 int tail=c.cardsInHand+draw+c.discardPile.size()+c.exhaustPile.size();
 for(int j=0;j<2;++j)tail+=int(c.stasisCards[j].id!=sts::CardId::INVALID);
 int ne=head+tail;device_state::require(ne<=1024);
 for(int ch=(head+63)/64;ch<(ne+63)/64;++ch){
  int at=atomicAdd(allocator+1,64);chunks[i*POOL_CHUNKS+ch]=at;
  if(at>entity_capacity-64){o.entities.pool_failed=true;atomicOr(allocator+2,2);}
 }
 desc[i*8+7]=tail;meta[i*4+1]=ne;meta[i*4+3]=!o.entities.pool_failed;atomicMax(allocator+3,ne);
}
int parallel_compare(const sts::search::PlayoutEntity&a,const sts::search::PlayoutEntity&b){
 if(a.id!=b.id)return a.id<b.id?-1:1;
 for(int k=0;k<16;++k){if(a.numbers[k]<b.numbers[k])return -1;if(a.numbers[k]>b.numbers[k])return 1;}
 return 0;
}
extern "C" __global__ void normal_features(unsigned char*arena,int*worlds,int*live,int*deck_sizes,
 float*state,unsigned*bits,float*action_numbers,int*keys,int*source_rows,int*targets,
 int*ids,int*zones,int*monsters,float*numbers,int*meta,int*chunks,int*allocator,
 const int*frame_count,DECK_EXTRA int*desc,int base,int root_count,int action_capacity,int entity_capacity){
 if(allocator[2])return;
 for(int i=blockIdx.x;i<frame_count[0];i+=gridDim.x){
  const auto&b=*reinterpret_cast<const sts::BattleContext*>(packed_address(arena,live[base+i]));
  #ifdef STS_FULL480_DECK
  if(meta[i*4]==1)continue;
  device_state::require(meta[i*4+3]);
  int root=worlds[base+i]%root_count;
  for(int row=0;row<meta[i*4+1];++row){
   int at=deck_input::physical(chunks,i,row),slot=ids[at];
   for(int dim=threadIdx.x;dim<64;dim+=blockDim.x)
    numbers[at*64+dim]=deck_input::token(b,weights,raw+i*deck_input::RAW,cat+i*deck_input::CAT,root_inputs+root*68,root_relics+root*deck_input::RELIC_WIDTH,root_counters+root*deck_input::RELIC_WIDTH,prefixes[live[base+i]],slot,dim);
  }
  continue;
#endif
  if(!parallel_capable(b))continue;
  device_state::require(meta[i*4+3]);
  const auto&c=b.cards;int na=meta[i*4],start=meta[i*4+2],head=desc[i*8],known=desc[i*8+6],tail=desc[i*8+7];
  int master=deck_sizes[worlds[base+i]%root_count],draw=c.drawPile.size(),hidden=draw-known;
  if(threadIdx.x==0){sts::search::SmallFeatures f{};sts::search::parallel_state(b,f);for(int k=0;k<32;++k)state[i*32+k]=f.state[k];}
  for(int a=threadIdx.x;a<na;a+=blockDim.x){
   sts::search::Action action(bits[start+a]);auto f=sts::search::parallel_action(b,action);
   for(int k=0;k<24;++k)action_numbers[(start+a)*24+k]=f.x[k];
   for(int k=0;k<10;++k){keys[(start+a)*10+k]=k<f.keyCount?f.keys[k]:-1;source_rows[(start+a)*10+k]=-1;}
   int target=-1;auto kind=action.getActionType();
   if(kind==sts::search::ActionType::CARD){
    int index=action.getSourceIdx();device_state::require(index>=0&&index<c.cardsInHand);
    source_rows[(start+a)*10]=head+index;if(c.hand[index].requiresTarget())target=action.getTargetIdx();
   }else if(kind==sts::search::ActionType::POTION){
    int index=action.getSourceIdx();device_state::require(index>=0&&index<b.potionCapacity);
    source_rows[(start+a)*10]=desc[i*8+1+index];if(sts::potionRequiresTarget(b.potions[index]))target=action.getTargetIdx();
   }else device_state::require(kind==sts::search::ActionType::END_TURN);
   device_state::require(target>=-1&&target<5);targets[start+a]=target;
  }
  for(int q=threadIdx.x;q<tail;q+=blockDim.x){
   using Z=sts::search::PlayoutZone;sts::search::PlayoutEntity v{};int at=head+q,j=q;
   if(j<c.cardsInHand)v=sts::search::parallel_card(b,master,c.hand[j],Z::Hand,j+1);
   else if((j-=c.cardsInHand)<hidden){
    v=sts::search::parallel_card(b,master,c.drawPile[j],Z::HiddenDraw,0);int rank=0;
    for(int other=0;other<hidden;++other){
     auto u=sts::search::parallel_card(b,master,c.drawPile[other],Z::HiddenDraw,0);
     int cmp=parallel_compare(u,v);rank+=(cmp<0||(cmp==0&&other<j));
    }
    at=head+c.cardsInHand+rank;
   }else if((j-=hidden)<known)v=sts::search::parallel_card(b,master,c.drawPile[draw-1-j],Z::KnownDraw,j+1);
   else if((j-=known)<int(c.discardPile.size()))v=sts::search::parallel_card(b,master,c.discardPile[j],Z::Discard,j+1);
   else if((j-=c.discardPile.size())<int(c.exhaustPile.size()))v=sts::search::parallel_card(b,master,c.exhaustPile[j],Z::Exhaust,j+1);
   else{
    j-=c.exhaustPile.size();int index=c.stasisCards[0].id!=sts::CardId::INVALID?j:1;
    device_state::require(index>=0&&index<2&&c.stasisCards[index].id!=sts::CardId::INVALID);
    v=sts::search::parallel_card(b,master,c.stasisCards[index],Z::Stasis,index+1);
   }
   int dest=chunks[i*POOL_CHUNKS+at/64]+at%64;
   device_state::require(dest>=0&&dest<entity_capacity);
   ids[dest]=v.id;zones[dest]=v.zone;monsters[dest]=v.monster;
   for(int k=0;k<16;++k)numbers[dest*16+k]=v.numbers[k];
  }
 }
}
