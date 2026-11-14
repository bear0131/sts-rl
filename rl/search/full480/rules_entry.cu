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

#include "small_redeterminize.cuh"
constexpr int POLICY_STATS=12;
namespace mirror_semantics {
using namespace sts;using namespace sts::search;
struct StrongPublicHash {
    std::uint64_t value;
    StrongPublicHash(std::uint64_t initial): value(initial) {}
    operator std::uint64_t() const { return value; }
};
inline void hashCombine(StrongPublicHash &hash, std::uint64_t value) {
    hash.value = noslMix(hash.value ^ noslMix(value));
}


template <typename H, typename T>
inline void hashValue(H &hash, T value) {
    hashCombine(hash, static_cast<std::uint64_t>(value));
}


int cardRetentionExhaustSignature(const CardInstance &card) {
    // Keep the exact old hash input for ordinary printed cards. The spare bit
    // distinguishes an independent exhaust mutation without changing normal
    // search tie-breaking / determinization identities for existing weights.
    return (card.retain ? 1 : 0)
        | (card.exhaust != doesCardExhaust(card.id, card.upgraded) ? 2 : 0);
}

int persistentCardIdentity(const CardInstance &card) {
    return card.id == CardId::RITUAL_DAGGER || card.id == CardId::GENETIC_ALGORITHM
        ? card.uniqueId : -1;
}


enum class ActionCardCollection : int {
    HAND = 1,
    DRAW,
    DISCARD,
    EXHAUST,
};

int cardSelectionCollection(CardSelectTask task) {
    switch (task) {
        case CardSelectTask::ARMAMENTS:
        case CardSelectTask::DUAL_WIELD:
        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            return static_cast<int>(ActionCardCollection::HAND);
        case CardSelectTask::EXHUME:
            return static_cast<int>(ActionCardCollection::EXHAUST);
        case CardSelectTask::HEADBUTT:
        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
        case CardSelectTask::MEDITATE:
            return static_cast<int>(ActionCardCollection::DISCARD);
        case CardSelectTask::SECRET_TECHNIQUE:
        case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK:
            return static_cast<int>(ActionCardCollection::DRAW);
        default:
            return 0;
    }
}

template <typename H>
void hashActionCard(H &hash, const CardInstance &card,
                    bool includeCost) {
    hashValue(hash, static_cast<int>(card.getId()));
    hashValue(hash, static_cast<int>(card.specialData));
    hashValue(hash, static_cast<int>(card.cost));
    hashValue(hash, static_cast<int>(card.costForTurn));
    hashValue(hash, card.upgraded ? 1 : 0);
    hashValue(hash, card.freeToPlayOnce ? 1 : 0);
    hashValue(hash, cardRetentionExhaustSignature(card));
    hashValue(hash, persistentCardIdentity(card));
    if (includeCost) {
        hashValue(hash, card.freeToPlayOnce ? 0 : card.costForTurn);
    }
    if (card.id == CardId::MIND_BLAST) {
        hashValue(hash, 0x4d696e64426c6173ULL);
        hashValue(hash, card.cachedBaseDamage);
    }
}

const CardInstance *actionCardAt(const BattleContext &bc, int collection,
                                 int index) {
    switch (static_cast<ActionCardCollection>(collection)) {
        case ActionCardCollection::HAND:
            return index >= 0 && index < bc.cards.cardsInHand
                ? &bc.cards.hand[index] : nullptr;
        case ActionCardCollection::DRAW:
            return index >= 0 && index < static_cast<int>(bc.cards.drawPile.size())
                ? &bc.cards.drawPile[index] : nullptr;
        case ActionCardCollection::DISCARD:
            return index >= 0 && index < static_cast<int>(bc.cards.discardPile.size())
                ? &bc.cards.discardPile[index] : nullptr;
        case ActionCardCollection::EXHAUST:
            return index >= 0 && index < static_cast<int>(bc.cards.exhaustPile.size())
                ? &bc.cards.exhaustPile[index] : nullptr;
        default:
            return nullptr;
    }
}

// Action ids use hand/pile indexes, but those indexes are not stable across
// NOSL samples when the corresponding collection is treated as a composition.
// This key lets a selected edge be remapped to the equivalent current index.
template <typename H = std::uint64_t>
__noinline__ std::uint64_t publicActionKey(const BattleContext &bc, const sts::search::Action &action) {
    H hash = 0xcbf29ce484222325ULL;
    hashValue(hash, static_cast<int>(action.getActionType()));
    if (action.getActionType() == kSelectNext) {
        hashValue(hash, action.getSourceIdx());
        return hash;
    }
    switch (action.getActionType()) {
        case ActionType::CARD: {
            hashValue(hash, action.getTargetIdx());
            const auto *card = actionCardAt(
                bc, static_cast<int>(ActionCardCollection::HAND),
                action.getSourceIdx());
            if (card != nullptr) {
                hashActionCard(hash, *card, true);
            } else {
                hashValue(hash, action.getSourceIdx());
            }
            break;
        }
        case ActionType::POTION:
            // Potion slots are observable and remain positional.
            hashValue(hash, action.getSourceIdx());
            hashValue(hash, action.getTargetIdx());
            if (action.getSourceIdx() >= 0 &&
                action.getSourceIdx() < bc.potionCapacity) {
                hashValue(hash, static_cast<int>(
                    bc.potions[action.getSourceIdx()]));
            }
            break;
        case ActionType::SINGLE_CARD_SELECT: {
            hashValue(hash, static_cast<int>(
                bc.cardSelectInfo.cardSelectTask));
            const int collection = cardSelectionCollection(
                bc.cardSelectInfo.cardSelectTask);
            const auto *card = actionCardAt(
                bc, collection, action.getSelectIdx());
            if (card != nullptr) {
                hashActionCard(
                    hash, *card,
                    collection == static_cast<int>(ActionCardCollection::HAND));
            } else if (bc.cardSelectInfo.cardSelectTask == CardSelectTask::CODEX ||
                       bc.cardSelectInfo.cardSelectTask == CardSelectTask::DISCOVERY) {
                const int index = action.getSelectIdx();
                if (index >= 0 && index < static_cast<int>(bc.cardSelectInfo.cards.size())) {
                    hashValue(hash, static_cast<int>(bc.cardSelectInfo.cards[index]));
                } else {
                    hashValue(hash, index);
                }
            } else {
                hashValue(hash, action.getSelectIdx());
            }
            break;
        }
        case ActionType::MULTI_CARD_SELECT: {
            hashValue(hash, static_cast<int>(
                bc.cardSelectInfo.cardSelectTask));
            std::array<std::uint64_t, CardManager::MAX_HAND_SIZE> selected{};
            std::size_t selectedCount = 0;
            for (const int index : action.getSelectedIdxs()) {
                const auto *card = actionCardAt(
                    bc, static_cast<int>(ActionCardCollection::HAND), index);
                if (card == nullptr) {
                    device_state::require(false);
                }
                // A set of card names/current costs is not a set of complete
                // instances (upgrades, free-once, exhaust, etc. can differ).
                // Reuse the single-card identity, retaining multiplicity.
                H cardKey = 0xcbf29ce484222325ULL;
                hashActionCard(cardKey, *card, true);
                selected[selectedCount++] = cardKey;
            }
            std::sort(selected.begin(), selected.begin() + selectedCount);
            hashValue(hash, selectedCount);
            for (std::size_t i = 0; i < selectedCount; ++i) {
                hashValue(hash, selected[i]);
            }
            break;
        }
        case ActionType::END_TURN:
        default:
            break;
    }
    return hash;
}

}

// Keep action enumeration and semantic validation out of the execution frame.
// Their temporary vectors must be destroyed before nested game callbacks execute.
namespace battle_actions {
__noinline__ bool legal(const sts::BattleContext &state,unsigned bits){
 for(auto action:device_state::legal_actions(state))if(action.bits==bits)return true;
 return false;
}
__noinline__ unsigned resolve(const sts::BattleContext &state,unsigned bits,unsigned long long wanted){
 sts::search::Action action(bits);
 if(action.isValidAction(state)&&mirror_semantics::publicActionKey<mirror_semantics::StrongPublicHash>(state,action)==wanted)return bits;
 for(auto candidate:device_state::legal_actions(state)){
  if(mirror_semantics::publicActionKey<mirror_semantics::StrongPublicHash>(state,candidate)==wanted)return candidate.bits;
 }
 device_state::require(false);return 0;
}
}

// Finite root-action-world jobs. All states and queued callbacks belong to this module.
namespace finite_search {
inline double quiet_nan(){union{unsigned long long bits;double value;}v;v.bits=0x7ff8000000000000ull;return v.value;}
inline void finish(const sts::BattleContext&s,unsigned long long*st,int&alive,int&outcome,double&value,int boss,int maximum,int limit){
 outcome=int(s.outcome);
 if(s.outcome!=sts::Outcome::UNDECIDED){
  alive=0;value=boss?(s.outcome==sts::Outcome::PLAYER_VICTORY?1.:0.):double(std::max(0,s.player.curHp))/double(maximum);
  if(s.outcome==sts::Outcome::PLAYER_VICTORY)st[3]=1;
  else if(s.outcome==sts::Outcome::PLAYER_LOSS)st[4]=1;
  else if(s.outcome==sts::Outcome::PLAYER_ESCAPE)st[9]=1;
  else device_state::require(false);
 }else if(st[7]>=static_cast<unsigned long long>(limit)){alive=0;value=quiet_nan();st[5]=1;}
 else {alive=1;value=quiet_nan();}
}
}
// Persistent root segments. Root and chain objects share one owning module.
namespace stream_search {
__device__ void finish(const sts::BattleContext&s,unsigned long long*st,int&alive,int&outcome,double&value,int boss,int maximum,int limit,int*done){
 finite_search::finish(s,st,alive,outcome,value,boss,maximum,limit);
 if(!alive)atomicAdd(done,1);
}
}
extern "C" __global__ void stream_clear(unsigned char*arena,int*constructed,int*alive,int*root_ids,int count,int budget_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;
 int i=root_ids[k/budget_cap]*budget_cap+k%budget_cap;
 device_state::require(!alive[i]);
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext))->~BattleContext();constructed[i]=0;}
}
extern "C" __global__ void stream_load(unsigned char*roots,int*constructed,int*root_ids,unsigned long long*words,int*lengths,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int r=root_ids[k];
 auto*address=roots+r*sizeof(sts::BattleContext);
 if(constructed[r])reinterpret_cast<sts::BattleContext*>(address)->~BattleContext();
 auto&s=*new(address)sts::BattleContext{};
 device_state::Archive<true> a{words+k*8192,lengths[k]};a.field(s);device_state::require(a.at==lengths[k]);
 device_state::require(s.outcome==sts::Outcome::UNDECIDED&&s.inputState==sts::InputState::PLAYER_NORMAL&&s.actionQueue.size==0);
 constructed[r]=1;
}
extern "C" __global__ void stream_start(unsigned char*roots,unsigned char*arena,int*root_ids,unsigned long long*job_generation,
 unsigned long long*root_generation,unsigned long long*root_nonce,int*budgets,int*action_counts,unsigned*ordered_bits,
 unsigned long long*stats,int*constructed,int*alive,unsigned long long*world_seeds,unsigned long long*policy_seeds,unsigned*initial_bits,
 int count,int budget_cap,int action_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;
 int r=root_ids[k/budget_cap],j=k%budget_cap,i=r*budget_cap+j;
 device_state::require(!constructed[i]&&!alive[i]);
 if(j>=budgets[r])return;
 int na=action_counts[r];device_state::require(na>0&&na<=action_cap);
 auto&root=*reinterpret_cast<sts::BattleContext*>(roots+r*sizeof(sts::BattleContext));
 auto&s=*new(arena+i*sizeof(sts::BattleContext))sts::BattleContext(root);constructed[i]=1;alive[i]=1;
 auto*st=stats+i*POLICY_STATS;for(int x=0;x<POLICY_STATS;++x)st[x]=0;
 job_generation[i]=root_generation[r];
 unsigned long long seed=sts::search::noslMix(sts::search::noslMix(root_nonce[r])+static_cast<unsigned long long>(j/na));
 world_seeds[i]=seed;policy_seeds[i]=sts::search::noslMix(seed^0x504149524544ull);initial_bits[i]=ordered_bits[r*action_cap+j%na];
 sts::search::smallRedeterminize(s,seed);
}
extern "C" __global__ void stream_begin(unsigned char*arena,int*root_ids,int*budgets,unsigned*initial_bits,unsigned long long*stats,
 int*alive,int*outcomes,double*values,int*root_boss,int*root_maximum,int*root_limits,int*root_done,unsigned long long*ordered_semantic,int*action_counts,unsigned*root_executed,int count,int budget_cap,int action_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;int r=root_ids[k/budget_cap],j=k%budget_cap;
 if(j>=budgets[r])return;int i=r*budget_cap+j;
 auto&s=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));auto*st=stats+i*POLICY_STATS;
 const unsigned bits=battle_actions::resolve(s,initial_bits[i],ordered_semantic[r*action_cap+j%action_counts[r]]);
 root_executed[i]=bits;sts::search::Action(bits).execute(s);
 stream_search::finish(s,st,alive[i],outcomes[i],values[i],root_boss[r],root_maximum[r],root_limits[r],root_done+r);
}
extern "C" __global__ void stream_snapshot(unsigned char*arena,int*constructed,int*selected,unsigned long long*words,int*meta,unsigned*actions,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=selected[k];device_state::require(constructed[i]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
 device_state::Archive<false> a{words+k*8192,8192};a.field(s);auto legal=device_state::legal_actions(s);device_state::require(legal.size()<=2048);
 auto*m=meta+k*6;m[0]=a.at;m[1]=s.actionQueue.size;m[2]=int(a.max_vector);m[3]=int(s.inputState);m[4]=int(s.outcome);m[5]=legal.size();
 for(int j=0;j<int(legal.size());++j)actions[k*2048+j]=legal[j].bits;
}
extern "C" __global__ void stream_destroy(unsigned char*arena,int*constructed,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext))->~BattleContext();constructed[i]=0;}
}

extern "C" __global__ void stream_apply(unsigned char*arena,int*live,int*job_roots,float*score,unsigned*bits,int*meta,
 unsigned long long*policy_seeds,double*temperatures,unsigned long long*stats,int*alive,int*outcomes,double*values,
 unsigned*chosen,double*draws,int*root_boss,int*root_maximum,int*root_limits,unsigned long long*job_generation,unsigned long long*root_generation,int*root_done,int count,int base){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;int i=live[base+row],root=job_roots[i];
 device_state::require(job_generation[i]==root_generation[root]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));auto*st=stats+i*POLICY_STATS;
 int na=meta[row*4],start=meta[row*4+2];device_state::require(meta[row*4+3]&&na>0&&na<=2048&&alive[i]&&s.outcome==sts::Outcome::UNDECIDED);
 int pick=0;draws[i]=-1.;
 if(na>1){
  auto random=sts::search::noslMix(policy_seeds[i]+0x9e3779b97f4a7c15ull*st[1]);
  double u=double(random>>11)*0x1.0p-53;draws[i]=u;
  auto*x=score+start;float high=x[0];for(int a=1;a<na;++a)high=high>x[a]?high:x[a];
  double temperature=temperatures[root];device_state::require(temperature>0&&isfinite(temperature));
  double sum=0;for(int a=0;a<na;++a)sum+=exp((double(x[a])-double(high))/temperature);
  device_state::require(sum>0&&isfinite(sum));double threshold=u*sum,acc=0;pick=na-1;
  for(int a=0;a<na;++a){acc+=exp((double(x[a])-double(high))/temperature);if(threshold<acc){pick=a;break;}}
  st[10]+=na;
 }
 chosen[i]=bits[start+pick];sts::search::Action(chosen[i]).execute(s);
 ++st[0];++st[na>1?1:2];++st[7];st[8]=st[8]>static_cast<unsigned long long>(na)?st[8]:na;
 stream_search::finish(s,st,alive[i],outcomes[i],values[i],root_boss[root],root_maximum[root],root_limits[root],root_done+root);
}


extern "C" __global__ void battle_step(unsigned char*arena,int*constructed,unsigned long long*epochs,unsigned long long*revisions,
 int*ids,unsigned long long*expected_epoch,unsigned long long*expected_revision,unsigned*bits,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];
 device_state::require(constructed[i]&&epochs[i]==expected_epoch[k]&&revisions[i]==expected_revision[k]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
 device_state::require(s.outcome==sts::Outcome::UNDECIDED);
 device_state::require(battle_actions::legal(s,bits[k]));sts::search::Action(bits[k]).execute(s);++revisions[i];
}
extern "C" __global__ void battle_drop(unsigned char*arena,int*constructed,int*ids,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];
 device_state::require(constructed[i]);reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext))->~BattleContext();constructed[i]=0;
}
extern "C" __global__ void battle_copy(unsigned char*source,int*source_constructed,unsigned long long*epochs,unsigned long long*revisions,
 int*source_ids,unsigned long long*expected_epoch,unsigned long long*expected_revision,unsigned char*roots,int*root_constructed,int*root_ids,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int s=source_ids[k],r=root_ids[k];
 device_state::require(source_constructed[s]&&epochs[s]==expected_epoch[k]&&revisions[s]==expected_revision[k]);
 auto&original=*reinterpret_cast<sts::BattleContext*>(source+s*sizeof(sts::BattleContext));
 device_state::require(original.outcome==sts::Outcome::UNDECIDED);
 auto*address=roots+r*sizeof(sts::BattleContext);
 if(root_constructed[r])reinterpret_cast<sts::BattleContext*>(address)->~BattleContext();
 new(address)sts::BattleContext(original);root_constructed[r]=1;
}
extern "C" __global__ void battle_keys(unsigned char*arena,int*constructed,int*ids,unsigned long long*keys,int*counts,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];device_state::require(constructed[i]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));auto legal=device_state::legal_actions(s);
 device_state::require(legal.size()<=2048);counts[k]=legal.size();
 for(int a=0;a<int(legal.size());++a)keys[k*2048+a]=mirror_semantics::publicActionKey<mirror_semantics::StrongPublicHash>(s,legal[a]);
}

extern "C" __global__ void rules_layout(unsigned long long*x){if(threadIdx.x==0&&blockIdx.x==0){x[0]=sizeof(sts::BattleContext);x[1]=sizeof(sts::Action);}}


inline unsigned char* packed_address(unsigned char*table,int index){
 const auto address=reinterpret_cast<const unsigned long long*>(table)[index];
 device_state::require(address!=0 && address%alignof(sts::BattleContext)==0);
 return reinterpret_cast<unsigned char*>(address);
}
extern "C" __global__ void packed_layout(unsigned long long*out){
 if(blockIdx.x==0 && threadIdx.x==0){out[0]=1;out[1]=sizeof(sts::BattleContext);out[2]=sizeof(void*);}
}
extern "C" __global__ void packed_clear(unsigned char*arena,int*constructed,int*alive,int*root_ids,int count,int budget_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;
 int i=root_ids[k/budget_cap]*budget_cap+k%budget_cap;
 device_state::require(!alive[i]);
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(packed_address(arena,i))->~BattleContext();constructed[i]=0;}
}

extern "C" __global__ void packed_start(unsigned char*roots,unsigned char*arena,int*root_ids,unsigned long long*job_generation,
 unsigned long long*root_generation,unsigned long long*root_nonce,int*budgets,int*action_counts,unsigned*ordered_bits,
 unsigned long long*stats,int*constructed,int*alive,unsigned long long*world_seeds,unsigned long long*policy_seeds,unsigned*initial_bits,
 int count,int budget_cap,int action_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;
 int r=root_ids[k/budget_cap],j=k%budget_cap,i=r*budget_cap+j;
 device_state::require(!constructed[i]&&!alive[i]);
 if(j>=budgets[r])return;
 int na=action_counts[r];device_state::require(na>0&&na<=action_cap);
 auto&root=*reinterpret_cast<sts::BattleContext*>(roots+r*sizeof(sts::BattleContext));
 auto&s=*new(packed_address(arena,i))sts::BattleContext(root);constructed[i]=1;alive[i]=1;
 auto*st=stats+i*POLICY_STATS;for(int x=0;x<POLICY_STATS;++x)st[x]=0;
 job_generation[i]=root_generation[r];
 unsigned long long seed=sts::search::noslMix(sts::search::noslMix(root_nonce[r])+static_cast<unsigned long long>(j/na));
 world_seeds[i]=seed;policy_seeds[i]=sts::search::noslMix(seed^0x504149524544ull);initial_bits[i]=ordered_bits[r*action_cap+j%na];
 sts::search::smallRedeterminize(s,seed);
}

extern "C" __global__ void packed_begin(unsigned char*arena,int*root_ids,int*budgets,unsigned*initial_bits,unsigned long long*stats,
 int*alive,int*outcomes,double*values,int*root_boss,int*root_maximum,int*root_limits,int*root_done,unsigned long long*ordered_semantic,int*action_counts,unsigned*root_executed,int count,int budget_cap,int action_cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count*budget_cap)return;int r=root_ids[k/budget_cap],j=k%budget_cap;
 if(j>=budgets[r])return;int i=r*budget_cap+j;
 auto&s=*reinterpret_cast<sts::BattleContext*>(packed_address(arena,i));auto*st=stats+i*POLICY_STATS;
 const unsigned bits=battle_actions::resolve(s,initial_bits[i],ordered_semantic[r*action_cap+j%action_counts[r]]);
 root_executed[i]=bits;sts::search::Action(bits).execute(s);
 stream_search::finish(s,st,alive[i],outcomes[i],values[i],root_boss[r],root_maximum[r],root_limits[r],root_done+r);
}

extern "C" __global__ void packed_snapshot(unsigned char*arena,int*constructed,int*selected,unsigned long long*words,int*meta,unsigned*actions,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=selected[k];device_state::require(constructed[i]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(packed_address(arena,i));
 device_state::Archive<false> a{words+k*8192,8192};a.field(s);auto legal=device_state::legal_actions(s);device_state::require(legal.size()<=2048);
 auto*m=meta+k*6;m[0]=a.at;m[1]=s.actionQueue.size;m[2]=int(a.max_vector);m[3]=int(s.inputState);m[4]=int(s.outcome);m[5]=legal.size();
 for(int j=0;j<int(legal.size());++j)actions[k*2048+j]=legal[j].bits;
}

extern "C" __global__ void packed_destroy(unsigned char*arena,int*constructed,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(packed_address(arena,i))->~BattleContext();constructed[i]=0;}
}


extern "C" __global__ void packed_apply(unsigned char*arena,int*live,int*job_roots,float*score,unsigned*bits,int*meta,
 unsigned long long*policy_seeds,double*temperatures,unsigned long long*stats,int*alive,int*outcomes,double*values,
 unsigned*chosen,double*draws,int*root_boss,int*root_maximum,int*root_limits,unsigned long long*job_generation,unsigned long long*root_generation,int*root_done,int count,int base){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;int i=live[base+row],root=job_roots[i];
 device_state::require(job_generation[i]==root_generation[root]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(packed_address(arena,i));auto*st=stats+i*POLICY_STATS;
 int na=meta[row*4],start=meta[row*4+2];device_state::require(meta[row*4+3]&&na>0&&na<=2048&&alive[i]&&s.outcome==sts::Outcome::UNDECIDED);
 int pick=0;draws[i]=-1.;
 if(na>1){
  auto random=sts::search::noslMix(policy_seeds[i]+0x9e3779b97f4a7c15ull*st[1]);
  double u=double(random>>11)*0x1.0p-53;draws[i]=u;
  auto*x=score+start;float high=x[0];for(int a=1;a<na;++a)high=high>x[a]?high:x[a];
  double temperature=temperatures[root];device_state::require(temperature>0&&isfinite(temperature));
  double sum=0;for(int a=0;a<na;++a)sum+=exp((double(x[a])-double(high))/temperature);
  device_state::require(sum>0&&isfinite(sum));double threshold=u*sum,acc=0;pick=na-1;
  for(int a=0;a<na;++a){acc+=exp((double(x[a])-double(high))/temperature);if(threshold<acc){pick=a;break;}}
  st[10]+=na;
 }
 chosen[i]=bits[start+pick];sts::search::Action(chosen[i]).execute(s);
 ++st[0];++st[na>1?1:2];++st[7];st[8]=st[8]>static_cast<unsigned long long>(na)?st[8]:na;
 stream_search::finish(s,st,alive[i],outcomes[i],values[i],root_boss[root],root_maximum[root],root_limits[root],root_done+root);
}




extern "C" __global__ void packed_apply_dynamic(unsigned char*arena,int*live,int*job_roots,float*score,unsigned*bits,int*meta,
 unsigned long long*policy_seeds,double*temperatures,unsigned long long*stats,int*alive,int*outcomes,double*values,
 unsigned*chosen,double*draws,int*root_boss,int*root_maximum,int*root_limits,unsigned long long*job_generation,unsigned long long*root_generation,int*root_done,int*dispatch,int*summary,long long*control){
 if(!control[2]||control[3]||summary[2])return;const int count=dispatch[0],base=0;
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;int i=live[base+row],root=job_roots[i];
 device_state::require(job_generation[i]==root_generation[root]);
 auto&s=*reinterpret_cast<sts::BattleContext*>(packed_address(arena,i));auto*st=stats+i*POLICY_STATS;
 int na=meta[row*4],start=meta[row*4+2];device_state::require(meta[row*4+3]&&na>0&&na<=2048&&alive[i]&&s.outcome==sts::Outcome::UNDECIDED);
 int pick=0;draws[i]=-1.;
 if(na>1){
  auto random=sts::search::noslMix(policy_seeds[i]+0x9e3779b97f4a7c15ull*st[1]);
  double u=double(random>>11)*0x1.0p-53;draws[i]=u;
  auto*x=score+start;float high=x[0];for(int a=1;a<na;++a)high=high>x[a]?high:x[a];
  double temperature=temperatures[root];device_state::require(temperature>0&&isfinite(temperature));
  double sum=0;for(int a=0;a<na;++a)sum+=exp((double(x[a])-double(high))/temperature);
  device_state::require(sum>0&&isfinite(sum));double threshold=u*sum,acc=0;pick=na-1;
  for(int a=0;a<na;++a){acc+=exp((double(x[a])-double(high))/temperature);if(threshold<acc){pick=a;break;}}
  st[10]+=na;
 }
 chosen[i]=bits[start+pick];sts::search::Action(chosen[i]).execute(s);
 ++st[0];++st[na>1?1:2];++st[7];st[8]=st[8]>static_cast<unsigned long long>(na)?st[8]:na;
 stream_search::finish(s,st,alive[i],outcomes[i],values[i],root_boss[root],root_maximum[root],root_limits[root],root_done+root);
}
extern "C" __global__ void chain_step_layout(unsigned long long*x){if(!blockIdx.x&&!threadIdx.x){x[0]=1;x[1]=sizeof(sts::BattleContext);x[2]=sizeof(long long);}}
