// Current model input consumer for the original prepare/features/pack pipeline.
#pragma once
#include "deck_schema.h"
#include "compact-observation-device.h"

namespace deck_input {
using namespace sts;
constexpr int TOKENS=121;
constexpr int RAW=compact_leaf::kFloatSize;
constexpr int CAT=compact_leaf::kCatSize;
constexpr int PS=compact_leaf::kPlayerStatus;
constexpr int MS=compact_leaf::kMonsterStatus;
constexpr int HAND=6+PS+24+1+30+5*MS;
constexpr int PICK=HAND+20+3*compact_leaf::kCardIds;

inline float scaled_cost(float cost){return (cost==-3.f?-2.f:cost)/ENERGY_SCALE;}
inline int physical(const int*chunks,int row,int slot){return chunks[row*16+slot/64]+slot%64;}

template<class Actions> inline void prepare(const BattleContext&b,const Actions&actions,int i,int prefix,const unsigned long long*w,
 unsigned*bits,int*keys,int*source_rows,int*targets,int*ids,int*zones,int*monsters,
 int*meta,int*chunks,int*allocator,int action_capacity,int entity_capacity,
 float*raw,int*cat,int*legal,float*selected,int*remaining){
 auto packed=compact_leaf::pack<false>(b);
 for(int j=0;j<HAND+20;++j)raw[i*RAW+j]=packed.scalars[j];
 for(int j=PICK;j<RAW;++j)raw[i*RAW+j]=packed.scalars[j];
 for(int j=0;j<CAT;++j)cat[i*CAT+j]=packed.categorical[j];
 const int na=actions.size();device_state::require(na>0&&na<=198);
 unsigned by_id[198]{};bool mask[198]{};
 for(int k=0;k<na;++k){const auto a=actions[k];
  int id=sts::search::jointActionId(a,b)-sts::search::kJointEnd;
  device_state::require(id>=0&&id<198&&!mask[id]);mask[id]=true;by_id[id]=a.bits;
 }
 int start=atomicAdd(allocator,na);meta[i*4]=na;meta[i*4+1]=0;meta[i*4+2]=start;meta[i*4+3]=0;
 atomicMax(allocator+4,na);
 if(start>action_capacity-na){atomicOr(allocator+2,1);return;}
 int needed=0;
 for(int j=0;j<10;++j)needed=max(needed,int(packed.scalars[HAND+2*j])+1);
 if(needed>upgrade_levels(w)[0]){
  // Bit 2 already prevents packing and apply; bit 64 identifies table growth.
  atomicMax(allocator+3,needed);atomicOr(allocator+2,66);return;
 }
 int a=0;
 for(int id=0;id<198;++id){
  legal[i*198+id]=mask[id];if(!mask[id])continue;
  int at=start+a++;bits[at]=by_id[id];
  for(int k=0;k<10;++k){keys[at*10+k]=k==0?id:-1;source_rows[at*10+k]=-1;}
  int source=-1,target=-1;
  if(id>=1&&id<=90){int g=(id-1)/6,t=(id-1)%6;source=g<10?1+g:16+g-10;if(t<5)target=11+t;}
  else if(id>=91&&id<187)source=25+id-91;
  else if(id>=187&&id<197)source=1+id-187;
  source_rows[at*10]=source;targets[at]=target;
 }
 for(int j=0;j<10;++j)selected[i*10+j]=(prefix>>j)&1;
 remaining[i]=max(0,int(packed.scalars[PICK])-__popc(unsigned(prefix)));
 bool queried[TOKENS]{},present[TOKENS]{};queried[0]=true;
 for(int j=0;j<TOKENS;++j){
  present[j]=true;
  if(j>=1&&j<=10)present[j]=packed.categorical[j-1]!=0;
  else if(j>=11&&j<=15)present[j]=packed.categorical[j-1]!=0;
  else if(j>=16&&j<=20){int p=packed.categorical[j+9];present[j]=p!=0&&p!=1;}
  else if(j>=25)present[j]=packed.categorical[127+j-25]>0;
 }
 for(int j=0;j<na;++j){int src=source_rows[(start+j)*10],tgt=targets[start+j];if(src>=0)queried[src]=true;if(tgt>=0)queried[tgt]=true;}
 for(int j=0;j<10;++j)if((prefix>>j)&1)queried[1+j]=true;
 int ne=0;for(int j=0;j<TOKENS;++j)ne+=present[j]||queried[j];
 meta[i*4+1]=ne;atomicMax(allocator+3,ne);
 for(int ch=0;ch<(ne+63)/64;++ch){
  int at=atomicAdd(allocator+1,64);chunks[i*16+ch]=at;
  if(at>entity_capacity-64){atomicOr(allocator+2,2);return;}
 }
 // IDs carry canonical logical slots; zones carry key presence. Queries are
 // derived independently, including legal PAD candidates and prefix members.
 int row=0;
 for(int j=0;j<TOKENS;++j)if(present[j]||queried[j]){
  int at=physical(chunks,i,row++);
  ids[at]=j;zones[at]=present[j];monsters[at]=queried[j];
 }
 meta[i*4+3]=1;
}

inline float relic_value(const unsigned long long*w,int id,float initial,const float*s){
 for(int k=0;k<LIVE_RELIC_COUNT;++k)if(live_relic_ids(w)[k]==id){
  int column=live_relic_counter_columns(w)[k],mod=live_relic_counter_moduli(w)[k],stop=live_relic_counter_stop_turns(w)[k];
  float v=column==-5?s[RAW-1]:column==-4?s[RAW-2]:column==-3?s[4]+1.f:column<0?s[PICK+197-column]:s[6+PS+column];
  if(mod)v=fmodf(v,float(mod));if(stop&&v>=stop)v=0.f;
  return column==-1||column==-2?v:v/COUNTER_SCALE;
 }
 return initial;
}

inline float player_feature(const unsigned long long*w,const float*s,const float*root,int prefix,int k){
 if(k<6){
  if(k<2)return fminf(s[k],HP_SCALE)/HP_SCALE;
  if(k==2)return s[2]/BLOCK_SCALE;if(k==3)return s[3]/ENERGY_SCALE;
  if(k==4)return (s[4]+1.f)/TURN_SCALE;return s[1]?s[0]/s[1]:0.f;
 }
 if(k<SCALAR_COMBAT_COUNTERS){int j=player_status_columns(w)[k-SCALAR_PLAYER_BUFFS];return j<0?0.f:s[6+j]/BUFF_SCALE;}
 if(k<SCALAR_POST_ENEMY_TURN_BOUNDARY){int j=k-SCALAR_COMBAT_COUNTERS;float v=s[6+PS+j];return binary_counters(w)[j]?float(v!=0):v/COUNTER_SCALE;}
 if(k==SCALAR_POST_ENEMY_TURN_BOUNDARY)return s[6+PS+24];
 if(k==SCALAR_ASCENSION)return root[67];
 if(k==SCALAR_GOLD)return s[5]/GOLD_SCALE;
 if(k==SCALAR_PICK_REMAINING)return float(max(0,int(s[PICK])-__popc(unsigned(prefix))))/PICK_SCALE;
 if(k==SCALAR_CAN_PICK_ZERO)return s[PICK+1];if(k==SCALAR_CAN_PICK_ANY)return s[PICK+2];
 if(k>=SCALAR_RUN_KEYS&&k<SCALAR_RUN_KEYS+3)return root[64+k-SCALAR_RUN_KEYS];
 if(k>=SCALAR_COMBAT_EXIT_ANCHOR&&k<SCALAR_EXTRA)return 0.f;
 int j=k-SCALAR_EXTRA;
 if(j==0)return fminf(s[1],HP_SCALE)/HP_SCALE-fminf(s[0],HP_SCALE)/HP_SCALE;
 if(j==1)return log1pf(fmaxf(0.f,s[4]+1.f))/4.f;
 float hp=s[1]?s[0]/s[1]:0.f;return 1.f/(1.f+expf(-12.f*(hp-.25f*float(j-1))));
}

inline float pile_value(const BattleContext&b,const unsigned long long*w,int slot,int dim){
 const auto&pile=slot==22?b.cards.drawPile:slot==23?b.cards.discardPile:b.cards.exhaustPile;
 float base=0.f,upgrade=0.f,magnitude=0.f;int count=0;
 for(const auto&item:pile){int id=card(w)[int(item.id)],up=item.getUpgradeCount();
  base+=pile_base(w)[id*64+dim];if(up>0)upgrade+=pile_upgrade(w)[id*64+dim];magnitude+=float(up)/UPGRADE_SCALE;count+=id!=0;
 }
 return ((base+upgrade)+magnitude*pile_magnitude(w)[dim])/float(max(1,count))+log1pf(float(count))*pile_count(w)[dim]+pile_bias(w)[dim];
}
struct PackedPiles {const float*counts;const float*upgraded;const float*sums;};
inline float pile_value(const PackedPiles&b,const unsigned long long*w,int slot,int dim){
 float base=0.f,upgrade=0.f,magnitude=0.f;int count=0;const int offset=(slot-22)*compact_leaf::kCardIds;
 for(int native=0;native<compact_leaf::kCardIds;++native){const int n=int(b.counts[offset+native]);if(!n)continue;const int id=card(w)[native];
  base+=float(n)*pile_base(w)[id*64+dim];upgrade+=b.upgraded[offset+native]*pile_upgrade(w)[id*64+dim];magnitude+=b.sums[offset+native]/UPGRADE_SCALE;count+=id?n:0;
 }
 return ((base+upgrade)+magnitude*pile_magnitude(w)[dim])/float(max(1,count))+log1pf(float(count))*pile_count(w)[dim]+pile_bias(w)[dim];
}
template<class Piles> inline float token(const Piles&b,const unsigned long long*w,const float*s,const int*c,
 const float*root,const int*relic_ids,const float*relic_counts,int prefix,int slot,int dim){
 int role=slot==0?0:slot<=10?1:slot<=15?2:slot<=20?3:slot==21?4:slot<=24?slot-17:8;
 float value=roles(w)[role*64+dim];
 if(slot==0){
  float p=player_bias(w)[dim];
  for(int k=0;k<SCALAR_WIDTH;++k)p+=player_weight(w)[dim*SCALAR_WIDTH+k]*player_feature(w,s,root,prefix,k);
  int turn=std::clamp(int(s[4]+1.f),0,31),reason_id=select_context(w)[c[30]];
  return value+((p+turn_bucket(w)[turn*64+dim])+reason(w)[reason_id*64+dim])+root[dim];
 }
 if((slot>=1&&slot<=10)||slot>=25){
  bool hand=slot<=10;int j=hand?slot-1:slot-25;
  int id=card(w)[c[(hand?0:31)+j]];float up=s[hand?HAND+2*j:PICK+4+j]/UPGRADE_SCALE;
  float card_value=card_token(w)[(id*2+int(up>0))*64+dim]+up*card_magnitude(w)[dim];
  float cost=scaled_cost(s[hand?HAND+2*j+1:PICK+100+j]);
  if(hand){
   int level=int(s[HAND+2*j]),levels=upgrade_levels(w)[0];device_state::require(level>=0&&level<levels);
   const float*v=hand_weight(w)+dim*4;
   float extra=hand_bias(w)[dim]+cost*v[0]+damage(w)[id*levels+level]/DMG_SCALE*v[1]+block(w)[id*levels+level]/BLOCK_SCALE*v[2]+float((prefix>>j)&1)*v[3];
   return value+(card_value+extra);
  }
  float extra=choice_bias(w)[dim]+cost*choice_weight(w)[dim*2]+float(c[127+j]>0&&id==0)*choice_weight(w)[dim*2+1];
  return value+((card_value+extra)+zone(w)[c[223]*64+dim]);
 }
 if(slot<=15){
  int j=slot-11,mi=monster(w)[c[10+j]],move_id=move(w)[c[15+j]],intent=c[20+j];
  bool explosion=move_id==EXPLODER_MOVE;if(explosion)intent=0;
  float total=monster_bias(w)[dim]+monster_id_token(w)[mi*64+dim]+monster_move_token(w)[move_id*64+dim]+monster_intent_token(w)[intent*64+dim];
  const float*m=s+6+PS+24+1+j*6;const float*weight=monster_weight(w)+dim*(6+MONSTER_BUFF_WIDTH);
  const float numeric[6]={m[0],fminf(m[1],HP_SCALE)/HP_SCALE,m[2]?m[1]/m[2]:0.f,m[3]/BLOCK_SCALE,explosion?0.f:m[4]/DMG_SCALE,explosion?0.f:m[5]/HITS_SCALE};
  for(int k=0;k<6;++k)total+=numeric[k]*weight[k];
  for(int k=0;k<MONSTER_BUFF_WIDTH;++k){int at=monster_status_columns(w)[k];float v=(at<0||m[0]<=0)?0.f:s[6+PS+24+1+30+j*MS+at]/BUFF_SCALE;total+=v*weight[6+k];}
  return value+total;
 }
 if(slot<=20)return value+potion_token(w)[potion(w)[c[25+slot-16]]*64+dim];
 if(slot==21){
  float total=0.f;
  for(int j=0;j<RELIC_WIDTH;++j){int id=relic_ids[j];if(!id)continue;
   float x=relic_value(w,id,relic_counts[j],s),count=rintf(x*COUNTER_SCALE);
   const float*g=relic_counter(w)+id*640+dim;float term=g[0]*x+g[64]*copysignf(log1pf(fabsf(x)),x);
   constexpr int periods[8]={2,3,4,5,6,7,8,10};
   for(int k=0;k<8;++k){float period=float(periods[k]);float rem=count-floorf(count/period)*period;term+=g[(k+2)*64]*rem/period;}
   total+=relic_identity(w)[id*64+dim]+term;
  }
  return value+total;
 }
 return value+pile_value(b,w,slot,dim);
}
}
