// Adapter only: reuse the accepted resident simulator's generated rules.
extern "C" __global__ void deck_layout(long long*out){
 if(threadIdx.x==0&&blockIdx.x==0){out[0]=sizeof(sts::BattleContext);out[1]=compact_leaf::kCatSize;out[2]=compact_leaf::kFloatSize;out[3]=sts::search::kAutoregressiveSize;out[4]=compact_leaf::kCardIds;out[5]=sts::Deck::MAX_SIZE;out[6]=1;out[7]=8192;}
}
extern "C" __global__ void deck_load(unsigned char*arena,int*constructed,unsigned long long*words,int*lengths,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 auto*address=arena+i*sizeof(sts::BattleContext);
 if(constructed[i])reinterpret_cast<sts::BattleContext*>(address)->~BattleContext();
 auto&s=*new(address)sts::BattleContext{};
 device_state::Archive<true>a{words+i*8192,lengths[i]};a.field(s);device_state::require(a.at==lengths[i]);
 device_state::require(s.actionQueue.size==0);constructed[i]=1;
}
template<bool Clear>
__device__ __forceinline__ void deck_observe_extras(const sts::BattleContext&b,int*prefix,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,int i){
 constexpr int A=sts::search::kAutoregressiveSize,C=compact_leaf::kCardIds;
 if(Clear)for(int k=0;k<A;++k){bits[i*A+k]=0;mask[i*A+k]=0;}
 status[i]=int(b.outcome);
 if(b.outcome==sts::Outcome::UNDECIDED){
  const auto actions=sts::search::autoregressiveActions(b,prefix[i]);
  for(auto a:actions){int id=sts::search::jointActionId(a,b)-sts::search::kJointEnd;device_state::require(id>=0&&id<A);mask[i*A+id]=1;bits[i*A+id]=a.bits;}
 }
 if(Clear)for(int k=0;k<6*C;++k)upgrades[i*6*C+k]=0;
 if(Clear)for(int k=0;k<sts::Deck::MAX_SIZE;++k)misc[i*sts::Deck::MAX_SIZE+k]=-1;
 auto permanent=[&](const sts::CardInstance&c){if(c.getId()==sts::CardId::RITUAL_DAGGER&&c.uniqueId>=0&&c.uniqueId<sts::Deck::MAX_SIZE)misc[i*sts::Deck::MAX_SIZE+c.uniqueId]=c.specialData;};
 auto pile=[&](const auto&cards,int zone){for(const auto&c:cards){int id=int(c.getId()),up=c.getUpgradeCount();upgrades[i*6*C+zone*C+id]+=up>0;upgrades[i*6*C+(3+zone)*C+id]+=float(up);permanent(c);}};
 pile(b.cards.drawPile,0);pile(b.cards.discardPile,1);pile(b.cards.exhaustPile,2);
 for(int k=0;k<b.cards.cardsInHand;++k)permanent(b.cards.hand[k]);
}
__device__ __forceinline__ void deck_observe_row(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,int i){
 device_state::require(constructed[i]);
 const auto&b=*reinterpret_cast<const sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
 auto row=compact_leaf::pack(b);
 for(int k=0;k<compact_leaf::kCatSize;++k)cat[i*compact_leaf::kCatSize+k]=row.categorical[k];
 for(int k=0;k<compact_leaf::kFloatSize;++k)scalars[i*compact_leaf::kFloatSize+k]=row.scalars[k];
 deck_observe_extras<true>(b,prefix,bits,mask,status,upgrades,misc,i);
}
extern "C" __global__ void deck_observe(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 deck_observe_row(arena,constructed,prefix,cat,scalars,bits,mask,status,upgrades,misc,i);
}
// Port the old normal_features split: one warp shares the canonical row
// codec, the other owns legal-action assembly and pile metadata.
struct DeckOutputView {int*categorical;float*scalars;int selectionCount=0;};
extern "C" __global__ void deck_observe_live_block(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,int*live,long long*offset,int n){
 int lane=blockIdx.x;if(lane>=n||!live[lane])return;
 int i=int(*offset)+lane,t=threadIdx.x;
 constexpr int A=sts::search::kAutoregressiveSize,C=compact_leaf::kCardIds,D=sts::Deck::MAX_SIZE;
 for(int k=t;k<compact_leaf::kCatSize;k+=blockDim.x)cat[i*compact_leaf::kCatSize+k]=0;
 for(int k=t;k<compact_leaf::kFloatSize;k+=blockDim.x)scalars[i*compact_leaf::kFloatSize+k]=0;
 for(int k=t;k<A;k+=blockDim.x){bits[i*A+k]=0;mask[i*A+k]=0;}
 for(int k=t;k<6*C;k+=blockDim.x)upgrades[i*6*C+k]=0;
 for(int k=t;k<D;k+=blockDim.x)misc[i*D+k]=-1;
 __syncthreads();
 if(t<=32){
  device_state::require(constructed[i]);
  const auto&b=*reinterpret_cast<const sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
  DeckOutputView row{cat+i*compact_leaf::kCatSize,scalars+i*compact_leaf::kFloatSize};
  if(t<32)compact_leaf::packInto(b,row,t,32);
  else deck_observe_extras<false>(b,prefix,bits,mask,status,upgrades,misc,i);
 }
}
extern "C" __global__ void deck_apply(unsigned char*arena,int*constructed,int*prefix,int*selected,unsigned*bits,int*mask,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||selected[i]<0)return;
 device_state::require(constructed[i]);auto&b=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
 int id=selected[i];device_state::require(id<sts::search::kAutoregressiveSize&&mask[i*sts::search::kAutoregressiveSize+id]&&b.outcome==sts::Outcome::UNDECIDED);
 sts::search::executeAutoregressiveAction(sts::search::Action(bits[i*sts::search::kAutoregressiveSize+id]),b,prefix[i]);
}
extern "C" __global__ void deck_destroy(unsigned char*arena,int*constructed,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext))->~BattleContext();constructed[i]=0;}
}

extern "C" __global__ void deck_snapshot(unsigned char*arena,int*constructed,unsigned long long*words,int*meta,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 device_state::require(constructed[i]);
 auto&b=*reinterpret_cast<sts::BattleContext*>(arena+i*sizeof(sts::BattleContext));
 device_state::Archive<false>a{words+i*8192,8192};a.field(b);
 meta[i*5]=a.at;meta[i*5+1]=b.actionQueue.size;meta[i*5+2]=int(a.max_vector);meta[i*5+3]=int(b.inputState);meta[i*5+4]=int(b.outcome);
}
extern "C" __global__ void deck_fork(unsigned char*arena,int*constructed,int*prefix,int*source,int*dest,unsigned long long*worlds,int count){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
 int s=source[j],d=dest[j];device_state::require(s!=d&&constructed[s]);
 auto*target=reinterpret_cast<sts::BattleContext*>(arena+d*sizeof(sts::BattleContext));
 if(constructed[d])target->~BattleContext();
 new(target)sts::BattleContext(*reinterpret_cast<const sts::BattleContext*>(arena+s*sizeof(sts::BattleContext)));
 constructed[d]=1;prefix[d]=prefix[s];
 sts::search::noslRedeterminize(*target,worlds[j]);
}
extern "C" __global__ void deck_first(unsigned char*arena,int*constructed,int*prefix,int*source,int*dest,unsigned*bits,int*draw,unsigned*available,int*mask,int*selected,int count){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
 int s=source[j],d=dest[j];device_state::require(s!=d&&constructed[s]&&constructed[d]);
 auto&target=*reinterpret_cast<sts::BattleContext*>(arena+d*sizeof(sts::BattleContext));
 const auto&root=*reinterpret_cast<const sts::BattleContext*>(arena+s*sizeof(sts::BattleContext));
 unsigned b=bits[j];if((b>>29)==2&&draw[j])b=sts::search::Action::remapDrawSelection(target,root,b);
 int found=-1;for(int k=0;k<sts::search::kAutoregressiveSize;++k)if(mask[d*sts::search::kAutoregressiveSize+k]&&available[d*sts::search::kAutoregressiveSize+k]==b)found=k;
 device_state::require(found>=0);selected[d]=found;
}
