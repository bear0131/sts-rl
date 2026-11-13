// Paged pool uses the same rules and action codec as the contiguous reference.
// Only state addressing and bounded scalar/pile frame storage differ.
namespace deck_paged_storage {
#define STS_DECK_PAGED_ADDRESSING
#define deck_apply deck_paged_apply
#define deck_apply_compact deck_paged_apply_compact
#define deck_apply_row deck_paged_apply_row
#define deck_destroy deck_paged_destroy
#define deck_first deck_paged_first
#define deck_fork deck_paged_fork
#define deck_layout deck_paged_layout
#define deck_load deck_paged_load
#define deck_load_variable deck_paged_load_variable
#define deck_observe deck_paged_observe
#define deck_observe_actions deck_paged_observe_actions
#define deck_observe_block deck_paged_observe_block
#define deck_observe_extras deck_paged_observe_extras
#define deck_observe_indexed_block deck_paged_observe_indexed_block
#define deck_observe_live_block deck_paged_observe_live_block
#define deck_observe_metadata deck_paged_observe_metadata
#define deck_observe_row deck_paged_observe_row
#define deck_observe_search_block deck_paged_observe_search_block
#define deck_snapshot deck_paged_snapshot
#define deck_snapshot_sizes deck_paged_snapshot_sizes
#define deck_snapshot_variable deck_paged_snapshot_variable
#define deck_state_address deck_paged_state_address
#include "cuda_battle_kernels.cuh"
#undef STS_DECK_PAGED_ADDRESSING
#undef deck_apply
#undef deck_apply_compact
#undef deck_apply_row
#undef deck_destroy
#undef deck_first
#undef deck_fork
#undef deck_layout
#undef deck_load
#undef deck_load_variable
#undef deck_observe
#undef deck_observe_actions
#undef deck_observe_block
#undef deck_observe_extras
#undef deck_observe_indexed_block
#undef deck_observe_live_block
#undef deck_observe_metadata
#undef deck_observe_row
#undef deck_observe_search_block
#undef deck_snapshot
#undef deck_snapshot_sizes
#undef deck_snapshot_variable
#undef deck_state_address
template<bool Clear>
__device__ __forceinline__ void deck_frame_metadata(const sts::BattleContext&b,float*upgrades,int*misc,int i,int o){
 constexpr int C=compact_leaf::kCardIds;
 if(Clear)for(int k=0;k<6*C;++k)upgrades[o*6*C+k]=0;
 if(Clear)for(int k=0;k<sts::Deck::MAX_SIZE;++k)misc[i*sts::Deck::MAX_SIZE+k]=-1;
 auto permanent=[&](const sts::CardInstance&c){if(c.getId()==sts::CardId::RITUAL_DAGGER&&c.uniqueId>=0&&c.uniqueId<sts::Deck::MAX_SIZE)misc[i*sts::Deck::MAX_SIZE+c.uniqueId]=c.specialData;};
 auto pile=[&](const auto&cards,int zone){for(const auto&c:cards){int id=int(c.getId()),up=c.getUpgradeCount();upgrades[o*6*C+zone*C+id]+=up>0;upgrades[o*6*C+(3+zone)*C+id]+=float(up);permanent(c);}};
 pile(b.cards.drawPile,0);pile(b.cards.discardPile,1);pile(b.cards.exhaustPile,2);
 for(int k=0;k<b.cards.cardsInHand;++k)permanent(b.cards.hand[k]);
}

__device__ __forceinline__ void deck_frame_block(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,int i,int o){
 int t=threadIdx.x;
 constexpr int A=sts::search::kAutoregressiveSize,C=compact_leaf::kCardIds,D=sts::Deck::MAX_SIZE;
 for(int k=t;k<compact_leaf::kCatSize;k+=blockDim.x)cat[i*compact_leaf::kCatSize+k]=0;
 for(int k=t;k<compact_leaf::kFloatSize;k+=blockDim.x)scalars[o*compact_leaf::kFloatSize+k]=0;
 for(int k=t;k<A;k+=blockDim.x){bits[i*A+k]=0;mask[i*A+k]=0;}
 for(int k=t;k<6*C;k+=blockDim.x)upgrades[o*6*C+k]=0;
 for(int k=t;k<D;k+=blockDim.x)misc[i*D+k]=-1;
 __syncthreads();
 if(t<=32){
  device_state::require(constructed[i]);
  const auto&b=*reinterpret_cast<const sts::BattleContext*>(deck_paged_state_address(arena,i));
  DeckOutputView row{cat+i*compact_leaf::kCatSize,scalars+o*compact_leaf::kFloatSize};
  if(t<32)compact_leaf::packInto(b,row,t,32);
  else {deck_paged_observe_actions<false>(b,prefix,bits,mask,status,i);deck_frame_metadata<false>(b,upgrades,misc,i,o);}
 }
}

extern "C" __global__ void deck_paged_frame_indexed(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,long long*slots,int*indices,int*count,long long*offset,int n){
 int row=blockIdx.x;if(row>=n||row>=*count)return;
 if(threadIdx.x==0)slots[int(*offset)+indices[row]]=row;
 deck_frame_block(arena,constructed,prefix,cat,scalars,bits,mask,status,upgrades,misc,int(*offset)+indices[row],row);
}

extern "C" __global__ void deck_paged_frame_search(unsigned char*arena,int*constructed,int*prefix,int*cat,float*scalars,unsigned*bits,int*mask,int*status,float*upgrades,int*misc,long long*slots,int*indices,int*count,long long*offset,int*depth,int*limit,int n){
 int row=blockIdx.x;if(row>=n||row>=*count)return;
 int logical=indices[row],i=int(*offset)+logical,t=threadIdx.x;
 if(t==0)slots[i]=row;
 constexpr int A=sts::search::kAutoregressiveSize,C=compact_leaf::kCardIds,D=sts::Deck::MAX_SIZE;
 __shared__ int choices;
 for(int k=t;k<A;k+=blockDim.x){bits[i*A+k]=0;mask[i*A+k]=0;}
 __syncthreads();
 device_state::require(constructed[i]);
 const auto&b=*reinterpret_cast<const sts::BattleContext*>(deck_paged_state_address(arena,i));
 if(t==32)choices=deck_paged_observe_actions<false>(b,prefix,bits,mask,status,i);
 __syncthreads();
 if(b.outcome!=sts::Outcome::UNDECIDED||(choices==1&&depth[logical]<*limit))return;
 for(int k=t;k<compact_leaf::kCatSize;k+=blockDim.x)cat[i*compact_leaf::kCatSize+k]=0;
 for(int k=t;k<compact_leaf::kFloatSize;k+=blockDim.x)scalars[row*compact_leaf::kFloatSize+k]=0;
 for(int k=t;k<6*C;k+=blockDim.x)upgrades[row*6*C+k]=0;
 for(int k=t;k<D;k+=blockDim.x)misc[i*D+k]=-1;
 __syncthreads();
 DeckOutputView output{cat+i*compact_leaf::kCatSize,scalars+row*compact_leaf::kFloatSize};
 if(t<32)compact_leaf::packInto(b,output,t,32);
 else if(t==32)deck_frame_metadata<false>(b,upgrades,misc,i,row);
}

extern "C" __global__ void deck_paged_load_indexed(unsigned char*arena,int*constructed,unsigned long long*words,int*lengths,int*stride,int*indices,int n){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;int i=indices[j];
 auto*address=deck_paged_state_address(arena,i);
 if(constructed[i])reinterpret_cast<sts::BattleContext*>(address)->~BattleContext();
 auto&b=*new(address)sts::BattleContext{};
 device_state::Archive<true>a{words+j*(*stride),lengths[j]};a.field(b);device_state::require(a.at==lengths[j]);
 device_state::require(b.actionQueue.size==0);constructed[i]=1;
}
extern "C" __global__ void deck_paged_destroy_indexed(unsigned char*arena,int*constructed,int*indices,int n){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;int i=indices[j];
 if(constructed[i]){reinterpret_cast<sts::BattleContext*>(deck_paged_state_address(arena,i))->~BattleContext();constructed[i]=0;}
}
}
