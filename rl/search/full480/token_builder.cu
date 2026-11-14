#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ void require(bool good){if(!good)asm volatile("trap;");}
__device__ __forceinline__ uint32_t h32(uint32_t x){
 x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;return x^(x>>16);
}
__device__ __forceinline__ int primary(int id,int zone,int mon,const float*x){
 if(zone==2)return 18+int(rintf(x[8]*40.f))-1;
 if(zone==13)return 13+int(rintf(x[0]*5.f))-1;
 if(zone==14&&id/512==5)return 28+mon;
 return -1;
}
__device__ __forceinline__ int entity_slot(int id,int zone,int mon,const float*x){
 constexpr int slots[15]={1,2,18,6,7,8,9,10,12,11,5,3,4,13,28};
 require(zone>=0&&zone<15&&id>=0&&id<6144&&mon>=-1&&mon<5);
 if(zone==1&&id/512==1)return 1;
 if(zone==2||zone==13)return primary(id,zone,mon,x);
 if(zone==14){require(mon>=0);return 28+mon;}
 return slots[zone];
}
__device__ __forceinline__ int physical(const int*chunks,int sid,int r){
 return chunks[sid*16+r/64]+r%64;
}
#define ARGS const int*worlds,const int*ids,const int*zones,const int*monsters,const float*numbers,const int*meta,const int*chunks, \
 const int*inv_ids,const int*inv_zones,const int*inv_monsters,const float*inv_numbers,const int*inv_counts,const int*inv_slots, \
 const int*source_rows,const int*target_slots,int*canonical_sources,int*canonical_targets,int*query_map, \
 float*features,int64_t*symbols,int64_t*slots,bool*present,int64_t*queries,int*blocks1,int*blocks2,int*owners, \
 int64_t*starts,int64_t*lens,float*scores,int*allocator,const int*dispatch,int*totals,int*queues,int*queue_counts, \
 int ROOTS,int I,int T,int F,int CT,int CQ,int CB1,int CB2
#define FORWARD worlds,ids,zones,monsters,numbers,meta,chunks,inv_ids,inv_zones,inv_monsters,inv_numbers,inv_counts,inv_slots, \
 source_rows,target_slots,canonical_sources,canonical_targets,query_map,features,symbols,slots,present,queries,blocks1,blocks2,owners, \
 starts,lens,scores,allocator,dispatch,totals,queues,queue_counts,ROOTS,I,T,F,CT,CQ,CB1,CB2

template<int CAP,int STAGE> __device__ void pack(ARGS){
 using ExtraScan=cub::BlockScan<int,256>;
 using TokenScan=cub::BlockScan<unsigned long long,256>;
 __shared__ union ScanSpace{typename ExtraScan::TempStorage extra;typename TokenScan::TempStorage token;} scan;
 __shared__ int sy[161],ct[161],queried[161],rank[161],order[161],qrank[161],last[33],source_map[1024];
 __shared__ int has_extra,nextra,nt,nq,ts,qs,bb1,bb2,overflow;
 extern __shared__ float feature[];
 int tid=threadIdx.x;
 if(allocator[2]&3)return;
 int count=STAGE==0?dispatch[0]:queue_counts[STAGE-1];
 for(int task=blockIdx.x;task<count;task+=gridDim.x){
  int sid=STAGE==0?task:queues[(STAGE-1)*F+task];
  int na=meta[sid*4],ne=meta[sid*4+1],begin=meta[sid*4+2],root=worlds[sid]%ROOTS,ni=inv_counts[root];
  require(meta[sid*4+3]&&na>0&&na<=2048&&ne<=1024&&T==161);
#ifdef STS_FULL480_DECK
  ni=0;
#endif
  for(int a=tid;a<na;a+=blockDim.x)owners[begin+a]=sid;
  if(na==1&&meta[sid*4+3]!=2){
   if(tid<4){lens[sid*4+tid]=0;starts[sid*4+tid]=0;}
   if(tid==0)scores[begin]=0.f;
   __syncthreads();continue;
  }
  if(tid<161){sy[tid]=0;ct[tid]=0;queried[tid]=(tid==0);}
  if(tid<33)last[tid]=-1;
  if(tid==0){has_extra=0;nextra=0;}
  __syncthreads();
#ifdef STS_FULL480_DECK
  for(int r=tid;r<ne;r+=blockDim.x){
   int at=physical(chunks,sid,r),sl=ids[at];require(sl>=0&&sl<121);
   ct[sl]=zones[at];queried[sl]=monsters[at];
  }
  __syncthreads();
#else
  for(int r=tid;r<ne+ni;r+=blockDim.x){
   bool own=r<ne;int at=own?physical(chunks,sid,r):root*I+r-ne;
   int eid=own?ids[at]:inv_ids[at],z=own?zones[at]:inv_zones[at],mon=own?monsters[at]:inv_monsters[at];
   const float*x=own?numbers+at*16:inv_numbers+at*16;
   int sl=own?entity_slot(eid,z,mon,x):inv_slots[at],pr=primary(eid,z,mon,x);
   require(sl>=0&&sl<33);atomicAdd(ct+sl,1);
   if(pr>=0)atomicMax(last+sl,r);
   if(own)source_map[r]=pr;
  }
  __syncthreads();
  if(tid<33&&last[tid]>=0){
   int r=last[tid],at=r<ne?physical(chunks,sid,r):root*I+r-ne;
   sy[tid]=(r<ne?ids[at]:inv_ids[at])+1;
  }
  for(int k=tid;k<na*10;k+=blockDim.x){
   int r=source_rows[begin*10+k];require(r>=-1&&r<ne);
   if(r>=0&&atomicCAS(source_map+r,-1,-2)==-1)atomicExch(&has_extra,1);
  }
  __syncthreads();
  if(has_extra){
   int in[4],out[4],aggregate;
   #pragma unroll
   for(int k=0;k<4;++k){int r=tid*4+k;in[k]=(r<ne&&source_map[r]==-2);}
   ExtraScan(scan.extra).ExclusiveSum(in,out,aggregate);
   if(tid==0){nextra=aggregate;require(nextra<=128);}
   #pragma unroll
   for(int k=0;k<4;++k)if(in[k]){
    int r=tid*4+k,sl=33+out[k];source_map[r]=sl;ct[sl]=1;sy[sl]=ids[physical(chunks,sid,r)]+1;
   }
   __syncthreads();
  }
  for(int k=tid;k<na*10;k+=blockDim.x){
   int r=source_rows[begin*10+k],sl=r>=0?source_map[r]:-1;
   canonical_sources[begin*10+k]=sl;
   if(sl>=0){require(sl<161);atomicExch(queried+sl,1);}
  }
  for(int a=tid;a<na;a+=blockDim.x){
   int raw=target_slots[begin+a];require(raw>=-1&&raw<5);int sl=raw>=0?28+raw:-1;
   canonical_targets[begin+a]=sl;if(sl>=0)atomicExch(queried+sl,1);
  }
  __syncthreads();
#endif
  bool used=tid<161&&(tid==0||ct[tid]>0||queried[tid]);
  bool query=tid<161&&queried[tid];
  unsigned long long prefix,aggregate,flags=static_cast<unsigned long long>(used)+(static_cast<unsigned long long>(query)<<32);
  TokenScan(scan.token).ExclusiveSum(flags,prefix,aggregate);
  if(tid==0){nt=int(aggregate&0xffffffffu);nq=int(aggregate>>32);require(nt>=nq&&nq>=1&&nt<=161);}
  if(tid<161){
   rank[tid]=used?int(prefix&0xffffffffu):-1;qrank[tid]=query?int(prefix>>32):-1;
   query_map[sid*T+tid]=qrank[tid];
   if(used)order[rank[tid]]=tid;
  }
  __syncthreads();
  if(nt>CAP){
   require(STAGE<2);
   if(tid==0){int at=atomicAdd(queue_counts+STAGE,1);require(at<F);queues[STAGE*F+at]=sid;}
   __syncthreads();continue;
  }
  int n1=(nt+15)/16,n2=(nq+15)/16;
  if(tid==0){
   ts=atomicAdd(totals,nt);qs=atomicAdd(totals+1,nq);bb1=atomicAdd(totals+2,n1);bb2=atomicAdd(totals+3,n2);
   atomicAdd(totals+4,1);
   overflow=(ts>CT-nt?4:0)|(qs>CQ-nq?8:0)|(bb1>CB1-n1?16:0)|(bb2>CB2-n2?32:0);
   if(overflow)atomicOr(allocator+2,overflow);
   starts[sid*4]=ts;starts[sid*4+1]=qs;starts[sid*4+2]=bb1;starts[sid*4+3]=bb2;
   lens[sid*4]=nt;lens[sid*4+1]=nq;lens[sid*4+2]=n1;lens[sid*4+3]=n2;
  }
  __syncthreads();
  if(overflow){__syncthreads();continue;}
  for(int k=tid;k<nt*128;k+=blockDim.x)feature[k]=0.f;
  __syncthreads();
#ifdef STS_FULL480_DECK
  for(int k=tid;k<ne*64;k+=blockDim.x){
   int at=physical(chunks,sid,k/64),sl=rank[ids[at]];require(sl>=0);
   feature[sl*128+k%64]=numbers[at*64+k%64];
  }
#else
  int j=tid%16;
  for(int r=tid/16;r<ne+ni;r+=blockDim.x/16){
   bool own=r<ne;int at=own?physical(chunks,sid,r):root*I+r-ne;
   int eid=own?ids[at]:inv_ids[at],z=own?zones[at]:inv_zones[at],mon=own?monsters[at]:inv_monsters[at];
   const float*x=own?numbers+at*16:inv_numbers+at*16;
   int sl=own?entity_slot(eid,z,mon,x):inv_slots[at],extra=own&&source_map[r]>=33?source_map[r]:-1;
   sl=rank[sl];if(extra>=0)extra=rank[extra];require(sl>=0);
   uint32_t identity=(uint32_t(eid)*15u+uint32_t(z))*17u,kind=((uint32_t(eid/512)+6144u)*15u+uint32_t(z))*17u;
   if(j==0){uint32_t h=h32(identity);float v=(h>>31)?-1.f:1.f;atomicAdd(feature+sl*128+h%127u,v);if(extra>=0)atomicAdd(feature+extra*128+h%127u,v);}
   if(x[j]!=0.f){
    uint32_t hi=h32(identity+uint32_t(j)+1u),hk=h32(kind+uint32_t(j)+1u);
    float vi=(hi>>31)?-x[j]:x[j],vk=(hk>>31)?-x[j]:x[j];
    atomicAdd(feature+sl*128+hi%127u,vi);atomicAdd(feature+sl*128+hk%127u,vk);
    if(extra>=0){atomicAdd(feature+extra*128+hi%127u,vi);atomicAdd(feature+extra*128+hk%127u,vk);}
   }
  }
#endif
  __syncthreads();
  for(int k=tid;k<nt*128;k+=blockDim.x){
   int c=k/128,d=k%128,old=order[c],cnt=ct[old];float cf=float(cnt);
   #ifdef STS_FULL480_DECK
   features[(int64_t(ts)+c)*128+d]=feature[k];
#else
   features[(int64_t(ts)+c)*128+d]=d==127?logf(cf+1.f):feature[k]/sqrtf(fmaxf(cf,1.f));
#endif
  }
  for(int c=tid;c<nt;c+=blockDim.x){
   int old=order[c];symbols[ts+c]=sy[old];slots[ts+c]=old<33?old:33;present[ts+c]=ct[old]>0||old==0;
   if(qrank[old]>=0)queries[qs+qrank[old]]=ts+c;
  }
  for(int b=tid;b<n1;b+=blockDim.x){blocks1[(bb1+b)*2]=sid;blocks1[(bb1+b)*2+1]=b*16;}
  for(int b=tid;b<n2;b+=blockDim.x){blocks2[(bb2+b)*2]=sid;blocks2[(bb2+b)*2+1]=b*16;}
  __syncthreads();
 }
}
extern "C" __global__ void pack_short(ARGS){pack<32,0>(FORWARD);}
extern "C" __global__ void pack_medium(ARGS){pack<64,1>(FORWARD);}
extern "C" __global__ void pack_long(ARGS){pack<161,2>(FORWARD);}
extern "C" __global__ void packed_finalize(const int*allocator,const int*totals,const int*dispatch,int*counts,int*summary){
 if(threadIdx.x==0&&blockIdx.x==0){
  int a=allocator[0],e=allocator[1],flags=allocator[2],real=totals[4];
  summary[0]=a;summary[1]=e;summary[2]=flags;summary[3]=allocator[3];summary[4]=allocator[4];
  for(int j=0;j<4;++j)summary[5+j]=totals[j];summary[9]=real;
  bool good=flags==0&&real>0;
  for(int j=0;j<4;++j)counts[j]=good?totals[j]:0;
  counts[4]=good?dispatch[0]:0;counts[5]=good?a:0;
 }
}
