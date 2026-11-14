import triton
import triton.language as tl
from .cooperative_gpu import hash32
@triton.jit
def pool_hash(ids,zones,numbers,meta,chunks,inv_ids,inv_zones,inv_numbers,inv_counts,inv_slots,
 entity_slots,extra_slots,selected,features,BASE,ROOTS:tl.constexpr,I:tl.constexpr,T:tl.constexpr,R:tl.constexpr):
 b=tl.program_id(0);sid=tl.load(selected+b);root=(BASE+sid)%ROOTS
 r=tl.program_id(1)*R+tl.arange(0,R);j=tl.arange(0,16)
 ne=tl.load(meta+sid*4+1);ni=tl.load(inv_counts+root);rr=r-ne;er=r<ne;ir=(rr>=0)&(rr<ni);live=r<ne+ni
 at=tl.load(chunks+sid*16+r//64,er,0)+r%64;ia=root*I+rr
 eid=tl.load(ids+at,er,0)+tl.load(inv_ids+ia,ir,0)
 z=tl.load(zones+at,er,0)+tl.load(inv_zones+ia,ir,0)
 s=tl.load(entity_slots+at,er,0)+tl.load(inv_slots+ia,ir,0);extra=tl.load(extra_slots+at,er,-1)
 x=tl.load(numbers+at[:,None]*16+j[None,:],er[:,None],0.)+tl.load(inv_numbers+ia[:,None]*16+j[None,:],ir[:,None],0.)
 identity=(eid*15+z)*17;kind=((eid//512+6144)*15+z)*17
 h=hash32(identity);val=tl.where((h>>31)!=0,-1.,1.)
 tl.atomic_add(features+(b*T+s)*128+h%127,val,live)
 tl.atomic_add(features+(b*T+extra)*128+h%127,val,live&(extra>=0))
 hi=hash32(identity[:,None]+j[None,:]+1);hk=hash32(kind[:,None]+j[None,:]+1)
 vi=tl.where((hi>>31)!=0,-x,x);vk=tl.where((hk>>31)!=0,-x,x);valid=live[:,None]&(x!=0.)
 tl.atomic_add(features+(b*T+s[:,None])*128+hi%127,vi,valid)
 tl.atomic_add(features+(b*T+s[:,None])*128+hk%127,vk,valid)
 tl.atomic_add(features+(b*T+extra[:,None])*128+hi%127,vi,valid&(extra[:,None]>=0))
 tl.atomic_add(features+(b*T+extra[:,None])*128+hk%127,vk,valid&(extra[:,None]>=0))
@triton.jit
def pool_normalize(features,counts,selected,REAL,T:tl.constexpr):
 row=tl.program_id(0);b=row//T;t=row%T;j=tl.arange(0,128)
 sid=tl.load(selected+b,b<REAL,0);count=tl.load(counts+sid*T+t,b<REAL,0).to(tl.float32)
 x=tl.load(features+row*128+j);v=x/tl.sqrt(tl.maximum(count,1.))
 tl.store(features+row*128+j,tl.where(j==127,tl.log(count+1.),v))
@triton.jit
def pool_pack_features(features,symbols,counts,token_order,query_order,raw_state,selected,
 packed,psymbols,slots,present,queries,pstate,REAL,T:tl.constexpr,C:tl.constexpr,Q:tl.constexpr):
 b=tl.program_id(0);c=tl.program_id(1)*4+tl.arange(0,4);j=tl.arange(0,128)
 sid=tl.load(selected+b,b<REAL,0);old=tl.load(token_order+sid*T+c,(b<REAL)&(c<C),-1);valid=(b<REAL)&(c<C)&(old>=0)
 f=tl.load(features+(b*T+old[:,None])*128+j[None,:],valid[:,None],0.)
 sym=tl.load(symbols+sid*T+old,valid,0);cnt=tl.load(counts+sid*T+old,valid,0)
 tl.store(packed+(b*C+c[:,None])*128+j[None,:],f,c[:,None]<C)
 tl.store(psymbols+b*C+c,sym,c<C);tl.store(slots+b*C+c,tl.where(valid,tl.minimum(old,33),0),c<C)
 tl.store(present+b*C+c,(valid&((cnt>0)|(old==0)))|((b>=REAL)&(c==0)),c<C)
 if tl.program_id(1)==0:
  q=tl.arange(0,triton.next_power_of_2(Q));v=tl.load(query_order+sid*T+q,(b<REAL)&(q<Q),0)
  tl.store(queries+b*Q+q,v,q<Q)
  k=tl.arange(0,32);x=tl.load(raw_state+sid*32+k,b<REAL,0.);tl.store(pstate+b*32+k,x)
@triton.jit
def pool_pack_actions(actions,keys,sources,targets,meta,query_map,selected,starts,
 pact,pkeys,owners,psources,ptargets,T:tl.constexpr,R:tl.constexpr):
 b=tl.program_id(0);a=tl.program_id(1)*R+tl.arange(0,R);j=tl.arange(0,32);k=tl.arange(0,16)
 sid=tl.load(selected+b);na=tl.load(meta+sid*4);begin=tl.load(meta+sid*4+2)
 valid=a<na;inp=begin+a;out=tl.load(starts+b)+a
 x=tl.load(actions+inp[:,None]*24+j[None,:],valid[:,None]&(j[None,:]<24),0.)
 kk=tl.load(keys+inp[:,None]*10+k[None,:],valid[:,None]&(k[None,:]<10),-1)
 src=tl.load(sources+inp[:,None]*10+k[None,:],valid[:,None]&(k[None,:]<10),-1)
 ss=tl.load(query_map+sid*T+src,valid[:,None]&(src>=0)&(k[None,:]<10),-1)
 tgt=tl.load(targets+inp,valid,-1);tt=tl.load(query_map+sid*T+tgt,valid&(tgt>=0),-1)
 tl.store(pact+out[:,None]*24+j[None,:],x,valid[:,None]&(j[None,:]<24))
 tl.store(pkeys+out[:,None]*10+k[None,:],kk,valid[:,None]&(k[None,:]<10))
 tl.store(psources+out[:,None]*10+k[None,:],ss,valid[:,None]&(k[None,:]<10))
 tl.store(ptargets+out,tt,valid);tl.store(owners+out,b,valid)
@triton.jit
def pool_scatter_scores(logits,out,meta,selected,starts,R:tl.constexpr):
 b=tl.program_id(0);a=tl.program_id(1)*R+tl.arange(0,R);sid=tl.load(selected+b)
 na=tl.load(meta+sid*4);begin=tl.load(meta+sid*4+2);local=tl.load(starts+b)+a
 score=tl.load(logits+local,a<na,0.);tl.store(out+begin+a,score,a<na)
