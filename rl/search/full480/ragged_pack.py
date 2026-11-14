import triton
import triton.language as tl
from cooperative_gpu import hash32

@triton.jit
def ragged_hash(ids,zones,numbers,meta,chunks,inv_ids,inv_zones,inv_numbers,inv_counts,inv_slots,
 entity_slots,extra_slots,features,worlds,input_map,starts,BASE,ROOTS:tl.constexpr,I:tl.constexpr,T:tl.constexpr,R:tl.constexpr):
 sid=tl.program_id(0);root=tl.load(worlds+BASE+sid)%ROOTS;offset=tl.load(starts+sid*4)
 active=tl.load(meta+sid*4)>1
 r=tl.program_id(1)*R+tl.arange(0,R);j=tl.arange(0,16)
 ne=tl.load(meta+sid*4+1);ni=tl.load(inv_counts+root);rr=r-ne;er=(r<ne)&active;ir=(rr>=0)&(rr<ni)&active;live=(r<ne+ni)&active
 at=tl.load(chunks+sid*16+r//64,er,0)+r%64;ia=root*I+rr
 eid=tl.load(ids+at,er,0)+tl.load(inv_ids+ia,ir,0)
 z=tl.load(zones+at,er,0)+tl.load(inv_zones+ia,ir,0)
 s=tl.load(entity_slots+at,er,0)+tl.load(inv_slots+ia,ir,0);extra=tl.load(extra_slots+at,er,-1)
 s=tl.load(input_map+sid*T+s,live,-1);extra=tl.load(input_map+sid*T+extra,live&(extra>=0),-1)
 x=tl.load(numbers+at[:,None]*16+j[None,:],er[:,None],0.)+tl.load(inv_numbers+ia[:,None]*16+j[None,:],ir[:,None],0.)
 identity=(eid*15+z)*17;kind=((eid//512+6144)*15+z)*17
 h=hash32(identity);val=tl.where((h>>31)!=0,-1.,1.)
 tl.atomic_add(features+(offset+s)*128+h%127,val,live)
 tl.atomic_add(features+(offset+extra)*128+h%127,val,live&(extra>=0))
 hi=hash32(identity[:,None]+j[None,:]+1);hk=hash32(kind[:,None]+j[None,:]+1)
 vi=tl.where((hi>>31)!=0,-x,x);vk=tl.where((hk>>31)!=0,-x,x);valid=live[:,None]&(x!=0.)
 tl.atomic_add(features+(offset+s[:,None])*128+hi%127,vi,valid)
 tl.atomic_add(features+(offset+s[:,None])*128+hk%127,vk,valid)
 tl.atomic_add(features+(offset+extra[:,None])*128+hi%127,vi,valid&(extra[:,None]>=0))
 tl.atomic_add(features+(offset+extra[:,None])*128+hk%127,vk,valid&(extra[:,None]>=0))

@triton.jit
def ragged_pack(features,symbols,counts,token_order,query_order,meta,starts,lens,psymbols,slots,present,query_tokens,blocks1,blocks2,owners,T:tl.constexpr):
 sid=tl.program_id(0);c=tl.program_id(1)*4+tl.arange(0,4);j=tl.arange(0,128)
 nt=tl.load(lens+sid*4);nq=tl.load(lens+sid*4+1);ts=tl.load(starts+sid*4);qs=tl.load(starts+sid*4+1)
 old=tl.load(token_order+sid*T+c,c<nt,-1);valid=c<nt
 f=tl.load(features+(ts+c[:,None])*128+j[None,:],valid[:,None],0.)
 sym=tl.load(symbols+sid*T+old,valid,0);cnt=tl.load(counts+sid*T+old,valid,0);cf=cnt.to(tl.float32)
 f=f/tl.sqrt(tl.maximum(cf,1.))[:,None];f=tl.where(j[None,:]==127,tl.log(cf+1.)[:,None],f)
 tl.store(features+(ts+c[:,None])*128+j[None,:],f,valid[:,None])
 tl.store(psymbols+ts+c,sym,valid);tl.store(slots+ts+c,tl.minimum(old,33),valid)
 tl.store(present+ts+c,(cnt>0)|(old==0),valid)
 qt=tl.load(query_order+sid*T+c,c<nq,0)
 tl.store(query_tokens+qs+c,ts+qt,c<nq)
 if tl.program_id(1)==0:
  bi=tl.arange(0,16);n1=tl.load(lens+sid*4+2);n2=tl.load(lens+sid*4+3)
  s1=tl.load(starts+sid*4+2);s2=tl.load(starts+sid*4+3)
  tl.store(blocks1+(s1+bi)*2,sid,bi<n1);tl.store(blocks1+(s1+bi)*2+1,bi*16,bi<n1)
  tl.store(blocks2+(s2+bi)*2,sid,bi<n2);tl.store(blocks2+(s2+bi)*2+1,bi*16,bi<n2)
  na=tl.load(meta+sid*4);begin=tl.load(meta+sid*4+2)
  for offset in range(0,na,128):
   a=offset+tl.arange(0,128);tl.store(owners+begin+a,sid,a<na)

@triton.jit
def ragged_score(acts,keys,owners,source,target,meta,query_map,starts,lens,z,coef,token,embedding,bias,out,N,
 T:tl.constexpr,K:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 row=tl.program_id(0)*B+tl.arange(0,B);valid=row<N
 d=tl.arange(0,32);j=tl.arange(0,16)
 own=tl.load(owners+row,valid,0);na=tl.load(meta+own*4,valid,0);live=valid&(na>1)
 base=tl.load(starts+own*4+1,live,0)*34;nq=tl.load(lens+own*4+1,live,0)
 z0=tl.load(z+own[:,None]*32+d[None,:],live[:,None],0.)
 action=tl.load(acts+row[:,None]*24+d[None,:],live[:,None]&(d[None,:]<24),0.)
 coeff=tl.load(coef+own[:,None]*24+d[None,:],live[:,None]&(d[None,:]<24),0.)
 main=tl.sum(action*coeff,1);ek=tl.full((B,),0,tl.float32);nk=tl.full((B,),0,tl.int32)
 for i in tl.static_range(K):
  key=tl.load(keys+row*K+i,live,-1).to(tl.int64);ok=live&(key>=0)
  v=tl.load(embedding+key[:,None]*32+d[None,:],ok[:,None],0.)
  ek+=tl.sum(v*z0,1)+tl.load(bias+key,ok,0.);nk+=ok.to(tl.int32)
 main+=ek/tl.maximum(1,nk)
 ss=tl.full((B,),0,tl.float32);left=tl.full((B,16),0,tl.float32);ns=tl.full((B,),0,tl.int32)
 for i in tl.static_range(S):
  old=tl.load(source+row*S+i,live,-1)
  sid=tl.load(query_map+own*T+old,live&(old>=0),-1);ok=live&(sid>=0)&(sid<nq)
  addr=base+sid*34
  ss+=tl.load(token+addr,ok,0.);left+=tl.load(token+addr[:,None]+1+j[None,:],ok[:,None],0.);ns+=ok.to(tl.int32)
 factor=1./tl.maximum(1,ns)
 ss=tl.where(ns>0,ss*factor,tl.load(token+base,live,0.))
 left=tl.where(ns[:,None]>0,left*factor[:,None],tl.load(token+base[:,None]+1+j[None,:],live[:,None],0.))
 old=tl.load(target+row,live,-1);tgt=tl.load(query_map+own*T+old,live&(old>=0),-1)
 ok=live&(tgt>=0)&(tgt<nq);addr=base+tgt*34
 ts=tl.load(token+addr+17,ok,0.);right=tl.load(token+addr[:,None]+18+j[None,:],ok[:,None],0.)
 tl.store(out+row,tl.where(live,main+ss+ts+.25*tl.sum(left*right,1),0.),valid)
