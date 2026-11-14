import triton
import triton.language as tl
@triton.jit
def hash32(x):
 x=x.to(tl.uint32);x=(x^(x>>16))*0x7feb352d;x=(x^(x>>15))*0x846ca68b
 return x^(x>>16)
@triton.jit
def hashed_entities(ids,zones,numbers,meta,inv_ids,inv_zones,inv_numbers,inv_counts,
 entity_slots,extra_slots,inventory_slots,features,E:tl.constexpr,I:tl.constexpr,T:tl.constexpr,R:tl.constexpr):
 b=tl.program_id(0);r=tl.program_id(1)*R+tl.arange(0,R);j=tl.arange(0,16)
 ne=tl.load(meta+b*2+1);ni=tl.load(inv_counts+b);rr=r-ne;live=r<ne+ni
 er=r<ne;ir=(rr>=0)&(rr<ni)
 id=tl.load(ids+b*E+r,er,0)+tl.load(inv_ids+b*I+rr,ir,0)
 z=tl.load(zones+b*E+r,er,0)+tl.load(inv_zones+b*I+rr,ir,0)
 s=tl.load(entity_slots+b*E+r,er,0)+tl.load(inventory_slots+b*I+rr,ir,0)
 extra=tl.load(extra_slots+b*E+r,er,-1)
 x=tl.load(numbers+(b*E+r[:,None])*16+j[None,:],er[:,None],0.)+tl.load(inv_numbers+(b*I+rr[:,None])*16+j[None,:],ir[:,None],0.)
 identity=(id*15+z)*17;kind=((id//512+6144)*15+z)*17
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
def normalize_tokens(features,counts,TOTAL:tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,128);count=tl.load(counts+row).to(tl.float32)
 x=tl.load(features+row*128+j);out=x/tl.sqrt(tl.maximum(count,1.))
 out=tl.where(j==127,tl.log(count+1.),out);tl.store(features+row*128+j,out)
@triton.jit
def pack_features(features,symbols,counts,mapping,packed,psymbols,slots,present,T:tl.constexpr,C:tl.constexpr,TT:tl.constexpr):
 b=tl.program_id(0);c=tl.program_id(1)*4+tl.arange(0,4);t=tl.arange(0,TT);j=tl.arange(0,128)
 mapped=tl.load(mapping+b*T+t,t<T,-1)
 old=tl.max(tl.where((mapped[None,:]==c[:,None])&(t[None,:]<T),t[None,:],-1),1)
 valid=(c<C)&(old>=0);off=b*T+old
 f=tl.load(features+off[:,None]*128+j[None,:],valid[:,None],0.)
 sym=tl.load(symbols+off,valid,0);cnt=tl.load(counts+off,valid,0)
 tl.store(packed+(b*C+c[:,None])*128+j[None,:],f,c[:,None]<C)
 tl.store(psymbols+b*C+c,sym,c<C);tl.store(slots+b*C+c,tl.where(valid,tl.minimum(old,33),0),c<C)
 tl.store(present+b*C+c,valid&((cnt>0)|(old==0)),c<C)
@triton.jit
def pack_actions(actions,keys,sources,targets,meta,starts,mapping,pact,pkeys,owners,psources,ptargets,
 A:tl.constexpr,T:tl.constexpr,R:tl.constexpr):
 b=tl.program_id(0);a=tl.program_id(1)*R+tl.arange(0,R);j=tl.arange(0,32);k=tl.arange(0,16)
 na=tl.load(meta+b*2);valid=a<na;out=tl.load(starts+b)+a;inp=b*A+a
 x=tl.load(actions+inp[:,None]*24+j[None,:],valid[:,None]&(j[None,:]<24),0.)
 kk=tl.load(keys+inp[:,None]*10+k[None,:],valid[:,None]&(k[None,:]<10),-1)
 src=tl.load(sources+inp[:,None]*10+k[None,:],valid[:,None]&(k[None,:]<10),-1)
 ss=tl.load(mapping+b*T+src,valid[:,None]&(src>=0)&(k[None,:]<10),-1)
 tgt=tl.load(targets+inp,valid,-1);tt=tl.load(mapping+b*T+tgt,valid&(tgt>=0),-1)
 tl.store(pact+out[:,None]*24+j[None,:],x,valid[:,None]&(j[None,:]<24))
 tl.store(pkeys+out[:,None]*10+k[None,:],kk,valid[:,None]&(k[None,:]<10))
 tl.store(psources+out[:,None]*10+k[None,:],ss,valid[:,None]&(k[None,:]<10))
 tl.store(ptargets+out,tt,valid);tl.store(owners+out,b,valid)
