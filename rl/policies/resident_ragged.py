"""Persistent ragged 2x64 kernels migrated from the accepted GPU search.

The norm/QKV, attention tile/buckets and mix+FFN below retain the old kernel
arithmetic. New packing adapts current fixed native slots to those contracts.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def relic_lookup(IDS, COUNTERS, LIVE, TABLE, IDENTITY, GATES, STATIC, OUT,
                 R: tl.constexpr, RR: tl.constexpr, LIMIT: tl.constexpr):
 row=tl.program_id(0);r=tl.arange(0,RR);d=tl.arange(0,64)
 ids=tl.load(IDS+row*R+r,r<R,0).to(tl.int64)
 live=(r<R)&(ids!=0)&tl.load(LIVE+ids)
 x=tl.load(COUNTERS+row*R+r,live,0.)
 count=libdevice.nearbyint(x*10.)
 lookup=live&(count>=0)&(count<LIMIT)&(x==count/10.)
 index=count.to(tl.int64)
 value=tl.load(TABLE+(ids[:,None]*LIMIT+index[:,None])*64+d[None,:],lookup[:,None],0.)
 fallback=live&~lookup
 if tl.sum(fallback.to(tl.int32),0)>0:
  direct=tl.load(IDENTITY+ids[:,None]*64+d[None,:],fallback[:,None],0.)
  contribution=tl.zeros((RR,64),tl.float32)
  for feature in tl.static_range(10):
   if feature==0:basis=x
   elif feature==1:basis=tl.where(x>0,1.,tl.where(x<0,-1.,0.))*libdevice.log1p(tl.abs(x))
   else:
    period=10 if feature==9 else feature
    basis=(count-tl.floor(count/period)*period)/period
   weight=tl.load(GATES+(ids[:,None]*10+feature)*64+d[None,:],fallback[:,None],0.)
   contribution=contribution+weight*basis[:,None]
  value=tl.where(fallback[:,None],direct+contribution,value)
 total=tl.load(STATIC+row*64+d)+tl.sum(value,0)
 tl.store(OUT+row*64+d,total)


@torch.compiler.disable
def frozen_relic_summary(ids, counters, static, live, table, identity, gates):
    """Root-static vector plus live counters; unusual values retain full basis."""
    if torch.is_grad_enabled() or not counters.is_cuda or counters.dtype != torch.float32:
        raise ValueError('frozen relic projection requires CUDA FP32 inference')
    result = torch.empty((len(ids),64),device=counters.device,dtype=torch.float32)
    relic_lookup[(len(ids),)](ids.contiguous(),counters.contiguous(),live,table,
        identity,gates,static.contiguous(),result,ids.shape[1],
        triton.next_power_of_2(ids.shape[1]),table.shape[1],num_warps=4,enable_fp_fusion=False)
    return result


@triton.jit
def pile_projection(COUNT, UPGRADED, MAGNITUDES, BASE, DELTA, UPGRADE, OUT, SIZES,
                    M: tl.constexpr, K: tl.constexpr):
 r=tl.program_id(0)*32+tl.arange(0,32)
 k=tl.arange(0,32);d=tl.arange(0,64)
 base_sum=tl.zeros((32,64),tl.float32)
 upgraded_sum=tl.zeros((32,64),tl.float32)
 magnitude=tl.zeros((32,),tl.float32);size=tl.zeros((32,),tl.float32)
 for block in range(tl.cdiv(K,32)):
  cols=block*32+k
  valid=(r[:,None]<M)&(cols[None,:]<K)
  count=tl.load(COUNT+r[:,None]*K+cols[None,:],valid,0.)
  upgraded=tl.load(UPGRADED+r[:,None]*K+cols[None,:],valid,0.)
  amounts=tl.load(MAGNITUDES+r[:,None]*K+cols[None,:],valid,0.)
  base=tl.load(BASE+cols[:,None]*64+d[None,:],cols[:,None]<K,0.)
  delta=tl.load(DELTA+cols[:,None]*64+d[None,:],cols[:,None]<K,0.)
  base_sum+=tl.dot(count,base,input_precision='tf32x3')
  upgraded_sum+=tl.dot(upgraded,delta,input_precision='tf32x3')
  magnitude+=tl.sum(amounts,1)
  size+=tl.sum(tl.where(cols[None,:]!=0,count,0.),1)
 value=base_sum+upgraded_sum+magnitude[:,None]*tl.load(UPGRADE+d)[None,:]
 tl.store(OUT+r[:,None]*64+d[None,:],value,r[:,None]<M)
 tl.store(SIZES+r,size,r<M)


@triton.jit
def indexed_pile_projection(SCALARS, UPGRADES, ROWS, ROW_COUNT, BASE, DELTA, UPGRADE,
                            OUT, SIZES, M: tl.constexpr, K: tl.constexpr,
                            STRIDE: tl.constexpr, START: tl.constexpr, SCALE: tl.constexpr):
 r=tl.program_id(0)*32+tl.arange(0,32);k=tl.arange(0,32);d=tl.arange(0,64)
 active=(r<M)&(r//3<tl.load(ROW_COUNT))
 physical=tl.load(ROWS+r//3,active,0);zone=r%3
 base_sum=tl.zeros((32,64),tl.float32);upgraded_sum=tl.zeros((32,64),tl.float32)
 magnitude=tl.zeros((32,),tl.float32);size=tl.zeros((32,),tl.float32)
 for block in range(tl.cdiv(K,32)):
  cols=block*32+k;valid=active[:,None]&(cols[None,:]<K)
  count=tl.load(SCALARS+physical[:,None]*STRIDE+START+zone[:,None]*K+cols[None,:],valid,0.)
  upgraded=tl.load(UPGRADES+(physical[:,None]*6+zone[:,None])*K+cols[None,:],valid,0.)
  amounts=tl.load(UPGRADES+(physical[:,None]*6+3+zone[:,None])*K+cols[None,:],valid,0.)/SCALE
  base=tl.load(BASE+cols[:,None]*64+d[None,:],cols[:,None]<K,0.)
  delta=tl.load(DELTA+cols[:,None]*64+d[None,:],cols[:,None]<K,0.)
  base_sum+=tl.dot(count,base,input_precision='tf32x3')
  upgraded_sum+=tl.dot(upgraded,delta,input_precision='tf32x3')
  magnitude+=tl.sum(amounts,1);size+=tl.sum(tl.where(cols[None,:]!=0,count,0.),1)
 value=base_sum+upgraded_sum+magnitude[:,None]*tl.load(UPGRADE+d)[None,:]
 tl.store(OUT+r[:,None]*64+d[None,:],value,r[:,None]<M);tl.store(SIZES+r,size,r<M)


@torch.compiler.disable
def frozen_indexed_piles(scalars, upgrades, rows, row_count, base, delta, upgrade, start, scale):
    """Project native piles in place; never copy their dense histograms per frame."""
    batch=len(rows);cards=upgrades.shape[-1]
    result=torch.empty((batch,3,64),device=scalars.device,dtype=torch.float32)
    sizes=torch.empty((batch,3,1),device=scalars.device,dtype=torch.float32)
    indexed_pile_projection[(triton.cdiv(batch*3,32),)](scalars,upgrades,rows,row_count,base,delta,
        upgrade,result,sizes,batch*3,cards,scalars.stride(0),start,scale,num_warps=4)
    return result,sizes


@torch.compiler.disable
def frozen_pile_summary(counts, upgraded, magnitudes, base, delta, upgrade):
    """Old FP32/tf32x3 frozen projections, with current exact pile counts."""
    if torch.is_grad_enabled() or counts.dtype != torch.float32 or not counts.is_cuda:
        raise ValueError('frozen pile projection requires CUDA FP32 inference')
    batch, piles, cards = counts.shape
    result = torch.empty((batch,piles,64),device=counts.device,dtype=torch.float32)
    sizes = torch.empty((batch,piles,1),device=counts.device,dtype=torch.float32)
    pile_projection[(triton.cdiv(batch*piles,32),)](counts.contiguous(),upgraded.contiguous(),
        magnitudes.contiguous(),base,delta,upgrade,result,sizes,batch*piles,cards,num_warps=4)
    return result,sizes


@triton.jit
def norm64(x,W,B,EPS:tl.constexpr):
 d=tl.arange(0,64);u=x-tl.sum(x,1)[:,None]/64.
 return u*tl.rsqrt(tl.sum(u*u,1)/64.+EPS)[:,None]*tl.load(W+d)[None,:]+tl.load(B+d)[None,:]

@triton.jit
def attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H:tl.constexpr,D:tl.constexpr,SECOND:tl.constexpr,KT:tl.constexpr):
 qi=local+tl.arange(0,16);ki=tl.arange(0,KT);di=tl.arange(0,D)
 qtoken=ts+qi
 if SECOND:qtoken=tl.load(query_tokens+qs+qi,qi<nq,0)
 q=tl.load(qkv+qtoken[:,None]*(3*H*D)+h*D+di[None,:],qi[:,None]<nq,0.)
 k=tl.load(qkv+(ts+ki[None,:])*(3*H*D)+H*D+h*D+di[:,None],ki[None,:]<nt,0.)
 live=(ki<nt)&(tl.load(present+ts+ki,ki<nt,False)!=0)
 scores=tl.where(live[None,:],tl.dot(q,k,input_precision='ieee')*(D**-0.5),-float('inf'))
 probs=tl.exp(scores-tl.max(scores,1)[:,None]);probs=probs/tl.sum(probs,1)[:,None]
 v=tl.load(qkv+(ts+ki[:,None])*(3*H*D)+2*H*D+h*D+di[None,:],ki[:,None]<nt,0.)
 return tl.dot(probs,v,input_precision='ieee')

@triton.jit
def qkv_project(X,NW,NB,W,B,Y,COUNTS,EPS:tl.constexpr):
 n=tl.load(COUNTS);d=tl.arange(0,64);cols=tl.program_id(1)*64+d
 weight=tl.load(W+cols[None,:]*64+d[:,None])
 for block in range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0)):
  r=block*32+tl.arange(0,32)
  x=tl.load(X+r[:,None]*64+d[None,:],r[:,None]<n,0.)
  z=norm64(x,NW,NB,EPS)
  y=tl.dot(z,weight,input_precision='tf32x3')+tl.load(B+cols)[None,:]
  tl.store(Y+r[:,None]*192+cols[None,:],y,r[:,None]<n)

@triton.jit
def mix_feed(A,X,QUERY,MW,MB,NW,NB,F1,B1,F2,B2,OUT,COUNTS,SECOND:tl.constexpr,EPS:tl.constexpr):
 n=tl.load(COUNTS+(1 if SECOND else 0));d=tl.arange(0,64);h=tl.arange(0,128)
 for block in tl.range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0),disable_licm=True):
  r=block*32+tl.arange(0,32);xr=r
  if SECOND:xr=tl.load(QUERY+r,r<n,0)
  a=tl.load(A+r[:,None]*64+d[None,:],r[:,None]<n,0.)
  residual=tl.load(X+xr[:,None]*64+d[None,:],r[:,None]<n,0.)
  wm=tl.load(MW+d[None,:]*64+d[:,None])
  mixed=tl.dot(a,wm,input_precision='tf32x3')+tl.load(MB+d)[None,:]
  residual=residual+mixed;v=norm64(residual,NW,NB,EPS)
  w1=tl.load(F1+h[None,:]*64+d[:,None])
  hidden=tl.maximum(tl.dot(v,w1,input_precision='tf32x3')+tl.load(B1+h)[None,:],0.)
  w2=tl.load(F2+d[None,:]*128+h[:,None])
  out=residual+(tl.dot(hidden,w2,input_precision='tf32x3')+tl.load(B2+d)[None,:])
  tl.store(OUT+r[:,None]*64+d[None,:],out,r[:,None]<n)

@triton.jit
def attend_bucket(QKV,PRESENT,QUERY,STARTS,LENS,BLOCKS,OUT,COUNTS,SECOND:tl.constexpr,BUCKET:tl.constexpr):
 nb=tl.load(COUNTS+(3 if SECOND else 2))
 for taskhead in range(tl.program_id(0),nb*4,tl.num_programs(0)):
  task=taskhead//4;h=taskhead%4;sid=tl.load(BLOCKS+task*2);local=tl.load(BLOCKS+task*2+1)
  ts=tl.load(STARTS+sid*4);nt=tl.load(LENS+sid*4);qs=ts;nq=nt
  if SECOND:qs=tl.load(STARTS+sid*4+1);nq=tl.load(LENS+sid*4+1)
  eligible=(nt<=32) if BUCKET==0 else ((nt>32)&(nt<=64) if BUCKET==1 else nt>64)
  if eligible:
   if BUCKET==0:
    if nt<=16:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,16)
    else:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,32)
   elif BUCKET==1:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,64)
   else:
    if nt<=128:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,128)
    else:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,256)
   qi=local+tl.arange(0,16);di=tl.arange(0,16)
   tl.store(OUT+(qs+qi[:,None])*64+h*16+di[None,:],y,qi[:,None]<nq)


@triton.jit
def prepare_lens(PRESENT, QUERIES, ROWS, KEEP, KEYS, VALID_QUERY, LENS,
                 T: tl.constexpr, TT: tl.constexpr, BOUNDED: tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,TT)
 active=True
 if BOUNDED:active=row<tl.load(ROWS)
 key=tl.load(PRESENT+row*T+j,(j<T)&active,False)
 query=tl.load(QUERIES+row*T+j,(j<T)&active,False)
 keep=key|query
 tl.store(KEEP+row*T+j,keep,j<T)
 tl.store(KEYS+row*T+j,key,j<T)
 tl.store(VALID_QUERY+row*T+j,query,j<T)
 nt=tl.sum(keep.to(tl.int64),0);nq=tl.sum(query.to(tl.int64),0)
 tl.store(LENS+row*4,nt);tl.store(LENS+row*4+1,nq)
 tl.store(LENS+row*4+2,(nt+15)//16);tl.store(LENS+row*4+3,(nq+15)//16)


@triton.jit
def prefix_local(LENS,ENDS,TOTALS,F:tl.constexpr,B:tl.constexpr):
 block=tl.program_id(0);r=block*B+tl.arange(0,B);j=tl.arange(0,4)
 x=tl.load(LENS+r[:,None]*4+j[None,:],r[:,None]<F,0)
 y=tl.cumsum(x,0)
 tl.store(ENDS+r[:,None]*4+j[None,:],y,r[:,None]<F)
 total=tl.sum(x,0)
 tl.store(TOTALS+block*4+j,total)

@triton.jit
def prefix_starts(ENDS,LENS,STARTS,COUNTS,TOTALS,F:tl.constexpr,B:tl.constexpr,CB:tl.constexpr):
 block=tl.program_id(0);r=block*B+tl.arange(0,B);j=tl.arange(0,4);b=tl.arange(0,CB)
 total=tl.load(TOTALS+b[:,None]*4+j[None,:],b[:,None]<block,0)
 prefix=tl.sum(total,0)
 y=tl.load(ENDS+r[:,None]*4+j[None,:],r[:,None]<F,0)
 length=tl.load(LENS+r[:,None]*4+j[None,:],r[:,None]<F,0)
 tl.store(STARTS+r[:,None]*4+j[None,:],y+prefix[None,:]-length,r[:,None]<F)
 if block==0:
  all_totals=tl.load(TOTALS+b[:,None]*4+j[None,:],b[:,None]<tl.cdiv(F,B),0)
  tl.store(COUNTS+j,tl.sum(all_totals,0))


@triton.jit
def check_capacity(COUNTS, STATUS, CT: tl.constexpr, CQ: tl.constexpr):
 nt=tl.load(COUNTS);nq=tl.load(COUNTS+1)
 overflow=(nt>CT)|(nq>CQ)
 tl.store(STATUS+5,nt);tl.store(STATUS+6,nq);tl.store(STATUS+7,overflow.to(tl.int32))
 # A capacity retry is read-only. Downstream persistent kernels see no work.
 if overflow:
  tl.store(COUNTS+tl.arange(0,4),0)


@triton.jit
def pack_slots(X, PRESENT, KEEP, QUERIES, STARTS, LENS, PACKED, KEYS,
               QUERY, DENSE_SLOT, BLOCKS1, BLOCKS2, STATUS, T: tl.constexpr,
               TT: tl.constexpr, BOUNDED: tl.constexpr):
 if BOUNDED:
  if tl.load(STATUS+7)!=0:return
 row = tl.program_id(0); j = tl.arange(0, TT); d = tl.arange(0, 64)
 keep = tl.load(KEEP+row*T+j, j<T, 0).to(tl.int32)
 query = tl.load(QUERIES+row*T+j, j<T, 0).to(tl.int32)
 ts=tl.load(STARTS+row*4);qs=tl.load(STARTS+row*4+1)
 dst=ts+tl.cumsum(keep,0)-1;qd=qs+tl.cumsum(query,0)-1
 value=tl.load(X+(row*T+j[:,None])*64+d[None,:],(j[:,None]<T)&(keep[:,None]!=0),0.)
 tl.store(PACKED+dst[:,None]*64+d[None,:],value,keep[:,None]!=0)
 present=tl.load(PRESENT+row*T+j,j<T,False)
 tl.store(KEYS+dst,present,keep!=0)
 tl.store(QUERY+qd,dst,query!=0)
 tl.store(DENSE_SLOT+qd,row*T+j,query!=0)
 for layer in tl.static_range(2):
  count=tl.load(LENS+row*4+2+layer);start=tl.load(STARTS+row*4+2+layer)
  blocks=BLOCKS1 if layer==0 else BLOCKS2
  for b in range(count):
   tl.store(blocks+(start+b)*2,row);tl.store(blocks+(start+b)*2+1,b*16)


@triton.jit
def unpack_queries(X, SLOTS, OUT, COUNTS, NW, NB, EPS: tl.constexpr,
                   NORMALIZE: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,64)
 valid=(i<CAP)&(i<tl.load(COUNTS+1))
 slot=tl.load(SLOTS+i,valid,0)
 value=tl.load(X+i[:,None]*64+d[None,:],valid[:,None],0.)
 if NORMALIZE:
  value=norm64(value,NW,NB,EPS)
 tl.store(OUT+slot[:,None]*64+d[None,:],value,valid[:,None])


@triton.jit
def entity_flags(HAND, MONSTER, POTION, VALID, REQUIRED, ROWS, row, j,
                 H: tl.constexpr, M: tl.constexpr, P: tl.constexpr, W: tl.constexpr):
 active=row<tl.load(ROWS)
 h=(j>=1)&(j<1+H);m=(j>=1+H)&(j<1+H+M)
 p=(j>=1+H+M)&(j<1+H+M+P);start=1+H+M+P
 select=(j>=start+4)&(j<start+4+W)
 key=(j==0)|((j>=start)&(j<start+4))
 key|=tl.load(HAND+row*H+j-1,h&active,0)!=0
 key|=tl.load(MONSTER+row*M+j-1-H,m&active,0)!=0
 key|=tl.load(POTION+row*P+j-1-H-M,p&active,0)!=0
 key|=tl.load(VALID+row*W+j-start-4,select&active,False)
 query=(key&(j<start))|tl.load(REQUIRED+row*W+j-start-4,select&active,False)
 return key&active,query&active


@triton.jit
def entity_lens(HAND, MONSTER, POTION, VALID, REQUIRED, ROWS, LENS,
                H: tl.constexpr, M: tl.constexpr, P: tl.constexpr,
                W: tl.constexpr, TT: tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,TT)
 key,query=entity_flags(HAND,MONSTER,POTION,VALID,REQUIRED,ROWS,row,j,H,M,P,W)
 nt=tl.sum((key|query).to(tl.int64),0);nq=tl.sum(query.to(tl.int64),0)
 tl.store(LENS+row*4,nt);tl.store(LENS+row*4+1,nq)
 tl.store(LENS+row*4+2,(nt+15)//16);tl.store(LENS+row*4+3,(nq+15)//16)


@triton.jit
def pack_entity_features(GLOBAL, RELIC, PILES, HAND, HU, HC, HD, HB, HP,
                         MONSTER, MM, MI, MS, MB, POTION, CHOICE, CU, CC, ZONE,
                         VALID, REQUIRED, ROWS, STARTS, LENS, FEATURES, META,
                         KEYS, QUERY, OUTPUT_MAP, BLOCKS1, BLOCKS2, STATUS,
                         H: tl.constexpr, M: tl.constexpr, P: tl.constexpr,
                         W: tl.constexpr, TT: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr):
 if tl.load(STATUS+7)!=0:return
 row=tl.program_id(0);j=tl.arange(0,TT);d=tl.arange(0,64)
 key,query=entity_flags(HAND,MONSTER,POTION,VALID,REQUIRED,ROWS,row,j,H,M,P,W)
 keep=key|query;start=1+H+M+P
 h=(j>=1)&(j<1+H);m=(j>=1+H)&(j<1+H+M)
 p=(j>=1+H+M)&(j<start);pile=(j>start)&(j<start+4)
 select=(j>=start+4)&(j<start+4+W)
 # Feature storage aliases the later QKV workspace. Static and global
 # vectors are already projected; dynamic features stay narrow until packed.
 x=tl.load(GLOBAL+row*64+d[None,:],(j[:,None]==0)&keep[:,None],0.)
 x+=tl.load(RELIC+row*64+d[None,:],(j[:,None]==start)&keep[:,None],0.)
 x+=tl.load(PILES+(row*3+j[:,None]-start-1)*64+d[None,:],pile[:,None]&keep[:,None],0.)
 hi=row*H+j-1;ci=row*W+j-start-4;mi=row*M+j-1-H
 hx=tl.where(d[None,:]==0,tl.load(HC+hi,h&keep,0.)[:,None],0.)
 hx+=tl.where(d[None,:]==1,tl.load(HD+hi,h&keep,0.)[:,None],0.)
 hx+=tl.where(d[None,:]==2,tl.load(HB+hi,h&keep,0.)[:,None],0.)
 hx+=tl.where(d[None,:]==3,tl.load(HP+hi,h&keep,0.)[:,None],0.)
 hx+=tl.where(d[None,:]==4,tl.load(HU+hi,h&keep,0.)[:,None],0.)
 x+=hx
 x+=tl.load(MS+mi[:,None]*NS+d[None,:],m[:,None]&keep[:,None]&(d[None,:]<NS),0.)
 x+=tl.load(MB+mi[:,None]*NB+d[None,:]-NS,m[:,None]&keep[:,None]&(d[None,:]>=NS)&(d[None,:]<NS+NB),0.)
 cid=tl.load(CHOICE+ci,select&keep,0)
 cx=tl.where(d[None,:]==0,tl.load(CC+ci,select&keep,0.)[:,None],0.)
 cv=tl.load(VALID+ci,select&keep,False)&(cid==0)
 cx+=tl.where(d[None,:]==1,cv[:,None].to(tl.float32),0.)
 cx+=tl.where(d[None,:]==2,tl.load(CU+ci,select&keep,0.)[:,None],0.)
 x+=cx
 role=tl.where(h,1,tl.where(m,2,tl.where(p,3,tl.where(j==start,4,
      tl.where(pile,4+j-start,tl.where(select,8,0))))))
 identity=tl.load(HAND+hi,h&keep,0)+tl.load(MONSTER+mi,m&keep,0)
 identity+=tl.load(POTION+row*P+j-1-H-M,p&keep,0)+cid
 ts=tl.load(STARTS+row*4);qs=tl.load(STARTS+row*4+1)
 dst=ts+tl.cumsum(keep.to(tl.int32),0)-1;qd=qs+tl.cumsum(query.to(tl.int32),0)-1
 tl.store(FEATURES+dst[:,None]*64+d[None,:],x,keep[:,None])
 tl.store(META+dst*5,role,keep);tl.store(META+dst*5+1,identity,keep)
 tl.store(META+dst*5+2,tl.load(MM+mi,m&keep,0),keep)
 tl.store(META+dst*5+3,tl.load(MI+mi,m&keep,0),keep)
 tl.store(META+dst*5+4,tl.load(ZONE+row),keep)
 tl.store(KEYS+dst,key,keep);tl.store(QUERY+qd,dst,query)
 consumer=(j<start)|select;slot=tl.where(select,j-4,j)
 tl.store(OUTPUT_MAP+row*(start+W)+slot,tl.where(query,qd,-1),consumer)
 for layer in tl.static_range(2):
  count=tl.load(LENS+row*4+2+layer);base=tl.load(STARTS+row*4+2+layer)
  blocks=BLOCKS1 if layer==0 else BLOCKS2
  for b in range(count):
   tl.store(blocks+(base+b)*2,row);tl.store(blocks+(base+b)*2+1,b*16)


@triton.jit
def entity_input_tokens(FEATURES, META, CARD, UPGRADE, HW, HB, CW, CB, ZONES,
                        MIDS, MOVES, INTENTS, MW, MB, POTIONS, ROLES, OUT, COUNTS,
                        MK: tl.constexpr, MKT: tl.constexpr):
 d=tl.arange(0,64);k=tl.arange(0,16);mk=tl.arange(0,MKT)
 nt=tl.load(COUNTS)
 for base in range(tl.program_id(0)*32,nt,tl.num_programs(0)*32):
  r=base+tl.arange(0,32);live=r<nt
  role=tl.load(META+r*5,live,0);identity=tl.load(META+r*5+1,live,0)
  x=tl.load(FEATURES+r[:,None]*64+d[None,:],live[:,None],0.)
  h=live&(role==1);c=live&(role==8);m=live&(role==2);p=live&(role==3)
  if tl.sum(h.to(tl.int32),0)>0:
   f=tl.load(FEATURES+r[:,None]*64+k[None,:],h[:,None]&(k[None,:]<4),0.)
   w=tl.load(HW+d[None,:]*4+k[:,None],k[:,None]<4,0.)
   contribution=tl.dot(f,w,input_precision='tf32x3')+tl.load(HB+d)[None,:]
   up=tl.load(FEATURES+r*64+4,h,0.)
   cards=tl.load(CARD+(identity[:,None]*2+(up[:,None]>0))*64+d[None,:],h[:,None],0.)
   token=(cards+up[:,None]*tl.load(UPGRADE+d)[None,:])+contribution
   x=tl.where(h[:,None],token,x)
  if tl.sum(c.to(tl.int32),0)>0:
   f=tl.load(FEATURES+r[:,None]*64+k[None,:],c[:,None]&(k[None,:]<2),0.)
   w=tl.load(CW+d[None,:]*2+k[:,None],k[:,None]<2,0.)
   contribution=tl.dot(f,w,input_precision='tf32x3')+tl.load(CB+d)[None,:]
   up=tl.load(FEATURES+r*64+2,c,0.);zone=tl.load(META+r*5+4,c,0)
   cards=tl.load(CARD+(identity[:,None]*2+(up[:,None]>0))*64+d[None,:],c[:,None],0.)
   token=(cards+up[:,None]*tl.load(UPGRADE+d)[None,:])+contribution
   token+=tl.load(ZONES+zone[:,None]*64+d[None,:],c[:,None],0.)
   x=tl.where(c[:,None],token,x)
  if tl.sum(m.to(tl.int32),0)>0:
   f=tl.load(FEATURES+r[:,None]*64+mk[None,:],m[:,None]&(mk[None,:]<MK),0.)
   w=tl.load(MW+d[None,:]*MK+mk[:,None],mk[:,None]<MK,0.)
   token=tl.dot(f,w,input_precision='tf32x3')+tl.load(MB+d)[None,:]
   move=tl.load(META+r*5+2,m,0);intent=tl.load(META+r*5+3,m,0)
   token+=tl.load(MIDS+identity[:,None]*64+d[None,:],m[:,None],0.)
   token+=tl.load(MOVES+move[:,None]*64+d[None,:],m[:,None],0.)
   token+=tl.load(INTENTS+intent[:,None]*64+d[None,:],m[:,None],0.)
   x=tl.where(m[:,None],token,x)
  if tl.sum(p.to(tl.int32),0)>0:
   x=tl.where(p[:,None],tl.load(POTIONS+identity[:,None]*64+d[None,:],p[:,None],0.),x)
  x+=tl.load(ROLES+role[:,None]*64+d[None,:],live[:,None],0.)
  tl.store(OUT+r[:,None]*64+d[None,:],x,live[:,None])


@triton.jit
def unpack_entities(X, MAP, OUT, NW, NB, STATUS, EPS: tl.constexpr,
                    C: tl.constexpr, SELECT: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,64)
 index=tl.load(MAP+i,(i<N)&(tl.load(STATUS+7)==0),-1)
 value=tl.load(X+index[:,None]*64+d[None,:],(i[:,None]<N)&(index[:,None]>=0),0.)
 value=norm64(value,NW,NB,EPS)
 # Absent fixed native slots keep norm(0); absent selection queries are zero,
 # matching the original pointer-key bias and required-candidate mask.
 value=tl.where(((i%C>=SELECT)&(index<0))[:,None],0.,value)
 tl.store(OUT+i[:,None]*64+d[None,:],value,i[:,None]<N)


def _packed_blocks(blocks, packed, keys, query, starts, lens, first, second, counts, cap, bcap, qkv=None):
    if qkv is None:
        qkv=torch.empty((cap,192),device=packed.device)
    attention=torch.empty((cap,64),device=packed.device)
    sms=torch.cuda.get_device_properties(packed.device).multi_processor_count
    grid=min(triton.cdiv(cap,32),sms*8)
    for i,b in enumerate(blocks):
        qkv_project[(grid,3)](packed,b.norm1.weight,b.norm1.bias,b.qkv.weight,b.qkv.bias,
            qkv,counts,b.norm1.eps,num_warps=4,num_stages=1)
        for bucket in range(3):
            attend_bucket[(min(sms*32,4*bcap),)](qkv,keys,query,starts,lens,
                first if i==0 else second,attention,counts,i==1,bucket,num_warps=1,num_stages=1)
        mix_feed[(grid,)](attention,packed,query,b.mix.weight,b.mix.bias,b.norm2.weight,
            b.norm2.bias,b.ff1.weight,b.ff1.bias,b.ff2.weight,b.ff2.bias,
            packed if i==0 else attention,counts,i==1,b.norm2.eps,num_warps=4,num_stages=1)
    return attention


@torch.compiler.disable
def frozen_entity_blocks(net, obs, global_token, relics, piles, valid, required, row_count, workspace):
    """Direct compact feature -> input -> query heads, with graph-owned aliasing."""
    if torch.is_grad_enabled() or net.training or len(net.blocks)!=2:
        raise ValueError('compact entity pack requires frozen two-block inference')
    if not global_token.is_cuda or global_token.dtype!=torch.float32:
        raise ValueError('compact entity pack requires CUDA FP32')
    batch=len(global_token);h=obs['hand_ids'].shape[1];m=obs['monster_id'].shape[1]
    p=obs['potions'].shape[1];w=obs['select_card_ids'].shape[1];length=1+h+m+p+4+w
    ns=obs['monster_scalars'].shape[-1];nb=obs['monster_buffs'].shape[-1]
    if ns+nb>64:
        raise ValueError('compact entity feature ABI supports up to 64 monster features')
    def arr(shape,dtype=torch.float32):return torch.empty(shape,device=global_token.device,dtype=dtype)
    def field(key):return obs[key].contiguous()
    hand_ids,monster_ids,potion_ids,valid,required=(x.contiguous() for x in
        (obs['hand_ids'],obs['monster_id'],obs['potions'],valid[:,:w],required[:,:w]))
    lens=arr((batch,4),torch.int64);starts=torch.empty_like(lens);ends=torch.empty_like(lens)
    totals=arr((triton.cdiv(batch,256),4),torch.int64);counts=arr((6,),torch.int32).zero_()
    entity_lens[(batch,)](hand_ids,monster_ids,potion_ids,valid,required,row_count,lens,
        h,m,p,w,triton.next_power_of_2(length),num_warps=4)
    prefix_local[(triton.cdiv(batch,256),)](lens,ends,totals,batch,256)
    prefix_starts[(triton.cdiv(batch,256),)](ends,lens,starts,counts,totals,batch,256,
        triton.next_power_of_2(triton.cdiv(batch,256)))
    cap,qcap,status=workspace
    check_capacity[(1,)](counts,status,cap,qcap)
    packed=arr((cap,64));keys=arr((cap,),torch.bool);query=arr((qcap,),torch.int64)
    qkv=arr((cap,192));features=qkv.flatten()[:cap*64].view(cap,64)
    meta=arr((cap,5),torch.int32)
    bcap=triton.cdiv(cap,16)+batch
    first,second=(arr((bcap,2),torch.int32) for _ in range(2))
    consumers=length-4;mapping=arr((batch,consumers),torch.int64)
    pack_entity_features[(batch,)](global_token.contiguous(),relics.contiguous(),piles.contiguous(),
        hand_ids,field('hand_upgrades'),field('hand_costs'),field('hand_damage'),field('hand_block'),
        field('select_hand_mask'),monster_ids,field('monster_move'),field('monster_intent_cat'),
        field('monster_scalars'),field('monster_buffs'),potion_ids,field('select_card_ids'),
        field('select_card_upgrades'),field('select_card_costs'),field('select_zone'),valid,required,
        row_count,starts,lens,features,meta,keys,query,mapping,first,second,status,
        h,m,p,w,triton.next_power_of_2(length),ns,nb,num_warps=4,enable_fp_fusion=False)
    card,upgrade=obs['_resident_card_projection']
    mid,move,intent,mw=obs['_resident_monster_projection']
    sms=torch.cuda.get_device_properties(global_token.device).multi_processor_count
    entity_input_tokens[(min(triton.cdiv(cap,32),sms*8),)](features,meta,card,upgrade,
        net.hand_context.weight,net.hand_context.bias,net.choice_context.weight,net.choice_context.bias,
        net.choice_zone.weight,mid,move,intent,mw,net.monster_projection.bias,net.potion.weight,
        net.roles.weight,packed,counts,ns+nb,triton.next_power_of_2(ns+nb),
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    # input_tokens consumed every feature before QKV overwrites the same store.
    attention=_packed_blocks(net.blocks,packed,keys,query,starts,lens,first,second,counts,cap,bcap,qkv)
    output=arr((batch,consumers,64))
    unpack_entities[(triton.cdiv(batch*consumers,32),)](attention,mapping,output,net.norm.weight,
        net.norm.bias,status,net.norm.eps,consumers,consumers-w,batch*consumers,32,num_warps=4)
    return output[:,0],dict(hand=output[:,1:1+h],monster=output[:,1+h:1+h+m],
        potion=output[:,1+h+m:1+h+m+p],select=output[:,1+h+m+p:])


@torch.compiler.disable
def frozen_ragged_blocks(blocks, x, present, queries, row_count=None, workspace=None, norm=None):
    """Current entities packed into the original dynamic 2x64 kernel ABI."""
    if (torch.is_grad_enabled() or any(b.training for b in blocks) or len(blocks)!=2
            or not x.is_cuda or x.dtype!=torch.float32 or x.shape[-1]!=64):
        raise ValueError('ragged search requires frozen CUDA FP32 2x64 blocks')
    batch,length,_=x.shape
    keep=torch.empty_like(present)
    keys_dense,queries_dense=torch.empty_like(present),torch.empty_like(queries)
    lens=torch.empty((batch,4),device=x.device,dtype=torch.int64)
    prepare_lens[(batch,)](present,queries,lens if row_count is None else row_count,
        keep,keys_dense,queries_dense,lens,length,triton.next_power_of_2(length),row_count is not None,
        num_warps=4)
    ends,starts=torch.empty_like(lens),torch.empty_like(lens)
    totals=torch.empty((triton.cdiv(batch,256),4),device=x.device,dtype=lens.dtype)
    counts=torch.zeros(6,device=x.device,dtype=torch.int32)
    prefix_local[(triton.cdiv(batch,256),)](lens,ends,totals,batch,256)
    prefix_starts[(triton.cdiv(batch,256),)](ends,lens,starts,counts,totals,batch,256,
        triton.next_power_of_2(triton.cdiv(batch,256)))
    cap,qcap,status=(batch*length,batch*length,counts) if workspace is None else workspace
    if workspace is not None:
        check_capacity[(1,)](counts,status,cap,qcap)
    def arr(shape,dtype=torch.float32):return torch.empty(shape,device=x.device,dtype=dtype)
    packed=arr((cap,64));keys=arr((cap,),torch.bool)
    query=arr((qcap,),torch.int64);slots=arr((qcap,),torch.int64)
    bcap=triton.cdiv(cap,16)+batch
    first,second=(arr((bcap,2),torch.int32) for _ in range(2))
    pack_slots[(batch,)](x,keys_dense,keep,queries_dense,starts,lens,packed,keys,query,slots,
        first,second,status,length,triton.next_power_of_2(length),workspace is not None,num_warps=4)
    attention=_packed_blocks(blocks,packed,keys,query,starts,lens,first,second,counts,cap,bcap)
    # Input entities died at pack_slots; the second mix consumes each compact
    # attention row once. Reuse both stores as in the original frame workspace.
    dense=x.zero_() if norm is None else x.copy_(norm.bias)
    unpack_queries[(triton.cdiv(qcap,32),)](attention,slots,dense,counts,
        counts if norm is None else norm.weight, counts if norm is None else norm.bias,
        0.0 if norm is None else norm.eps, norm is not None,qcap,32,num_warps=4)
    return dense
