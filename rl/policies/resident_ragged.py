"""Persistent ragged 2x64 kernels migrated from the accepted GPU search.

The norm/QKV, attention tile/buckets and mix+FFN below retain the old kernel
arithmetic. New packing adapts current fixed native slots to those contracts.
"""
import torch
import triton
import triton.language as tl

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
def unpack_queries(X, SLOTS, OUT, COUNTS, CAP: tl.constexpr, B: tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,64)
 valid=(i<CAP)&(i<tl.load(COUNTS+1))
 slot=tl.load(SLOTS+i,valid,0)
 value=tl.load(X+i[:,None]*64+d[None,:],valid[:,None],0.)
 tl.store(OUT+slot[:,None]*64+d[None,:],value,valid[:,None])


@torch.compiler.disable
def frozen_ragged_blocks(blocks, x, present, queries, row_count=None, workspace=None):
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
    qkv=arr((cap,192));attention=arr((cap,64))
    sms=torch.cuda.get_device_properties(x.device).multi_processor_count
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
    # Input entities died at pack_slots; the second mix consumes each compact
    # attention row once. Reuse both stores as in the original frame workspace.
    dense=x.zero_()
    unpack_queries[(triton.cdiv(qcap,32),)](attention,slots,dense,counts,qcap,32,num_warps=4)
    return dense
