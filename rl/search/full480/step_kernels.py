"""Stable device compaction and transactional frame accounting."""
import triton
import triton.language as tl

# Control: active, cursor, enabled, error, compact, ticks, applied, choices,
# forced, model_frames, tokens, queries, actions, attention_blocks, frames,
# overflows, first_failed_summary[10], failed_count.
@triton.jit
def compact_local(ALIVE,PREFIX,TOTALS,C,N:tl.constexpr,B:tl.constexpr):
 if tl.load(C+4):
  b=tl.program_id(0);i=b*B+tl.arange(0,B)
  x=(tl.load(ALIVE+i,i<N,0)!=0).to(tl.int32)
  y=tl.cumsum(x,0)
  tl.store(PREFIX+i,y,i<N)
  tl.store(TOTALS+b,tl.sum(x,0))

@triton.jit
def compact_scatter(ALIVE,PREFIX,TOTALS,LIVE,WORLDS,C,N:tl.constexpr,ROOT_BUDGET:tl.constexpr,B:tl.constexpr,CB:tl.constexpr):
 if tl.load(C+4):
  b=tl.program_id(0);i=b*B+tl.arange(0,B);q=tl.arange(0,CB)
  offset=tl.sum(tl.load(TOTALS+q,q<b,0),0)
  valid=(i<N)&(tl.load(ALIVE+i,i<N,0)!=0)
  dst=offset+tl.load(PREFIX+i,i<N,0)-1
  tl.store(LIVE+dst,i,valid)
  tl.store(WORLDS+dst,i//ROOT_BUDGET,valid)

@triton.jit
def compact_finish(TOTALS,C,NB:tl.constexpr,CB:tl.constexpr):
 if tl.load(C+4):
  q=tl.arange(0,CB);n=tl.sum(tl.load(TOTALS+q,q<NB,0),0)
  tl.store(C,n);tl.store(C+1,0);tl.store(C+4,0)

@triton.jit
def frame_prepare(LIVE,WORLDS,OUT_LIVE,OUT_WORLDS,DISPATCH,C,F:tl.constexpr,B:tl.constexpr):
 active=tl.load(C);cursor=tl.load(C+1)
 enabled=(tl.load(C+2)!=0)&(tl.load(C+3)==0)
 count=tl.where(enabled,tl.minimum(F,active-cursor),0).to(tl.int32)
 i=tl.program_id(0)*B+tl.arange(0,B);valid=(i<F)&(i<count)
 job=tl.load(LIVE+cursor+i,valid,0);world=tl.load(WORLDS+cursor+i,valid,0)
 tl.store(OUT_LIVE+i,job,i<F);tl.store(OUT_WORLDS+i,world,i<F)
 if tl.program_id(0)==0:tl.store(DISPATCH,count)

@triton.jit
def frame_commit(SUMMARY,DISPATCH,C):
 n=tl.load(DISPATCH).to(tl.int64)
 if (tl.load(C+2)!=0)&(tl.load(C+3)==0)&(n>0):
  flags=tl.load(SUMMARY+2)
  if flags!=0:
   tl.store(C+3,flags);tl.store(C+15,tl.load(C+15)+1)
   q=tl.arange(0,16);v=tl.load(SUMMARY+q,q<10,0).to(tl.int64)
   tl.store(C+16+q,v,q<10);tl.store(C+26,n)
  else:
   real=tl.load(SUMMARY+9).to(tl.int64);cursor=tl.load(C+1)+n
   tl.store(C+1,cursor);tl.store(C+6,tl.load(C+6)+n)
   tl.store(C+7,tl.load(C+7)+real);tl.store(C+8,tl.load(C+8)+n-real)
   tl.store(C+14,tl.load(C+14)+1)
   if real>0:
    tl.store(C+9,tl.load(C+9)+1)
    tl.store(C+10,tl.load(C+10)+tl.load(SUMMARY+5))
    tl.store(C+11,tl.load(C+11)+tl.load(SUMMARY+6))
    tl.store(C+12,tl.load(C+12)+tl.load(SUMMARY))
    tl.store(C+13,tl.load(C+13)+tl.load(SUMMARY+7)+tl.load(SUMMARY+8))
   if cursor==tl.load(C):
    tl.store(C+4,1);tl.store(C+5,tl.load(C+5)+1)
