import triton
import triton.language as tl
@triton.jit
def prepare_lens(META,GEOMETRY,DISPATCH,ALLOC,LENS,REAL,SCORES,F:tl.constexpr,B:tl.constexpr):
 r=tl.program_id(0)*B+tl.arange(0,B)
 n=tl.load(DISPATCH);valid=(r<F)&(r<n)&(tl.load(ALLOC+2)==0)
 na=tl.load(META+r*4,valid,0);choice=valid&(na>1)
 nt=tl.load(GEOMETRY+r*4,choice,0).to(tl.int64)
 nq=tl.load(GEOMETRY+r*4+1,choice,0).to(tl.int64)
 j=tl.arange(0,4)
 v=tl.where(j[None,:]==0,nt[:,None],tl.where(j[None,:]==1,nq[:,None],tl.where(j[None,:]==2,(nt[:,None]+15)//16,(nq[:,None]+15)//16)))
 tl.store(LENS+r[:,None]*4+j[None,:],v,r[:,None]<F)
 tl.atomic_add(REAL,tl.sum(choice.to(tl.int32),0))
 offset=tl.load(META+r*4+2,valid&(na==1),0)
 tl.store(SCORES+offset,0.,valid&(na==1))
@triton.jit
def finalize(ENDS,LENS,STARTS,ALLOC,REAL,DISPATCH,COUNTS,SUMMARY,F:tl.constexpr,CT:tl.constexpr,CQ:tl.constexpr,CB1:tl.constexpr,CB2:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 e=tl.load(ENDS+i,i<F*4,0);l=tl.load(LENS+i,i<F*4,0)
 tl.store(STARTS+i,e-l,i<F*4)
 if tl.program_id(0)==0:
  a=tl.load(ALLOC);entities=tl.load(ALLOC+1);flags=tl.load(ALLOC+2)
  me=tl.load(ALLOC+3);ma=tl.load(ALLOC+4)
  nt=tl.load(ENDS+(F-1)*4);nq=tl.load(ENDS+(F-1)*4+1)
  nb1=tl.load(ENDS+(F-1)*4+2);nb2=tl.load(ENDS+(F-1)*4+3)
  real=tl.load(REAL);n=tl.load(DISPATCH)
  flags=flags|tl.where(nt>CT,4,0)|tl.where(nq>CQ,8,0)|tl.where(nb1>CB1,16,0)|tl.where(nb2>CB2,32,0)
  j=tl.arange(0,16)
  values=tl.where(j==0,a,tl.where(j==1,entities,tl.where(j==2,flags,tl.where(j==3,me,tl.where(j==4,ma,tl.where(j==5,nt,tl.where(j==6,nq,tl.where(j==7,nb1,tl.where(j==8,nb2,real)))))))))
  tl.store(SUMMARY+j,values,j<10)
  good=(flags==0)&(real>0)
  values=tl.where(j==0,nt,tl.where(j==1,nq,tl.where(j==2,nb1,tl.where(j==3,nb2,tl.where(j==4,n,a)))))
  tl.store(COUNTS+j,tl.where(good,values,0),j<6)

@triton.jit
def prefix_local(LENS,ENDS,TOTALS,F:tl.constexpr,B:tl.constexpr):
 block=tl.program_id(0);r=block*B+tl.arange(0,B);j=tl.arange(0,4)
 x=tl.load(LENS+r[:,None]*4+j[None,:],r[:,None]<F,0)
 y=tl.cumsum(x,0)
 tl.store(ENDS+r[:,None]*4+j[None,:],y,r[:,None]<F)
 total=tl.sum(x,0)
 tl.store(TOTALS+block*4+j,total)
@triton.jit
def prefix_add(ENDS,TOTALS,F:tl.constexpr,B:tl.constexpr,CB:tl.constexpr):
 block=tl.program_id(0);r=block*B+tl.arange(0,B);j=tl.arange(0,4);b=tl.arange(0,CB)
 total=tl.load(TOTALS+b[:,None]*4+j[None,:],b[:,None]<block,0)
 prefix=tl.sum(total,0)
 y=tl.load(ENDS+r[:,None]*4+j[None,:],r[:,None]<F,0)
 tl.store(ENDS+r[:,None]*4+j[None,:],y+prefix[None,:],r[:,None]<F)
