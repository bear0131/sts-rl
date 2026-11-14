"""FP32 complete ragged model, persistent task grids and reusable outputs."""
import math
import torch
import triton
import triton.language as tl
from ragged_model import attention_tile

@triton.jit
def norm64(x,W,B,EPS:tl.constexpr):
 d=tl.arange(0,64);u=x-tl.sum(x,1)[:,None]/64.
 return u*tl.rsqrt(tl.sum(u*u,1)/64.+EPS)[:,None]*tl.load(W+d)[None,:]+tl.load(B+d)[None,:]

@triton.jit
def input_tokens(FEAT,SYMBOL,SLOT,W,B,SE,LE,X,COUNTS):
 n=tl.load(COUNTS);d=tl.arange(0,128);o=tl.arange(0,64)
 weight=tl.load(W+o[None,:]*128+d[:,None])
 for block in range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0)):
  r=block*32+tl.arange(0,32);live=r<n
  a=tl.load(FEAT+r[:,None]*128+d[None,:],live[:,None],0.)
  value=tl.dot(a,weight,input_precision='ieee')+tl.load(B+o)[None,:]
  sy=tl.load(SYMBOL+r,live,0);sl=tl.load(SLOT+r,live,0)
  value=value+tl.load(SE+sl[:,None]*64+o[None,:],live[:,None],0.)
  value=value+tl.load(LE+sy[:,None]*64+o[None,:],live[:,None],0.)
  tl.store(X+r[:,None]*64+o[None,:],value,live[:,None])

@triton.jit
def qkv_project(X,NW,NB,W,B,Y,COUNTS,EPS:tl.constexpr):
 n=tl.load(COUNTS);d=tl.arange(0,64);cols=tl.program_id(1)*64+d
 weight=tl.load(W+cols[None,:]*64+d[:,None])
 for block in range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0)):
  r=block*32+tl.arange(0,32)
  x=tl.load(X+r[:,None]*64+d[None,:],r[:,None]<n,0.)
  z=norm64(x,NW,NB,EPS)
  y=tl.dot(z,weight,input_precision='ieee')+tl.load(B+cols)[None,:]
  tl.store(Y+r[:,None]*192+cols[None,:],y,r[:,None]<n)

@triton.jit
def attend(QKV,PRESENT,QUERY,STARTS,LENS,BLOCKS,OUT,COUNTS,SECOND:tl.constexpr):
 nb=tl.load(COUNTS+(3 if SECOND else 2))
 for taskhead in range(tl.program_id(0),nb*4,tl.num_programs(0)):
  task=taskhead//4;h=taskhead%4;sid=tl.load(BLOCKS+task*2);local=tl.load(BLOCKS+task*2+1)
  ts=tl.load(STARTS+sid*4);nt=tl.load(LENS+sid*4);qs=ts;nq=nt
  if SECOND:qs=tl.load(STARTS+sid*4+1);nq=tl.load(LENS+sid*4+1)
  if nt<=16:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,16)
  elif nt<=32:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,32)
  elif nt<=64:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,64)
  elif nt<=128:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,128)
  else:y=attention_tile(QKV,PRESENT,QUERY,ts,qs,nt,nq,local,h,4,16,SECOND,256)
  qi=local+tl.arange(0,16);di=tl.arange(0,16)
  tl.store(OUT+(qs+qi[:,None])*64+h*16+di[None,:],y,qi[:,None]<nq)

@triton.jit
def mix_feed(A,X,QUERY,MW,MB,NW,NB,F1,B1,F2,B2,OUT,COUNTS,SECOND:tl.constexpr,EPS:tl.constexpr):
 n=tl.load(COUNTS+(1 if SECOND else 0));d=tl.arange(0,64);h=tl.arange(0,128)
 for block in tl.range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0),disable_licm=True):
  r=block*32+tl.arange(0,32);xr=r
  if SECOND:xr=tl.load(QUERY+r,r<n,0)
  a=tl.load(A+r[:,None]*64+d[None,:],r[:,None]<n,0.)
  residual=tl.load(X+xr[:,None]*64+d[None,:],r[:,None]<n,0.)
  wm=tl.load(MW+d[None,:]*64+d[:,None])
  mixed=tl.dot(a,wm,input_precision='ieee')+tl.load(MB+d)[None,:]
  residual=residual+mixed;v=norm64(residual,NW,NB,EPS)
  w1=tl.load(F1+h[None,:]*64+d[:,None])
  hidden=tl.maximum(tl.dot(v,w1,input_precision='ieee')+tl.load(B1+h)[None,:],0.)
  w2=tl.load(F2+d[None,:]*128+h[:,None])
  out=residual+(tl.dot(hidden,w2,input_precision='ieee')+tl.load(B2+d)[None,:])
  tl.store(OUT+r[:,None]*64+d[None,:],out,r[:,None]<n)

@triton.jit
def final_head(X,NW,NB,W,Y,COUNTS,EPS:tl.constexpr):
 n=tl.load(COUNTS+1);d=tl.arange(0,64);o=tl.arange(0,64)
 w=tl.load(W+o[None,:]*64+d[:,None],o[None,:]<34,0.)
 for block in range(tl.program_id(0),tl.cdiv(n,32),tl.num_programs(0)):
  r=block*32+tl.arange(0,32)
  x=tl.load(X+r[:,None]*64+d[None,:],r[:,None]<n,0.);v=norm64(x,NW,NB,EPS)
  y=tl.dot(v,w,input_precision='ieee')
  tl.store(Y+r[:,None]*34+o[None,:],y,(r[:,None]<n)&(o[None,:]<34))

@triton.jit
def state_head(STATE,W1,B1,W2,B2,PROJ,LIN,Z,COEF,COUNTS,H:tl.constexpr,HH:tl.constexpr):
 n=tl.load(COUNTS+4);d=tl.arange(0,32);h=tl.arange(0,HH)
 w1=tl.load(W1+h[None,:]*32+d[:,None],h[None,:]<H,0.)
 w2=tl.load(W2+d[None,:]*H+h[:,None],h[:,None]<H,0.)
 wp=tl.load(PROJ+d[:,None]*24+d[None,:],d[None,:]<24,0.)
 for block in range(tl.program_id(0),tl.cdiv(n,16),tl.num_programs(0)):
  r=block*16+tl.arange(0,16)
  x=tl.load(STATE+r[:,None]*32+d[None,:],r[:,None]<n,0.)
  hidden=tl.maximum(tl.dot(x,w1,input_precision='ieee')+tl.load(B1+h,h<H,0.)[None,:],0.)
  z=tl.dot(hidden,w2,input_precision='ieee')+tl.load(B2+d)[None,:]
  coef=tl.dot(z,wp,input_precision='ieee')+tl.load(LIN+d,d<24,0.)[None,:]
  tl.store(Z+r[:,None]*32+d[None,:],z,r[:,None]<n)
  tl.store(COEF+r[:,None]*24+d[None,:],coef,(r[:,None]<n)&(d[None,:]<24))

@triton.jit
def action_head(acts,keys,owners,source,target,meta,query_map,starts,lens,z,coef,token,embedding,bias,out,COUNTS,
 T:tl.constexpr,K:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 N=tl.load(COUNTS+5)
 for block in range(tl.program_id(0),tl.cdiv(N,B),tl.num_programs(0)):
  row=block*B+tl.arange(0,B);valid=row<N
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
