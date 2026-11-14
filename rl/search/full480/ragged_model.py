import math
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from ragged_pack import ragged_score

@triton.jit
def projected(x,ln,lnbias,w,bias,out,R,D:tl.constexpr,O:tl.constexpr,M:tl.constexpr,K:tl.constexpr,EPS:tl.constexpr,PREC:tl.constexpr):
 rows=tl.program_id(0)*M+tl.arange(0,M);dims=tl.arange(0,D);cols=tl.program_id(1)*K+tl.arange(0,K)
 a=tl.load(x+rows[:,None]*D+dims[None,:],rows[:,None]<R,0.)
 c=a-tl.sum(a,1)[:,None]/D;norm=c*tl.rsqrt(tl.sum(c*c,1)/D+EPS)[:,None]
 z=norm*tl.load(ln+dims)[None,:]+tl.load(lnbias+dims)[None,:]
 weight=tl.load(w+cols[None,:]*D+dims[:,None],cols[None,:]<O,0.)
 y=tl.dot(z,weight,input_precision=PREC)+tl.load(bias+cols,cols<O,0.)[None,:]
 tl.store(out+rows[:,None]*O+cols[None,:],y,(rows[:,None]<R)&(cols[None,:]<O))

@triton.jit
def fed(x,nw,nb,w1,b1,w2,b2,y,R,M:tl.constexpr,D:tl.constexpr,FH:tl.constexpr,EPS:tl.constexpr,PREC:tl.constexpr):
 r=tl.program_id(0)*M+tl.arange(0,M);d=tl.arange(0,D);h=tl.arange(0,FH)
 inp=tl.load(x+r[:,None]*D+d[None,:],r[:,None]<R,0.)
 mean=tl.sum(inp,1)/D;u=inp-mean[:,None];u=u*tl.rsqrt(tl.sum(u*u,1)/D+EPS)[:,None]
 u=u*tl.load(nw+d)[None,:]+tl.load(nb+d)[None,:]
 up=tl.load(w1+h[None,:]*D+d[:,None])
 hidden=tl.maximum(tl.dot(u,up,input_precision=PREC)+tl.load(b1+h)[None,:],0.)
 down=tl.load(w2+d[None,:]*FH+h[:,None])
 final=inp+(tl.dot(hidden,down,input_precision=PREC)+tl.load(b2+d)[None,:])
 tl.store(y+r[:,None]*D+d[None,:],final,r[:,None]<R)

@triton.jit
def attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H:tl.constexpr,D:tl.constexpr,SECOND:tl.constexpr,KT:tl.constexpr):
 qi=local+tl.arange(0,16);ki=tl.arange(0,KT);di=tl.arange(0,D)
 qtoken=ts+qi
 if SECOND:qtoken=tl.load(query_tokens+qs+qi,qi<nq,0)
 q=tl.load(qkv+qtoken[:,None]*(3*H*D)+h*D+di[None,:],qi[:,None]<nq,0.)
 k=tl.load(qkv+(ts+ki[None,:])*(3*H*D)+H*D+h*D+di[:,None],ki[None,:]<nt,0.)
 live=(ki<nt)&(tl.load(present+ts+ki,ki<nt,False)!=0)
 scores=tl.where(live[None,:],tl.dot(q,k,input_precision='ieee')*(1.0/math.sqrt(D)),-float('inf'))
 probs=tl.exp(scores-tl.max(scores,1)[:,None]);probs=probs/tl.sum(probs,1)[:,None]
 v=tl.load(qkv+(ts+ki[:,None])*(3*H*D)+2*H*D+h*D+di[None,:],ki[:,None]<nt,0.)
 return tl.dot(probs,v,input_precision='ieee')

@triton.jit
def attention(qkv,present,query_tokens,starts,lens,blocks,out,H:tl.constexpr,D:tl.constexpr,SECOND:tl.constexpr):
 task=tl.program_id(0)//H;h=tl.program_id(0)%H
 sid=tl.load(blocks+task*2);local=tl.load(blocks+task*2+1)
 ts=tl.load(starts+sid*4);nt=tl.load(lens+sid*4)
 qs=ts;nq=nt
 if SECOND:qs=tl.load(starts+sid*4+1);nq=tl.load(lens+sid*4+1)
 if nt<=16:y=attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H,D,SECOND,16)
 elif nt<=32:y=attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H,D,SECOND,32)
 elif nt<=64:y=attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H,D,SECOND,64)
 elif nt<=128:y=attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H,D,SECOND,128)
 else:y=attention_tile(qkv,present,query_tokens,ts,qs,nt,nq,local,h,H,D,SECOND,256)
 qi=local+tl.arange(0,16);di=tl.arange(0,D)
 tl.store(out+(qs+qi[:,None])*(H*D)+h*D+di[None,:],y,qi[:,None]<nq)

def project(block,x,precision):
 n,d=x.shape;out=torch.empty((n,3*d),device=x.device,dtype=x.dtype)
 projected[(triton.cdiv(n,32),triton.cdiv(3*d,64))](x,block.norm1.weight,block.norm1.bias,block.qkv.weight,block.qkv.bias,out,
  n,d,3*d,32,64,block.norm1.eps,precision,num_warps=4)
 return out

def feed(block,x,precision):
 n,d=x.shape;out=torch.empty_like(x);fh=block.ff1.weight.shape[0]
 fed[(triton.cdiv(n,32),)](x,block.norm2.weight,block.norm2.bias,block.ff1.weight,block.ff1.bias,block.ff2.weight,block.ff2.bias,out,
  n,32,d,fh,block.norm2.eps,precision,num_warps=4)
 return out

@torch.inference_mode()
def forward(scorer,feat,symbols,slots,present,query_tokens,starts,lens,blocks1,blocks2,
 state,actions,keys,owners,sources,targets,meta,query_map,scores,action_count):
 net=scorer.net;precision=scorer.precision
 assert len(net.blocks)==2 and precision=='tf32x3'
 x=net.input(feat)+net.slots(slots)+scorer.lookup[symbols]
 d=x.shape[1];assert d==64
 for index,block in enumerate(net.blocks):
  h=block.heads;assert h*d//h==d and d//h in (16,32,64)
  qkv=project(block,x,precision)
  blocks=blocks1 if index==0 else blocks2
  n=len(feat) if index==0 else len(query_tokens)
  a=torch.empty((n,d),device=x.device,dtype=x.dtype)
  attention[(len(blocks)*h,)](qkv,present,query_tokens,starts,lens,blocks,a,h,d//h,index==1,num_warps=1)
  residual=x if index==0 else x[query_tokens]
  x=feed(block,residual+block.mix(a),precision)
 x=net.final(x)
 z=net.fc2(F.relu(net.fc1(state)));coef=z@net.projection.weight+net.linear.weight
 token=F.linear(x,scorer.head_weight)
 ragged_score[(triton.cdiv(action_count,16),)](actions,keys,owners,sources,targets,meta,query_map,starts,lens,z,coef,token,
  net.keys.weight,net.bias.weight,scores,action_count,161,10,10,16,num_warps=4)
