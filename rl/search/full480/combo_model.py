"""One bounded CUDA graph owns every captured pointer until explicit release."""
import collections,os
import torch
import triton
from .combo_kernels import (input_tokens,qkv_project,mix_feed,final_head,state_head,
 action_head,attend_bucket,deck_global,deck_subset,deck_actions)

class DeckWeights:
 """Current heads around the original fused two-block execution."""
 consumer='deck-latent-combat-v1'
 precision='tf32x3'
 def __init__(self,core):
  if core.combat_encoder!='transformer64x2':raise ValueError('full480 requires transformer64x2')
  self.net=core.combat;self.version=None;self.refresh()
 @torch.no_grad()
 def refresh(self):
  n=self.net
  if n.training or next(n.parameters()).dtype!=torch.float32:raise ValueError('full480 requires frozen FP32 evaluation weights')
  self.lookup=n.inference_card_table()
  zero=n.norm.weight.new_zeros
  head=torch.cat((torch.eye(64,device=n.norm.weight.device),n.pointer_key.weight,
   n.complete_head.source_query.weight,n.complete_head.target_key.weight,n.complete_head.subset.hand.weight))
  bias=torch.cat((zero(64),n.pointer_key.bias,zero(128),n.complete_head.subset.hand.bias))
  state=torch.cat((n.pointer_query.weight,n.complete_head.global_query.weight,
   n.complete_head.subset.state.weight,n.end.weight,n.value_head[0].weight))
  sbias=torch.cat((n.pointer_query.bias,zero(64),n.complete_head.subset.state.bias,n.end.bias,n.value_head[0].bias))
  for key,value in dict(head_weight=head,head_bias=bias,state_weight=state,state_bias=sbias).items():
   old=getattr(self,key,None)
   if old is None:setattr(self,key,value.detach().clone())
   else:old.copy_(value)
  self.version=tuple((id(p),p._version) for p in n.parameters())
 def validate(self):
  if self.version!=tuple((id(p),p._version) for p in self.net.parameters()):
   raise RuntimeError('full480 weights changed before collection drain and refresh')
 def tensors(self):return (self.lookup,self.head_weight,self.head_bias,self.state_weight,self.state_bias)

class GraphScorer:
 def __init__(self,flat):
  self.variant='attention-dotall'
  self.flat=flat;self.net=flat.net;self.graph=None;self.bound=None;self.buffers={};self.key=None
  self.capacity=(0,0,0);self.counts=collections.Counter();self.closed=False;self.weights=();self.kernel_metadata={}
  self.sms=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
  assert len(self.net.blocks)==2 and flat.precision=='tf32x3'
  self.deck=getattr(flat,'consumer',None)=='deck-latent-combat-v1';self.deck_inputs=None
  if not self.deck:
   assert self.net.input.weight.shape==(64,128) and self.net.fc2.weight.shape[0]==32
   assert flat.head_weight.shape==(34,64)
  else:assert flat.head_weight.shape==(288,64)
  norm=self.net.norm if self.deck else self.net.final
  for b in self.net.blocks:
   assert b.heads==4 and b.ff1.weight.shape==(128,64) and b.norm1.eps==b.norm2.eps==norm.eps
 def bind_deck(self,legal,selected,remaining):
  assert self.deck and legal.shape[1]==198 and selected.shape==(len(legal),10)
  assert all(t.is_cuda and t.is_contiguous() for t in (legal,selected,remaining))
  inputs=(legal,selected,remaining)
  if self.deck_inputs is not None and tuple(t.data_ptr() for t in inputs)!=tuple(t.data_ptr() for t in self.deck_inputs):self.release()
  self.deck_inputs=inputs
 def release(self):
  if self.graph is not None:
   torch.cuda.synchronize();self.graph.reset();self.graph=None;self.counts['evictions']+=1
  self.bound=None;self.buffers={};self.key=None;self.capacity=(0,0,0);self.weights=()
 def close(self):
  self.release();self.deck_inputs=None;self.closed=True
 def workspace_bytes(self,extra=()):
  storages={}
  for t in (*self.buffers.values(),*extra):
   st=t.untyped_storage();storages[(str(t.device),st.data_ptr())]=st.nbytes()
  return sum(storages.values())
 def statistics(self):
  return dict(self.counts,variant=self.variant,cached=int(self.graph is not None),capacity=list(self.capacity),closed=self.closed,kernels=self.kernel_metadata,workspace_bytes=self.workspace_bytes())
 def grid(self,cap,tile=32,warps=4):
  return (max(1,min(triton.cdiv(cap,tile),self.sms*(32 if warps==1 else 8))),)
 def allocate(self,key,bound,nt,nq,ns,*,qkv_workspace=None):
  ct=triton.next_power_of_2(max(1,nt))
  if qkv_workspace is not None:
   feat=bound[0]
   assert qkv_workspace.is_cuda and qkv_workspace.dtype==torch.float32 and qkv_workspace.device==feat.device
   assert qkv_workspace.is_contiguous() and tuple(qkv_workspace.shape)==(ct,192)
   assert feat.dtype==torch.float32 and feat.is_contiguous() and tuple(feat.shape)==(ct,128)
   assert feat.data_ptr()==qkv_workspace.data_ptr() and feat.untyped_storage().data_ptr()==qkv_workspace.untyped_storage().data_ptr()
   assert qkv_workspace.untyped_storage().nbytes()>=ct*192*4
  self.release();self.key=key;self.bound=bound;self.weights=(*(self.flat.tensors() if self.deck else (self.flat.lookup,self.flat.head_weight)),*self.net.parameters())
  self.capacity=tuple(triton.next_power_of_2(max(1,n)) for n in [nt,nq,ns]);ct,cq,cs=self.capacity
  def arr(shape,dtype=torch.float32):return torch.empty(shape,dtype=dtype,device='cuda')
  self.buffers=dict(x=arr((ct,64)),qkv=arr((ct,192)) if qkv_workspace is None else qkv_workspace,attention=arr((ct,64)),
   second=arr((cq,64)),token=arr((cq,288 if self.deck else 34)),z=arr((cs,225 if self.deck else 32)),coef=arr((cs,24)),
   starts=arr((cs,4),torch.int64),lens=arr((cs,4),torch.int64),counts=arr((6,),torch.int32))
  self.counts['allocations']+=1
 def launch(self,name,kernel,grid,*args,warps=4):
  compiled=kernel[grid](*args,num_warps=warps,num_stages=1)
  self.kernel_metadata[name]=dict(registers=compiled.n_regs,spills=compiled.n_spills,shared_bytes=compiled.metadata.shared)
 def execute(self):
  (feat,symbols,slots,present,query,blocks1,blocks2,state,actions,keys,owners,sources,targets,meta,query_map,scores)=self.bound
  net=self.net;w=self.buffers;c=w['counts'];ct,cq,cs=self.capacity
  if self.deck:
   self.launch('input',input_tokens,self.grid(ct),feat,symbols,slots,feat,feat,feat,feat,w['x'],c,True,warps=4)
  else:self.launch('input',input_tokens,self.grid(ct),feat,symbols,slots,net.input.weight,net.input.bias,net.slots.weight,self.flat.lookup,w['x'],c,warps=4)
  for i,b in enumerate(net.blocks):
   qgrid=self.grid(ct);qgrid=(qgrid[0],3)
   self.launch('qkv'+str(i),qkv_project,qgrid,w['x'],b.norm1.weight,b.norm1.bias,b.qkv.weight,b.qkv.bias,w['qkv'],c,b.norm1.eps,warps=4)
   blocks=blocks1 if i==0 else blocks2
   # The grid is capacity-independent; dynamic bounds execute only actual attention tasks.
   for bucket in range(3):
    self.launch('attention'+str(i)+'_bin'+str(bucket),attend_bucket,(min(self.sms*32,4*(triton.cdiv(ct if i==0 else cq,16)+cs)),),w['qkv'],present,query,w['starts'],w['lens'],blocks,w['attention'],c,i==1,bucket,warps=1)
   out=w['x'] if i==0 else w['second']
   self.launch('mix_feed'+str(i),mix_feed,self.grid(ct if i==0 else cq),w['attention'],w['x'],query,b.mix.weight,b.mix.bias,
    b.norm2.weight,b.norm2.bias,b.ff1.weight,b.ff1.bias,b.ff2.weight,b.ff2.bias,out,c,i==1,b.norm2.eps,warps=4)
  if self.deck:
   assert self.deck_inputs is not None
   legal,selected,remaining=self.deck_inputs;flat=self.flat;subset=net.complete_head.subset
   self.launch('final',final_head,(self.grid(cq)[0],5),w['second'],net.norm.weight,net.norm.bias,flat.head_weight,w['token'],c,net.norm.eps,288,flat.head_bias,warps=4)
   self.launch('state',deck_global,(self.grid(cs,16)[0],4),w['token'],w['starts'],query_map,meta,flat.state_weight,flat.state_bias,w['z'],c,warps=4)
   self.launch('subset',deck_subset,(min(cs,self.sms*8),),w['token'],w['starts'],query_map,meta,w['z'],legal,selected,remaining,
    subset.prefix.weight,subset.choice_key.weight,subset.stop.weight,subset.stop.bias,subset.completion_counts,
    net.value_head[2].weight,net.value_head[2].bias,w['coef'],c,warps=4)
   self.launch('action',deck_actions,self.grid(len(actions),16),keys,owners,meta,query_map,w['starts'],w['z'],w['coef'],w['token'],legal,scores,c,warps=4)
   return
  self.launch('final',final_head,self.grid(cq),w['second'],net.final.weight,net.final.bias,self.flat.head_weight,w['token'],c,net.final.eps,warps=4)
  h=net.fc1.weight.shape[0]
  self.launch('state',state_head,self.grid(cs,16),state,net.fc1.weight,net.fc1.bias,net.fc2.weight,net.fc2.bias,net.projection.weight,net.linear.weight,
   w['z'],w['coef'],c,h,triton.next_power_of_2(h),warps=4)
  self.launch('action',action_head,self.grid(len(actions),16),actions,keys,owners,sources,targets,meta,query_map,w['starts'],w['lens'],w['z'],w['coef'],w['token'],
   net.keys.weight,net.bias.weight,scores,c,161,10,10,16,warps=4)
 @torch.inference_mode()
 def __call__(self,flat,feat,symbols,slots,present,query,starts,lens,blocks1,blocks2,
              state,actions,keys,owners,sources,targets,meta,query_map,scores,action_count,*,graph=True):
  assert not self.closed and flat is self.flat
  nt,nq,ns=len(feat),len(query),len(state)
  assert nt>0 and nq>0 and nq<=nt and ns>0 and action_count>0
  assert starts.shape==lens.shape==(ns,4) and starts.dtype==lens.dtype==torch.int64
  bound=(feat,symbols,slots,present,query,blocks1,blocks2,state,actions,keys,owners,sources,targets,meta,query_map,scores)
  assert all(t.is_cuda and t.is_contiguous() for t in bound)
  if self.deck:flat.validate()
  key=tuple(t.data_ptr() for t in bound)+(flat.lookup.data_ptr(),flat.head_weight.data_ptr())+tuple(t.data_ptr() for t in self.net.parameters())
  if self.deck:key+=tuple(t.data_ptr() for t in (*flat.tensors(),*self.deck_inputs))
  if self.key!=key or any(n>cap for n,cap in zip((nt,nq,ns),self.capacity)):
   self.allocate(key,bound,nt,nq,ns)
  w=self.buffers
  w['starts'][:ns].copy_(starts);w['lens'][:ns].copy_(lens)
  w['counts'].copy_(torch.tensor([nt,nq,len(blocks1),len(blocks2),ns,action_count],dtype=torch.int32))
  if graph:
   if self.graph is None:
    self.execute();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    try:
     with torch.cuda.graph(g):self.execute()
    except BaseException:
     g.reset();raise
    self.graph=g;self.counts['captures']+=1
   self.graph.replay();self.counts['replays']+=1
  else:self.execute();self.counts['eager_calls']+=1
  self.counts['calls']+=1;self.counts['actual_tokens']+=nt;self.counts['actual_queries']+=nq
