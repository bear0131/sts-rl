"""Bounded readonly observation-to-score CUDA graph; apply stays outside."""
from .run_device import artifact_directory
from pathlib import Path
import sys,collections
P=artifact_directory()

from .graph_runtime import *
from .frame_modules import FrameModules
from .frame_kernels import prepare_lens,finalize,prefix_local,prefix_add

class FramePipeline:
 def __init__(self,pool,model,token_capacity=None,query_capacity=None):
  self.pool=pool;self.model=model;self.modules=FrameModules(pool.modules.identity['rules']['abi_layout'])
  self.F=pool.frame_cap;self.requested_ct=triton.next_power_of_2(token_capacity or self.F*32)
  self.requested_cq=triton.next_power_of_2(query_capacity or self.F*8)
  self.graph=None;self.buffers={};self.key=None;self.bound=();self.capacity=(0,0,self.F)
  self.counts=collections.Counter();self.closed=False;self.history=[]
 def signature(self):
  result=[]
  for t in (self.pool.flat.lookup,self.pool.flat.head_weight,*self.pool.net.parameters()):
   try:version=t._version
   except RuntimeError:version=None
   result.append((t.data_ptr(),version))
  return tuple(result)
 def external(self):
  p=self.pool
  return (p.arena,p.deck,p.state,p.bits,p.actions,p.keys,p.source_rows,p.targets,p.ids,p.zones,p.monsters,p.numbers,p.meta,p.chunks,p.allocator,
   *p.inv,p.inv_slots,p.symbols,p.counts,p.canonical_sources,p.canonical_targets,p.entity_slots,p.extra_slots,p.token_order,p.query_order,p.query_map,p.input_map,p.geometry,p.scores)
 def fingerprint(self):
  p=self.pool
  return (p.I,p.action_cap,p.entity_cap,self.requested_ct,self.requested_cq,tuple(t.data_ptr() for t in self.external()),
   p.flat.lookup.data_ptr(),p.flat.head_weight.data_ptr(),tuple(t.data_ptr() for t in p.net.parameters()))
 def release(self):
  self.pool._frame_ready=False
  if self.graph is not None:
   torch.cuda.synchronize();self.graph.reset();self.graph=None;self.counts['evictions']+=1
  self.model.release();self.buffers={};self.bound=();self.key=None;self.capacity=(0,0,self.F)
 def allocate(self,key):
  self.release();p=self.pool;ct=self.requested_ct;cq=self.requested_cq;F=self.F
  def arr(shape,dtype=torch.float32):return torch.empty(shape,dtype=dtype,device='cuda')
  # Input features die after input_tokens, before the first QKV projection.
  # The graph owns this shared storage; every replay packs fresh features first.
  token_workspace=arr((ct,192))
  self.buffers=dict(features=token_workspace.reshape(-1)[:ct*128].view(ct,128),symbols=arr((ct,),torch.int64),slots=arr((ct,),torch.int64),present=arr((ct,),torch.bool),
   queries=arr((cq,),torch.int64),blocks1=arr((triton.cdiv(ct,16)+F,2),torch.int32),blocks2=arr((triton.cdiv(cq,16)+F,2),torch.int32),
   owners=arr((p.action_cap,),torch.int32),live=arr((F,),torch.int32),worlds=arr((F,),torch.int32),dispatch=arr((1,),torch.int32),
   normal_rows=arr((F,8),torch.int32),totals=arr((5,),torch.int32),queues=arr((2*F,),torch.int32),queue_counts=arr((2,),torch.int32),summary=arr((10,),torch.int32))
  w=self.buffers
  self.model.allocate(key,(w['features'],w['symbols'],w['slots'],w['present'],w['queries'],w['blocks1'],w['blocks2'],
   p.state,p.actions,p.keys,w['owners'],p.canonical_sources,p.canonical_targets,p.meta,p.query_map,p.scores),ct,cq,F,qkv_workspace=token_workspace)
  self.bound=self.external();self.key=key;self.capacity=(ct,cq,F);self.counts['allocations']+=1
 def execute(self):
  p=self.pool;w=self.buffers;m=self.model.buffers;F=self.F
  p.allocator.zero_();w['totals'].zero_();w['queue_counts'].zero_()
  observer_args=[p.arena,w['worlds'],w['live'],p.deck,p.state,p.bits,p.actions,p.keys,p.source_rows,p.targets,
   p.ids,p.zones,p.monsters,p.numbers,p.meta,p.chunks,p.allocator,w['dispatch']]
  shape=[0,p.m,p.action_cap,p.entity_cap]
  self.modules.launch('frame_observe_special',observer_args,shape,triton.cdiv(F,32))
  self.modules.launch('normal_prepare',observer_args+[w['normal_rows']],shape,triton.cdiv(F,32))
  self.modules.launch('normal_features',observer_args+[w['normal_rows']],shape,min(F,self.model.sms*16),threads=64)
  tensors=[w['worlds'],p.ids,p.zones,p.monsters,p.numbers,p.meta,p.chunks]+p.inv+[p.inv_slots,
   p.source_rows,p.targets,p.canonical_sources,p.canonical_targets,p.query_map,
   w['features'],w['symbols'],w['slots'],w['present'],w['queries'],w['blocks1'],w['blocks2'],w['owners'],
   m['starts'],m['lens'],p.scores,p.allocator,w['dispatch'],w['totals'],w['queues'],w['queue_counts']]
  for name,cap in [('pack_short',32),('pack_medium',64),('pack_long',161)]:
   self.modules.launch(name,tensors,[p.m,p.I,161,F,self.capacity[0],self.capacity[1],len(w['blocks1']),len(w['blocks2'])],
    min(F,self.model.sms*8),threads=256,shared=cap*128*4)
  self.modules.launch('packed_finalize',[p.allocator,w['totals'],w['dispatch'],m['counts'],w['summary']],[],1)
  self.model.execute()
 @torch.inference_mode()
 def run(self,base,count):
  assert not self.closed and 0<count<=self.F
  p=self.pool
  for attempt in range(8):
   key=self.fingerprint()
   if key!=self.key:self.allocate(key)
   w=self.buffers
   w['live'][:count].copy_(p.live[base:base+count]);w['worlds'][:count].copy_(p.logical_ids[base:base+count])
   w['dispatch'].fill_(count)
   if self.graph is None:
    self.execute();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    try:
     with torch.cuda.graph(g):self.execute()
    except BaseException:
     g.reset();raise
    self.graph=g;self.counts['captures']+=1
   self.graph.replay();self.counts['replays']+=1
   values=list(map(int,w['summary'].cpu().tolist()))
   a,e,flags,me,ma,nt,nq,nb1,nb2,real=values
   assert flags>=0 and flags&~63==0 and a>0 and e>=0 and 0<=real<=count,values
   if not flags:
    assert nt>=real and nq>=real and nq<=nt and (real>0 or nt==nq==nb1==nb2==0)
    self.counts['valid_frames']+=1
    return values
   row=dict(base=base,count=count,attempt=attempt,summary=values,capacity=list(self.capacity),action_capacity=p.action_cap,entity_capacity=p.entity_cap)
   self.history.append(row);self.counts['overflow_retries']+=1
   self.release()
   if a>p.action_cap:p.allocate_actions(triton.next_power_of_2(a))
   if e>p.entity_cap:p.allocate_entities(triton.next_power_of_2(e))
   if nt>self.requested_ct:self.requested_ct=triton.next_power_of_2(nt)
   if nq>self.requested_cq:self.requested_cq=triton.next_power_of_2(nq)
   assert not flags&48 or nt>0 or nq>0
  raise RuntimeError('Frame graph overflow retry budget exhausted')
 def statistics(self):
  return dict(self.counts,cached=int(self.graph is not None),capacity=list(self.capacity),closed=self.closed,
   kernels=self.model.kernel_metadata,workspace_bytes=self.model.workspace_bytes(self.buffers.values()),workspace_alias_version=1,
   aliased_input_bytes=0 if not self.buffers else self.buffers['features'].numel()*self.buffers['features'].element_size(),
   overflow_history=self.history,modules_closed=self.modules.closed,identity=self.modules.identity)
 def close(self):
  if self.closed:return
  self.release();self.model.close();self.modules.close();self.closed=True
