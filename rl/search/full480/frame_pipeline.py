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
  self.pool=pool;self.model=model;self.modules=FrameModules(pool.modules.identity['rules']['abi_layout'],deck=pool.deck_consumer,
   schema=pool.flat.input_tables.header() if pool.deck_consumer else None)
  self.F=pool.frame_cap;self.requested_ct=triton.next_power_of_2(token_capacity or self.F*32)
  self.requested_cq=triton.next_power_of_2(query_capacity or self.F*8)
  self.graph=None;self.buffers={};self.key=None;self.bound=();self.capacity=(0,0,self.F)
  self.counts=collections.Counter();self.closed=False;self.history=[]
  self.leaf_inference=None;self.owns_leaf_inference=False
 def signature(self):
  if self.pool.deck_consumer:self.pool.flat.validate();self.pool.flat.input_tables.validate()
  result=[]
  for t in (self.pool.flat.lookup,self.pool.flat.head_weight,*self.pool.net.parameters()):
   try:version=t._version
   except RuntimeError:version=None
   result.append((t.data_ptr(),version))
  if p:=getattr(self.pool.flat,'input_tables',None):
   for t in p.tables.values():result.append((t.data_ptr(),t._version))
  return tuple(result)
 def external(self):
  p=self.pool
  result=(p.arena,p.deck,p.state,p.bits,p.actions,p.keys,p.source_rows,p.targets,p.ids,p.zones,p.monsters,p.numbers,p.meta,p.chunks,p.allocator,
   *p.inv,p.inv_slots,p.symbols,p.counts,p.canonical_sources,p.canonical_targets,p.entity_slots,p.extra_slots,p.token_order,p.query_order,p.query_map,p.input_map,p.geometry,p.scores)
  if p.deck_consumer:result+=(p.selection_prefix,p.flat.input_tables.bindings,p.root_inputs,p.root_relics,p.root_counters,p.chain_inputs,p.value_mode,*p.flat.tensors())
  return result
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
  if p.deck_consumer:
   from env.core.run_env import _nosl_compact_lut
   layout=_nosl_compact_lut()['layout']
   w.update(raw=arr((F,layout['scalar_width'])),cat=arr((F,layout['categorical_width']),torch.int32),
    legal=arr((F,198),torch.int32),selected=arr((F,10)),remaining=arr((F,),torch.int32))
   self.model.bind_deck(w['legal'],w['selected'],w['remaining'])
  self.model.allocate(key,(w['features'],w['symbols'],w['slots'],w['present'],w['queries'],w['blocks1'],w['blocks2'],
   p.state,p.actions,p.keys,w['owners'],p.canonical_sources,p.canonical_targets,p.meta,p.query_map,p.scores),ct,cq,F,qkv_workspace=token_workspace)
  self.bound=self.external();self.key=key;self.capacity=(ct,cq,F);self.counts['allocations']+=1
 def execute(self):
  p=self.pool;w=self.buffers;m=self.model.buffers;F=self.F
  p.allocator.zero_();w['totals'].zero_();w['queue_counts'].zero_()
  observer_args=[p.arena,w['worlds'],w['live'],p.deck,p.state,p.bits,p.actions,p.keys,p.source_rows,p.targets,
   p.ids,p.zones,p.monsters,p.numbers,p.meta,p.chunks,p.allocator,w['dispatch']]
  if p.deck_consumer:
   observer_args+=[p.selection_prefix,p.flat.input_tables.bindings,w['raw'],w['cat'],w['legal'],w['selected'],w['remaining'],p.root_inputs,p.root_relics,p.root_counters,p.chain_inputs,p.value_mode]
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
 def run(self,base,count,*,selected=None,worlds=None,value_only=False):
  assert not self.closed and 0<count<=self.F
  p=self.pool
  if p.deck_consumer:p.flat.validate();p.flat.input_tables.validate()
  if p.deck_consumer:p.value_mode.fill_(int(value_only))
  for attempt in range(8):
   key=self.fingerprint()
   if key!=self.key:self.allocate(key)
   w=self.buffers
   w['live'][:count].copy_(p.live[base:base+count] if selected is None else selected)
   w['worlds'][:count].copy_(p.logical_ids[base:base+count] if worlds is None else worlds)
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
   assert flags>=0 and flags&~127==0 and a>0 and e>=0 and 0<=real<=count,values
   if not flags:
    assert nt>=real and nq>=real and nq<=nt and (real>0 or nt==nq==nb1==nb2==0)
    self.counts['valid_frames']+=1
    return values
   row=dict(base=base,count=count,attempt=attempt,summary=values,capacity=list(self.capacity),action_capacity=p.action_cap,entity_capacity=p.entity_cap)
   self.history.append(row);self.counts['overflow_retries']+=1
   self.release()
   if flags&64:p.flat.input_tables.grow_upgrades(me)
   if a>p.action_cap:p.allocate_actions(triton.next_power_of_2(a))
   if e>p.entity_cap:p.allocate_entities(triton.next_power_of_2(e))
   if nt>self.requested_ct:self.requested_ct=triton.next_power_of_2(nt)
   if nq>self.requested_cq:self.requested_cq=triton.next_power_of_2(nq)
   assert not flags&48 or nt>0 or nq>0
  raise RuntimeError('Frame graph overflow retry budget exhausted')
 def completion_inference(self):
  if self.leaf_inference is None:
   self.leaf_inference=getattr(self.pool.flat,'inference',None)
   if self.leaf_inference is None:
    from rl.search.rollout_inference import UpperPolicyInference
    self.leaf_inference=UpperPolicyInference(self.pool.flat.core,'cuda');self.owns_leaf_inference=True
  return self.leaf_inference
 @torch.inference_mode()
 def complete_values(self,handles):
  """Score finished original pages before reap releases their leases."""
  p=self.pool;inference=self.completion_inference()
  slots=[h.slot for h in handles]
  jobs=np.concatenate([s*p.budget_cap+np.arange(p.bank[s]['budget']) for s in slots])
  selected=p.upload(jobs,np.int32);owners=p.job_roots[selected.long()]
  outcomes=p.outcomes[selected.long()].cpu().numpy()
  p.values[selected.long()]=0.
  roots={s:i for i,s in enumerate(slots)}
  context_data={key:np.stack([p.bank[s]['observation'][key] for s in slots]) for key in p.flat.core.context.keys}
  root_context=inference.search_context(context_data).anchor.cpu().numpy()
  for offset in range(0,len(jobs),self.F):
   ids=selected[offset:offset+self.F];root_ids=owners[offset:offset+self.F]
   state=outcomes[offset:offset+self.F];host_roots=root_ids.cpu().numpy()
   anchors=np.flatnonzero((state==1)|(state==3))
   cutoffs=np.flatnonzero(state==0)
   if len(anchors):
    chosen=ids[torch.as_tensor(anchors,device='cuda')];source=root_ids[torch.as_tensor(anchors,device='cuda')]
    width=p.root_relics.shape[1];packed=torch.empty((len(anchors),10+width),device='cuda')
    self.modules.launch('deck_anchor_inputs',[p.arena,chosen,source,p.flat.input_tables.bindings,
      p.root_relics,p.root_counters,packed],[len(anchors)],triton.cdiv(len(anchors),32))
    raw=packed.cpu().numpy();root_rows=np.array([roots[int(s)] for s in host_roots[anchors]])
    projected=root_context[root_rows].copy()
    dynamic=np.flatnonzero(np.isin(host_roots[anchors],list(p.dynamic_context_roots)))
    if len(dynamic):
     dynamic_rows=torch.as_tensor(dynamic,device='cuda');misc=torch.empty((len(dynamic),p.root_misc.shape[1]),device='cuda')
     self.modules.launch('deck_context_misc',[p.arena,chosen[dynamic_rows],source[dynamic_rows],p.root_misc,misc],
       [len(dynamic),misc.shape[1]],triton.cdiv(len(dynamic),32))
     changed={key:value[root_rows[dynamic]].copy() for key,value in context_data.items()}
     changed['outer_deck_misc']=misc.cpu().numpy()
     projected[dynamic]=inference.search_context(changed).anchor.cpu().numpy()
    inputs=dict(player_scalars=raw[:,:2],potions=raw[:,2:7].astype(np.int64),gold=raw[:,7:8],
      combat_exit_anchor=raw[:,8:10],relic_counters=raw[:,10:],
      relics=p.root_relics[source.long()].cpu().numpy(),
      run_keys=np.stack([p.bank[s]['observation']['run_keys'] for s in slots])[root_rows])
    p.values[chosen.long()]=torch.as_tensor(inference.anchor_values(inputs,projected).reshape(-1),device='cuda',dtype=p.values.dtype)
   if len(cutoffs):
    chosen=ids[torch.as_tensor(cutoffs,device='cuda')];source=root_ids[torch.as_tensor(cutoffs,device='cuda')]
    self.refresh_contexts(chosen)
    self.run(0,len(cutoffs),selected=chosen,worlds=source,value_only=True)
    p.values[chosen.long()]=self.model.buffers['coef'][:len(cutoffs),12].to(p.values.dtype)
   self.counts.update(anchor_rows=len(anchors),cutoff_rows=len(cutoffs),terminal_rows=int((state==2).sum()))
 @torch.inference_mode()
 def refresh_contexts(self,selected=None):
  p=self.pool
  if not p.dynamic_context_roots:return
  if selected is None:selected=torch.nonzero(p.context_dirty & p.alive,as_tuple=False).flatten().to(torch.int32)
  else:selected=selected[p.context_dirty[selected.long()].bool()]
  count=len(selected)
  if not count:return
  owners=p.job_roots[selected.long()];roots=owners.cpu().tolist()
  width=p.root_misc.shape[1];misc=p.arr((count,width),torch.float32)
  self.modules.launch('deck_context_misc',[p.arena,selected,owners,p.root_misc,misc],[count,width],triton.cdiv(count,32))
  data={key:np.stack([p.bank[r]['observation'][key] for r in roots]) for key in p.flat.core.context.keys}
  data['outer_deck_misc']=misc.cpu().numpy()
  context=self.completion_inference().search_context(data)
  index=selected.long()
  p.chain_inputs[index,:64]=context.combat;p.context_dirty[index]=0
  self.counts['context_refresh_rows']+=count
 def statistics(self):
  return dict(self.counts,cached=int(self.graph is not None),capacity=list(self.capacity),closed=self.closed,
   kernels=self.model.kernel_metadata,workspace_bytes=self.model.workspace_bytes(self.buffers.values()),workspace_alias_version=1,
   aliased_input_bytes=0 if not self.buffers else self.buffers['features'].numel()*self.buffers['features'].element_size(),
   overflow_history=self.history,modules_closed=self.modules.closed,identity=self.modules.identity)
 def close(self):
  if self.closed:return
  self.release();self.model.close();self.modules.close()
  if self.owns_leaf_inference:self.leaf_inference.__exit__(None,None,None)
  self.leaf_inference=None;self.closed=True
