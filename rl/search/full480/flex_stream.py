from .run_device import artifact_directory
from pathlib import Path
import sys,os,ctypes,json,hashlib,collections
import numpy as np
P=artifact_directory()
from .search_runtime import *
from .root_ledger import RootLedger
from .state_pages import DevicePages
P=artifact_directory()
class StreamModules:
 def __init__(self):
  self.handles=[];self.funcs={};self.references=0;self.closed=False;self.identity={}
  for kind,directory,names,probe in [
   ('observer',P.parent/'devicerootsearch141',['search_observe','pool_layout','pool_inventory_slots'],'observer_layout'),
   ('rules',P,['stream_clear','stream_load','stream_start','stream_begin','stream_apply','stream_snapshot','stream_destroy'],'rules_layout')]:
   receipt=directory/('observer-compile-0-result.json' if kind=='observer' else 'rule-compile-0-result.json')
   result=json.loads(receipt.read_text());assert result['completed'] and result['exit']==0
   path=directory/(kind+'-0.cubin');digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==result['cubin_sha256']
   handle=ptr();check(lib.cuModuleLoad(ctypes.byref(handle),str(path).encode()),'load '+kind);self.handles.append(handle)
   for name in names+[probe]:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),handle,name.encode()),name);self.funcs[name]=f
   out=torch.empty(2,dtype=torch.int64,device='cuda');address=ptr(out.data_ptr());args=(ptr*1)(ctypes.cast(ctypes.byref(address),ptr))
   check(lib.cuLaunchKernel(self.funcs.pop(probe),1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'ABI')
   layout=out.cpu().tolist();assert layout==[16688,112],layout
   self.identity[kind]=dict(cubin_sha256=digest,abi_layout=layout,compile_options=result['options'])
 def close(self):
  if self.closed:return
  assert self.references==0
  torch.cuda.synchronize()
  for h in reversed(self.handles):check(lib.cuModuleUnload(h),'unload stream')
  self.handles.clear();self.closed=True
class StreamingPool(ScoringPool):
 def __init__(self,capacity=480,budget_cap=1000,*,precision='tf32x3',tile=16384,frame=65536,inventory_cap=32,modules=None,net=None,flat=None):
  assert capacity>0 and 1<=budget_cap<=100000 and tile>=32 and tile&(tile-1)==0 and frame>0 and inventory_cap>0
  self.m=capacity;self.budget_cap=budget_cap;self.n=capacity*budget_cap;self.ledger=RootLedger(capacity)
  self.bank=[None]*capacity;self.root_inventory=self.bank;self.precision=precision;self.tile=tile;self.frame_cap=min(self.n,frame)
  self.closed=False;self.token_layout='compact';self.scoring_mode='choices';self.skipped_forced_rows=0
  self.modules=StreamModules() if modules is None else modules;self.owns_modules=modules is None
  assert not self.modules.closed;self.modules.references+=1;self.funcs=self.modules.funcs
  self.net=load_net() if net is None else net;self.flat=Fused(self.net,precision) if flat is None else flat
  self.deck_consumer=getattr(self.flat,'consumer',None)=='deck-latent-combat-v1'
  self.layout=self.modules.identity['rules']['abi_layout'][0];self.I=inventory_cap;self.action_width=2048
  def arr(shape,dtype=torch.int32):return torch.empty(shape,dtype=dtype,device='cuda')
  self.arr=arr
  if self.deck_consumer:
   inputs=self.flat.input_tables;R=inputs.defines['RELIC_WIDTH']
   self.selection_prefix=arr((self.n,)).zero_()
   self.root_prefix=arr((self.m,)).zero_()
   self.paired_words=arr((self.m,self.budget_cap,5),torch.int64)
   self.pcg=arr((self.n,4),torch.int64)
   self.context_hash=arr((self.n,inputs.prototypes['outer_deck_misc'].shape[0]),torch.int64)
   self.context_dirty=arr((self.n,)).zero_();self.context_roots=arr((self.m,)).zero_()
   self.chain_inputs=arr((self.n,68),torch.float32)
   self.chain_anchor=arr((self.n,256),torch.float32)
   self.root_misc=arr((self.m,inputs.prototypes['outer_deck_misc'].shape[0]),torch.float32)
   self.value_mode=arr((1,)).zero_()
   self.dynamic_context_roots=set()
   self.root_inputs=arr((self.m,68),torch.float32)
   self.root_relics=arr((self.m,R));self.root_counters=arr((self.m,R),torch.float32)
  self.roots=arr((self.m,self.layout),torch.uint8);self.arena=arr((self.n,),torch.int64).zero_()
  assert self.modules.identity['rules']['chain_addressing_version']==1
  self.state_pages=DevicePages(self.m,self.budget_cap,self.layout,lambda count:arr((count,self.layout),torch.uint8));self.page_leases={}
  self.root_constructed=arr((self.m,)).zero_();self.constructed=arr((self.n,)).zero_();self.alive=arr((self.n,)).zero_()
  self.job_roots=torch.arange(self.n,dtype=torch.int32,device='cuda')//self.budget_cap
  self.root_generation=arr((self.m,),torch.int64).zero_();self.job_generation=arr((self.n,),torch.int64)
  self.root_nonce=arr((self.m,),torch.int64);self.budgets=arr((self.m,)).zero_();self.action_counts=arr((self.m,))
  self.ordered_bits=arr((self.m,self.action_width));self.initial_bits=arr((self.n,))
  self.world_seeds=arr((self.n,),torch.int64);self.policy_seeds=arr((self.n,),torch.int64)
  self.root_done=arr((self.m,)).zero_();self.root_boss=arr((self.m,));self.root_maximum=arr((self.m,));self.root_limits=arr((self.m,))
  self.temperatures=arr((self.m,),torch.float64);self.deck=arr((self.m,))
  self.inv=[arr((self.m,self.I)).zero_() for _ in range(3)]+[arr((self.m,self.I,16),torch.float32).zero_(),arr((self.m,)).zero_()]
  self.inv_slots=arr((self.m,self.I))
  F=self.frame_cap
  self.state=arr((F,32),torch.float32);self.meta=arr((F,4));self.chunks=arr((F,16));self.allocator=arr((5,))
  self.symbols=arr((F,T));self.counts=arr((F,T));self.token_order=arr((F,T));self.query_order=arr((F,T));self.query_map=arr((F,T));self.input_map=arr((F,T));self.geometry=arr((F,4))
  self.stats=arr((self.n,STAT),torch.int64);self.chosen=arr((self.n,));self.draws=arr((self.n,),torch.float64)
  self.outcomes=arr((self.n,));self.values=arr((self.n,),torch.float64)
  self.cache=collections.OrderedDict();self.histogram=collections.Counter();self.retry_history=[];self.frame_history=[]
  self.neural_rows=0;self.real_neural_rows=0;self.model_calls=0;self.iterations=0;self.completed_counts=collections.Counter()
  self.allocate_actions(F*8);self.allocate_entities(F*128)
  self.refresh()
 def launch(self,name,tensors,ints=(),wide=(),grid=None):
  packed={'stream_clear','stream_start','stream_begin','stream_apply','stream_snapshot','stream_destroy'}
  if name in packed:
   arena_index=1 if name=='stream_start' else 0
   if tensors[arena_index] is self.arena:name='packed_'+name[len('stream_'):]
  if self.deck_consumer:
   if name=='packed_start':tensors=tensors+[self.paired_words,self.pcg,self.selection_prefix,self.root_prefix,self.context_hash,self.context_dirty,self.context_roots]
   elif name in ('packed_begin','packed_apply','packed_apply_dynamic'):tensors=tensors+[self.pcg,self.selection_prefix,self.context_hash,self.context_dirty,self.context_roots]
  return ScoringPool.launch(self,name,tensors,ints,wide,grid)
 def prepare_pages(self,handles,rows):
  assert all(h.slot not in self.page_leases for h in handles)
  specs=[((h.slot,h.generation),int(rows[h.key]['budget'])) for h in handles]
  leases,pointers=self.state_pages.acquire(specs)
  try:
   indices=np.concatenate([h.slot*self.budget_cap+np.arange(rows[h.key]['budget'],dtype=np.int64) for h in handles])
   values=np.concatenate([pointers[(h.slot,h.generation)] for h in handles])
   self.arena.index_copy_(0,self.upload(indices,np.int64),self.upload(values,np.uint64))
  except BaseException:
   self.state_pages.release(leases.values());raise
  for h in handles:self.page_leases[h.slot]=leases[(h.slot,h.generation)]
 def retire_roots(self,output):
  if not output:return
  slots=[r['root_slot'] for r in output.values()]
  leases=[self.page_leases[s] for s in slots]
  for r,lease in zip(output.values(),leases,strict=True):
   assert lease.key==(r['root_slot'],r['root_generation']) and lease.count==r['simulations']
  ids=self.upload(slots,np.int32)
  self.launch('stream_clear',[self.arena,self.constructed,self.alive,ids],[len(slots),self.budget_cap],grid=len(slots)*self.budget_cap)
  self.arena.view(self.m,self.budget_cap).index_fill_(0,ids.long(),0)
  # Same-stream destruction and pointer clearing precede every later reuse.
  self.state_pages.release(leases)
  for s in slots:del self.page_leases[s]
 def upload(self,value,dtype=None):
  a=np.ascontiguousarray(value,dtype=dtype)
  if a.dtype==np.uint64:a=a.view(np.int64)
  if a.dtype==np.uint32:a=a.view(np.int32)
  return torch.from_numpy(a).cuda()
 def grow_inventory(self,required):
  if required<=self.I:return
  size=triton.next_power_of_2(required);old=self.I
  for k in range(4):
   x=self.inv[k];v=self.arr((self.m,size)+tuple(x.shape[2:]),x.dtype).zero_();v[:,:old].copy_(x);self.inv[k]=v
  self.inv_slots=self.arr((self.m,size));self.I=size
 def submit(self,rows):
  assert not self.closed
  if not rows:return {}
  # Validate every root before claiming or modifying any owned slot.
  for r in rows.values():
   assert r['eligible'] and r['snapshot']['portable'] and r['snapshot']['input_state']==1 and r['snapshot']['outcome']==0
   assert 1<=r['budget']<=self.budget_cap and 0<=r['depth_limit']<=200 and r['max_hp']>0
   assert np.isfinite(r['temperature']) and r['temperature']>0
   na=len(r['action_bits']);assert 1<=na<=self.action_width and sorted(r['order'])==list(range(na))
   assert len(r['snapshot']['words'])<=8192
   inv=r['inventory'];assert set(inv)=={'ids','zones','monsters','numbers'}
   assert len(inv['ids'])==len(inv['zones'])==len(inv['monsters'])==len(inv['numbers'])
  handles=self.ledger.claim(rows.keys())
  try:self.prepare_pages(handles,rows)
  except BaseException:
   for h in handles:self.ledger.release(h)
   raise
  root_ids=self.upload([h.slot for h in handles],np.int32);indices=root_ids.long();count=len(handles)
  self.launch('stream_clear',[self.arena,self.constructed,self.alive,root_ids],[count,self.budget_cap],grid=count*self.budget_cap)
  self.grow_inventory(max(len(r['inventory']['ids']) for r in rows.values()))
  host_words=np.zeros((count,8192),np.uint64);lengths=[];ordered=np.zeros((count,self.action_width),np.uint32)
  for k,(h,r) in enumerate(zip(handles,rows.values())):
   self.bank[h.slot]=r;w=r['snapshot']['words'];host_words[k,:len(w)]=w;lengths.append(len(w))
   ordered[k,:len(r['order'])]=np.asarray(r['action_bits'],np.uint32)[r['order']]
  self.launch('stream_load',[self.roots,self.root_constructed,root_ids,self.upload(host_words),self.upload(lengths,np.int32)],[count],grid=count)
  for name,key,dtype in [('root_nonce','nonce',np.uint64),('budgets','budget',np.int32),('root_boss','boss',np.int32),('root_maximum','max_hp',np.int32),('root_limits','depth_limit',np.int32),('temperatures','temperature',np.float64),('deck','master_deck_size',np.int32)]:
   getattr(self,name).index_copy_(0,indices,self.upload([r[key] for r in rows.values()],dtype))
  self.action_counts.index_copy_(0,indices,self.upload([len(r['action_bits']) for r in rows.values()],np.int32))
  self.root_generation.index_copy_(0,indices,self.upload([h.generation for h in handles],np.uint64));self.root_done[indices]=0
  self.ordered_bits.index_copy_(0,indices,self.upload(ordered))
  for k,(name,tail,dtype) in enumerate([('ids',(),np.int32),('zones',(),np.int32),('monsters',(),np.int32),('numbers',(16,),np.float32)]):
   v=np.zeros((count,self.I)+tail,dtype)
   for j,r in enumerate(rows.values()):v[j,:len(r['inventory'][name])]=r['inventory'][name]
   self.inv[k].index_copy_(0,indices,self.upload(v))
  self.inv[4].index_copy_(0,indices,self.upload([len(r['inventory']['ids']) for r in rows.values()],np.int32))
  self.launch('pool_inventory_slots',self.inv+[self.inv_slots],[self.m,self.I],grid=self.m)
  self.launch('stream_start',[self.roots,self.arena,root_ids,self.job_generation,self.root_generation,self.root_nonce,self.budgets,self.action_counts,self.ordered_bits,
   self.stats,self.constructed,self.alive,self.world_seeds,self.policy_seeds,self.initial_bits],[count,self.budget_cap,self.action_width],grid=count*self.budget_cap)
  self.launch('stream_begin',[self.arena,root_ids,self.budgets,self.initial_bits,self.stats,self.alive,self.outcomes,self.values,self.root_boss,self.root_maximum,self.root_limits,self.root_done],
   [count,self.budget_cap],grid=count*self.budget_cap)
  self.refresh()
  return {h.key:h for h in handles}
 def refresh(self):
  self.live=torch.nonzero(self.alive,as_tuple=False).flatten().to(torch.int32);self.active_count=self.live.numel()
  self.logical_ids=self.job_roots[self.live]
  return self.active_count
 observe=SearchPool.observe
 def public(self):
  self.logical_host=self.logical_ids.cpu().numpy()
  return ScoringPool.public(self)
 def apply(self):
  self.launch('stream_apply',[self.arena,self.live,self.job_roots,self.scores,self.bits,self.meta,self.policy_seeds,self.temperatures,self.stats,
   self.alive,self.outcomes,self.values,self.chosen,self.draws,self.root_boss,self.root_maximum,self.root_limits,self.job_generation,self.root_generation,self.root_done],
   [self.count,self.base],grid=self.count)
 def tick(self):
  assert self.active_count>0 and not self.closed
  for base in range(0,self.active_count,self.frame_cap):self.observe(base);self.score();self.apply()
  self.iterations+=1;self.refresh()
 @property
 def pending(self):return bool(len(self.ledger))
 def reap(self):
  if not self.pending:return {}
  done=self.root_done.cpu().numpy()
  handles=[]
  for h in self.ledger.pending.values():
   b=self.bank[h.slot]['budget'];assert 0<=done[h.slot]<=b
   if done[h.slot]==b:handles.append(h)
  if not handles:return {}
  if self.deck_consumer:self.frame_pipeline.complete(handles)
  indices=self.upload([h.slot for h in handles],np.int64)
  stats=self.stats.view(self.m,self.budget_cap,STAT)[indices].cpu().numpy().view(np.uint64)
  values=self.values.view(self.m,self.budget_cap)[indices].cpu().numpy()
  outcomes=self.outcomes.view(self.m,self.budget_cap)[indices].cpu().numpy()
  output={}
  for k,h in enumerate(handles):
   r=self.bank[h.slot];b=r['budget'];na=len(r['action_bits'])
   plan=dict(root=np.zeros(b,np.int32),action_index=np.asarray(r['order'],np.int32)[np.arange(b)%na])
   result=aggregate([r],plan,values[k,:b],outcomes[k,:b],stats[k,:b])[0]
   result['root_generation']=h.generation;result['root_slot']=h.slot
   assert result.get('atomic_decisions',result['actual_transitions'])-b==result['nonforced_decisions']+result['forced_actions']
   self.completed_counts.update(roots=1,simulations=b,actual_transitions=result['actual_transitions'],nonforced=result['nonforced_decisions'],forced=result['forced_actions'],cuts=result['depth_cuts'])
   output[h.key]=result;self.ledger.release(h);self.bank[h.slot]=None
   if self.deck_consumer:self.dynamic_context_roots.discard(h.slot)
  self.retire_roots(output)
  return output
 def receive(self,steps=1):
  assert steps>=0
  out=self.reap()
  if out:return out
  for _ in range(steps):
   if not self.active_count:break
   self.tick();out=self.reap()
   if out:return out
  assert out or self.active_count or not self.pending,'unfinished roots with no active tasks'
  return out
 def snapshot(self,selected=None,roots=False):
  mask=self.root_constructed if roots else self.constructed
  if selected is None:selected=torch.nonzero(mask,as_tuple=False).flatten().to(torch.int32)
  elif not isinstance(selected,torch.Tensor):selected=self.upload(selected,np.int32)
  count=selected.numel()
  words=self.arr((count,8192),torch.int64);meta=self.arr((count,6));actions=self.arr((count,2048))
  if count:self.launch('stream_snapshot',[self.roots if roots else self.arena,mask,selected,words,meta,actions],[count],grid=count)
  return words.cpu().numpy().view(np.uint64),meta.cpu().numpy(),actions.cpu().numpy().view(np.uint32)
 def close(self):
  if self.closed:return
  assert not self.pending,'cannot silently abandon unfinished root searches'
  assert self.real_neural_rows==self.completed_counts['nonforced']
  assert self.skipped_forced_rows==self.completed_counts['forced']
  self.launch('stream_destroy',[self.arena,self.constructed],[self.n],grid=self.n)
  self.launch('stream_destroy',[self.roots,self.root_constructed],[self.m],grid=self.m)
  torch.cuda.synchronize();self.state_pages.close();self.closed=True;self.modules.references-=1
  if self.owns_modules:self.modules.close()
