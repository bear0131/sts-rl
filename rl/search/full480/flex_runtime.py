from pathlib import Path
import sys,os,ctypes,json,hashlib,collections
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'devicestreamfull141'))
from flex_adapter import *
P=Path(__file__).resolve().parent
class BattleModules:
 def __init__(self):
  self.handles=[];self.funcs={};self.references=0;self.closed=False;self.identity={}
  for kind,directory,names,probe in [
   ('observer',P,['search_observe','pool_layout','pool_inventory_slots'],'observer_layout'),
   ('rules',P,['stream_clear','stream_load','stream_start','stream_begin','stream_apply','stream_snapshot','stream_destroy','battle_step','battle_drop','battle_copy','battle_keys','packed_clear','packed_start','packed_begin','packed_apply','packed_snapshot','packed_destroy','packed_apply_dynamic','chain_step_layout'],'rules_layout')]:
   receipt=directory/('search-observer-compile-0-result.json' if kind=='observer' else 'rule-compile-0-result.json')
   result=json.loads(receipt.read_text());assert result['completed'] and result['exit']==0
   path=directory/('search-observer-0.cubin' if kind=='observer' else 'rules-0.cubin');digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==result['cubin_sha256']
   handle=ptr();check(lib.cuModuleLoad(ctypes.byref(handle),str(path).encode()),'load '+kind);self.handles.append(handle)
   for name in names+[probe,'packed_layout']:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),handle,name.encode()),name);self.funcs[name]=f
   out=torch.empty(2,dtype=torch.int64,device='cuda');address=ptr(out.data_ptr());args=(ptr*1)(ctypes.cast(ctypes.byref(address),ptr))
   check(lib.cuLaunchKernel(self.funcs.pop(probe),1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'ABI')
   layout=out.cpu().tolist();assert layout==[16864,112],layout
   if self.identity:assert layout==self.identity['observer']['abi_layout']
   mode=torch.empty(3,dtype=torch.int64,device='cuda');address=ptr(mode.data_ptr());args=(ptr*1)(ctypes.cast(ctypes.byref(address),ptr))
   check(lib.cuLaunchKernel(self.funcs.pop('packed_layout'),1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'packed addressing ABI')
   assert mode.cpu().tolist()==[1,16864,8]
   self.identity[kind]=dict(cubin_sha256=digest,abi_layout=layout,compile_options=result['options'],chain_addressing_version=1)
   if kind=='rules':
    step=torch.empty(3,dtype=torch.int64,device='cuda');address=ptr(step.data_ptr());args=(ptr*1)(ctypes.cast(ctypes.byref(address),ptr))
    check(lib.cuLaunchKernel(self.funcs.pop('chain_step_layout'),1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'chain-step ABI')
    assert step.cpu().tolist()==[1,16864,8]
    self.identity[kind]['chain_step_version']=1

 def close(self):
  if self.closed:return
  assert self.references==0
  torch.cuda.synchronize()
  for h in reversed(self.handles):check(lib.cuModuleUnload(h),'unload stream')
  self.handles.clear();self.closed=True
class BattleArena:
 upload=StreamingPool.upload
 launch=ScoringPool.launch
 snapshot=StreamingPool.snapshot
 def __init__(self,capacity,modules=None):
  assert capacity>0
  self.m=self.n=capacity;self.modules=BattleModules() if modules is None else modules;self.owns_modules=modules is None
  assert not self.modules.closed;self.modules.references+=1;self.funcs=self.modules.funcs;self.closed=False
  self.arr=lambda shape,dtype=torch.int32:torch.empty(shape,dtype=dtype,device='cuda')
  self.arena=self.arr((capacity,self.modules.identity['rules']['abi_layout'][0]),torch.uint8);self.roots=self.arena
  self.constructed=self.arr((capacity,)).zero_();self.root_constructed=self.constructed
  self.epochs=self.arr((capacity,),torch.int64).zero_();self.revisions=self.arr((capacity,),torch.int64).zero_()
  self.last_epochs=[0]*capacity;self.versions={}
 def validate(self,index,epoch,revision):
  assert not self.closed and 0<=index<self.m and self.versions.get(index)==(epoch,revision),(index,epoch,revision,self.versions.get(index))
 def load(self,rows):
  assert not self.closed
  if not rows:return {}
  for i,s in rows.items():
   assert type(i)==int and 0<=i<self.m
   assert s['portable'] and s['input_state']==1 and s['outcome']==0 and len(s['words'])<=8192
  ids=self.upload(list(rows),np.int32);words=np.zeros((len(rows),8192),np.uint64);lengths=[]
  for k,(i,s) in enumerate(rows.items()):
   words[k,:len(s['words'])]=s['words'];lengths.append(len(s['words']))
  self.launch('stream_load',[self.arena,self.constructed,ids,self.upload(words),self.upload(lengths,np.int32)],[len(rows)],grid=len(rows))
  epochs=[]
  for i in rows:
   self.last_epochs[i]+=1;self.versions[i]=(self.last_epochs[i],0);epochs.append(self.last_epochs[i])
  indices=ids.long();self.epochs.index_copy_(0,indices,self.upload(epochs,np.uint64));self.revisions[indices]=0
  return {i:self.versions[i] for i in rows}
 def step(self,rows):
  assert not self.closed
  if not rows:return
  for i,(epoch,revision,bits) in rows.items():self.validate(i,epoch,revision);assert 0<=bits<2**32
  n=len(rows);ids=self.upload(list(rows),np.int32);epochs=self.upload([v[0] for v in rows.values()],np.uint64);revs=self.upload([v[1] for v in rows.values()],np.uint64)
  bits=self.upload([v[2] for v in rows.values()],np.uint32)
  self.launch('battle_step',[self.arena,self.constructed,self.epochs,self.revisions,ids,epochs,revs,bits],[n],grid=n)
  for i,(ep,rev,_) in rows.items():self.versions[i]=(ep,rev+1)
 def drop(self,indices):
  assert not self.closed
  indices=list(indices);assert len(set(indices))==len(indices)
  for i in indices:assert i in self.versions
  if indices:self.launch('battle_drop',[self.arena,self.constructed,self.upload(indices,np.int32)],[len(indices)],grid=len(indices))
  for i in indices:del self.versions[i]
 def keys(self,indices):
  ids=list(indices)
  for i in ids:assert i in self.versions
  if not ids:return []
  keys=self.arr((len(ids),2048),torch.int64);counts=self.arr((len(ids),))
  self.launch('battle_keys',[self.arena,self.constructed,self.upload(ids,np.int32),keys,counts],[len(ids)],grid=len(ids))
  k=keys.cpu().numpy().view(np.uint64);ns=counts.cpu().tolist()
  return [row[:n].copy() for row,n in zip(k,ns)]
 def close(self):
  if self.closed:return
  self.launch('stream_destroy',[self.arena,self.constructed],[self.m],grid=self.m);torch.cuda.synchronize()
  self.versions.clear();self.closed=True;self.modules.references-=1
  if self.owns_modules:self.modules.close()

class BattlePool(CollectionPool):
 def __init__(self,mirror,budget_cap=1000,**kw):
  assert 'modules' not in kw
  super().__init__(mirror.m,budget_cap,modules=mirror.modules,**kw)
  self.mirror=mirror;self.ordered_semantic=self.arr((self.m,self.action_width),torch.int64);self.root_executed=self.arr((self.n,))
 def submit(self,rows):
  assert not self.closed
  if not rows:return {}
  # Validate every root before claiming or modifying any owned slot.
  for r in rows.values():
   if 'mirror' in r:self.mirror.validate(*r['mirror'])
   else:assert r['eligible'] and r['snapshot']['portable'] and r['snapshot']['input_state']==1
   assert r['snapshot']['input_state'] in (1,2) and r['snapshot']['outcome']==0 and len(r['semantic'])==len(r['action_bits'])
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
  entries=list(zip(handles,rows.values()));ordered=np.zeros((count,self.action_width),np.uint32);semantics=np.zeros((count,self.action_width),np.uint64)
  imported=[];mirrored=[]
  for k,(h,r) in enumerate(entries):
   self.bank[h.slot]=r
   ordered[k,:len(r['order'])]=np.asarray(r['action_bits'],np.uint32)[r['order']]
   semantics[k,:len(r['order'])]=np.asarray(r['semantic'],np.uint64)[r['order']]
   (mirrored if 'mirror' in r else imported).append(k)
  if imported:
   host_words=np.zeros((len(imported),8192),np.uint64);lengths=[];destination=[]
   for j,k in enumerate(imported):
    h,r=entries[k];w=r['snapshot']['words'];host_words[j,:len(w)]=w;lengths.append(len(w));destination.append(h.slot)
   self.launch('stream_load',[self.roots,self.root_constructed,self.upload(destination,np.int32),self.upload(host_words),self.upload(lengths,np.int32)],[len(imported)],grid=len(imported))
  if mirrored:
   source=[entries[k][1]['mirror'] for k in mirrored]
   destination=[entries[k][0].slot for k in mirrored]
   self.launch('battle_copy',[self.mirror.arena,self.mirror.constructed,self.mirror.epochs,self.mirror.revisions,
    self.upload([v[0] for v in source],np.int32),self.upload([v[1] for v in source],np.uint64),self.upload([v[2] for v in source],np.uint64),
    self.roots,self.root_constructed,self.upload(destination,np.int32)],[len(mirrored)],grid=len(mirrored))
  self.ordered_semantic.index_copy_(0,indices,self.upload(semantics))
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
  self.launch('stream_begin',[self.arena,root_ids,self.budgets,self.initial_bits,self.stats,self.alive,self.outcomes,self.values,self.root_boss,self.root_maximum,self.root_limits,self.root_done,self.ordered_semantic,self.action_counts,self.root_executed],
   [count,self.budget_cap,self.action_width],grid=count*self.budget_cap)
  self.refresh()
  return {h.key:h for h in handles}
