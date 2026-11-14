from pathlib import Path
import sys,os,ctypes,json,hashlib,collections,time
import numpy as np
P=Path(__file__).resolve().parent;SOURCE_P=P.parent/'devicecompact141'
sys.path.insert(0,str(SOURCE_P));sys.path.insert(0,str(P.parent/'devicechoice141'))
from owned_pool import ResidentPool as ScoringPool
from owned_pool import *
from finite_plan import build_plan,aggregate
P=Path(__file__).resolve().parent
lib.cuModuleUnload.argtypes=[ptr]
class SearchModules:
 def __init__(self,mode='0'):
  assert mode in ['0','mid'];self.handles=[];self.funcs={};self.references=0;self.closed=False;self.identity={}
  for kind,names,probe in [
   ('observer',['search_observe','pool_layout','pool_inventory_slots'],'observer_layout'),
   ('rules',['search_init','search_apply','bridge_init','bridge_snapshot','bridge_destroy'],'rules_layout')]:
   variant='0' if kind=='observer' else mode
   receipt=P/(('observer-compile-' if kind=='observer' else 'rule-compile-')+variant+'-result.json')
   result=json.loads(receipt.read_text());assert result['completed'] and result['exit']==0
   path=P/(kind+'-'+variant+'.cubin');digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==result['cubin_sha256']
   handle=ptr();check(lib.cuModuleLoad(ctypes.byref(handle),str(path).encode()),'load '+kind);self.handles.append(handle)
   for name in names+[probe]:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),handle,name.encode()),name);self.funcs[name]=f
   out=torch.empty(2,dtype=torch.int64,device='cuda');address=ptr(out.data_ptr());args=(ptr*1)(ctypes.cast(ctypes.byref(address),ptr))
   check(lib.cuLaunchKernel(self.funcs.pop(probe),1,1,1,1,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'ABI')
   layout=out.cpu().tolist();assert layout==[16688,112],layout
   self.identity[kind]=dict(mode=variant,cubin_sha256=digest,abi_layout=layout,compile_options=result['options'])
 def close(self):
  if self.closed:return
  assert self.references==0,'live states still reference this module'
  torch.cuda.synchronize()
  for handle in reversed(self.handles):check(lib.cuModuleUnload(handle),'unload finite search')
  self.handles.clear();self.closed=True
class SearchPool(ScoringPool):
 def __init__(self,roots,plan=None,*,precision='tf32x3',tile=4096,frame=None,modules=None,net=None,flat=None,initial_actions=None,initial_entities=None):
  assert roots and tile>=32 and tile&(tile-1)==0
  self.bank=roots;self.root_inventory=roots;self.m=len(roots);self.plan=build_plan(roots) if plan is None else plan
  self.n=len(self.plan['root']);assert self.n>0
  assert all(r['eligible'] and r['snapshot']['portable'] and r['snapshot']['input_state']==1 for r in roots)
  self.precision=precision;self.tile=tile;self.frame_cap=min(self.n,frame or self.n);self.closed=False;self.token_layout='compact'
  self.scoring_mode='choices';self.skipped_forced_rows=0
  self.modules=SearchModules(os.environ.get('RULE_MODE','0')) if modules is None else modules
  self.owns_modules=modules is None;assert not self.modules.closed;self.modules.references+=1;self.funcs=self.modules.funcs
  self.net=load_net() if net is None else net;self.flat=Fused(self.net,precision) if flat is None else flat
  self.layout=16688;self.I=max(len(r['inventory']['ids']) for r in roots)
  def arr(shape,dtype=torch.int32):return torch.empty(shape,dtype=dtype,device='cuda')
  self.arr=arr
  def upload(v,dtype=None):
   a=np.ascontiguousarray(v,dtype=dtype)
   if a.dtype==np.uint64:a=a.view(np.int64)
   if a.dtype==np.uint32:a=a.view(np.int32)
   return torch.from_numpy(a).cuda()
  self.job_roots=upload(self.plan['root']);self.root_bits=upload(self.plan['action_bits'])
  self.world_seeds=upload(self.plan['world_seed']);self.policy_seeds=upload(self.plan['policy_seed'])
  self.logical_ids=self.job_roots;self.logical_host=self.plan['root'];self.order_digest=hashlib.sha256(self.logical_host.tobytes()).hexdigest()
  self.roots=arr((self.m,self.layout),torch.uint8);self.arena=arr((self.n,self.layout),torch.uint8)
  h=np.zeros((self.m,8192),np.uint64);lengths=[]
  for i,r in enumerate(roots):
   w=r['snapshot']['words'];assert len(w)<=8192;h[i,:len(w)]=w;lengths.append(len(w))
  self.launch('bridge_init',[self.roots,upload(h),upload(lengths,np.int32)],[self.m],grid=self.m);torch.cuda.synchronize()
  self.deck=upload([r['master_deck_size'] for r in roots],np.int32)
  self.root_boss=upload([r['boss'] for r in roots],np.int32);self.root_maximum=upload([r['max_hp'] for r in roots],np.int32)
  self.root_limits=upload([r['depth_limit'] for r in roots],np.int32);self.temperatures=upload([r['temperature'] for r in roots],np.float64)
  self.inv=[]
  for name,tail,dtype in [('ids',(),np.int32),('zones',(),np.int32),('monsters',(),np.int32),('numbers',(16,),np.float32)]:
   v=np.zeros((self.m,self.I)+tail,dtype)
   for i,r in enumerate(roots):v[i,:len(r['inventory'][name])]=r['inventory'][name]
   self.inv.append(upload(v))
  self.inv.append(upload([len(r['inventory']['ids']) for r in roots],np.int32));self.inv_slots=arr((self.m,self.I))
  self.launch('pool_inventory_slots',self.inv+[self.inv_slots],[self.m,self.I],grid=self.m)
  F=self.frame_cap
  self.state=arr((F,32),torch.float32);self.meta=arr((F,4));self.chunks=arr((F,16));self.allocator=arr((5,))
  self.symbols=arr((F,T));self.counts=arr((F,T));self.token_order=arr((F,T));self.query_order=arr((F,T));self.query_map=arr((F,T));self.input_map=arr((F,T));self.geometry=arr((F,4))
  self.stats=arr((self.n,STAT),torch.int64);self.chosen=arr((self.n,));self.draws=arr((self.n,),torch.float64)
  self.alive=arr((self.n,));self.outcomes=arr((self.n,));self.values=arr((self.n,),torch.float64)
  self.cache=collections.OrderedDict();self.histogram=collections.Counter();self.retry_history=[];self.frame_history=[]
  self.neural_rows=0;self.real_neural_rows=0;self.model_calls=0;self.iterations=0
  self.allocate_actions(initial_actions or F*8);self.allocate_entities(initial_entities or F*128)
  self.launch('search_init',[self.roots,self.arena,self.job_roots,self.root_bits,self.world_seeds,self.stats,self.alive,self.outcomes,self.values,
   self.root_boss,self.root_maximum,self.root_limits],[self.n])
  self.refresh()
 def refresh(self):
  self.live=torch.nonzero(self.alive,as_tuple=False).flatten().to(torch.int32);self.active_count=self.live.numel()
  self.logical_ids=self.job_roots[self.live]
  return self.active_count
 def observe(self,base=0,count=None):
  count=min(self.frame_cap,self.active_count-base) if count is None else count
  assert base>=0 and 0<count<=self.frame_cap and base+count<=self.active_count
  self.base=base;self.count=count
  for attempt in range(8):
   self.allocator.zero_()
   self.launch('search_observe',[self.arena,self.logical_ids,self.live,self.deck,self.state,self.bits,self.actions,self.keys,self.source_rows,self.targets,
    self.ids,self.zones,self.monsters,self.numbers,self.meta,self.chunks,self.allocator],
    [count,base,self.m,self.action_cap,self.entity_cap],grid=count)
   values=self.allocator.cpu().tolist();needed_a,needed_e,flags,max_e,max_a=map(int,values)
   assert needed_a>0 and needed_e>=0 and flags in [0,1,2,3],values
   if not flags:self.frame_values=values;return
   self.retry_history.append(dict(base=base,count=count,attempt=attempt,action_cap=self.action_cap,entity_cap=self.entity_cap,needed_actions=needed_a,needed_entities=needed_e,flags=flags))
   if needed_a>self.action_cap:self.allocate_actions(triton.next_power_of_2(needed_a))
   if needed_e>self.entity_cap:self.allocate_entities(triton.next_power_of_2(needed_e))
  raise RuntimeError('Finite observation growth exceeded8 attempts')
 def apply(self):
  self.launch('search_apply',[self.arena,self.live,self.job_roots,self.scores,self.bits,self.meta,self.policy_seeds,self.temperatures,self.stats,
   self.alive,self.outcomes,self.values,self.chosen,self.draws,self.root_boss,self.root_maximum,self.root_limits],
   [self.count,self.base],grid=self.count)
 def public(self):
  self.logical_host=self.logical_ids.cpu().numpy()
  return super().public()
 def tick(self):
  assert self.active_count>0
  for base in range(0,self.active_count,self.frame_cap):self.observe(base);self.score();self.apply()
  self.iterations+=1;self.refresh()
 def run(self):
  for _ in range(max(r['depth_limit'] for r in self.bank)):
   if not self.active_count:break
   self.tick()
  assert self.active_count==0,'finite jobs did not finish within declared depth'
  return self.results()
 def results(self):
  assert self.active_count==0
  stats=self.stats.cpu().numpy().view(np.uint64);values=self.values.cpu().numpy();outcomes=self.outcomes.cpu().numpy()
  assert int(stats[:,0].sum())==int(stats[:,1].sum()+stats[:,2].sum())
  assert self.real_neural_rows==int(stats[:,1].sum()) and self.skipped_forced_rows==int(stats[:,2].sum())
  return aggregate(self.bank,self.plan,values,outcomes,stats)
 def reset(self,*args,**kwargs):raise RuntimeError('A finite search cannot silently refill its completed jobs')
 def prepare(self,*args,**kwargs):return self.refresh()
 def close(self):
  if self.closed:return
  super().close();self.modules.references-=1
  if self.owns_modules:self.modules.close()
