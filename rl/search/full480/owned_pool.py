from pathlib import Path
import ctypes,json,pickle,collections,hashlib,os
import numpy as np
from device_policy import load_net,Fused,bridge,P,lib,ptr,check,module,torch,DEPTH,STAT
from pool_gpu import *
T=161
TOKEN_BINS=[16,24,32,48,64,96,128,161]
QUERY_BINS=[1,2,4,8,16,32,64,128,161]
class ResidentPool:
 def __init__(self,n,seed,precision='ieee',tile=4096,frame=None,initial_actions=None,initial_entities=None,world_order='root',token_layout='compact',kernel_functions=None):
  assert n>0 and tile>0 and tile&(tile-1)==0
  self.n=n;self.seed=seed;self.precision=precision;self.tile=tile;self.frame_cap=min(n,frame or n);self.closed=False
  self.scoring_mode=os.environ.get('POOL_SCORING','choices');assert self.scoring_mode in ('all','choices')
  self.skipped_forced_rows=0
  self.net=load_net();self.flat=Fused(self.net,precision)
  self.bank=pickle.loads((P/'real-branches.pkl').read_bytes());self.m=len(self.bank)
  assert world_order in ('linear','root'),world_order
  assert token_layout in ('canonical','compact'),token_layout
  self.token_layout=token_layout
  self.world_order=world_order;logical=np.arange(n,dtype=np.int32)
  if world_order=='root':logical=logical[np.argsort(logical%self.m,kind='stable')]
  assert np.array_equal(np.sort(logical),np.arange(n,dtype=np.int32))
  self.logical_host=logical;self.logical_ids=torch.from_numpy(logical).cuda()
  self.order_digest=hashlib.sha256(logical.tobytes()).hexdigest()
  inventory={x['index']:x for x in pickle.loads((P/'observation-roots.pkl').read_bytes())}
  self.root_inventory=[inventory[x['metadata']['root']] for x in self.bank]
  self.I=max(len(x['inventory']['ids']) for x in self.root_inventory)
  self.layout=json.loads((P/'device-smoke.json').read_text())['device_layout']['values'][0]
  self.funcs={}
  for name in ['bridge_init','bridge_snapshot','bridge_destroy','pool_prepare','pool_observe','pool_layout','pool_apply','pool_inventory_slots']:
   f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),module,name.encode()),name);self.funcs[name]=f
  if kernel_functions is not None:
   assert set(kernel_functions).issubset(self.funcs),set(kernel_functions)-set(self.funcs)
   self.funcs.update(kernel_functions)
  def arr(shape,dtype=torch.int32):return torch.empty(shape,dtype=dtype,device='cuda')
  self.arr=arr;self.roots=arr((self.m,self.layout),torch.uint8);self.arena=arr((n,self.layout),torch.uint8)
  h=np.zeros((self.m,8192),np.uint64)
  for i,row in enumerate(self.bank):h[i,:len(row['snapshot']['words'])]=row['snapshot']['words']
  words=torch.from_numpy(h.view(np.int64)).cuda()
  lengths=torch.tensor([len(x['snapshot']['words']) for x in self.bank],device='cuda',dtype=torch.int32)
  self.launch('bridge_init',[self.roots,words,lengths],[self.m],grid=self.m);torch.cuda.synchronize()
  self.deck=torch.tensor([x['master_deck_size'] for x in self.root_inventory],device='cuda',dtype=torch.int32)
  self.inv=[]
  for name,tail,dtype in [('ids',(),np.int32),('zones',(),np.int32),('monsters',(),np.int32),('numbers',(16,),np.float32)]:
   v=np.zeros((self.m,self.I)+tail,dtype)
   for i,x in enumerate(self.root_inventory):v[i,:len(x['inventory'][name])]=x['inventory'][name]
   self.inv.append(torch.from_numpy(v).cuda())
  self.inv.append(torch.tensor([len(x['inventory']['ids']) for x in self.root_inventory],device='cuda',dtype=torch.int32))
  self.inv_slots=arr((self.m,self.I))
  self.launch('pool_inventory_slots',self.inv+[self.inv_slots],[self.m,self.I],grid=self.m)
  F=self.frame_cap
  self.state=arr((F,32),torch.float32);self.meta=arr((F,4));self.chunks=arr((F,16));self.allocator=arr((5,))
  self.symbols=arr((F,T));self.counts=arr((F,T));self.token_order=arr((F,T));self.query_order=arr((F,T));self.query_map=arr((F,T));self.input_map=arr((F,T));self.geometry=arr((F,4))
  self.stats=arr((n,STAT),torch.int64);self.chosen=arr((n,));self.draws=arr((n,),torch.float64)
  self.features=arr((tile,T,128),torch.float32) if token_layout=='canonical' else None;self.cache=collections.OrderedDict();self.histogram=collections.Counter()
  self.retry_history=[];self.neural_rows=0;self.real_neural_rows=0;self.model_calls=0;self.frame_history=[]
  self.allocate_actions(initial_actions or F*8);self.allocate_entities(initial_entities or F*128)
  self.prepare(1)
 def launch(self,name,tensors,ints=(),wide=(),grid=None):
  params=[ptr(x.data_ptr()) for x in tensors]+[ctypes.c_int(x) for x in ints]+[ctypes.c_ulonglong(x) for x in wide]
  args=(ptr*len(params))(*(ctypes.cast(ctypes.byref(x),ptr) for x in params));n=self.n if grid is None else grid
  check(lib.cuLaunchKernel(self.funcs[name],(n+31)//32,1,1,32,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),name)
 def allocate_actions(self,cap):
  self.action_cap=int(cap);a=self.arr
  self.bits=a((cap,));self.actions=a((cap,24),torch.float32);self.keys=a((cap,10))
  self.source_rows=a((cap,10));self.targets=a((cap,));self.canonical_sources=a((cap,10));self.canonical_targets=a((cap,));self.scores=a((cap,),torch.float32)
 def allocate_entities(self,cap):
  cap=triton.cdiv(int(cap),64)*64;self.entity_cap=cap;a=self.arr
  self.ids=a((cap,));self.zones=a((cap,));self.monsters=a((cap,));self.numbers=a((cap,16),torch.float32)
  self.entity_slots=a((cap,));self.extra_slots=a((cap,))
 def prepare(self,initialize=0):
  params=[ptr(x.data_ptr()) for x in [self.roots,self.arena,self.stats,self.logical_ids]]
  params += [ctypes.c_int(self.n),ctypes.c_int(self.m),ctypes.c_int(DEPTH),ctypes.c_ulonglong(self.seed),ctypes.c_int(initialize)]
  args=(ptr*len(params))(*(ctypes.cast(ctypes.byref(x),ptr) for x in params))
  check(lib.cuLaunchKernel(self.funcs['pool_prepare'],(self.n+31)//32,1,1,32,1,1,0,ptr(torch.cuda.current_stream().cuda_stream),args,None),'prepare')
 def observe(self,base=0,count=None):
  count=min(self.frame_cap,self.n-base) if count is None else count
  assert base>=0 and 0<count<=self.frame_cap and base+count<=self.n
  self.base=base;self.count=count
  for attempt in range(8):
   self.allocator.zero_()
   self.launch('pool_observe',[self.arena,self.logical_ids,self.deck,self.state,self.bits,self.actions,self.keys,self.source_rows,self.targets,
    self.ids,self.zones,self.monsters,self.numbers,self.meta,self.chunks,self.allocator],
    [count,base,self.m,self.action_cap,self.entity_cap],grid=count)
   values=self.allocator.cpu().tolist();needed_a,needed_e,flags,max_e,max_a=map(int,values)
   assert needed_a>0 and needed_e>=0 and flags in [0,1,2,3],values
   if not flags:
    self.frame_values=values;return
   self.retry_history.append(dict(base=base,count=count,attempt=attempt,action_cap=self.action_cap,entity_cap=self.entity_cap,needed_actions=needed_a,needed_entities=needed_e,flags=flags))
   if needed_a>self.action_cap:self.allocate_actions(triton.next_power_of_2(needed_a))
   if needed_e>self.entity_cap:self.allocate_entities(triton.next_power_of_2(needed_e))
  raise RuntimeError(('pool growth retry budget exceeded',self.retry_history[-1]))
 def layout_frame(self):
  self.launch('pool_layout',[self.logical_ids,self.ids,self.zones,self.monsters,self.numbers,self.meta,self.chunks]+self.inv+[self.inv_slots,
   self.source_rows,self.targets,self.symbols,self.counts,self.canonical_sources,self.canonical_targets,self.entity_slots,self.extra_slots,
   self.token_order,self.query_order,self.query_map,self.input_map,self.geometry],
   [self.count,self.base,self.m,self.I,T],grid=self.count)
  keys=self.geometry[:self.count,3].long();bins=len(TOKEN_BINS)*len(QUERY_BINS)
  if self.scoring_mode=='choices':keys=torch.where(self.meta[:self.count,0]>1,keys,bins)
  self.order=torch.argsort(keys)
  counts=torch.bincount(keys,minlength=bins+1).cpu().tolist()
  assert sum(counts)==self.count
  self.bucket_counts=counts[:bins];self.scored_count=sum(self.bucket_counts)
  self.skipped_forced_rows+=counts[bins]
  # A single legal action has probability1 for every possible score; apply ignores it.
  # Zero only provides defined diagnostic logits for rows that never run the network.
  if self.scoring_mode=='choices':self.scores.zero_()
 def workspace(self,n,C,Q,A):
  key=(n,C,Q,A)
  if key not in self.cache:
   a=self.arr
   self.cache[key]=[a((A,24),torch.float32).zero_(),a((A,10)).fill_(-1),a((A,)).zero_(),a((A,10)).fill_(-1),a((A,)).fill_(-1),
    a((n,C,128),torch.float32),a((n,C)),a((n,C)),a((n,C),torch.bool),a((n,Q)),a((n,32),torch.float32)]
   if len(self.cache)>4:self.cache.popitem(last=False)
  self.cache.move_to_end(key);self.histogram[str(key)]+=1
  return self.cache[key]
 @torch.inference_mode()
 def score(self,callback=None):
  self.layout_frame();offset=0;max_rows=self.frame_values[3]+self.I
  for key,bucket_count in enumerate(self.bucket_counts):
   if not bucket_count:continue
   C=TOKEN_BINS[key//len(QUERY_BINS)];Q=QUERY_BINS[key%len(QUERY_BINS)]
   for local in range(0,bucket_count,self.tile):
    real=min(self.tile,bucket_count-local);pad=max(32,triton.next_power_of_2(real));assert pad<=self.tile
    selected=self.order[offset+local:offset+local+real]
    sizes=self.meta[selected,0].long();end=sizes.cumsum(0);starts=end-sizes
    total,max_actions=torch.stack((end[-1],sizes.max())).cpu().tolist()
    cap=max(32,triton.next_power_of_2(total));v=self.workspace(pad,C,Q,cap)
    if self.token_layout=='canonical':
     self.features[:pad].zero_()
     pool_hash[(real,triton.cdiv(max_rows,32))](self.ids,self.zones,self.numbers,self.meta,self.chunks,
      self.inv[0],self.inv[1],self.inv[3],self.inv[4],self.inv_slots,self.entity_slots,self.extra_slots,selected,self.features,self.logical_ids,
      self.base,self.m,self.I,T,32,num_warps=4)
     pool_normalize[(pad*T,)](self.features,self.counts,selected,real,T,num_warps=4)
     pool_pack_features[(pad,triton.cdiv(C,4))](self.features,self.symbols,self.counts,self.token_order,self.query_order,self.state,selected,
      *v[5:],real,T,C,Q,num_warps=4)
    else:
     v[5].zero_()
     pool_compact_hash[(real,triton.cdiv(max_rows,32))](self.ids,self.zones,self.numbers,self.meta,self.chunks,
      self.inv[0],self.inv[1],self.inv[3],self.inv[4],self.inv_slots,self.entity_slots,self.extra_slots,selected,v[5],self.logical_ids,self.input_map,
      self.base,self.m,self.I,T,C,32,num_warps=4)
     pool_compact_pack[(pad,triton.cdiv(C,4))](v[5],self.symbols,self.counts,self.token_order,self.query_order,self.state,selected,
      *v[5:],real,T,C,Q,num_warps=4)
    pool_pack_actions[(real,triton.cdiv(max_actions,16))](self.actions,self.keys,self.canonical_sources,self.canonical_targets,self.meta,self.query_map,
     selected,starts,*v[:5],T,16,num_warps=4)
    act,keys,owners,source,target,feat,symbols,slots,present,queries,state=v
    logits=self.flat(state,act,keys,owners,feat,symbols,slots,present,queries,source,target)
    if callback:callback(selected,starts,v,logits,total,dict(real=real,padded=pad,tokens=C,queries=Q,actions=cap))
    pool_scatter_scores[(real,triton.cdiv(max_actions,16))](logits,self.scores,self.meta,selected,starts,16,num_warps=4)
    self.neural_rows+=pad;self.real_neural_rows+=real;self.model_calls+=1
   offset+=bucket_count
  assert offset==self.scored_count
 def apply(self):
  self.launch('pool_apply',[self.arena,self.logical_ids,self.scores,self.bits,self.meta,self.stats,self.chosen,self.draws],
   [self.count,self.base,self.n,DEPTH],[self.seed],grid=self.count)
 def tick(self):
  self.prepare()
  for base in range(0,self.n,self.frame_cap):
   self.observe(base);self.score();self.apply()
 def reset(self,seed):
  self.skipped_forced_rows=0;self.seed=seed;self.prepare(2);self.histogram.clear();self.neural_rows=0;self.real_neural_rows=0;self.model_calls=0
 def snapshot(self,base=0,count=None):
  count=self.n-base if count is None else count
  words=self.arr((count,8192),torch.int64);meta=self.arr((count,6));acts=self.arr((count,2048))
  self.launch('bridge_snapshot',[self.arena[base:],words,meta,acts],[count],grid=count)
  return words.cpu().numpy().view(np.uint64),meta.cpu().numpy(),acts.cpu().numpy().view(np.uint32)
 def public(self):
  ta,te=map(int,self.frame_values[:2]);hm=self.meta[:self.count].cpu().numpy();chunks=self.chunks[:self.count].cpu().numpy()
  state=self.state[:self.count].cpu().numpy()
  a=[x[:ta].cpu().numpy() for x in [self.bits,self.actions,self.keys,self.source_rows,self.targets]]
  e=[x[:te].cpu().numpy() for x in [self.ids,self.zones,self.monsters,self.numbers]]
  out=[]
  for i in range(self.count):
   na,ne,start,valid=map(int,hm[i]);assert valid
   logical=np.arange(ne);physical=chunks[i,logical//64]+logical%64;root=int(self.logical_host[self.base+i])%self.m
   out.append(dict(state=state[i],action_bits=a[0][start:start+na].view(np.uint32),actions=a[1][start:start+na],keys=a[2][start:start+na],
    source_rows=a[3][start:start+na],target_slots=a[4][start:start+na],
    entities=dict(ids=e[0][physical],zones=e[1][physical],monsters=e[2][physical],numbers=e[3][physical]),inventory=self.root_inventory[root]['inventory']))
  return out
 def logits_rows(self):
  ta=int(self.frame_values[0]);s=self.scores[:ta].cpu().double().numpy();m=self.meta[:self.count].cpu().numpy()
  return [s[int(r[2]):int(r[2]+r[0])] for r in m]
 def close(self):
  if self.closed:return
  self.launch('bridge_destroy',[self.arena],[self.n],grid=self.n)
  self.launch('bridge_destroy',[self.roots],[self.m],grid=self.m)
  torch.cuda.synchronize();self.closed=True
