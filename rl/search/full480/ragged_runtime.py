from pathlib import Path
import sys
RAGGED_P=Path(__file__).resolve().parent
sys.path.insert(0,str(RAGGED_P.parent/'devicebattle141'))
from battle_runtime import *
from ragged_pack import ragged_hash,ragged_pack
from ragged_model import forward as ragged_forward

class RaggedBattlePool(BattlePool):
 def _buffer(self,name,count,tail=(),dtype=torch.float32):
  if not hasattr(self,'ragged_buffers'):self.ragged_buffers={}
  old=self.ragged_buffers.get(name)
  if old is None or len(old)<count:
   self.ragged_buffers[name]=torch.empty((triton.next_power_of_2(max(1,count)),)+tail,dtype=dtype,device='cuda')
  return self.ragged_buffers[name][:count]
 @torch.inference_mode()
 def score(self,callback=None):
  if callback is not None:raise ValueError('ragged qualification reads complete logits_rows, not bucket callbacks')
  self.launch('pool_layout',[self.logical_ids,self.ids,self.zones,self.monsters,self.numbers,self.meta,self.chunks]+self.inv+[self.inv_slots,
   self.source_rows,self.targets,self.symbols,self.counts,self.canonical_sources,self.canonical_targets,self.entity_slots,self.extra_slots,
   self.token_order,self.query_order,self.query_map,self.input_map,self.geometry],
   [self.count,self.base,self.m,self.I,T],grid=self.count)
  choice=self.meta[:self.count,0]>1
  lengths=torch.where(choice[:,None],self.geometry[:self.count,:2].long(),0)
  lens=torch.cat((lengths,(lengths+15)//16),dim=1).contiguous()
  ends=lens.T.contiguous().cumsum(1).T.contiguous();starts=(ends-lens).contiguous()
  nt,nq,nb1,nb2,real=map(int,torch.cat((ends[-1],choice.sum().reshape(1))).cpu().tolist())
  assert 0<=real<=self.count and nt>=real and nq>=real
  self.scored_count=real;self.skipped_forced_rows+=self.count-real
  action_count=int(self.frame_values[0]);self.scores[:action_count].zero_()
  if not real:
   assert nt==nq==nb1==nb2==0
   return
  features=self._buffer('features',nt,(128,));features.zero_()
  symbols=self._buffer('symbols',nt,dtype=torch.int64);slots=self._buffer('slots',nt,dtype=torch.int64)
  present=self._buffer('present',nt,dtype=torch.bool);query_tokens=self._buffer('queries',nq,dtype=torch.int64)
  blocks1=self._buffer('blocks1',nb1,(2,),torch.int32);blocks2=self._buffer('blocks2',nb2,(2,),torch.int32)
  owners=self._buffer('owners',action_count,dtype=torch.int32)
  max_rows=self.frame_values[3]+self.I
  ragged_hash[(self.count,triton.cdiv(max_rows,32))](self.ids,self.zones,self.numbers,self.meta,self.chunks,
   self.inv[0],self.inv[1],self.inv[3],self.inv[4],self.inv_slots,self.entity_slots,self.extra_slots,
   features,self.logical_ids,self.input_map,starts,self.base,self.m,self.I,T,32,num_warps=4)
  ragged_pack[(self.count,triton.cdiv(T,4))](features,self.symbols,self.counts,self.token_order,self.query_order,self.meta,starts,lens,
   symbols,slots,present,query_tokens,blocks1,blocks2,owners,T,num_warps=4)
  ragged_forward(self.flat,features,symbols,slots,present,query_tokens,starts,lens,blocks1,blocks2,
   self.state[:self.count],self.actions,self.keys,owners,self.canonical_sources,self.canonical_targets,self.meta,self.query_map,self.scores,action_count)
  self.real_neural_rows+=real;self.neural_rows+=real;self.model_calls+=1
  self.histogram['ragged_'+str((triton.next_power_of_2(nt),triton.next_power_of_2(nq)))]+=1
  if not hasattr(self,'ragged_counts'):self.ragged_counts=collections.Counter()
  self.ragged_counts.update(frames=1,tokens=nt,queries=nq,states=real,actions=action_count,attention_blocks=nb1+nb2)
 def close(self):
  if self.closed:return
  super().close()
  if hasattr(self,'ragged_buffers'):self.ragged_buffers.clear()

BattlePool=RaggedBattlePool
