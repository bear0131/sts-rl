"""Lazy complete readonly frame, followed by the original search action."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'devicegraph141'))
import layout_bindings
from graph_runtime import *
from graph_runtime import BattlePool as BasePool
from step_pipeline import StepPipeline as FramePipeline,refresh as step_refresh
from combo_model import GraphScorer as ComboScorer

class BattlePool(BasePool):
 def __init__(self,*a,frame_tokens=None,frame_queries=None,**kw):
  self.frame_pipeline=None;self._observed=False;self._frame_ready=False;self._scored=False
  super().__init__(*a,**kw)
  assert all(p.dtype==torch.float32 for p in self.net.parameters())
  self.graph_scorer.close();self.graph_scorer=ComboScorer(self.flat)
  self.frame_pipeline=FramePipeline(self,self.graph_scorer,frame_tokens,frame_queries);self.graph_scorer=self.frame_pipeline
 def refresh(self):return step_refresh(self)
 def tick(self):return self.frame_pipeline.tick()
 def allocate_actions(self,n):
  if getattr(self,'frame_pipeline',None) is not None:self.frame_pipeline.release()
  return super().allocate_actions(n)
 def allocate_entities(self,n):
  if getattr(self,'frame_pipeline',None) is not None:self.frame_pipeline.release()
  return super().allocate_entities(n)
 def grow_inventory(self,n):
  if n>self.I and getattr(self,'frame_pipeline',None) is not None:self.frame_pipeline.release()
  return super().grow_inventory(n)
 def observe(self,base=0,count=None):
  count=min(self.frame_cap,self.active_count-base) if count is None else count
  assert not self.closed and base>=0 and 0<count<=self.frame_cap and base+count<=self.active_count
  assert not self._scored,'cannot overwrite a credited but unapplied frame'
  self.base=base;self.count=count;self._observed=True;self._frame_ready=False
 def _materialize(self):
  assert self._observed
  signature=self.frame_pipeline.signature()
  if not self._frame_ready or signature!=self._frame_signature or any(v is None for _,v in signature):
   self._frame_values=self.frame_pipeline.run(self.base,self.count);self.frame_values=self._frame_values[:5]
   self._frame_signature=signature;self._frame_ready=True
  return self._frame_values
 def public(self):
  self._materialize();return super().public()
 def score(self,callback=None):
  assert callback is None and not self._scored
  a,e,flags,me,ma,nt,nq,nb1,nb2,real=self._materialize()
  self.scored_count=real;self.skipped_forced_rows+=self.count-real
  if real:
   self.real_neural_rows+=real;self.neural_rows+=real;self.model_calls+=1
   self.histogram['frame_'+str((triton.next_power_of_2(nt),triton.next_power_of_2(nq)))]+=1
   if not hasattr(self,'ragged_counts'):self.ragged_counts=collections.Counter()
   self.ragged_counts.update(frames=1,tokens=nt,queries=nq,states=real,actions=a,attention_blocks=nb1+nb2)
  self._scored=True
 def apply(self):
  assert self._scored and self._frame_ready and self._observed
  super().apply();self._scored=False;self._frame_ready=False;self._observed=False
