"""One captured frame includes observation, model, apply, and final compaction."""
import collections
from .frame_pipeline import FramePipeline,torch,triton
from .step_kernels import compact_local,compact_scatter,compact_finish,frame_prepare,frame_commit

def compact(pool):
 n=pool.n;b=1024;nb=triton.cdiv(n,b);cb=triton.next_power_of_2(nb)
 compact_local[(nb,)](pool.alive,pool.step_prefix,pool.step_totals,pool.step_status,n,b)
 compact_scatter[(nb,)](pool.alive,pool.step_prefix,pool.step_totals,pool.live,pool.logical_ids,pool.step_status,n,pool.budget_cap,b,cb)
 compact_finish[(1,)](pool.step_totals,pool.step_status,nb,cb)

def refresh(pool):
 if not hasattr(pool,'step_status'):
  pool.step_status=torch.zeros(32,device='cuda',dtype=torch.int64)
  pool.step_prefix=torch.empty(pool.n,device='cuda',dtype=torch.int32)
  pool.step_totals=torch.empty(triton.cdiv(pool.n,1024),device='cuda',dtype=torch.int32)
  pool.live=torch.zeros(pool.n,device='cuda',dtype=torch.int32)
  pool.logical_ids=torch.zeros(pool.n,device='cuda',dtype=torch.int32)
  pool.step_counts=collections.Counter();pool.step_last=[0]*32
 else:
  assert pool.step_last[1]==0 and pool.step_last[3]==0,'refresh during unfinished frame'
 pool.step_status[4].fill_(1);compact(pool)
 pool.step_last=list(map(int,pool.step_status.cpu().tolist()))
 pool.active_count=pool.step_last[0]
 return pool.active_count

class StepPipeline(FramePipeline):
 def __init__(self,*a,**kw):
  self.step_graph=None;self.step_closed=False
  super().__init__(*a,**kw)
 def release(self):
  if self.step_graph is not None:
   torch.cuda.synchronize();self.step_graph.reset();self.step_graph=None;self.counts['step_evictions']+=1
  super().release()
 def fingerprint(self):
  p=self.pool
  extra=(p.live,p.logical_ids,p.alive,p.step_status,p.step_prefix,p.step_totals,p.job_roots,p.policy_seeds,p.temperatures,
   p.stats,p.outcomes,p.values,p.chosen,p.draws,p.root_boss,p.root_maximum,p.root_limits,p.job_generation,p.root_generation,p.root_done)
  return super().fingerprint()+(tuple(x.data_ptr() for x in extra),)
 def execute_step(self):
  p=self.pool;w=self.buffers
  frame_prepare[(triton.cdiv(self.F,256),)](p.live,p.logical_ids,w['live'],w['worlds'],w['dispatch'],p.step_status,self.F,256)
  FramePipeline.execute(self)
  p.launch('packed_apply_dynamic',[p.arena,w['live'],p.job_roots,p.scores,p.bits,p.meta,p.policy_seeds,p.temperatures,p.stats,
   p.alive,p.outcomes,p.values,p.chosen,p.draws,p.root_boss,p.root_maximum,p.root_limits,p.job_generation,p.root_generation,p.root_done,
   w['dispatch'],w['summary'],p.step_status],grid=self.F)
  frame_commit[(1,)](w['summary'],w['dispatch'],p.step_status)
  compact(p)
 @torch.inference_mode()
 def ensure_step_graph(self):
  if self.pool.deck_consumer:self.pool.flat.validate();self.pool.flat.input_tables.validate()
  key=self.fingerprint()
  if key!=self.key:self.allocate(key)
  if self.step_graph is None:
   p=self.pool
   p.step_status[2].zero_()
   self.execute_step();torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   try:
    with torch.cuda.graph(g):self.execute_step()
   except BaseException:
    g.reset();raise
   self.step_graph=g;self.counts['step_captures']+=1
   p.step_status[2].fill_(1)
 def grow_failed_frame(self,s):
  a,e,flags,me,ma,nt,nq,nb1,nb2,real=s[16:26]
  assert flags==s[3] and flags>0 and flags&~63==0
  self.history.append(dict(kind='step',cursor=s[1],count=s[26],summary=s[16:26],capacity=list(self.capacity),
   action_capacity=self.pool.action_cap,entity_capacity=self.pool.entity_cap))
  self.counts['step_overflow_retries']+=1
  p=self.pool;self.release()
  if a>p.action_cap:p.allocate_actions(triton.next_power_of_2(a))
  if e>p.entity_cap:p.allocate_entities(triton.next_power_of_2(e))
  if nt>self.requested_ct:self.requested_ct=triton.next_power_of_2(nt)
  if nq>self.requested_cq:self.requested_cq=triton.next_power_of_2(nq)
  p.step_status[3].zero_();s[3]=0
 @torch.inference_mode()
 def tick(self):
  p=self.pool
  assert p.active_count>0 and not p.closed and not p._scored and not p._observed
  before=p.step_last.copy();s=before.copy()
  assert s[1]==0 and s[3]==0 and s[0]==p.active_count
  for attempt in range(8):
   self.ensure_step_graph()
   frames=triton.cdiv(s[0]-s[1],self.F);assert frames>0
   for _ in range(frames):
    self.step_graph.replay();self.counts['step_replays']+=1
   s=list(map(int,p.step_status.cpu().tolist()));p.step_counts['host_reads']+=1
   if not s[3]:break
   self.grow_failed_frame(s)
  else:raise RuntimeError('Step graph overflow retry budget exhausted')
  assert s[5]==before[5]+1 and s[1]==0 and s[4]==0
  assert s[6]-before[6]==before[0]
  delta=[x-y for x,y in zip(s,before)]
  p.real_neural_rows+=delta[7];p.neural_rows+=delta[7];p.skipped_forced_rows+=delta[8];p.model_calls+=delta[9]
  if not hasattr(p,'ragged_counts'):p.ragged_counts=collections.Counter()
  p.ragged_counts.update(frames=delta[9],tokens=delta[10],queries=delta[11],states=delta[7],actions=delta[12],attention_blocks=delta[13])
  p.step_counts.update(applied=delta[6],nonforced=delta[7],forced=delta[8],frames=delta[14],overflow_frames=delta[15],ticks=1)
  p.step_last=s;p.active_count=s[0];p.iterations+=1
 def statistics(self):
  s=super().statistics();s.update(chain_step_version=1,step_graph_cached=int(self.step_graph is not None),
   cached=int(self.graph is not None)+int(self.step_graph is not None),step_counts=dict(self.pool.step_counts))
  return s
