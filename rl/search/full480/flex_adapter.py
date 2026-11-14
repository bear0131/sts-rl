"""Native collector result boundary and full-Transformer CPU fallback."""
from .run_device import artifact_directory
from pathlib import Path
import sys,time
P=artifact_directory()
from .flex_stream import *
import slaythespire as sts
from .run_device import cpu_bridge
cpu=cpu_bridge()
from rl.policies.token_regret import encode_public_roots,pack_rows
P=artifact_directory()
class CollectionPool(StreamingPool):
 def reap(self):
  out=super().reap()
  if out:
   slots=self.upload([r['root_slot'] for r in out.values()],np.int64)
   depths=self.stats.view(self.m,self.budget_cap,STAT)[slots,:,7].cpu().numpy()
   for k,r in enumerate(out.values()):
    d=depths[k,:r['simulations']];assert (d>=0).all() and (d<=200).all()
    r['playout_lengths']=np.bincount(d,minlength=201)
    assert r['playout_lengths'].sum()==r['simulations'] and np.dot(r['playout_lengths'],np.arange(201))==r['statistics'][7]
  return out
def native_result(features,result,turn_limit,seconds):
 np.testing.assert_array_equal(features['action_bits'],result['action_bits'])
 s=result['statistics'];n=result['simulations'];hist=np.asarray(result['playout_lengths'],np.int64)
 assert hist.shape==(201,) and hist.sum()==n and np.dot(hist,np.arange(201))==s[0]
 z=dict(features,action=int(result['action_bits'][0]),visits=result['visits'],terminal_visits=result['terminal_visits'],values=result['values'],value_statistic='terminal-return-lower-bound')
 z['stats']=dict(simulations=n,tree_transitions=0,playout_transitions=result['actual_transitions'],terminal_wins=int(s[3]),terminal_losses=int(s[4]),
  terminal_escapes=int(s[9]),depth_cuts=result['depth_cuts'],tree_cuts=0,chance_replays=0,nodes=0,policy_calls=result['nonforced_decisions'],
  action_scores=int(s[10]),forced_policy_skips=result['forced_actions'],prior_calls=0,draw_edges=0,guard_transitions=0,
  turn_card_limit_decisions=int(turn_limit),turn_card_limit_simulated=0,playout_lengths=hist,search_seconds=float(seconds))
 assert z['stats']['terminal_wins']+z['stats']['terminal_losses']+z['stats']['terminal_escapes']+z['stats']['depth_cuts']==n
 return z
class CpuFallback:
 def __init__(self,jobs,environments,net,*,capacity=128,threads=8):
  assert jobs and capacity>0;self.jobs=list(jobs);self.net=net;self.rows=0;self.calls=0;self.complete=False
  self.temps=[j[2].temperature for j in jobs]
  self.roots=[sts.WaveRoot(environments[i].thick.gc,environments[i].thick.bc,cfg,nonce) for i,tier,cfg,nonce in jobs]
  for i,r in enumerate(self.roots):r.inventory_key=i
  self.wave=sts.WaveDiag(self.roots,min(capacity,sum(j[2].simulations for j in jobs)),threads)
  self.wave.full_tokens=True;self.wave.fast_codec=False;self.wave.gpu_choice=True;self.wave.aos_mode=True;self.wave.short_atoms=False
 @torch.inference_mode()
 def step(self):
  assert not self.complete
  n,acts,t,q=self.wave.ready()
  if not n:
   self.complete=True;results=self.wave.results()
   assert sum(r['stats']['simulations'] for r in results)==sum(j[2].simulations for j in self.jobs)
   assert sum(r['stats']['policy_calls'] for r in results)==self.rows
   return list(zip(self.jobs,results,strict=True))
  assert n>0 and acts>=2*n
  entities=self.wave.entity_count()
  def ints(shape):return np.empty(shape,np.int32)
  fields=[np.empty((n,32),np.float32),np.empty((acts,24),np.float32),ints((acts,10)),ints((acts,)),ints((n,max(1,q))),ints((acts,10)),ints((acts,)),
   ints((entities,19)),ints((entities,)),ints((entities,)),ints((n,)),ints((n,)),ints((n,)),np.empty(n,np.float64)]
  self.wave.pack_aos(fields)
  raw=[self.wave.raw(i) for i in range(n)];tokens=encode_public_roots(raw)
  for r,token in zip(raw,tokens,strict=True):r['tokens']=token
  scores=self.net.forward_packed(*pack_rows(raw,'cuda')).cpu().double().numpy()
  picks=[]
  for i,r in enumerate(raw):
   a=len(r['action_bits']);assert a>1 and a==fields[12][i]
   x=scores[i,:a];assert np.isfinite(x).all();temp=self.temps[int(fields[10][i])]
   p=np.exp((x-x.max())/temp);p/=p.sum();u=float(fields[13][i]);assert 0<=u<1
   picks.append(min(a-1,int(np.searchsorted(p.cumsum(),u,side='right'))))
  self.wave.choose(np.asarray(picks,np.int32));self.rows+=n;self.calls+=1
  return []
