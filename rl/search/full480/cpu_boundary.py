"""CPU boundary extracted from qualified flex/fallback implementations; no CUDA imports."""
from .run_device import artifact_directory
from pathlib import Path
import sys,os,time,collections
import numpy as np
import slaythespire as sts
P=artifact_directory()

from .run_device import cpu_bridge
cpu=cpu_bridge()

def native_result(features,result,turn_limit,seconds):
 np.testing.assert_array_equal(features['action_bits'],result['action_bits'])
 s=result['statistics'];n=result['simulations'];hist=np.asarray(result['playout_lengths'],np.int64)
 assert hist.shape==(201,) and hist.sum()==n and np.dot(hist,np.arange(201))==s[7]
 z=dict(features,action=int(result['action_bits'][0]),visits=result['visits'],terminal_visits=result['terminal_visits'],values=result['values'],value_statistic='terminal-return-lower-bound')
 if 'combat_context' in result:z['combat_context']=result['combat_context']
 z['stats']=dict(simulations=n,tree_transitions=0,playout_transitions=result['actual_transitions'],terminal_wins=int(s[3]),terminal_losses=int(s[4]),
  terminal_escapes=int(s[9]),depth_cuts=result['depth_cuts'],tree_cuts=0,chance_replays=0,nodes=0,policy_calls=result['nonforced_decisions'],
  action_scores=int(s[10]),forced_policy_skips=result['forced_actions'],prior_calls=0,draw_edges=0,guard_transitions=0,
  turn_card_limit_decisions=int(turn_limit),turn_card_limit_simulated=0,playout_lengths=hist,search_seconds=float(seconds))
 assert z['stats']['terminal_wins']+z['stats']['terminal_losses']+z['stats']['terminal_escapes']+z['stats']['depth_cuts']==n
 return z

class GameMirror:
 def __init__(self,envs,arena,*,verify=True):
  self.envs=envs;self.arena=arena;self.identities={};self.verify=verify;self.counts=collections.Counter();self.before_states={};self.before_versions={}
 def ensure(self,indices):
  drop=[];load={};identities={}
  for i in indices:
   e=self.envs[i];bc=e.thick.bc if e.thick is not None and e._phase()=='combat' else None
   if bc is not None and self.identities.get(i) is bc:continue
   if i in self.identities:drop.append(i);del self.identities[i]
   if bc is not None:
    s=cpu.State(bc).snapshot()
    if s['portable'] and s['input_state']==1 and s['outcome']==0:
     load[i]=s;identities[i]=bc
    else:self.counts['initial_unimportable']+=1
  if drop:self.arena.drop(drop);self.counts['dropped_battles']+=len(drop)
  if load:self.arena.load(load);self.identities.update(identities);self.counts['loaded_battles']+=len(load)
 def check(self,states):
  if not states:return
  words,meta,bits=self.arena.snapshot(list(states))
  for j,(i,state) in enumerate(states.items()):
   snap=state.snapshot();actions=state.actions()
   np.testing.assert_array_equal(meta[j],[len(snap['words']),snap['queue_size'],snap['max_vector'],snap['input_state'],snap['outcome'],len(actions)])
   np.testing.assert_array_equal(words[j,:len(snap['words'])],snap['words'])
   np.testing.assert_array_equal(bits[j,:len(actions)],actions)
   self.counts['verified_states']+=1;self.counts['verified_words']+=len(snap['words'])
 def before(self,ordered):
  assert not self.before_versions
  self.ensure(ordered)
  self.before_versions={i:self.arena.versions[i] for i in ordered if i in self.identities}
  self.before_states={i:cpu.State(self.identities[i]) for i in self.before_versions} if self.verify else {}
 def after(self,ordered,infos):
  assert len(ordered)==len(infos)
  moves={}
  for i,info in zip(ordered,infos,strict=True):
   if i not in self.before_versions:continue
   if info.get('simulator_transition') is False:
    assert 'native_action' not in info
    self.counts['selection_prefix_noop']+=1
   else:
    assert 'native_action' in info,('missing original atomic action',i,info.keys())
    bits=int(info['native_action']);moves[i]=(*self.before_versions[i],bits)
    if self.verify:self.before_states[i].step(bits)
  self.arena.step(moves);self.counts['actual_actions']+=len(moves);self.counts['advance_batches']+=bool(moves)
  if self.verify:self.check(self.before_states)
  self.before_versions.clear();self.before_states.clear()
  self.ensure(ordered)
  if self.verify:self.check({i:cpu.State(self.identities[i]) for i in ordered if i in self.identities})
 def source(self,index):
  if index not in self.identities:return None
  if self.verify:self.check({index:cpu.State(self.identities[index])})
  self.counts['root_clones']+=1
  return (index,*self.arena.versions[index])
 def close(self):
  assert not self.before_versions
  self.arena.close();self.identities.clear()

class CpuFallback:
 def __init__(self,jobs,environments,service,*,capacity=128,threads=8,reference=False,rows=None):
  assert jobs and capacity>0;self.jobs=list(jobs);self.service=service;self.reference=reference;self.pending=None;self.rows=0;self.calls=0;self.complete=False
  self.temps=[j[2].temperature for j in jobs]
  self.root_contexts=None;self.paired=rows is not None;self.root_rows=rows;self.environments=environments
  if self.paired:
   from .run_device import cpu_fallback
   native=cpu_fallback()
   self.roots=[native.WaveRoot(environments[i].thick.gc,environments[i].thick.bc,cfg,nonce,True) for i,tier,cfg,nonce in jobs]
   for root,(i,_,_,_) in zip(self.roots,jobs,strict=True):
    row=rows[i];root.configure_paired(row['action_bits'],row['prefix'],row['worlds'])
   from rl.policies.deck_latent import RunContextEncoder
   keys=(*RunContextEncoder.keys,'run_keys','ascension','relic_counters')
   self.context_globals={key:np.stack([rows[i]['observation'][key] for i,_,_,_ in jobs]) for key in keys}
  else:
   native=sts
   self.roots=[sts.WaveRoot(environments[i].thick.gc,environments[i].thick.bc,cfg,nonce) for i,tier,cfg,nonce in jobs]
  for i,r in enumerate(self.roots):r.inventory_key=i
  self.wave=native.WaveDiag(self.roots,min(capacity,sum(j[2].simulations for j in jobs)),threads)
  self.wave.full_tokens=True;self.wave.fast_codec=False;self.wave.gpu_choice=True;self.wave.aos_mode=True;self.wave.short_atoms=False
  self.wave.raw_mode=True;self.wave.aos_mode=False
  assert len(set(self.temps))==1
  self.audit=os.environ.get("FALLBACK_VERIFY","off")=="all"
  self.metrics=dict(pending_wall_seconds=0.,cpu_step_seconds=0.,pending_polls=0,rows_checked=0,raw_contract_violations=0,prob_max_abs=0.,max_kl=0.,top1_changes=0,draw_changes=0,same_logit_sampler_changes=0,frames=0,encode_score_seconds=0.,audit_seconds=0.,shapes={})
 def validate(self,fields,actual,picks,gold,raw):
  for i,row in enumerate(raw):
   start=int(fields[15][i]);n=int(fields[16][i]);temp=self.temps[0]
   assert n==len(row['action_bits'])
   x=actual[start:start+n];g=gold[i,:n];assert np.isfinite(x).all() and np.isfinite(g).all()
   px=np.exp((x-x.max())/temp);px/=px.sum();pg=np.exp((g-g.max())/temp);pg/=pg.sum()
   diff=float(np.abs(px-pg).max());kl=float(np.sum(pg*(np.log(pg)-np.log(px))))
   m=self.metrics;m['prob_max_abs']=max(m['prob_max_abs'],diff);m['max_kl']=max(m['max_kl'],kl)
   m['raw_contract_violations']+=int((np.abs(x-g)>3e-5+1e-5*np.abs(g)).sum())
   u=float(fields[17][i]);gp=min(n-1,int(np.searchsorted(pg.cumsum(),u,side='right')))
   xp=min(n-1,int(np.searchsorted(px.cumsum(),u,side='right')))
   m['top1_changes']+=int(x.argmax()!=g.argmax());m['draw_changes']+=int(picks[i]!=gp)
   m['same_logit_sampler_changes']+=int(picks[i]!=xp);m['rows_checked']+=1
   if not(diff<=2e-5 and kl<=1e-8 and xp==picks[i]==gp and self.metrics['top1_changes']==0):
    import pickle,json
    name=os.environ['WAVE_LIVE_TAG']
    (P/(name+'-numeric-failure.json')).write_text(json.dumps(self.metrics,indent=2))
    with (P/(name+'-numeric-failure.pkl')).open('xb') as f:pickle.dump(dict(fields=fields,raw=raw,gold=gold,actual=actual,picks=picks,temps=self.temps),f)
   assert diff<=2e-5 and kl<=1e-8,(i,diff,kl)
   assert xp==picks[i]==gp,(i,int(picks[i]),xp,gp)
  assert self.metrics['top1_changes']==0,self.metrics
 def step(self):
  start=time.monotonic();assert not self.complete
  if self.pending is not None:
   ident,fields,raw,shape,submitted=self.pending
   ready,answer=self.service.take_reply(ident)
   if not ready:
    self.metrics['pending_polls']+=1
    self.metrics['cpu_step_seconds']+=time.monotonic()-start
    return []
   self.pending=None;self.metrics['pending_wall_seconds']+=time.monotonic()-submitted
   picks=answer['picks']
   if 'root_contexts' in answer:self.root_contexts=answer['root_contexts']
   if self.audit and not self.reference and not self.paired:
    tick=time.monotonic();self.validate(fields,answer['logits'],picks,answer['gold'],raw);self.metrics['audit_seconds']+=time.monotonic()-tick
   if self.paired:
    self.wave.choose_paired(np.asarray(picks,np.int32),np.asarray(answer['values'],np.float64))
    self.rows+=len(fields['draws'])
   else:
    self.wave.choose(np.asarray(picks,np.int32));self.rows+=len(fields[0])
   self.calls+=1
   self.metrics['frames']+=1;self.metrics['shapes'][shape]=self.metrics['shapes'].get(shape,0)+1
   self.metrics['cpu_step_seconds']+=time.monotonic()-start
   return []
  b,n,t,q=self.wave.ready()
  if not b:
   self.complete=True;results=self.wave.paired_results() if self.paired else self.wave.results()
   if self.root_contexts is not None:
    for result,context in zip(results,self.root_contexts,strict=True):result['combat_context']=context
   assert sum(r['simulations'] if self.paired else r['stats']['simulations'] for r in results)==sum(j[2].simulations for j in self.jobs)
   assert sum(r['nonforced_decisions'] if self.paired else r['stats']['policy_calls'] for r in results)==self.rows
   self.metrics.update(rows=self.rows,calls=self.calls,completed=True)
   self.metrics['cpu_step_seconds']+=time.monotonic()-start
   return list(zip(self.jobs,results,strict=True))
  if self.paired:
   fields=self.wave.pack_raw()
   fields['root_context_request']=self.root_contexts is None
   tick=time.monotonic();ident=self.service.begin_call('fallback',(fields,1.,self.context_globals,self.reference))
   self.pending=(ident,fields,None,str((b,n)),tick)
   self.metrics['encode_score_seconds']+=time.monotonic()-tick
   self.metrics['cpu_step_seconds']+=time.monotonic()-start
   return []
  e=self.wave.entity_count();ints=lambda shape:np.empty(shape,np.int32)
  fields=[np.empty((b,32),np.float32),np.empty((n,24),np.float32),ints((n,10)),ints(n),np.zeros((b,t),np.int32),
   np.zeros((b,t),np.int32),np.zeros((b,t),np.int32),np.zeros((b,t),bool),np.zeros((b,max(1,q)),np.int32),
   ints((n,10)),ints(n),np.empty((e,16),np.float32),ints(e),ints(e),ints(e),ints(b),ints(b),np.empty(b,np.float64)]
  self.wave.pack_raw(fields)
  assert (fields[8]>=0).all() and (fields[8]<t).all(),'query index outside compact token extent'
  raw=[self.wave.raw(i) for i in range(b)] if self.audit or self.reference else None
  tick=time.monotonic();ident=self.service.begin_call('fallback',(fields,self.temps[0],raw,self.reference))
  self.pending=(ident,fields,raw,str((b,t,q)),tick)
  self.metrics['encode_score_seconds']+=time.monotonic()-tick
  self.metrics['cpu_step_seconds']+=time.monotonic()-start
  return []
