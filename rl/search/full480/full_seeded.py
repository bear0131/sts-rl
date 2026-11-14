"""Original full480 asynchronous stream and natural-game root consumers."""
from pathlib import Path
import sys,os,time,json,hashlib,collections,inspect
from .service_devicestepfull480141 import GpuService,np,torch
from .cpu_boundary import GameMirror,CpuFallback,native_result,sts,cpu
from .finite_plan import PAIRED_TERMINAL,paired_worlds
from rl.algorithms.upper_probe import state_digest
P=Path(os.environ.get('STS_FULL480_RUN_DIR','runs/local/full480'))
RUN=os.environ.get('WAVE_LIVE_TAG','collection')
count=collections.Counter();terminals={};actual_game_ids=[];fallback_reports=[]
current_stream=None;model_owner=None;t0=0.;last=0.
def save(name,data):
 P.mkdir(parents=True,exist_ok=True)
 (P/(RUN+'-'+name+'.json')).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x)))
def report(refresh=False):
 service=None if current_stream is None else current_stream.service
 status={} if service is None else service.statistics() if refresh else service.status
 return dict(tag=RUN,full_wall=time.monotonic()-t0,counts=dict(count),terminals=terminals,actual_game_ids=actual_game_ids,
  fallback_reports=fallback_reports,schedule=dict(getattr(model_owner,'_independent_rollout_progress',{})),
  service=status,host_service={} if service is None else dict(service.counts),pending_requests={} if service is None else dict(service.requests),pool=status.get('pool'),mirror={} if current_stream is None else dict(current_stream.game.counts))
def plan_job(job,envs,obs):
 i,tier,cfg,nonce=job;e=envs[i]
 r=cpu.State(e.thick.bc).root_plan(nonce,cfg.simulations,cfg.temperature,cfg.playout_depth,cfg.boss)
 semantic=dict(zip(r['action_bits'],r['semantic'],strict=True));mapping=sorted(e._joint_policy_actions())
 r.update(cpu.inventory(e.thick.gc))
 r.update(contract=PAIRED_TERMINAL,worlds=paired_worlds(nonce,len(mapping),cfg.simulations),prefix=sum(1<<j for j in (e._multi or ())),
  action_bits=np.asarray([bits for _,bits in mapping],np.uint32),
  semantic=np.asarray([semantic.get(bits,0) for _,bits in mapping],np.uint64),order=np.arange(len(mapping)),
  observation=obs,relics=obs['relics'],relic_counters=obs['relic_counters'])
 return r
class Stream:
 def __init__(self,batch,*,threads):
  global current_stream,model_owner,t0,last
  model=batch.model
  current_stream=self;model_owner=model;t0=last=time.monotonic();count.clear();terminals.clear();fallback_reports.clear()
  self.batch=batch;self.meta={};self.inline={};self.queue=[];self.fallback=None;self.threads=min(8,threads);self.fallback_rows={};self.cancelled=False
  P.mkdir(parents=True,exist_ok=True);path=P/(RUN+'-policy.pt')
  digest=state_digest(model.policy.state_dict());torch.save(model.policy,path)
  self.service=GpuService(len(batch.envs),path,expected_digest=digest,
   prototypes=batch.observer.prototypes,frozen_policy=model.policy,
   budget_cap=max(batch.pair[t+'_simulations'] for t in ('common','elite','boss')))
  self.game=GameMirror(batch.envs,self.service.arena,verify=os.environ.get('BATTLE_VERIFY','off')=='all')
 def __enter__(self):return self
 def cancel(self,reason):
  self.cancelled=True;self.service.close(cancel=reason)
 def __exit__(self,kind,value,tb):
  if kind is None and not self.cancelled:
   assert not self.pending and not self.meta
   self.service.close()
  elif not self.cancelled:
   try:self.service.close(cancel=repr(value))
   except BaseException as cleanup:print('SERVICE_CLEANUP_ERROR',repr(cleanup),flush=True)
  self.batch.inference.__exit__(kind,value,tb)
  self.game.close()
  assert not self.service.process.is_alive()
 @property
 def pending(self):return bool(self.meta or self.queue or self.fallback is not None or self.inline)
 def submit(self,calls):
  if not calls:return
  start=time.monotonic();self.game.ensure(calls);inline,jobs=self.batch.prepare(calls);self.inline.update(inline);rows={}
  observations=self.batch.observer.encode([self.batch.envs[j[0]] for j in jobs]) if jobs else {}
  for row,job in enumerate(jobs):
   i,tier,cfg,nonce=job;assert i not in self.meta;e=self.batch.envs[i]
   r=plan_job(job,self.batch.envs,{key:value[row] for key,value in observations.items()});features=dict(action_bits=r['action_bits'],observation=r['observation'],max_hp=r['max_hp'])
   self.meta[i]=(job,features,time.monotonic());count['allocated_'+tier]+=cfg.simulations
   source=self.game.source(i)
   if source is not None:
    from rl.policies.deck_latent import RunContextEncoder
    r['mirror']=source
    rows[i]=dict(r,observation={key:r['observation'][key] for key in (*RunContextEncoder.keys,'run_keys','ascension')})
    count['submitted_gpu']+=1
   else:
    self.queue.append(job);self.fallback_rows[i]=r;count['submitted_cpu']+=1
  if rows:
   count['overlap_submitted']+=len(rows) if self.service.outstanding else 0
   self.service.submit(rows);count['submit_batches']+=1
  count['submit_seconds']+=time.monotonic()-start
 def receive(self,timeout=0.):
  global last
  if any(time.monotonic()-at>3450. for _,_,at in self.meta.values()):
   raise TimeoutError('full480 search root deadline exhausted')
  start=time.monotonic();out,self.inline=self.inline,{};completed=[]
  wait=.001 if timeout and not out else 0.
  for i,result in self.service.receive(wait).items():
   job,packet,at=self.meta[i]
   completed.append((job,result));count['completed_gpu']+=1
  if self.fallback is None and self.queue:
   jobs,self.queue=self.queue,[];owned={j[0]:self.fallback_rows.pop(j[0]) for j in jobs}
   self.fallback=CpuFallback(jobs,self.batch.envs,self.service,capacity=128,threads=self.threads,rows=owned);count['cpu_groups']+=1
  if self.fallback is not None:
   tick=time.monotonic();cpu_ready=self.fallback.step();count['cpu_service_seconds']+=time.monotonic()-tick
   if self.fallback.complete:
    fallback_reports.append(self.fallback.metrics);save('fallback-checks',fallback_reports)
    completed.extend(cpu_ready);count['completed_cpu']+=len(cpu_ready);count['cpu_model_rows']+=self.fallback.rows;self.fallback=None
  if completed:
   tick=time.monotonic();out.update(self.accept_ready(completed));count['root_accept_seconds']+=time.monotonic()-tick
   for job,result in completed:
    i,tier,cfg,nonce=job;s=result['statistics'];assert result['simulations']==cfg.simulations
    count.update(roots=1,simulations=cfg.simulations,guide=result['nonforced_decisions'],steps=result['actual_transitions'],forced=result['forced_actions'],
     atomic_steps=result['atomic_decisions'],prefixes=result['selection_prefix_decisions'],cuts=result['depth_cuts'],terminal=int(s[3]+s[4]+s[9]),anchors=0)
    del self.meta[i]
  count['receive_seconds']+=time.monotonic()-start;count['receives']+=1
  if time.monotonic()-last>30:
   v=report(True);save('progress',v);print('NATURAL_LIVE',round(v['full_wall'],2),count['guide'],count['roots'],len(terminals),'gpu_pending',len(self.service.outstanding),'cpu',len(self.queue),flush=True);last=time.monotonic()
  return out
 def accept_ready(self,completed):
  ready=[]
  for job,result in completed:
   i,tier,cfg,nonce=job;_,features,at=self.meta[i]
   ready.append((job,native_result(features,result,sts.small_playout_turn_card_limit_reached(self.batch.envs[i].thick.bc),time.monotonic()-at)))
  return self.batch.accept_ready(ready)

class RemoteUpper:
 def __init__(self,*args):pass
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def statistics(self):return current_stream.service.call('upper_statistics')
def upper_begin(obs,masks):return current_stream.service.begin_call('upper',(obs,masks))
def upper_take(ident):
 ready,value=current_stream.service.take_reply(ident)
 return value if ready else None
def mirror_before(ordered):
 for i in ordered:assert i not in current_stream.meta
 current_stream.game.before(ordered)
def mirror_after(ordered,infos):current_stream.game.after(ordered,infos)

def record_start(session,env):
 actual_game_ids[:]=[int(e.thick.gc.seed) for e in env.venv.native_search_batch.envs]
 assert len(set(actual_game_ids))==env.num_envs
 save('game-seeds',dict(first_seed=session['first_seed'],actual_game_ids=actual_game_ids))

def record_end(session,indices,dones,infos):
 for i,done,info in zip(indices,dones,infos,strict=True):
  if not done:continue
  assert not info.get('TimeLimit.truncated') and not info.get('episode_cohort_censored') and i not in terminals
  z=info['run_stats'];terminals[i]=dict(game_id=actual_game_ids[i],seed=session['first_seed']+i,
   win=int(z['act']>3 or z['outcome']=='PLAYER_VICTORY'),floor=float(z['floor']),outcome=z['outcome'],
   actions=info['episode']['l'],completion_wall_seconds=time.monotonic()-t0)
