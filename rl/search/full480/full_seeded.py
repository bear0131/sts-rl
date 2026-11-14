"""Original full480 asynchronous stream, consuming fixed-length Deck PPO rows."""
from pathlib import Path
import sys,os,time,json,hashlib,collections,inspect
from .service_devicestepfull480141 import GpuService,np,torch
from .cpu_boundary import GameMirror,CpuFallback,sts,cpu
from .finite_plan import PAIRED_ANCHOR,paired_worlds
from rl.algorithms.upper_probe import state_digest
P=Path(os.environ.get('STS_FULL480_RUN_DIR','runs/local/full480'))
RUN=os.environ.get('WAVE_LIVE_TAG','collection')
count=collections.Counter();terminals={};actual_game_ids=[];fallback_reports=[]
current_stream=None;model_owner=None;t0=0.;last=0.
def save(name,data):
 (P/(RUN+'-'+name+'.json')).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x)))
def report(refresh=False):
 service=None if current_stream is None else current_stream.service
 status={} if service is None else service.statistics() if refresh else service.status
 return dict(tag=RUN,full_wall=time.monotonic()-t0,counts=dict(count),terminals=terminals,actual_game_ids=actual_game_ids,
  fallback_reports=fallback_reports,schedule=dict(getattr(model_owner,'_independent_rollout_progress',{})),
  service=status,host_service={} if service is None else dict(service.counts),pending_requests={} if service is None else dict(service.requests),pool=status.get('pool'),mirror={} if current_stream is None else dict(current_stream.game.counts))
def plan_job(job,envs,packet):
 i,tier,cfg,nonce=job;e=envs[i]
 assert cfg.paired_worldlines and 1<=cfg.playout_depth<=200
 r=cpu.State(e.thick.bc).root_plan(nonce,cfg.simulations,cfg.temperature,cfg.playout_depth,cfg.boss)
 semantic=dict(zip(r['action_bits'],r['semantic'],strict=True));mapping=sorted(e._joint_policy_actions())
 obs=packet['observation'];worlds=paired_worlds(nonce,len(mapping),cfg.simulations)
 r.update(cpu.inventory(e.thick.gc))
 r.update(contract=PAIRED_ANCHOR,worlds=worlds,prefix=sum(1<<j for j in (e._multi or ())),
  action_bits=np.asarray([bits for _,bits in mapping],np.uint32),
  semantic=np.asarray([semantic.get(bits,0) for _,bits in mapping],np.uint64),order=np.arange(len(mapping)),
  context_inputs=np.concatenate((packet['context'][0],obs['run_keys'],obs['ascension'])).astype(np.float32),
  anchor_context=packet['context'][1],observation=obs,relics=obs['relics'],relic_counters=obs['relic_counters'])
 return r
class Stream:
 def __init__(self,batch,*,threads,model,config):
  global current_stream,model_owner,t0,last
  current_stream=self;model_owner=model;t0=last=time.monotonic();count.clear();terminals.clear();fallback_reports.clear()
  self.batch=batch;self.meta={};self.inline={};self.queue=[];self.fallback=None;self.config=config;self.threads=threads;self.fallback_rows={};self.cancelled=False
  P.mkdir(parents=True,exist_ok=True);path=P/(RUN+'-policy.pt')
  digest=state_digest(model.policy.state_dict());torch.save(model.policy,path)
  self.service=GpuService(len(batch.envs),path,expected_digest=digest,
   prototypes={k:v[0] for k,v in model._last_obs.items()},frozen_policy=model.policy,
   budget_cap=max(198,config.normal_budget,config.boss_budget))
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
  self.game.close()
  assert not self.service.process.is_alive()
 @property
 def pending(self):return bool(self.meta or self.queue or self.fallback is not None or self.inline)
 def submit(self,calls):
  if not calls:return
  start=time.monotonic();self.game.ensure(calls);jobs=[];rows={}
  for i,packet in calls.items():
   e=self.batch.envs[i];boss=bool(packet['observation']['combat_is_boss'].item());budget=self.config.boss_budget if boss else self.config.normal_budget
   na=len(e._joint_policy_actions());cfg=sts.SmallPlayoutConfig();cfg.simulations=na*max(1,budget//na)
   cfg.playout_depth=self.config.max_depth;cfg.paired_worldlines=True;cfg.boss=boss
   jobs.append((i,'boss' if boss else 'common',cfg,packet['nonce']))
  for job in jobs:
   i,tier,cfg,nonce=job;assert i not in self.meta;e=self.batch.envs[i]
   self.meta[i]=(job,calls[i],time.monotonic())
   r=plan_job(job,self.batch.envs,calls[i]);count['allocated_'+tier]+=cfg.simulations
   source=self.game.source(i)
   if source is not None:
    # The real PPO row stays with the collector. GPU branches only consume the
    # frozen context fields; their live observations come from the old observer.
    from rl.policies.deck_latent import RunContextEncoder
    r['mirror']=source
    rows[i]=dict(r,observation={key:r['observation'][key] for key in (*RunContextEncoder.keys,'run_keys')})
    count['submitted_gpu']+=1
    count['mirrored_pending_roots']+=int(r['snapshot']['input_state']==2);count['mirrored_callback_roots']+=int(r['snapshot']['queue_size']>0)
   else:
    self.queue.append(job);self.fallback_rows[i]=r;count['submitted_cpu']+=1
    count['cpu_input_'+str(r['snapshot']['input_state'])]+=1;count['cpu_nonportable']+=int(not r['snapshot']['portable'])
  if rows:
   count['overlap_submitted']+=len(rows) if self.service.outstanding else 0
   self.service.submit(rows);count['submit_batches']+=1
  count['submit_seconds']+=time.monotonic()-start
 def receive(self,timeout=0.):
  global last
  if any(time.monotonic()-at>self.config.timeout_seconds for _,_,at in self.meta.values()):
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
     atomic_steps=result['atomic_decisions'],prefixes=result['selection_prefix_decisions'],cuts=result['depth_cuts'],terminal=int(s[3]+s[4]+s[9]),anchors=int(s[3]+s[9]))
    del self.meta[i]
  count['receive_seconds']+=time.monotonic()-start;count['receives']+=1
  if time.monotonic()-last>30:
   v=report(True);save('progress',v);print('NATURAL_LIVE',round(v['full_wall'],2),count['guide'],count['roots'],len(terminals),'gpu_pending',len(self.service.outstanding),'cpu',len(self.queue),flush=True);last=time.monotonic()
  return out
 def accept_ready(self,completed):
  from rl.algorithms.search_auxiliary import improved_prior
  from env.search.action_selection import search_filtered_choice
  out={}
  for job,result in completed:
   i,_,_,nonce=job;packet=self.meta[i][1];mapping=dict((int(bits),int(action)) for action,bits in self.batch.envs[i]._joint_policy_actions())
   choices=np.asarray([mapping[int(bits)] for bits in result['action_bits']],np.int64)
   logits=np.asarray(packet['logits'],np.float32);q=np.zeros_like(logits);q[choices]=result['values']
   target=improved_prior(torch.from_numpy(logits[None]),torch.from_numpy(q[None]),torch.from_numpy(packet['mask'][None]),
    torch.tensor([int(result['visits'][0])]),self.config.target_std)[0].numpy()
   chosen,logp=search_filtered_choice(logits[choices],np.asarray(result['values']),result['visits'],nonce)
   out[i]=dict(action=int(choices[chosen]),log_prob=logp,target=target,changed=int(choices[chosen])!=packet['policy_action'])
  return out

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
