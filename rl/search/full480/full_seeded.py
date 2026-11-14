"""Frozen full natural collection with CPU-only games and one GPU owner."""
from pathlib import Path
import sys,os,time,json,hashlib,collections,inspect
from service_devicestepfull480141 import GpuService,np,torch
from cpu_boundary import GameMirror,CpuFallback,native_result,sts,cpu
from rl.algorithms import independent_rollout as ind
from rl.training import distributed_sb3
from rl.training.resource_monitor import ResourceMonitor
from rl.algorithms.upper_probe import state_digest
P=Path(__file__).resolve().parent
assert Path(inspect.getfile(GpuService)).resolve()==P/'service_devicestepfull480141.py'
RUN=os.environ['WAVE_LIVE_TAG'];EXPECTED=int(os.environ['WAVE_LIVE_GAMES'])
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
def plan_job(job,envs):
 i,tier,cfg,nonce=job;e=envs[i]
 assert cfg.playout_depth==200 and cfg.paired_worldlines and cfg.simulations==(1000 if tier=='boss' else 250)
 r=cpu.State(e.thick.bc).root_plan(nonce,cfg.simulations,cfg.temperature,cfg.playout_depth,cfg.boss)
 r.update(cpu.inventory(e.thick.gc));return r
def qualify_first(job,envs,service):
 start=time.monotonic();i,tier,cfg,nonce=job;e=envs[i]
 config=sts.SmallPlayoutConfig()
 for name in ('boss','exploration','paired_worldlines','playout_depth','prior_mix','prior_temperature','temperature','tree_depth','use_prior'):setattr(config,name,getattr(cfg,name))
 config.simulations=8;nonce=2027060100000;probe=(i,tier,config,nonce)
 row=cpu.State(e.thick.bc).root_plan(nonce,8,config.temperature,200,config.boss);row.update(cpu.inventory(e.thick.gc))
 gpu=service.call('probe',row)
 native=CpuFallback([probe],envs,service,capacity=8,threads=1,reference=True);done=[]
 while not native.complete:
  assert time.monotonic()-start<300;done.extend(native.step())
  if native.pending is not None:service.poll(.001)
 assert len(done)==1
 a=native_result(sts.small_playout_features(e.thick.bc),gpu,sts.small_playout_turn_card_limit_reached(e.thick.bc),0);b=done[0][1]
 for key in ('state','actions','keys','action_bits','visits','terminal_visits'):np.testing.assert_array_equal(a[key],b[key])
 np.testing.assert_allclose(a['values'],b['values'],atol=1e-12,rtol=0)
 assert set(a['stats'])==set(b['stats'])
 for key in a['stats']:
  if key!='search_seconds':np.testing.assert_array_equal(a['stats'][key],b['stats'][key],err_msg=key)
 return dict(completed=True,nonce=nonce,budget=8,stats=a['stats'],values=a['values'],seconds=time.monotonic()-start)

class Stream:
 def __init__(self,batch,*,threads):
  global current_stream
  current_stream=self;self.batch=batch;self.meta={};self.inline={};self.queue=[];self.fallback=None;self.checked=False
  self.service=GpuService(len(batch.envs),P/(RUN+'-policy.pt'))
  self.game=GameMirror(batch.envs,self.service.arena,verify=os.environ.get('BATTLE_VERIFY','all')=='all')
 def __enter__(self):return self
 def __exit__(self,kind,value,tb):
  if kind is None:
   assert not self.pending and not self.meta
   self.service.close();self.game.close()
  else:
   try:self.service.close(cancel=repr(value))
   except BaseException as cleanup:print('SERVICE_CLEANUP_ERROR',repr(cleanup),flush=True)
  assert not self.service.process.is_alive() and not torch.cuda.is_initialized()
 @property
 def pending(self):return bool(self.meta or self.queue or self.fallback is not None or self.inline)
 def submit(self,calls):
  if not calls:return
  start=time.monotonic();self.game.ensure(calls);inline,jobs=self.batch.prepare(calls)
  assert not(set(inline)&set(self.inline)) and not(set(inline)&set(self.meta))
  self.inline.update(inline);rows={}
  for job in jobs:
   i,tier,cfg,nonce=job;assert i not in self.meta;e=self.batch.envs[i]
   self.meta[i]=(job,sts.small_playout_features(e.thick.bc),sts.small_playout_turn_card_limit_reached(e.thick.bc),time.monotonic())
   r=plan_job(job,self.batch.envs);count['allocated_'+tier]+=cfg.simulations
   source=self.game.source(i)
   if source is not None:
    r['mirror']=source;rows[i]=r;count['submitted_gpu']+=1
    count['mirrored_pending_roots']+=int(r['snapshot']['input_state']==2);count['mirrored_callback_roots']+=int(r['snapshot']['queue_size']>0)
    if not self.checked and r['eligible']:
     result=qualify_first(job,self.batch.envs,self.service);save('first-root-pair',result);count['diagnostic_seconds']+=result['seconds'];self.checked=True
   else:
    self.queue.append(job);count['submitted_cpu']+=1
    count['cpu_input_'+str(r['snapshot']['input_state'])]+=1;count['cpu_nonportable']+=int(not r['snapshot']['portable'])
  if rows:
   count['overlap_submitted']+=len(rows) if self.service.outstanding else 0
   self.service.submit(rows);count['submit_batches']+=1
  count['submit_seconds']+=time.monotonic()-start
 def receive(self,timeout=0.):
  global last
  start=time.monotonic();out,self.inline=self.inline,{};completed=[]
  wait=.001 if timeout and not out else 0.
  for i,result in self.service.receive(wait).items():
   job,features,limit,at=self.meta[i]
   completed.append((job,native_result(features,result,limit,time.monotonic()-at)));count['completed_gpu']+=1
  if self.fallback is None and self.queue:
   jobs,self.queue=self.queue,[];self.fallback=CpuFallback(jobs,self.batch.envs,self.service,capacity=128,threads=8);count['cpu_groups']+=1
  if self.fallback is not None:
   tick=time.monotonic();cpu_ready=self.fallback.step();count['cpu_service_seconds']+=time.monotonic()-tick
   if self.fallback.complete:
    fallback_reports.append(self.fallback.metrics);save('fallback-checks',fallback_reports)
    completed.extend(cpu_ready);count['completed_cpu']+=len(cpu_ready);count['cpu_model_rows']+=self.fallback.rows;self.fallback=None
  if completed:
   tick=time.monotonic();out.update(self.batch.accept_ready(completed));count['root_accept_seconds']+=time.monotonic()-tick
   for job,result in completed:
    i,tier,cfg,nonce=job;s=result['stats'];assert s['simulations']==cfg.simulations
    count.update(roots=1,simulations=s['simulations'],guide=s['policy_calls'],steps=s['playout_transitions'],forced=s['forced_policy_skips'],
     cuts=s['depth_cuts'],terminal=s['terminal_wins']+s['terminal_losses']+s['terminal_escapes'])
    del self.meta[i]
  count['receive_seconds']+=time.monotonic()-start;count['receives']+=1
  if time.monotonic()-last>30:
   v=report(True);save('progress',v);print('NATURAL_LIVE',round(v['full_wall'],2),count['guide'],count['roots'],len(terminals),'gpu_pending',len(self.service.outstanding),'cpu',len(self.queue),flush=True);last=time.monotonic()
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
def record_start(session,env):
 actual_game_ids[:]=[int(e.thick.gc.seed) for e in environments.native_search_batch.envs]
 assert len(set(actual_game_ids))==EXPECTED and session['first_seed']==int(os.environ['EXPECTED_FIRST_SEED'])
 save('game-seeds',dict(first_seed=session['first_seed'],actual_game_ids=actual_game_ids,expected=EXPECTED))
def record_end(session,indices,dones,infos):
 for i,done,info in zip(indices,dones,infos,strict=True):
  if not done:continue
  assert not info.get('TimeLimit.truncated') and not info.get('episode_cohort_censored') and i not in terminals
  z=info['run_stats'];terminals[i]=dict(game_id=actual_game_ids[i],seed=session['first_seed']+i,win=int(z['act']>3 or z['outcome']=='PLAYER_VICTORY'),floor=float(z['floor']),outcome=z['outcome'],actions=info['episode']['l'],completion_wall_seconds=time.monotonic()-t0)
def mirror_before(ordered):
 for i in ordered:assert i not in current_stream.meta
 current_stream.game.before(ordered)
def mirror_after(ordered,infos):current_stream.game.after(ordered,infos)

source=inspect.getsource(ind.collect);patched=source
for a,b in [
 ('    session = episode_rollout.begin(model, env)','    session = episode_rollout.begin(model, env)\n    record_start(session,env)'),
 ('            episode_rollout.record_terminals(session, dones, infos)','            record_end(session,indices,dones,infos)\n            episode_rollout.record_terminals(session, dones, infos)'),
 ('            observations, rewards, dones, infos = step_subset(ordered)','            mirror_before(ordered)\n            observations, rewards, dones, infos = step_subset(ordered)\n            mirror_after(ordered,infos)')]:
 assert patched.count(a)==1;patched=patched.replace(a,b)
upper_initial = "    upper_pending = None\n    stats.update(upper_pending_wall_seconds=0., upper_submit_cpu_seconds=0., upper_accept_cpu_seconds=0.,\n                 transitions_with_upper_pending=0, transitions_with_fallback_pending=0)\n    def consume_upper():\n        nonlocal upper_pending\n        if upper_pending is None:\n            return\n        ident, rows, submitted = upper_pending\n        started = time.perf_counter()\n        packet = upper_take(ident)\n        if packet is None:\n            return\n        assert all(i in upper_waiting and i not in actions and i not in ready for i in rows)\n        ps = packet['probabilities']\n        chosen = [randoms.action(i, ps[j]) for j, i in enumerate(rows)]\n        logps = packet['logprobs'][np.arange(len(rows)),chosen]\n        values = packet['values']\n        reward_rows = {int(j): p for j, p in zip(*packet['reward'])}\n        campfire_rows = {int(j): p for j, p in zip(*packet['campfire'])}\n        for j, i in enumerate(rows):\n            del upper_waiting[i]\n            actions[i] = chosen[j]\n            estimates[i] = (float(values[j]), float(logps[j]))\n            upper_policy[i] = (reward_rows.get(j), campfire_rows.get(j))\n        stats['upper_accept_cpu_seconds'] += time.perf_counter() - started\n        stats['upper_pending_wall_seconds'] += time.perf_counter() - submitted\n        stats['upper_batches'] += 1\n        stats['upper_rows'] += len(rows)\n        upper_pending = None\n"
upper_dispatch = "            consume_upper()\n            if upper_pending is None and upper_waiting and (len(upper_waiting) >= cfg.upper_batch or not stream.pending\n                    or time.monotonic() - min(upper_waiting.values()) >= cfg.upper_wait_seconds):\n                started = time.perf_counter()\n                rows = sorted(upper_waiting)[:cfg.upper_batch]\n                obs = {k: v[rows] for k, v in model._last_obs.items()}\n                action_masks = np.stack([masks[i] for i in rows])\n                upper_pending = (upper_begin(obs,action_masks),rows,started)\n                stats['upper_submit_cpu_seconds'] += time.perf_counter() - started\n            consume_upper()\n"
anchor='    model._independent_rollout_progress = stats'
assert patched.count(anchor)==1
patched=patched.replace(anchor,anchor+'\n'+upper_initial)
start=patched.index('            if upper_waiting and (len(upper_waiting) >= cfg.upper_batch')
end=patched.index('            waiting = not actions',start)
patched=patched[:start]+upper_dispatch+patched[end:]
anchor="            actions.update({i: d['action'] for i, d in decisions.items()})"
assert patched.count(anchor)==1
patched=patched.replace(anchor,anchor+'\n            consume_upper()')
anchor='            ordered = {i: actions[i] for i in indices}'
patched=patched.replace(anchor,anchor+"\n            if upper_pending is not None:\n                assert not set(upper_pending[1]).intersection(indices)\n                stats['transitions_with_upper_pending'] += len(indices)\n            if stream.fallback is not None and stream.fallback.pending is not None:\n                stats['transitions_with_fallback_pending'] += len(indices)")
patched=patched.replace('        if stream.pending or upper_waiting or actions:','        if stream.pending or upper_waiting or actions or upper_pending is not None:')
patched=patched.replace("        stats.update(upper_inference.statistics())","        stats['root_policy_seconds'] = stats['upper_submit_cpu_seconds'] + stats['upper_accept_cpu_seconds']\n        stats.update(upper_inference.statistics())")
ns=dict(vars(ind),SmallPlayoutStream=Stream,UpperPolicyInference=RemoteUpper,upper_begin=upper_begin,upper_take=upper_take,record_start=record_start,record_end=record_end,mirror_before=mirror_before,mirror_after=mirror_after)
exec(patched,ns);orig=ns['collect']
assert Path(sts.__file__).resolve().parent==Path(os.environ['STS_LIGHTSPEED_BUILD_DIR']).resolve()
assert os.environ['WAVE_COUNTER_DRAW']=='1'
class Done(Exception):pass
def collect(m,*a,**kw):
 global model_owner,t0,environments
 assert str(m.device)=='cpu' and not torch.cuda.is_initialized()
 environments=a[0].venv;model_owner=m;torch.set_num_threads(1)
 before=dict(kw['config']);assert before['mode']=='independent'
 cfg={**before,'upper_wait_seconds':0.,'upper_batch':min(64,EXPECTED)};kw['config']=cfg
 save('settings',dict(before=before,after=cfg,arithmetic='tf32x3 GPU chains; IEEE fused fallback; FP32 upper',counter_draw=True,
  original_collector_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_collector_sha256=hashlib.sha256(patched.encode()).hexdigest()))
 (P/(RUN+'-collector.py')).write_text(patched)
 digest=state_digest(m.policy.state_dict());assert digest=='6a0ace45db866af66fb2939fa966fbc5ff80e7154178fa536ea9ca78a1c0083e'
 paths=[P/n for n in [Path(__file__).name,'service_devicestepfull480141.py','worker_devicestepfull480141.py','cpu_boundary.py','combo_runtime.py','combo_model.py','combo_kernels.py','frame_pipeline.py','frame_kernels.py','frame_modules.py','observer_entry.cu','observer-0.cubin','token_builder.cu','token_builder.cubin']]
 paths += [P/n for n in ['fallback_frame.py','fallback_ieee_model.py','fallback_ieee_kernels.py']]
 paths+=[P.parent/'devicegraph141'/n for n in ['graph_runtime.py','graph_model.py','graph_kernels.py']]
 paths += [P/n for n in json.loads((P/'source-baseline-manifest.json').read_text())]
 paths+=[P.parent/'devicefallback141'/n for n in ['packed_fallback.py','fallback_model.py']]+[Path(sts.__file__),Path(cpu.__file__)]
 save('sources',dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},full_model_digest=digest,torch=torch.__version__))
 started=time.monotonic();torch.save(m.policy,P/(RUN+'-policy.pt'));serialization=time.monotonic()-started;t0=time.monotonic()
 try:
  with ResourceMonitor(interval=.2,gpu=True,torch_device='cpu') as monitor:orig(m,*a,**kw)
 except BaseException as failure:
  snapshot=report();snapshot.update(error=repr(failure),resources=monitor.summary,weights_unchanged=state_digest(m.policy.state_dict())==digest)
  save('failure',snapshot);raise
 p=report();p.update(resources=monitor.summary,cohort=m.episode_rollout_stats,small_playout_stats=m._small_playout_stats,serialization_seconds=serialization,
  parent_cuda_initialized=torch.cuda.is_initialized(),child_pid=current_stream.service.process.pid,child_exitcode=current_stream.service.process.exitcode)
 assert state_digest(m.policy.state_dict())==digest and p['service']['weights_unchanged'] and not p['parent_cuda_initialized']
 assert len(terminals)==EXPECTED and sum(t['win'] for t in terminals.values())==p['cohort']['wins']
 assert p['cohort']['completed_games']==EXPECTED and p['cohort']['censored_games']==p['cohort']['unfinished_games']==0
 assert count['simulations']==sum(count['allocated_'+tier] for tier in ('common','elite','boss'))
 assert count['guide']==sum(v for k,v in m._small_playout_stats.items() if k.endswith('/policy_calls'))
 assert count['guide']==p['pool']['real_neural_rows']+count['cpu_model_rows']
 assert count['roots']==count['completed_gpu']+count['completed_cpu']==count['submitted_gpu']+count['submitted_cpu']
 assert count['steps']-count['simulations']==count['guide']+count['forced']
 assert p['pool']['state_pages']['closed'] and p['pool']['state_pages']['owners']==0 and p['pool']['modules_closed']
 assert p['pool']['graph']['chain_step_version']==1 and p['pool']['graph']['closed']
 assert count['diagnostic_seconds']>0 and current_stream.game.counts['actual_actions']>0
 p.update(completed=True,weights_unchanged=True,effective_nonforced_per_second=count['guide']/p['full_wall'])
 save('complete',p);print('NATURAL_COMPLETE',p['full_wall'],EXPECTED,p['effective_nonforced_per_second'],flush=True)
 raise Done()
ind.SmallPlayoutStream=Stream;ind.collect=collect
if __name__=='__main__':
 try:distributed_sb3.main()
 except Done:pass
