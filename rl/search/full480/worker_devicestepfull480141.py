"""The only CUDA-owning process. Importing this module is CPU safe."""
def run_worker(commands,responses,settings):
 from pathlib import Path
 import os,sys,time,queue,pickle,traceback,collections,resource,subprocess,json,hashlib,importlib.util
 import numpy as np
 import torch
 P=Path(__file__).resolve().parent
 engine=arena=upper=scorer=None;cancelled=[];counts=collections.Counter();before=None;policy=None;loaded_modules={}
 def emit(kind,value,ident=None):responses.put((kind,ident,value),timeout=120)
 def abort(reason):
  nonlocal cancelled
  if scorer is not None:scorer.close()
  if engine is None or engine.closed:return
  engine.graph_scorer.close()
  cancelled=[dict(key=h.key,slot=h.slot,generation=h.generation,reason=reason) for h in engine.ledger.pending.values()]
  engine.launch('stream_destroy',[engine.arena,engine.constructed],[engine.n],grid=engine.n)
  engine.launch('stream_destroy',[engine.roots,engine.root_constructed],[engine.m],grid=engine.m)
  torch.cuda.synchronize()
  engine.state_pages.release(list(engine.page_leases.values()));engine.page_leases.clear();engine.state_pages.close()
  engine.closed=True;engine.modules.references-=1
  if hasattr(engine,'ragged_buffers'):engine.ragged_buffers.clear()
 def stats():
  return dict(pid=os.getpid(),parent_pid=settings['parent_pid'],counts=dict(counts),cancelled=cancelled,
   worker_path=str(Path(__file__).resolve()),loaded_modules=loaded_modules,full_model_digest=before,weights_unchanged=policy is not None and state_digest(policy.state_dict())==before,
   fallback_graph={} if scorer is None else scorer.statistics(),
   upper_graph={} if upper is None else upper.statistics(),pool={} if engine is None else dict(
    pending=len(engine.ledger),active=engine.active_count,graph=engine.graph_scorer.statistics(),counts=dict(engine.completed_counts),ragged_counts=dict(getattr(engine,'ragged_counts',{})),
    real_neural_rows=engine.real_neural_rows,forced_skips=engine.skipped_forced_rows,model_calls=engine.model_calls,
    iterations=engine.iterations,state_pages=engine.state_pages.statistics(),modules_closed=engine.modules.closed,identity=engine.modules.identity,capacity=engine.m,frame=engine.frame_cap,tile=engine.tile),
   torch_allocated=torch.cuda.memory_allocated(),torch_reserved=torch.cuda.memory_reserved(),torch_peak=torch.cuda.max_memory_allocated(),
   cpu_self_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
 try:
  torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  import importlib
  for module_name in ['root_ledger','state_pages','layout_bindings','frame_kernels','combo_kernels','combo_model','frame_modules','frame_pipeline','step_kernels','step_pipeline','combo_runtime','fallback_ieee_kernels','fallback_ieee_model','fallback_frame','flex_stream','flex_adapter','flex_runtime','flex_ragged']:
   module=importlib.import_module('.'+module_name,__package__)
   path=P/(module_name+'.py');assert Path(module.__file__).resolve()==path.resolve()
   loaded_modules[module_name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  from rl.policies.token_regret import encode_public_roots,pack_rows
  from .fallback_frame import FallbackFrame
  from .combo_runtime import BattleArena,BattlePool
  from .combo_model import DeckWeights
  from .deck_inputs import DeckInputTables
  from .run_device import artifact_directory
  artifacts=artifact_directory()
  from rl.algorithms.upper_probe import state_digest
  from rl.search.rollout_inference import UpperPolicyInference
  from rl.algorithms import independent_rollout as ind
  policy=torch.load(settings['policy_path'],map_location='cpu',weights_only=False)
  before=state_digest(policy.state_dict());assert before==settings['expected_digest']
  policy=policy.cuda().eval()
  with torch.inference_mode():
   deck=getattr(policy.core,'model_contract',None)=='deck-latent-combat-v1'
   upper=UpperPolicyInference(policy.core,'cuda');upper.__enter__()
   flat=None
   if deck:
    flat=DeckWeights(policy.core);flat.input_tables=DeckInputTables(policy.core,settings['prototypes']);flat.inference=upper
   arena=BattleArena(settings['capacity'],deck=deck)
   engine=BattlePool(arena,settings['budget_cap'],net=policy.core.combat if deck else policy.small_playout,flat=flat,frame=settings['frame'],tile=settings['tile'])
   actual=engine.frame_pipeline.modules.identity;directory=artifacts/'deck' if deck else artifacts
   expected={name:json.loads((directory/receipt).read_text())['cubin_sha256'] for name,receipt in [('observer','observer-compile-0-result.json'),('builder','builder-build.json')]}
   assert {name:actual[name]['sha256'] for name in expected}==expected,('wrong compiled inference backend',actual,expected)
   rules=engine.modules.identity['rules']
   assert rules['cubin_sha256']==json.loads((directory/'rule-compile-0-result.json').read_text())['cubin_sha256']
   assert rules['abi_layout']==[16864,112] and rules['chain_step_version']==1
   assert engine.graph_scorer.statistics()['chain_step_version']==1
   if deck:
    fallback_flat=DeckWeights(policy.core);fallback_flat.precision='ieee'
    scorer=FallbackFrame(policy.core.combat,flat=fallback_flat)
   else:scorer=FallbackFrame(policy.small_playout)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   pids=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()]
   assert set(pids)==({os.getpid(),settings['parent_pid']} if settings['parent_cuda'] else {os.getpid()}),pids
   startup=stats();startup['gpu_pids']=pids;emit('ready',startup)
   stop=False
   while not stop:
    if not engine.pending:packet=commands.get()
    else:
     try:packet=commands.get_nowait()
     except queue.Empty:packet=None
    for index in range(64):
     if packet is None:break
     kind,payload,ident=pickle.loads(packet);answer=None;start=time.perf_counter()
     if kind=='load':
      rows,expected=payload;answer=arena.load(rows);assert answer==expected
     elif kind=='step':arena.step(payload)
     elif kind=='drop':arena.drop(payload)
     elif kind=='snapshot':answer=arena.snapshot(payload)
     elif kind=='keys':answer=arena.keys(payload)
     elif kind=='submit':engine.submit(payload)
     elif kind=='reference':
      assert not engine.pending;engine.submit(payload);answer={}
      while engine.pending:answer.update(engine.receive())
     elif kind=='probe':
      row=dict(payload);mirror=BattleArena(1,modules=engine.modules);pool=None
      try:
       mirror.load({0:row['snapshot']});row['mirror']=(0,*mirror.versions[0])
       pool=BattlePool(mirror,row['budget'],precision='tf32x3',frame=max(1,row['budget']),tile=32,net=policy.small_playout,flat=engine.flat)
       pool.submit({0:row});got={}
       while pool.pending:got.update(pool.receive())
       answer=got[0];pool.close()
      finally:
       if pool is not None and not pool.closed:
        pool.launch('stream_destroy',[pool.arena,pool.constructed],[pool.n],grid=pool.n)
        pool.launch('stream_destroy',[pool.roots,pool.root_constructed],[pool.m],grid=pool.m)
        torch.cuda.synchronize();pool.graph_scorer.close()
        pool.state_pages.release(list(pool.page_leases.values()));pool.page_leases.clear();pool.state_pages.close()
        pool.closed=True;pool.modules.references-=1
       mirror.close()
     elif kind in ('upper','upper_reference'):
      obs,masks=payload
      if kind=='upper' and deck:logits,values,context=upper(obs,with_search_context=True)
      elif kind=='upper':logits,values=upper(obs)
      else:
       from rl.search.native import observation_as_tensor
       logits,values=policy.core.actor_critic(observation_as_tensor(obs,'cuda'))
      dist=policy.action_dist.proba_distribution(action_logits=logits.float());dist.apply_masking(masks)
      answer=dict(probabilities=dist.distribution.probs.cpu().numpy(),logprobs=dist.distribution.logits.cpu().numpy(),
       values=values.float().flatten().cpu().numpy(),reward=ind.reward_monitor.policy_probabilities(dist,obs),
       campfire=ind.campfire_monitor.policy_probabilities(dist,obs))
      if deck:
       if kind=='upper_reference':
        projected=upper.search_context(obs);context=(projected.combat.cpu().numpy(),projected.anchor.cpu().numpy())
       answer.update(logits=logits.float().cpu().numpy(),context=context)
     elif kind=='fallback':
      fields,temp,raw,reference=payload
      if deck:
       obs=fields['observations'];kinds=fields['kinds'];count=len(kinds)
       context_obs={key:value[fields['root_indices']] for key,value in fields['context_roots'].items()}
       context_obs['outer_deck_misc']=obs['outer_deck_misc']
       context=upper.search_context(context_obs);picks=np.zeros(count,np.int32);values=np.zeros(count,np.float64)
       anchors=np.flatnonzero(np.isin(kinds,[1,3]));chains=np.flatnonzero((kinds==0)|(kinds==4))
       if len(anchors):values[anchors]=upper.anchor_values({k:v[anchors] for k,v in obs.items()},context.anchor[anchors].cpu().numpy())
       if len(chains):
        observations={k:v[chains] for k,v in obs.items()};observations['_combat_context']=context.combat[chains].cpu().numpy()
        chosen,leaf_values,logits=scorer.score_deck(observations,fields['draws'][chains],kinds[chains]==4,fields['layout'])
        picks[chains]=chosen;values[chains]=leaf_values
       answer=dict(picks=picks,values=values)
      elif reference:
       assert raw is not None
       for row,token in zip(raw,encode_public_roots(raw),strict=True):row['tokens']=token
       gold=policy.small_playout.forward_packed(*pack_rows(raw,'cuda')).cpu().double().numpy()
       picks=[]
       for i,row in enumerate(raw):
        n=int(fields[16][i]);x=gold[i,:n];p=np.exp((x-x.max())/temp);p/=p.sum()
        picks.append(min(n-1,int(np.searchsorted(p.cumsum(),fields[17][i],side='right'))))
       answer=dict(picks=np.asarray(picks,np.int32),gold=gold,logits=None)
      else:
       picks,logits=scorer.score(fields,fields[4].shape[1],float(temp))
       answer=dict(picks=picks,logits=None,gold=None)
       if raw is not None:
        for row,token in zip(raw,encode_public_roots(raw),strict=True):row['tokens']=token
        answer.update(logits=logits.cpu().double().numpy(),gold=policy.small_playout.forward_packed(*pack_rows(raw,'cuda')).cpu().double().numpy())
     elif kind=='statistics':answer=stats()
     elif kind=='upper_statistics':answer=upper.statistics()
     elif kind=='delay':time.sleep(min(float(payload),.1))
     elif kind=='flush':answer=dict(pending=len(engine.ledger),active=engine.active_count)
     elif kind=='close':assert not engine.pending;stop=True
     elif kind=='cancel':abort(str(payload));stop=True
     else:raise ValueError('unknown GPU command '+kind)
     counts['commands_'+kind]+=1;counts['seconds_'+kind]+=time.perf_counter()-start
     if ident is not None:emit('reply',answer,ident)
     if stop or index==63:break
     try:packet=commands.get_nowait()
     except queue.Empty:packet=None
    if not stop and engine.pending:
     active_before=engine.active_count;start=time.perf_counter();ready=engine.receive(steps=1)
     elapsed=time.perf_counter()-start;counts['gpu_service_seconds']+=elapsed;counts['active_tick_time_sum']+=active_before*elapsed;counts['active_tick_sum']+=active_before;counts['ticks']+=1
     if ready:emit('roots',ready);counts['roots_published']+=len(ready)
   if scorer is not None:scorer.close()
   if not engine.closed:engine.close()
   arena.close();upper.__exit__(None,None,None);torch.cuda.synchronize()
   final=stats();assert final['weights_unchanged'];emit('closed',final)
 except BaseException:
  failure=traceback.format_exc()
  try:
   with torch.inference_mode():
    abort('worker exception')
    if arena is not None and not arena.closed:arena.close()
    if upper is not None:upper.__exit__(None,None,None)
  except BaseException:failure+='\nCleanup failure:\n'+traceback.format_exc()
  emit('error',failure)
  emit('closed',dict(error=failure,cancelled=cancelled,pid=os.getpid()))
