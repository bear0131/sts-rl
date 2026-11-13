"""stepfull480-01 command loop; the deck consumer owns no scheduling."""
def run_worker(commands,responses,settings):
 import os,time,queue,pickle,traceback,collections
 import torch
 from rl.search.deck_worker import DeckWorker
 engine=None;counts=collections.Counter()
 def emit(kind,value,ident=None):responses.put((kind,ident,value),timeout=120)
 try:
  torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with torch.no_grad():
   engine=DeckWorker(settings)
   emit('ready',engine.statistics())
   stop=False
   while not stop:
    if not engine.pending:packet=commands.get()
    else:
     try:packet=commands.get_nowait()
     except queue.Empty:packet=None
    for index in range(64):
     if packet is None:break
     kind,payload,ident=pickle.loads(packet);answer=None;start=time.perf_counter()
     if kind=='submit':engine.submit(payload)
     elif kind=='statistics':answer=engine.statistics()
     elif kind=='park':answer=engine.park()
     elif kind=='begin':
      counts.clear();answer=engine.begin(payload)
     elif kind in ('upper','fallback_begin','fallback','fallback_end','values'):
      answer=engine.infer(kind,payload)
     elif kind=='close':
      if engine.pending:raise RuntimeError('normal close cannot abandon roots')
      stop=True
     elif kind=='cancel':stop=True
     else:raise ValueError('unknown GPU command '+kind)
     counts['commands_'+kind]+=1;counts['seconds_'+kind]+=time.perf_counter()-start
     if kind in ('statistics','park'):answer['owner_counts']=dict(counts)
     if ident is not None:emit('reply',answer,ident)
     if stop or index==63:break
     try:packet=commands.get_nowait()
     except queue.Empty:packet=None
    if not stop and engine.pending:
     start=time.perf_counter();ready=engine.receive(steps=1)
     counts['gpu_service_seconds']+=time.perf_counter()-start;counts['ticks']+=1
     if ready:emit('roots',ready);counts['roots_published']+=len(ready)
   report=engine.close(cancel=kind=='cancel')
   report['owner_counts']=dict(counts);emit('closed',report)
 except BaseException:
  failure=traceback.format_exc()
  try:
   if engine is not None:engine.close(cancel=True)
  except BaseException:failure+='\nCleanup failure:\n'+traceback.format_exc()
  emit('error',failure)
