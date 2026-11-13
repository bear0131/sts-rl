"""Bounded FIFO owner copied from the accepted stepfull480-01 implementation.

The transport/drain/reply/close lifecycle stays in the legacy implementation.
Only current frozen-checkpoint begin/park and deck row payloads adapt it.
"""
import collections
import multiprocessing as mp
import pickle
import queue
import time
import weakref
from rl.search.resident_worker import run_worker

class GpuService:
 def __init__(self,capacity,snapshot,settings):
  self.capacity=capacity;self.closed=False;self.stopping=False;self.failure=None;self.finished=False
  self.seq=0;self.requests={};self.replies={};self.ready_roots={};self.outstanding=set();self.counts=collections.Counter();self.status={};self.cancelled=[]
  self.settings=dict(settings);self.parked=False
  ctx=mp.get_context('spawn')
  self.commands=ctx.Queue(maxsize=4*capacity+16);self.responses=ctx.Queue(maxsize=capacity+16)
  self.process=ctx.Process(target=run_worker,args=(self.commands,self.responses,dict(settings,capacity=capacity,snapshot=str(snapshot))),name='device-search-owner',daemon=True)
  self.process.start();deadline=time.monotonic()+120
  try:
   while not self.status:
    self.poll(.1)
    if time.monotonic()>deadline:raise TimeoutError('GPU process startup exceeded120s')
   self.check()
  except BaseException:
   self.process.terminate();self.process.join(timeout=10)
   self.commands.cancel_join_thread();self.commands.close();self.responses.close();raise
 def check(self):
  if self.failure:raise RuntimeError('GPU service failed:\n'+self.failure)
  if self.process.exitcode is not None and not self.finished:raise RuntimeError('GPU worker exited without completion '+str(self.process.exitcode))
 def poll(self,timeout=0.):
  first=True
  while True:
   try:kind,ident,value=self.responses.get(timeout=timeout) if first and timeout else self.responses.get_nowait()
   except queue.Empty:break
   first=False
   if kind=='ready':self.status=value
   elif kind=='reply':
    assert ident in self.requests and ident not in self.replies;self.replies[ident]=value
   elif kind=='roots':
    for i in value:
     assert i in self.outstanding and i not in self.ready_roots
    self.ready_roots.update(value)
   elif kind=='error':self.failure=value
   elif kind=='closed':self.status=value;self.cancelled=value['cancelled'];self.finished=True
   else:raise RuntimeError('unknown worker message '+kind)
  self.check()
 def send(self,kind,payload=None,reply=None):
  self.check();assert not self.closed and not self.stopping
  # Freeze NumPy data before the queue feeder thread runs.
  packet=pickle.dumps((kind,payload,reply),protocol=5);deadline=time.monotonic()+120
  while True:
   try:self.commands.put(packet,timeout=.05);break
   except queue.Full:
    self.poll()
    if time.monotonic()>deadline:raise TimeoutError('GPU command queue remained full')
  self.counts['commands_sent']+=1;self.counts['command_bytes']+=len(packet)
 def begin_call(self,kind,payload=None):
  if kind in ('upper','fallback'):
   assert kind not in self.requests.values(),'only one '+kind+' request may be in flight'
  self.seq+=1;ident=self.seq;self.requests[ident]=kind
  try:self.send(kind,payload,ident)
  except BaseException:self.requests.pop(ident);raise
  self.counts['rpc_started_'+kind]+=1
  self.counts['max_pending_requests']=max(self.counts['max_pending_requests'],len(self.requests))
  return ident
 def take_reply(self,ident):
  assert ident in self.requests,'unknown or already consumed request'
  self.poll()
  if ident not in self.replies:return False,None
  value=self.replies.pop(ident);kind=self.requests.pop(ident);self.counts['rpc_consumed_'+kind]+=1
  return True,value
 def call(self,kind,payload=None,timeout=300):
  ident=self.begin_call(kind,payload);deadline=time.monotonic()+timeout
  while True:
   ready,value=self.take_reply(ident)
   if ready:return value
   self.poll(.05)
   if time.monotonic()>deadline:raise TimeoutError('GPU RPC '+kind)
 def submit(self,rows):
  if self.parked or set(rows)&self.outstanding or len(self.outstanding)+len(rows)>self.capacity:
   raise ValueError('duplicate search, parked owner or root capacity exceeded')
  if rows:
   from env.search.resident_transport import pack_actor_roots
   if not hasattr(self,'native_host'):
    from env.search.cuda_battle import load_artifacts
    self.native_host=load_artifacts(self.settings['artifacts'])[2]
   self.send('submit',pack_actor_roots(rows,self.native_host));self.outstanding.update(rows)
 def park(self):
  if self.parked or self.outstanding or self.ready_roots or self.requests:raise RuntimeError('park requires consumed results')
  self.status=self.call('park');self.parked=True;return self.status
 def begin(self,snapshot):
  if not self.parked:raise RuntimeError('begin requires parked owner')
  self.status=self.call('begin',str(snapshot));self.parked=False;self.counts.clear()
 def receive(self,timeout=0.):
  self.poll(timeout);out,self.ready_roots=self.ready_roots,{}
  self.outstanding.difference_update(out);return out
 def statistics(self):
  if not self.closed:self.status=self.call('statistics')
  return self.status
 def close(self,cancel=None):
  if self.closed:return
  if cancel is None and self.failure is None and not self.finished:
   assert not self.outstanding and not self.ready_roots and not self.requests,'normal close cannot abandon in-flight requests/results'
  deadline=time.monotonic()+120;failure=self.failure
  try:
   if not self.finished and self.process.is_alive():
    if failure is None:
     if cancel is None:assert not self.outstanding and not self.ready_roots,'normal close cannot abandon results'
     self.send('close' if cancel is None else 'cancel',cancel);self.stopping=True
    while not self.finished and self.process.is_alive():
     try:self.poll(.05)
     except RuntimeError:failure=self.failure
     if time.monotonic()>deadline:raise TimeoutError('GPU child close exceeded120s')
   self.process.join(timeout=max(0.,deadline-time.monotonic()))
   if self.process.is_alive():raise TimeoutError('GPU child join exceeded120s')
  finally:
   if self.process.is_alive():self.process.terminate();self.process.join(timeout=10)
   self.cancelled_requests=dict(self.requests);self.requests.clear();self.replies.clear()
   self.closed=True;self.commands.cancel_join_thread();self.commands.close();self.responses.close()
  if self.failure:raise RuntimeError('GPU service failed:\n'+self.failure)
  assert self.finished and self.process.exitcode==0


def collection_service(model, capacity, snapshot, settings):
    """One owner across collections; only frozen compiler/model state stays idle.

    Handles are excluded from checkpoints. A restored model always builds a new
    process. The weak finalizer also closes owners used by library callers that
    do not go through the training CLI; the child is daemonized for parent death.
    """
    settings=dict(settings,environment_capacity=model.n_envs)
    service=getattr(model,'_resident_search_service',None)
    if service is not None:
        if not service.closed and (service.capacity!=capacity or service.settings!=settings):
            service.close()
        if service.closed:
            model._resident_search_finalizer.detach(); service=None
    if service is None:
        service=GpuService(capacity,snapshot,settings)
        model._resident_search_service=service
        model._resident_search_finalizer=weakref.finalize(model,service.close,cancel='model released')
    else:
        service.begin(snapshot)
    return service


def close_collection_service(model):
    finalizer=getattr(model,'_resident_search_finalizer',None)
    if finalizer is not None: finalizer()
