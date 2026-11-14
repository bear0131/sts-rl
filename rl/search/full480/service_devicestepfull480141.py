"""CPU-only owner of a bounded FIFO protocol to one CUDA process."""
from pathlib import Path
import sys,os,time,queue,pickle,multiprocessing as mp,collections,json,inspect
import numpy as np
import torch
from .worker_devicestepfull480141 import run_worker

class ArenaProxy:
 def __init__(self,owner,capacity):
  self.owner=owner;self.m=capacity;self.versions={};self.epochs=[0]*capacity;self.closed=False
 def validate(self,index,epoch,revision):
  assert not self.closed and self.versions.get(index)==(epoch,revision),(index,epoch,revision,self.versions.get(index))
 def load(self,rows):
  assert not self.closed
  for i,s in rows.items():assert type(i)==int and 0<=i<self.m and s['portable'] and s['input_state']==1 and s['outcome']==0
  if not rows:return {}
  expected={}
  for i in rows:self.epochs[i]+=1;self.versions[i]=(self.epochs[i],0);expected[i]=self.versions[i]
  self.owner.send('load',(rows,expected));return expected
 def step(self,rows):
  for i,(ep,rev,bits) in rows.items():self.validate(i,ep,rev);assert 0<=bits<2**32
  if rows:self.owner.send('step',dict(rows))
  for i,(ep,rev,_) in rows.items():self.versions[i]=(ep,rev+1)
 def drop(self,indices):
  ids=list(indices);assert len(ids)==len(set(ids)) and all(i in self.versions for i in ids)
  if ids:self.owner.send('drop',ids)
  for i in ids:del self.versions[i]
 def snapshot(self,indices):return self.owner.call('snapshot',list(indices))
 def keys(self,indices):return self.owner.call('keys',list(indices))
 def close(self):self.closed=True;self.versions.clear()

class GpuService:
 def __init__(self,capacity,policy_path,*,expected_digest,prototypes=None,frozen_policy=None,budget_cap=1000,frame=65536,tile=16384,queue_size=None):
  self.parent_cuda=torch.cuda.is_initialized();self.frozen_policy=frozen_policy
  if self.parent_cuda:
   assert frozen_policy is not None and not frozen_policy.training,'learner must hand over a frozen collection'
   torch.cuda.synchronize()
  self.parent_versions=None if frozen_policy is None else tuple((id(p),p._version) for p in frozen_policy.parameters())
  owner=Path(__file__).resolve().parent
  assert Path(inspect.getfile(run_worker)).resolve()==owner/'worker_devicestepfull480141.py'
  self.capacity=capacity;self.closed=False;self.stopping=False;self.failure=None;self.finished=False
  self.seq=0;self.requests={};self.replies={};self.ready_roots={};self.outstanding=set();self.counts=collections.Counter();self.status={};self.cancelled=[]
  ctx=mp.get_context('spawn')
  self.commands=ctx.Queue(maxsize=queue_size or 4*capacity+16);self.responses=ctx.Queue(maxsize=capacity+16)
  self.arena=ArenaProxy(self,capacity)
  self.process=ctx.Process(target=run_worker,args=(self.commands,self.responses,dict(capacity=capacity,policy_path=str(policy_path),budget_cap=budget_cap,frame=frame,tile=tile,parent_pid=os.getpid(),parent_cuda=self.parent_cuda,expected_digest=expected_digest,prototypes=prototypes)),name='device-search-owner')
  self.process.start();deadline=time.monotonic()+120
  try:
   while not self.status:
    self.poll(.1)
    if time.monotonic()>deadline:raise TimeoutError('GPU process startup exceeded120s')
   self.check()
   from .run_device import artifact_directory
   directory=artifact_directory()/('deck' if prototypes is not None else '')
   expected=json.loads((directory/'builder-build.json').read_text())['cubin_sha256']
   assert self.status['pool']['graph']['identity']['builder']['sha256']==expected
   assert self.status['worker_path']==str(owner/'worker_devicestepfull480141.py')
  except BaseException:
   self.process.terminate();self.process.join(timeout=10);raise
 def check(self):
  assert torch.cuda.is_initialized()==self.parent_cuda,'collection changed CUDA ownership'
  if self.frozen_policy is not None:assert not self.frozen_policy.training,'learner resumed before GPU collection drain'
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
  started=time.monotonic();packet=pickle.dumps((kind,payload,reply),protocol=5);deadline=time.monotonic()+120
  self.counts['pack_seconds_'+kind]+=time.monotonic()-started;self.counts['bytes_'+kind]+=len(packet)
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
  assert not (set(rows)&self.outstanding)
  for i,r in rows.items():
   m,ep,rev=r['mirror'];self.arena.validate(m,ep,rev)
  if rows:self.send('submit',rows);self.outstanding.update(rows)
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
  if self.frozen_policy is not None:
   assert self.parent_versions==tuple((id(p),p._version) for p in self.frozen_policy.parameters()),'weights changed during collection'
