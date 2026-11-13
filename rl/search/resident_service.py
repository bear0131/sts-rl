"""Bounded FIFO to the migrated single-owner search worker.

Protocol and lifecycle are taken from service_devicestepfull480141.GpuService.
The collector's current PPO actor remains its consumer; frozen search weights
are sent once at collection start and no CUDA tensor crosses the FIFO.
"""
import collections
import multiprocessing as mp
import pickle
import queue
import time
import weakref

from rl.search.resident_worker import run_worker


class GpuService:
    def __init__(self, capacity, snapshot, settings):
        self.capacity=capacity; self.closed=False; self.stopping=False; self.failure=None; self.finished=False
        self.seq=0; self.requests={}; self.replies={}; self.ready_roots={}; self.outstanding=set()
        self.counts=collections.Counter(); self.status={}; self.cancelled=[]
        self.settings=dict(settings); self.parked=False
        from env.search.cuda_battle import load_artifacts
        self.transport_host=load_artifacts(settings['artifacts'])[2]
        ctx=mp.get_context('spawn')
        self.commands=ctx.Queue(maxsize=4*capacity+16); self.responses=ctx.Queue(maxsize=capacity+16)
        self.process=ctx.Process(target=run_worker,args=(self.commands,self.responses,
            dict(settings,capacity=capacity,snapshot=str(snapshot))),name='device-search-owner',daemon=True)
        self.process.start(); deadline=time.monotonic()+120
        try:
            while not self.status:
                self.poll(.1)
                if time.monotonic()>deadline:
                    raise TimeoutError('GPU process startup exceeded 120s')
            self.check()
        except BaseException:
            self.process.terminate(); self.process.join(timeout=10)
            self.commands.cancel_join_thread(); self.commands.close(); self.responses.close()
            raise

    def check(self):
        if self.failure:
            raise RuntimeError('GPU service failed:\n'+self.failure)
        if self.process.exitcode is not None and not self.finished:
            raise RuntimeError('GPU worker exited without completion '+str(self.process.exitcode))

    def poll(self, timeout=0.):
        first=True
        while True:
            try:
                kind,ident,value=self.responses.get(timeout=timeout) if first and timeout else self.responses.get_nowait()
            except queue.Empty:
                break
            first=False
            if kind=='ready': self.status=value
            elif kind=='reply':
                if ident not in self.requests or ident in self.replies:
                    raise RuntimeError('duplicate or unknown GPU reply')
                self.replies[ident]=value
            elif kind=='roots':
                if not set(value)<=self.outstanding or set(value)&self.ready_roots.keys():
                    raise RuntimeError('duplicate or unknown completed search')
                self.ready_roots.update(value)
            elif kind=='error': self.failure=value
            elif kind=='closed': self.status=value; self.cancelled=value['cancelled']; self.finished=True
            else: raise RuntimeError('unknown worker message '+kind)
        self.check()

    def send(self, kind, payload=None, reply=None):
        self.check()
        if self.closed or self.stopping:
            raise RuntimeError('service is closing')
        packet=pickle.dumps((kind,payload,reply),protocol=5); deadline=time.monotonic()+120
        while True:
            try: self.commands.put(packet,timeout=.05); break
            except queue.Full:
                self.poll()
                if time.monotonic()>deadline: raise TimeoutError('GPU command queue remained full')
        self.counts['commands_sent']+=1; self.counts['command_bytes']+=len(packet)

    def call(self, kind, payload=None, timeout=300):
        self.seq+=1; ident=self.seq; self.requests[ident]=kind
        self.send(kind,payload,ident); deadline=time.monotonic()+timeout
        while ident not in self.replies:
            self.poll(.05)
            if time.monotonic()>deadline: raise TimeoutError('GPU RPC '+kind)
        del self.requests[ident]
        return self.replies.pop(ident)

    def submit(self, rows):
        if self.parked: raise RuntimeError('submit while search owner is parked')
        if set(rows)&self.outstanding or len(self.outstanding)+len(rows)>self.capacity:
            raise ValueError('duplicate search or root capacity exceeded')
        if rows:
            from env.search.resident_transport import pack_roots
            self.send('submit',pack_roots(rows,self.transport_host)); self.outstanding.update(rows)

    def receive(self, timeout=0.):
        self.poll(timeout); out,self.ready_roots=self.ready_roots,{}
        self.outstanding.difference_update(out)
        return out

    def statistics(self):
        if not self.closed: self.status=self.call('statistics')
        return self.status

    def park(self):
        if self.parked or self.outstanding or self.ready_roots or self.requests:
            raise RuntimeError('park requires all collection results to be consumed')
        self.status=self.call('park'); self.parked=True
        return self.status

    def begin(self, snapshot):
        if not self.parked: raise RuntimeError('begin requires a parked owner')
        self.status=self.call('begin',str(snapshot)); self.parked=False
        self.counts.clear()

    def close(self, cancel=None):
        if self.closed: return
        if cancel is None and self.failure is None and not self.finished:
            if self.outstanding or self.ready_roots or self.requests:
                raise RuntimeError('normal close cannot abandon in-flight requests/results')
        deadline=time.monotonic()+120; failure=self.failure
        try:
            if not self.finished and self.process.is_alive():
                if failure is None:
                    self.send('close' if cancel is None else 'cancel',cancel); self.stopping=True
                while not self.finished and self.process.is_alive():
                    try: self.poll(.05)
                    except RuntimeError: failure=self.failure
                    if time.monotonic()>deadline: raise TimeoutError('GPU child close exceeded 120s')
            self.process.join(timeout=max(0.,deadline-time.monotonic()))
            if self.process.is_alive(): raise TimeoutError('GPU child join exceeded 120s')
        finally:
            if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=10)
            self.requests.clear(); self.replies.clear(); self.closed=True
            self.commands.cancel_join_thread(); self.commands.close(); self.responses.close()
        if self.failure: raise RuntimeError('GPU service failed:\n'+self.failure)
        if not self.finished or self.process.exitcode!=0:
            raise RuntimeError('GPU worker did not close normally')


def collection_service(model, capacity, snapshot, settings):
    """One owner across collections; only frozen compiler/model state stays idle.

    Handles are excluded from checkpoints. A restored model always builds a new
    process. The weak finalizer also closes owners used by library callers that
    do not go through the training CLI; the child is daemonized for parent death.
    """
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
