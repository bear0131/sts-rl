"""Single CUDA owner: legacy bounded command acceptance followed by one tick."""
import os
import gc
import pickle
import queue
import time
import traceback
from types import SimpleNamespace


def run_worker(commands,responses,settings):
    import torch
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena,load_artifacts
    from rl.policies.deck_latent import DeckLatentCore
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.search.resident_pool import ResidentPool
    from rl.search.resident_step import ResidentStepPipeline
    engine=arena=scorer=core=compiled=None
    generation=0; parked_report=None; geometry=space=None
    active=0.; started=time.monotonic()
    try:
        torch.set_num_threads(1)
        folder,manifest,host=load_artifacts(settings['artifacts'])
        if getattr(host,'host_runtime_version',None) != 1:
            raise ValueError('resident worker requires native host runtime v1')
        if (manifest.get('observation_storage') != 'bounded-frame-v1'
                or manifest.get('root_import') != 'indexed-v1'):
            raise ValueError('streaming search requires bounded-frame-v1 and indexed-v1 artifacts')
        def begin(snapshot):
            nonlocal core,scorer,arena,engine,generation,geometry,space,active,started
            if engine is not None:
                raise RuntimeError('new weights require a fully parked search owner')
            data=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
            if core is None:
                geometry,space=data['geometry'],data['space']
                with torch.device('meta'):
                    core=DeckLatentCore(space,**geometry)
                core.load_state_dict(data['state_dict'],assign=True)
                for name,value in data['buffers'].items():
                    parent,_,field=name.rpartition('.')
                    core.get_submodule(parent)._buffers[field]=value
                core.to('cuda').eval()
            else:
                if data['geometry']!=geometry or data['space']!=space:
                    raise ValueError('parked owner model contract changed')
                # Keep parameter objects stable for the compiled callable, but
                # invalidate every derived projection/cache before rebuilding.
                state=core.state_dict(); buffers=dict(core.named_buffers())
                if (state.keys()!=data['state_dict'].keys() or buffers.keys()!=data['buffers'].keys()
                        or any(state[k].shape!=v.shape or state[k].dtype!=v.dtype
                               for k,v in data['state_dict'].items())
                        or any(buffers[k].shape!=v.shape or buffers[k].dtype!=v.dtype
                               for k,v in data['buffers'].items())):
                    raise ValueError('parked owner tensor schema changed')
                core.load_state_dict(data['state_dict'],assign=False)
                for name,value in data['buffers'].items(): buffers[name].copy_(value)
            policy=SimpleNamespace(core=core,observation_space=space,parameters=core.parameters)
            if scorer is None:
                scorer=ResidentDeckScorer(policy,settings['cache_bytes'],native_host=host)
            else:
                # Same owner identity permits reuse of Dynamo's compiled code.
                # No old root, graph, context ID or score cache survives park.
                scorer.__init__(policy,settings['cache_bytes'],native_host=host,compiled_gather=compiled)
            capacity=settings['capacity']; budget=settings['budget_cap']
            arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],
                capacity+capacity*((budget+15)//16)*16,
                card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,
                observation_capacity=min(capacity+capacity*((budget+15)//16)*16,
                                         ResidentStepPipeline.frame_capacity))
            engine=ResidentPool(arena,scorer,capacity,budget,settings['max_depth'],settings['timeout'])
            active=0.; started=time.monotonic(); generation+=1

        def report():
            if engine is None: return parked_report
            return dict(pool=engine.statistics(),active_tick_seconds=active,
                elapsed_seconds=time.monotonic()-started,pid=os.getpid(),
                generation=generation,
                context_refreshes=scorer.refreshes,anchor_cache=dict(scorer.anchor_cache.stats),
                torch_allocated=torch.cuda.memory_allocated(),torch_reserved=torch.cuda.memory_reserved(),
                torch_peak=torch.cuda.max_memory_allocated())
        with torch.no_grad():
            begin(settings['snapshot'])
            responses.put(('ready',None,dict(pid=os.getpid(),generation=generation,
                artifact_sha256=manifest['sha256']['deck.cubin'])))
            while True:
                try: packet=commands.get_nowait() if engine is not None and engine.pending else commands.get()
                except queue.Empty: packet=None
                # Exact old worker scheduling: bounded FIFO drain, then one
                # engine.receive tick; incoming roots never wait for pool drain.
                admitted={}
                for command_index in range(64):
                    if packet is None: break
                    kind,payload,ident=pickle.loads(packet)
                    if kind=='submit':
                        if engine is None: raise RuntimeError('submit while search owner is parked')
                        if admitted.keys() & payload.keys():
                            raise ValueError('duplicate queued root admission')
                        admitted.update(payload)
                    else:
                        # Coalesce only already available FIFO submissions.
                        # Control messages remain barriers and never overtake
                        # a root; do not wait for a batch or for live work to end.
                        if admitted:
                            engine.submit(admitted); admitted={}
                    if kind=='statistics': responses.put(('reply',ident,report()))
                    elif kind=='park':
                        if engine is None or engine.pending:
                            raise RuntimeError('park requires a drained active collection')
                        parked_report=report(); release_started=time.monotonic()
                        compiled=scorer.compiled_gather
                        engine.close()
                        # The pipeline has back-references into scorer/arena.
                        # Break all references before asking the allocator to
                        # yield memory to the separate PPO process.
                        scorer.__dict__.clear(); engine=arena=None
                        core._context_cache=core._actor_context=None
                        core.combat._card_inference_tables.clear()
                        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
                        parked_report['park']=dict(seconds=time.monotonic()-release_started,
                            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                            free_bytes=torch.cuda.mem_get_info()[0])
                        responses.put(('reply',ident,parked_report))
                    elif kind=='begin':
                        begin(payload)
                        responses.put(('reply',ident,dict(pid=os.getpid(),generation=generation)))
                    elif kind in ('close','cancel'):
                        cancelled=list(engine.ledger.pending) if engine is not None and kind=='cancel' else []
                        last=report()
                        if engine is not None: engine.close(cancel=kind=='cancel')
                        responses.put(('closed',None,dict(last,cancelled=cancelled)))
                        return
                    elif kind!='submit': raise ValueError('unsupported streaming command '+kind)
                    if command_index < 63:
                        try: packet=commands.get_nowait()
                        except queue.Empty: packet=None
                if admitted: engine.submit(admitted)
                if engine is not None and engine.pending:
                    t=time.monotonic(); ready=engine.receive(steps=1); active+=time.monotonic()-t
                    if ready: responses.put(('roots',None,ready))
    except BaseException:
        responses.put(('error',None,traceback.format_exc()))
        raise
    finally:
        if engine is not None and not engine.closed:
            engine.close(cancel=True)
        elif arena is not None and not arena.closed:
            if scorer is not None: scorer.close()
            arena.close()
