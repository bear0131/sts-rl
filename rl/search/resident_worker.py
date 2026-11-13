"""Single CUDA owner: legacy bounded command acceptance followed by one tick."""
import os
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
    engine=None
    try:
        torch.set_num_threads(1)
        data=torch.load(settings['snapshot'],map_location='cpu',weights_only=False,mmap=True)
        with torch.device('meta'):
            core=DeckLatentCore(data['space'],**data['geometry'])
        core.load_state_dict(data['state_dict'],assign=True)
        for name,value in data['buffers'].items():
            parent,_,field=name.rpartition('.')
            core.get_submodule(parent)._buffers[field]=value
        core.to('cuda').eval()
        policy=SimpleNamespace(core=core,observation_space=data['space'],parameters=core.parameters)
        folder,manifest,host=load_artifacts(settings['artifacts'])
        if getattr(host,'host_runtime_version',None) != 1:
            raise ValueError('resident worker requires native host runtime v1')
        scorer=ResidentDeckScorer(policy,settings['cache_bytes'],native_host=host)
        if (manifest.get('observation_storage') != 'bounded-frame-v1'
                or manifest.get('root_import') != 'indexed-v1'):
            raise ValueError('streaming search requires bounded-frame-v1 and indexed-v1 artifacts')
        capacity=settings['capacity']; budget=settings['budget_cap']
        arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],
            capacity+capacity*((budget+15)//16)*16,
            card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,
            observation_capacity=min(capacity+capacity*((budget+15)//16)*16,
                                     ResidentStepPipeline.frame_capacity))
        engine=ResidentPool(arena,scorer,capacity,budget,settings['max_depth'],settings['timeout'])
        active=0.; started=time.monotonic()
        def report():
            return dict(pool=engine.statistics(),active_tick_seconds=active,
                elapsed_seconds=time.monotonic()-started,pid=os.getpid(),
                context_refreshes=scorer.refreshes,anchor_cache=dict(scorer.anchor_cache.stats),
                torch_allocated=torch.cuda.memory_allocated(),torch_reserved=torch.cuda.memory_reserved(),
                torch_peak=torch.cuda.max_memory_allocated())
        responses.put(('ready',None,dict(pid=os.getpid(),artifact_sha256=manifest['sha256']['deck.cubin'])))
        with torch.no_grad():
            while True:
                try: packet=commands.get_nowait() if engine.pending else commands.get()
                except queue.Empty: packet=None
                # Exact old worker scheduling: bounded FIFO drain, then one
                # engine.receive tick; incoming roots never wait for pool drain.
                admitted={}
                for command_index in range(64):
                    if packet is None: break
                    kind,payload,ident=pickle.loads(packet)
                    if kind=='submit':
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
                    elif kind in ('close','cancel'):
                        cancelled=list(engine.ledger.pending) if kind=='cancel' else []
                        engine.close(cancel=kind=='cancel')
                        responses.put(('closed',None,dict(report(),cancelled=cancelled)))
                        return
                    elif kind!='submit': raise ValueError('unsupported streaming command '+kind)
                    if command_index < 63:
                        try: packet=commands.get_nowait()
                        except queue.Empty: packet=None
                if admitted: engine.submit(admitted)
                if engine.pending:
                    t=time.monotonic(); ready=engine.receive(steps=1); active+=time.monotonic()-t
                    if ready: responses.put(('roots',None,ready))
    except BaseException:
        responses.put(('error',None,traceback.format_exc()))
        raise
    finally:
        if engine is not None and not engine.closed:
            engine.close(cancel=True)
