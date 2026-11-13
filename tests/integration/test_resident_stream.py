import gc
import time

import numpy as np
import pytest
import torch

pytestmark=[pytest.mark.native,pytest.mark.cuda,pytest.mark.timeout(240)]


def test_real_mirror_replays_policy_actions_and_preserves_rng_and_prefix(
        complete_combat_factory, cuda_battle_artifacts):
    import pickle
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_mirror import ResidentBattleMirror
    from env.search.resident_transport import ResidentRootExporter, pack_roots, unpack_roots
    folder, manifest, host = cuda_battle_artifacts
    roots = [complete_combat_factory(['DEFEND_RED','WARCRY','HEADBUTT','TRUE_GRIT','EXHUME'],
        seed=seed, deck_latent_context=True, card_language=True) for seed in (42,43)]
    exporter = ResidentRootExporter(roots, folder)
    arena = CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],4,
        card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,observation_capacity=4)
    mirror = ResidentBattleMirror(arena,0,2)
    siblings = torch.empty((2,arena.battle_bytes),dtype=torch.uint8,device='cuda')
    arena.arena[2:4].copy_(siblings.data_ptr()+torch.arange(2,device='cuda')*arena.battle_bytes)
    rng = np.random.default_rng(20261111)
    steps = 0
    try:
        with pytest.raises(ValueError,match='distinct live slots'):
            arena.replay_slots([0],[0])
        for steps in range(60):
            payloads = exporter.export([0,1],generation=(1,2))
            assert not any(p.get('native_fallback') for p in payloads)
            if steps:
                assert all(p['snapshot'] is None for p in payloads)
            rows={i:dict(root=p,observation=r._obs(),context=(np.zeros(64,np.float32),)*2,
                        worlds=np.array([71+i],np.int64),budget=64,submitted_at=1.)
                  for i,(r,p) in enumerate(zip(roots,payloads))}
            received=unpack_roots(pickle.loads(pickle.dumps(pack_roots(rows,host),protocol=5)))
            sources=mirror.prepare([received[i]['root'] for i in range(2)],[0,1])
            words,meta=arena.snapshot(2)
            for i,r in enumerate(roots):
                expected=host.snapshot(r.thick.bc)
                np.testing.assert_array_equal(words[i,:len(expected['words'])],expected['words'])
                assert arena.prefix[i].item()==sum(1<<k for k in (r._multi or ()))
            before=words.copy(),meta.copy()
            arena.fork(sources,[2,3],[123,456])
            after=arena.snapshot(2)
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            actions=[int(rng.choice(np.flatnonzero(r.action_masks()))) for r in roots]
            exporter.before_step(actions)
            outcomes=[r.step(a) for r,a in zip(roots,actions)]
            if (any(x[2] or x[3] for x in outcomes)
                    or any(r._phase()!='combat' or r._at_combat_exit_anchor() for r in roots)):
                break
            if steps==4:gc.collect();torch.cuda.empty_cache()
        assert steps>=5
        assert mirror.counts['loads']==2 and mirror.counts['actual_actions']==2*steps
        # A restored collection gets its origin again; no state is checkpointed
        # as a CUDA address or dependent on the previous GPU owner.
        if all(r.thick is not None and r.thick.bc is not None
               and r._phase()=='combat' and not r._at_combat_exit_anchor() for r in roots):
            assert all(p['snapshot'] is not None for p in exporter.export([0,1],generation=(3,4)))
    finally:arena.close()


def test_mirror_clones_each_queued_version_before_advancing_next(
        complete_combat_factory, cuda_battle_artifacts):
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_transport import ResidentRootExporter
    from env.search.resident_mirror import ResidentBattleMirror
    from env.search.anchored_chains import paired_anchor_search
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.deck_latent import DeckLatentScorer
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.search.resident_pool import ResidentPool
    root=complete_combat_factory(['STRIKE_RED','DEFEND_RED','WARCRY'],seed=42,
        deck_latent_context=True,card_language=True)
    folder,manifest,host=cuda_battle_artifacts
    exporter=ResidentRootExporter([root],folder)
    torch.manual_seed(20261111)
    policy=DeckLatentPolicy(root.observation_space,root.action_space,lambda _:5e-5,
        d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    rows={};expected=[]
    for i in range(2):
        obs=root._obs();payload=exporter.export([0],generation=(1,2))[0]
        context=policy.core.prepare_search_context({k:torch.as_tensor(v[None],device='cuda') for k,v in obs.items()})
        values,_=paired_anchor_search([root],[64],DeckLatentScorer(policy,[obs]),
            seed=20261111+i,max_depth=16,max_active=256,timeout_seconds=90)
        expected.append(values[0])
        rows[i]=dict(root=payload,observation=obs,context=tuple(x.detach().cpu().numpy()[0]
            for x in (context.combat,context.anchor)),worlds=np.random.default_rng(20261111+i).integers(
                0,2**63-1,max(1,64//len(payload['actions'])),dtype=np.int64),budget=64,submitted_at=time.monotonic())
        if i==0:
            action=int(np.flatnonzero(root.action_masks())[0]);exporter.before_step([action]);root.step(action)
    assert list(ResidentBattleMirror.groups([r['root'] for r in rows.values()]))==[[0],[1]]
    state=root.export_training_state()
    arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],131,
        card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,observation_capacity=64)
    scorer=ResidentDeckScorer(policy,native_host=host)
    pool=ResidentPool(arena,scorer,2,64,16,180,mirror_capacity=1)
    try:
        pool.submit(rows);actual={}
        gc.collect();torch.cuda.empty_cache()
        while pool.pending:actual.update(pool.receive())
        for i in range(2):
            np.testing.assert_array_equal(actual[i]['actions'],expected[i]['actions'])
            np.testing.assert_array_equal(actual[i]['visits'],expected[i]['visits'])
            np.testing.assert_allclose(actual[i]['values'],expected[i]['values'],rtol=3e-5,atol=3e-6)
        assert root.export_training_state()==state
        assert pool.mirror.counts['loads']==1 and pool.mirror.counts['actual_actions']==1
        with pytest.raises(ValueError,match='backwards'):
            pool.mirror.prepare([rows[0]['root']],[0])
    finally:pool.close(cancel=True)


def test_stream_column_transport_preserves_every_byte_and_ragged_root(
        complete_combat_factory, cuda_battle_artifacts):
    import pickle
    from env.search.resident_chains import root_payload
    from env.search.resident_transport import pack_roots, unpack_roots
    _, _, host = cuda_battle_artifacts
    rows = {}
    for i, cards in enumerate((['STRIKE_RED', 'BASH'], ['DEFEND_RED', 'WARCRY', 'ANGER'])):
        env = complete_combat_factory(cards, seed=42+i, deck_latent_context=True, card_language=True)
        obs = env._obs()
        obs['transport_bits'] = np.array([0., -0., np.nan], np.float32)
        root = root_payload(env, host)
        root['replay'] = list(range(i))
        rows[(27, i)] = dict(root=root, observation=obs,
            context=(np.arange(64, dtype=np.float32)+i, np.arange(128, dtype=np.float32)-i),
            worlds=np.array([2**62+i, i], np.int64)[:i+1], budget=64+192*i,
            submitted_at=1234567.125+i)
    expected = pickle.loads(pickle.dumps(rows, protocol=5))
    packed = pack_roots(rows, host)
    decoded = unpack_roots(pickle.loads(pickle.dumps(packed, protocol=5)))
    assert list(decoded) == list(expected)
    def equal(a, b):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True): equal(x, y)
        else:
            assert a == b
    for key in expected:
        expected[key]['root']['replay'] = np.asarray(expected[key]['root']['replay'], np.int64)
        equal(expected[key], decoded[key])


def test_stream_root_export_consumes_saved_select_zone_without_reencoding(
        complete_combat_factory, cuda_battle_artifacts, monkeypatch):
    from env.search.resident_chains import root_payload
    from env.search.resident_transport import ResidentRootExporter
    folder, _, host = cuda_battle_artifacts
    root = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'WARCRY'], seed=42,
        deck_latent_context=True, card_language=True)
    obs = root._obs()
    expected = root_payload(root, host)
    exporter = ResidentRootExporter([root], folder,
        observations={'select_zone': obs['select_zone'][None].copy()})
    def forbidden():
        raise AssertionError('root export must not rebuild the current observation')
    monkeypatch.setattr(root, '_obs', forbidden)
    actual = exporter.export([0])[0]
    for field in ('actions', 'bits'):
        np.testing.assert_array_equal(expected[field], actual[field])
    for field in ('prefix', 'draw_selection', 'offset', 'skipped'):
        assert expected[field] == actual[field]
    np.testing.assert_array_equal(expected['snapshot']['words'], actual['snapshot']['words'])


def test_completed_payload_preserves_values_order_and_integer_counts():
    from types import SimpleNamespace
    from rl.search.resident_step import ResidentStepPipeline
    p=SimpleNamespace(arena=SimpleNamespace(device='cuda'),
        final=torch.tensor([0.,-0.,.123456789012345,1e-200],device='cuda',dtype=torch.float64),
        anchors=torch.tensor([0,1,0,1],device='cuda',dtype=torch.int32),
        depth=torch.tensor([1,200,17,91],device='cuda',dtype=torch.int32))
    for jobs in ([2,0,3],[1],[3,2,1,0]):
        got=ResidentStepPipeline.completed(p,np.array(jobs))
        np.testing.assert_array_equal(got[:,0].copy().view(np.uint64),p.final[jobs].cpu().numpy().view(np.uint64))
        np.testing.assert_array_equal(got[:,1],p.anchors[jobs].cpu().numpy())
        assert got[:,2].sum()==p.depth[jobs].sum().item()


def test_parked_owner_reloads_weights_without_stale_scores(complete_combat_factory,cuda_battle_artifacts,tmp_path):
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.resident_service import GpuService
    from env.search.resident_chains import root_payload
    from env.search.anchored_chains import paired_anchor_search
    from rl.search.deck_latent import DeckLatentScorer
    root=complete_combat_factory(['STRIKE_RED','DEFEND_RED','BASH'],seed=42,
        deck_latent_context=True,card_language=True)
    torch.manual_seed(20261111)
    policy=DeckLatentPolicy(root.observation_space,root.action_space,lambda _:5e-5,
        d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    folder,_,bridge=cuda_battle_artifacts
    payload=root_payload(root,bridge);snapshot=tmp_path/'frozen.pt'
    def freeze():
        torch.save(dict(space=root.observation_space,geometry=dict(width=32,layers=1,heads=4,
            combat_encoder='transformer64x2'),state_dict={k:v.detach().cpu() for k,v in policy.core.state_dict().items()},
            buffers={k:v.detach().cpu() for k,v in policy.core.named_buffers()}),snapshot)
    freeze()
    settings=dict(artifacts=str(folder),cache_bytes=64*1024**2,budget_cap=64,max_depth=8,timeout=120)
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    free_before=torch.cuda.mem_get_info()[0]
    service=GpuService(2,snapshot,settings);pid=service.process.pid
    try:
        for generation in (1,2):
            if generation==2:
                with torch.no_grad(): policy.core.combat.end.bias.add_(2.)
                freeze();service.begin(snapshot)
            rng=np.random.default_rng(20261111)
            with torch.no_grad():
                context=policy.core.prepare_search_context({k:torch.as_tensor(v[None],device='cuda') for k,v in root._obs().items()})
            row=dict(root=payload,observation=root._obs(),context=tuple(x.cpu().numpy()[0] for x in (context.combat,context.anchor)),
                worlds=rng.integers(0,2**63-1,max(1,64//len(payload['actions'])),dtype=np.int64),
                budget=64,submitted_at=time.monotonic())
            service.submit({'root':row})
            with pytest.raises(RuntimeError,match='consumed'):service.park()
            got={};deadline=time.monotonic()+120
            while service.outstanding:
                assert time.monotonic()<deadline
                got.update(service.receive(.05))
            parked=service.park()
            assert service.process.pid==pid and parked['generation']==generation
            assert parked['park']['reserved']<256*1024**2, 'idle owner must release graph/state storage'
            assert parked['park']['free_bytes']>=free_before-2*1024**3, 'driver stack must be returned for PPO'
            with pytest.raises(RuntimeError,match='parked'):service.submit({'bad':row})
            expected,_=paired_anchor_search([root],[64],DeckLatentScorer(policy,[root._obs()]),
                seed=20261111,max_depth=8,max_active=256,timeout_seconds=120)
            np.testing.assert_array_equal(got['root']['visits'],expected[0]['visits'])
            np.testing.assert_allclose(got['root']['values'],expected[0]['values'],rtol=3e-5,atol=3e-6)
            gc.collect();torch.cuda.empty_cache()
    finally:service.close(cancel='test end')
    assert not service.process.is_alive()


def test_stream_admission_matches_separate_fork_first(complete_combat_factory,cuda_battle_artifacts):
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_chains import root_payload
    folder,manifest,bridge=cuda_battle_artifacts
    roots=[complete_combat_factory(['STRIKE_RED','DEFEND_RED','BASH','WARCRY'],seed=x,
        deck_latent_context=True,card_language=True) for x in (42,43)]
    payloads=[root_payload(r,bridge) for r in roots]
    arenas=[CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],6,
        card_count=_nosl_compact_lut()['layout']['card_count']) for _ in range(2)]
    sources=np.array([0,0,1,1]);destinations=np.arange(2,6);worlds=np.array([31,31,53,67])
    bits=np.array([payloads[i]['bits'][0] for i in sources],np.uint32)
    draw=np.array([payloads[i]['draw_selection'] for i in sources])
    try:
        for arena in arenas: arena.load_slots([r['snapshot'] for r in payloads],[0,0],[0,1])
        arenas[0].fork(sources,destinations,worlds)
        arenas[0].first_selected(sources,destinations,bits,draw)
        arenas[1].start_selected(sources,destinations,worlds,bits,draw)
        for a,b in zip(arenas[0].snapshot(6),arenas[1].snapshot(6),strict=True):
            np.testing.assert_array_equal(a,b)
        left={k:v.clone() for k,v in arenas[0].observe(6).items()}
        right=arenas[1].observe(6)
        for k in left:torch.testing.assert_close(left[k],right[k],rtol=0,atol=0)
    finally:
        for arena in arenas:arena.close()


def test_native_context_reuses_equal_encoder_inputs_across_roots(complete_combat_factory,cuda_battle_artifacts):
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.resident_deck import ResidentDeckScorer
    root=complete_combat_factory(['STRIKE_RED','DEFEND_RED','RITUAL_DAGGER'],seed=42,
        deck_latent_context=True,card_language=True)
    torch.manual_seed(20261111)
    policy=DeckLatentPolicy(root.observation_space,root.action_space,lambda _:5e-5,
        d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    scorer=ResidentDeckScorer(policy,native_host=cuda_battle_artifacts[2])
    try:
        scorer.roots([root._obs(),root._obs()])
        owners=torch.tensor([0,1],device='cuda'); changed=torch.arange(2,device='cuda')
        misc=torch.zeros((2,96),device='cuda',dtype=torch.int32)
        projected=scorer.root['outer_deck_misc'].clone(); projected[:,0]+=.1
        with torch.no_grad():
            expected=policy.core.prepare_search_context(dict(scorer.root,outer_deck_misc=projected))
            actual,ids=scorer.resolve_changed_context(owners,misc,projected,changed,True,with_ids=True)
            torch.testing.assert_close(actual,expected.anchor,rtol=3e-5,atol=3e-6)
            assert scorer.refreshes==1 and ids[0]==ids[1]
            actual=scorer.resolve_changed_context(owners.flip(0),misc,projected,changed,False)
            torch.testing.assert_close(actual,expected.combat,rtol=3e-5,atol=3e-6)
            assert scorer.refreshes==1
            projected[1,0]+=.1
            scorer.resolve_changed_context(owners,misc,projected,changed,False)
            assert scorer.refreshes==2, 'real permanent growth must miss'
            altered=dict(root._obs());altered['floor']=altered['floor'].copy()+.1
            scorer.roots([root._obs(),altered])
            scorer.resolve_changed_context(owners,misc,projected,changed,False)
            assert scorer.refreshes==4, 'all static encoder inputs participate in identity'
    finally:scorer.close()


def test_fused_refill_and_completion_counts_keep_live_siblings():
    import triton
    from env.search import resident_step_kernels as K
    n=48;c=128
    slots=torch.tensor([2,33,9],device='cuda');sources=torch.tensor([1,0,1],device='cuda',dtype=torch.int32)
    root_misc=torch.arange(2*101,device='cuda',dtype=torch.float32).reshape(2,101)[:,:96]
    root_context=torch.arange(2*(c+7),device='cuda',dtype=torch.float32).reshape(2,c+7)[:,:c]
    expected=torch.full((n,96),-7.,device='cuda');context=torch.full((n,c),-8.,device='cuda')
    owners=torch.full((n,),8,device='cuda',dtype=torch.int64);wave=owners.clone()
    depth,cursor,anchors,changed,live=[torch.full((n,),9,device='cuda',dtype=torch.int32) for _ in range(5)]
    final=torch.full((n,),17.,device='cuda',dtype=torch.float64)
    K.refill[(3,)](slots,sources,owners,wave,root_misc,root_context,expected,context,
        depth,cursor,anchors,final,changed,live,3,c,root_misc.stride(0),root_context.stride(0),128,num_warps=4)
    other=torch.ones(n,device='cuda',dtype=torch.bool);other[slots]=False
    torch.testing.assert_close(expected[slots],root_misc[sources.long()],rtol=0,atol=0)
    torch.testing.assert_close(context[slots],root_context[sources.long()],rtol=0,atol=0)
    assert (expected[other]==-7).all() and (context[other]==-8).all()
    for tensor in (owners,wave):
        torch.testing.assert_close(tensor[slots],sources.long(),rtol=0,atol=0)
        assert (tensor[other]==8).all()
    for tensor,want in ((depth,1),(cursor,0),(anchors,0),(changed,0),(live,1),(final,0)):
        assert (tensor[slots]==want).all() and (tensor[other]==(17 if tensor is final else 9)).all()
    live.zero_();live[slots]=1;status=torch.zeros(13,device='cuda',dtype=torch.int32)
    K.remaining_roots[(3,)](live,status,16,triton.next_power_of_2(16))
    np.testing.assert_array_equal(status[10:].cpu().numpy(),[2,0,1])
    assert status[4].item()==3


@pytest.mark.parametrize('page_limit', [None, 64])
def test_stream_accepts_new_roots_before_drain_and_reuses_only_retired_pages(
        complete_combat_factory,cuda_battle_artifacts,page_limit):
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_chains import root_payload
    from env.search.anchored_chains import paired_anchor_search
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.deck_latent import DeckLatentScorer
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.search.resident_pool import ResidentPool
    roots=[complete_combat_factory(['STRIKE_RED','DEFEND_RED','BASH'],seed=seed,
        deck_latent_context=True,card_language=True) for seed in (42,43)]
    before=[r.export_training_state() for r in roots]
    torch.manual_seed(20261111)
    policy=DeckLatentPolicy(roots[0].observation_space,roots[0].action_space,lambda _:5e-5,
        d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    expected=[]
    for i,r in enumerate(roots):
        result,_=paired_anchor_search([r],[64],DeckLatentScorer(policy,[r._obs()]),
            seed=20261111+i,max_depth=32,max_active=256,timeout_seconds=90)
        expected.append(result[0])
    folder,manifest,bridge=cuda_battle_artifacts
    arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],2+128,
        card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,observation_capacity=64)
    scorer=ResidentDeckScorer(policy,graph_capacity=2,native_host=bridge)
    pool=ResidentPool(arena,scorer,2,64,32,180)
    if page_limit is not None:
        pool.state_pages.close()
        allocated = 0
        def limited_allocate(count):
            nonlocal allocated
            if allocated+count > page_limit:
                raise torch.OutOfMemoryError('test: live pages exhaust available VRAM')
            result=torch.empty((count,arena.battle_bytes),dtype=torch.uint8,device='cuda')
            allocated+=count
            return result
        pool.state_pages=bridge.DevicePages(2,64,arena.battle_bytes,limited_allocate,growth_pages=4)
    def row(i):
        payload=root_payload(roots[i],bridge)
        context=policy.core.prepare_search_context({k:torch.as_tensor(v[None],device='cuda')
            for k,v in roots[i]._obs().items()})
        return dict(root=payload,observation=roots[i]._obs(),
            context=tuple(x.detach().cpu().numpy()[0] for x in (context.combat,context.anchor)),
            worlds=np.random.default_rng(20261111+i).integers(0,2**63-1,
                max(1,64//len(payload['actions'])),dtype=np.int64),budget=64,submitted_at=time.monotonic())
    try:
        first=pool.submit({'first':row(0)})['first']
        assert pool.pending
        with pytest.raises(RuntimeError,match='abandon'): pool.close()
        pool.receive(steps=1)
        assert pool.pending, 'fixture must leave live work before second admission'
        # Another CUDA owner needs idle allocator memory while this graph and
        # its root/page leases remain live. Replay may not borrow freed scratch.
        gc.collect(); torch.cuda.empty_cache()
        second={'second':row(1)}
        before_failed=pool.state_pages.statistics()
        accepted=pool.admit(second)
        if page_limit is not None:
            assert accepted=={} and pool.counts['page_oom_retries']==1
            assert set(pool.ledger.pending)=={'first'}
            assert pool.state_pages.statistics()==before_failed
        else:
            assert pool.counts['submissions_while_pending']==1
            second.clear()
        actual={}
        while pool.pending or second:
            for key in pool.admit(second): del second[key]
            actual.update(pool.receive(steps=1))
        for key,i in (('first',0),('second',1)):
            np.testing.assert_array_equal(actual[key]['actions'],expected[i]['actions'])
            np.testing.assert_array_equal(actual[key]['visits'],expected[i]['visits'])
            np.testing.assert_allclose(actual[key]['values'],expected[i]['values'],rtol=3e-5,atol=3e-6)
        assert pool.state_pages.statistics()['owners']==0
        assert pool.counts['chains']==sum(int(x['visits'].sum()) for x in actual.values())
        if page_limit is not None:
            assert pool.counts['deferred_admissions']>0
            assert allocated==64
        gc.collect(); torch.cuda.empty_cache()
        replacement=pool.submit({'replacement':row(0)})['replacement']
        assert replacement.slot==first.slot and replacement.generation>first.generation
        replay={}
        while pool.pending: replay.update(pool.receive())
        np.testing.assert_allclose(replay['replacement']['values'],actual['first']['values'],rtol=3e-5,atol=3e-6)
        assert [r.export_training_state() for r in roots]==before
        pool.close()
        assert pool.state_pages.statistics()['closed'] and pool.state_pages.statistics()['owners']==0
    finally:
        pool.close(cancel=True)


def test_paged_apply_graph_keeps_bounded_observation_frame(
        complete_combat_factory, cuda_battle_artifacts, monkeypatch):
    from env.core.run_env import _nosl_compact_lut
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_chains import root_payload
    roots=[complete_combat_factory(['STRIKE_RED','DEFEND_RED','BASH'],seed=s,
        deck_latent_context=True,card_language=True) for s in (42,43,44)]
    folder,manifest,bridge=cuda_battle_artifacts
    payload=[root_payload(r,bridge) for r in roots]
    arenas=[];storage=[]
    try:
        for graphs in (0,1):
            arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],3,
                card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,
                observation_capacity=1,graph_capacity=graphs)
            arenas.append(arena)
            storage.append(torch.empty((3,arena.battle_bytes),device='cuda',dtype=torch.uint8))
            arena.arena.copy_(storage[-1].data_ptr()+torch.arange(3,device='cuda')*arena.battle_bytes)
            arena.load_slots([p['snapshot'] for p in payload],[p['prefix'] for p in payload],[0,1,2])
        for arena in arenas:
            original=arena._launch
            def checked(name,*args,_launch=original,**kwargs):
                assert name!='observe', 'paged apply must not launch an unbounded observer'
                return _launch(name,*args,**kwargs)
            monkeypatch.setattr(arena,'_launch',checked)
        for n in (3,2,3,1,3):
            for arena in arenas:
                actions=[]
                for row in range(n):
                    raw=arena.observe_selected([row])
                    legal=torch.nonzero(raw['mask'][row]).flatten().cpu().tolist()
                    actions.append(legal[0] if legal else -1)
                frame=arena.scalars.clone()
                arena.apply(actions)
                torch.testing.assert_close(arena.scalars,frame,rtol=0,atol=0)
            a,b=[arena.snapshot(3) for arena in arenas]
            np.testing.assert_array_equal(a[1],b[1])
            for row,length in enumerate(a[1][:,0]):
                np.testing.assert_array_equal(a[0][row,:length],b[0][row,:length])
            gc.collect();torch.cuda.empty_cache()
        assert arenas[1].counts['graph_evictions']>=2
    finally:
        for arena in reversed(arenas):arena.close()


def test_anchor_projection_matches_full_observation_without_mutating_siblings(
        complete_combat_factory, cuda_battle_artifacts):
    from env.core.run_env import _nosl_compact_lut
    from env.core.cuda_observation import CudaObservationDecoder
    from env.search.cuda_battle import CudaBattleArena
    from env.search.resident_chains import root_payload
    roots=[complete_combat_factory(['STRIKE_RED','DEFEND_RED','RITUAL_DAGGER','WARCRY'],seed=seed,
        relics=['Lizard Tail','Pen Nib','Happy Flower','Centennial Puzzle'],
        deck_latent_context=True,card_language=True) for seed in (42,43,44)]
    folder,manifest,bridge=cuda_battle_artifacts
    arena=CudaBattleArena(folder/'deck.cubin',manifest['sha256']['deck.cubin'],3,
        card_count=_nosl_compact_lut()['layout']['card_count'],paged=True,observation_capacity=3)
    storage=torch.empty((3,arena.battle_bytes),device='cuda',dtype=torch.uint8)
    arena.arena.copy_(storage.data_ptr()+torch.arange(3,device='cuda')*arena.battle_bytes)
    decoder=CudaObservationDecoder(roots[0].observation_space)
    root_obs={k:torch.as_tensor(np.stack([r._obs()[k] for r in roots]),device='cuda') for k in roots[0]._obs()}
    rng=np.random.default_rng(20261111)
    try:
        payload=[root_payload(r,bridge) for r in roots]
        arena.load_slots([p['snapshot'] for p in payload],[p['prefix'] for p in payload],[0,1,2])
        for step in range(120):
            raw=arena.observe_selected([0,1,2])
            full={k:v[:3].clone() for k,v in raw.items()}
            before=arena.snapshot(3)
            order=np.array([2,0],np.int64)
            compact,physical,slots=arena.observe_anchor_selected(order)
            actual=decoder.anchor_indexed(compact,root_obs,physical,physical,slots)
            expected=decoder.anchor(full,root_obs,torch.arange(3,device='cuda'))
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key][order],rtol=0,atol=0)
            torch.testing.assert_close(compact['permanent_misc'][physical],full['permanent_misc'][order],rtol=0,atol=0)
            # Sparse projection leaves other physical rows and simulator/RNG untouched.
            for key in ('categorical','permanent_misc','outcome'):
                torch.testing.assert_close(compact[key][1],full[key][1],rtol=0,atol=0)
            after=arena.snapshot(3)
            np.testing.assert_array_equal(before[1],after[1])
            # Variable-length transport does not initialize trailing padding.
            # Every serialized word, including simulator RNG, must be identical.
            for i,length in enumerate(before[1][:,0]):
                np.testing.assert_array_equal(before[0][i,:length],after[0][i,:length])
            arena.observe_selected([0,1,2])  # Actions belong to the full reference codec.
            masks=full['mask'].cpu().numpy(); actions=np.full(3,-1,np.int32)
            for i in range(3):
                if masks[i].any():actions[i]=rng.choice(np.flatnonzero(masks[i]))
            if (actions<0).all():break
            arena.apply(actions)
            if step==3:gc.collect();torch.cuda.empty_cache()
        assert step>5
    finally:arena.close()
