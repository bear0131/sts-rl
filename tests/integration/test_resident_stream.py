import gc
import time

import numpy as np
import pytest
import torch

pytestmark=[pytest.mark.native,pytest.mark.cuda,pytest.mark.timeout(240)]


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


def test_stream_accepts_new_roots_before_drain_and_reuses_only_retired_pages(
        complete_combat_factory,cuda_battle_artifacts):
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
        pool.submit({'second':row(1)})
        assert pool.counts['submissions_while_pending']==1
        actual={}
        while pool.pending: actual.update(pool.receive(steps=1))
        for key,i in (('first',0),('second',1)):
            np.testing.assert_array_equal(actual[key]['actions'],expected[i]['actions'])
            np.testing.assert_array_equal(actual[key]['visits'],expected[i]['visits'])
            np.testing.assert_allclose(actual[key]['values'],expected[i]['values'],rtol=3e-5,atol=3e-6)
        assert pool.state_pages.statistics()['owners']==0
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
