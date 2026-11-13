import gc
import time

import numpy as np
import pytest
import torch

pytestmark=[pytest.mark.native,pytest.mark.cuda,pytest.mark.timeout(240)]


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
    scorer=ResidentDeckScorer(policy,graph_capacity=2)
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
