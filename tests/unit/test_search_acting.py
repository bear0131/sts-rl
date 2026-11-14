import numpy as np
import pytest
import torch

from env.search.action_selection import search_filtered_choice


def test_search_filters_then_model_ranks_and_keeps_legacy_ties():
    q=np.array([.1,.396,.4,.4]); scores=np.array([100.,2.,1.,1.]); visits=np.ones(4)
    assert search_filtered_choice(scores,q,visits,123)==(1,0.)
    scores[1]=0.
    picked,logp=search_filtered_choice(scores,q,visits,123)
    expected=np.random.default_rng(123 ^ 0x524F4F54544945).choice([2,3])
    assert picked==expected and logp==pytest.approx(-np.log(2))
    visits[0]=0; q[0]=1.
    assert search_filtered_choice(scores,q,visits,123)==(picked,logp)


def test_search_actions_have_value_and_teacher_but_no_fictitious_ppo_ratio():
    from rl.algorithms.ppo import ppo_loss_components
    logits=torch.tensor([-.3,-.4],requires_grad=True)
    value=torch.tensor([.2,.3],requires_grad=True)
    c=ppo_loss_components(hierarchical=False,normalize_advantage=False,
        observations={'_search_executed':torch.tensor([1.,0.])},advantages=torch.ones(2),
        returns=torch.ones(2),old_values=torch.zeros(2),old_log_prob=torch.tensor([0.,-.4]),
        values=value,log_prob=logits,entropy=-logits,clip_range=.2,clip_range_vf=None)
    (c.policy_loss+.02*c.entropy_loss+c.value_loss).backward()
    assert logits.grad[0]==0 and logits.grad[1]!=0
    assert (value.grad!=0).all()
    assert c.approx_kl==0 and c.clip_fraction==0


def test_search_schedule_versions_actual_behavior():
    from types import SimpleNamespace
    from rl.algorithms.deck_schedule import DeckTrainingSchedule, apply_stage
    model=SimpleNamespace(policy=SimpleNamespace(),policy_kwargs={},deck_training_state=dict(
        stage='search',stage_start=240,validation={'target_std':.07}))
    apply_stage(model,DeckTrainingSchedule(search_qualified=True,
        search_action_selection='search-filter-policy-rank-v1'),246)
    assert model.training_objective_contract=='deck-latent-search-acting-v1'
    assert model.deck_search_config['action_selection']=='search-filter-policy-rank-v1'
    assert model.deck_search_config['fraction']==pytest.approx(.4)


def test_native_search_filter_matches_reference_for_boundaries_and_seeded_ties():
    from slaythespire import filtered_action_ties
    rng=np.random.default_rng(83147)
    for size in (1,3,17,96):
        for repeat in range(16):
            logits=rng.integers(-2,3,size).astype(np.float32)
            values=rng.integers(-100,100,size).astype(np.float64)/1000
            visits=rng.integers(0,5,size,dtype=np.int64);visits[0]=1
            if size>1:
                values[1]=values[0]-.005
            eligible=np.flatnonzero((visits>0)&(values>=values[visits>0].max()-.005))
            tied=eligible[logits[eligible]==logits[eligible].max()]
            assert filtered_action_ties(logits,values,visits)==list(tied)
            picked,logp=search_filtered_choice(logits,values,visits,repeat)
            expected=np.random.default_rng(repeat ^ 0x524F4F54544945).choice(tied)
            assert picked==expected and logp==-float(np.log(len(tied)))
            assert np.float64(logp).tobytes()==np.float64(-np.log(len(tied))).tobytes()
    for scores,values,visits in [([np.nan],[0.],[1]),([1.],[np.inf],[1]),([1.],[0.],[0]),([1.,2.],[0.],[1])]:
        with pytest.raises(ValueError,match='finite aligned'):
            filtered_action_ties(scores,values,visits)
