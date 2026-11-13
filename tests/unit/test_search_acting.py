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


def test_independent_slot_writes_keep_original_ppo_storage_and_returns():
    from gymnasium import spaces
    from rl.algorithms.search_auxiliary import SearchRolloutBuffer
    from rl.algorithms.shared_buffer import SharedRolloutBuffer
    space=spaces.Dict({'x':spaces.Box(-10,10,(2,),dtype=np.float32)})
    a=SharedRolloutBuffer(4,space,spaces.Discrete(3),n_envs=3,device='cpu',gamma=1.,gae_lambda=1.)
    b=SearchRolloutBuffer(4,space,spaces.Discrete(3),n_envs=3,device='cpu',gamma=1.,gae_lambda=1.)
    def row(t,i):
        return np.array([t,i],np.float32),(t+i)%3,float(t+i),float(t==0),float(t/10),float(-.5)
    for t in range(4):
        obs,act,rew,start,value,logp=map(np.asarray,zip(*(row(t,i) for i in range(3))))
        a.add({'x':obs},act,rew,start,torch.tensor(value),torch.tensor(logp),action_masks=np.ones((3,3),bool))
    for ids in ([2,0],[2],[1,0],[2,1],[0,1],[2],[0,1]):
        obs,act,rew,start,value,logp=map(np.asarray,zip(*(row(int(b.slot_positions[i]),i) for i in ids)))
        b.add_subset(ids,{'x':obs},act,rew,start,value,logp,action_masks=np.ones((len(ids),3),bool))
        obs[:]=-9
    assert b.full and np.all(b.slot_positions==4)
    for buffer in (a,b):
        buffer.compute_returns_and_advantage(torch.ones(3),np.zeros(3,bool))
    np.testing.assert_array_equal(a.observations['x'],b.observations['x'])
    for key in ('actions','rewards','episode_starts','values','log_probs','action_masks','returns','advantages'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
