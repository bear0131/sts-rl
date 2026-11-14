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


@pytest.mark.parametrize('backend', ['cuda-batched-native', 'cuda-resident'])
def test_withdrawn_sampler_rejects_old_config_without_silent_fallback(backend):
    from dataclasses import asdict
    from types import SimpleNamespace
    from rl.algorithms.deck_search import ShadowSearchCollection
    from rl.algorithms.search_auxiliary import SearchTrainingConfig

    # Saved configuration remains readable; execution stops before environment
    # access, model inference or accidental substitution of CPU/shadow search.
    saved = asdict(SearchTrainingConfig(backend=backend, cuda_artifacts='retired-build',
                                      action_selection='search-filter-policy-rank-v1'))
    model = SimpleNamespace(deck_search_config=saved)
    with pytest.raises(RuntimeError, match='withdrawn at user request; no CPU/shadow fallback'):
        ShadowSearchCollection(model, None, None)
