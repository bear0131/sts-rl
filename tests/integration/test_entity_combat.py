import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from env.core import native_actions as C
from rl.policies.entity_combat import EntityCombatPolicy
from rl.search.native import NativeSearchConfig, search

pytestmark = [pytest.mark.native,pytest.mark.timeout(90)]


def make_model(env):
    return MaskablePPO(EntityCombatPolicy,env,seed=42,n_steps=4,batch_size=4,device='cpu',
        policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4,quantiles=8)))


def test_entity_model_predicts_searches_and_reloads_without_overworld(combat_factory,tmp_path):
    env=combat_factory(['STRIKE_RED']*5,'Two Louse',combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    model=make_model(env)
    assert not hasattr(model.policy.core,'overworld')
    obs=env._obs();mask=env.action_masks()
    action,_=model.predict(obs,action_masks=mask,deterministic=True)
    assert mask[int(action)]
    path=tmp_path/'entity.zip';model.save(path)
    loaded=MaskablePPO.load(path,device='cpu')
    other,_=loaded.predict(obs,action_masks=mask,deterministic=True)
    assert int(action)==int(other)
    result=search(loaded,env,obs,NativeSearchConfig(simulations=4,depth=2),nonce=42,stochastic=False)
    assert mask[result['action']]
    assert 0 <= result['root_value'] <= 1


def test_entity_value_and_source_logits_respect_entity_permutations(combat_factory):
    env=combat_factory(['BASH','STRIKE_RED','DEFEND_RED','ANGER','CLEAVE'],'Two Louse',
        combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    model=make_model(env);model.policy.set_training_mode(False)
    obs={k:torch.as_tensor(v[None]) for k,v in env._obs().items()}
    perm=torch.tensor([1,0,2,3,4,5,6,7,8,9])
    changed={k:v.clone() for k,v in obs.items()}
    for key in ('hand_ids','hand_upgrades','hand_costs','hand_damage','hand_block','select_hand_mask'):
        changed[key]=changed[key][:,perm]
    changed['combat_legal'][:,1:61]=obs['combat_legal'][:,1:61].reshape(1,10,6)[:,perm].flatten(1)
    with torch.no_grad():
        a,v=model.policy.core.combat_actor_critic(obs)
        b,w=model.policy.core.combat_actor_critic(changed)
    torch.testing.assert_close(v,w,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(a[:,C.JOINT_PLAY:C.JOINT_POTION].reshape(1,10,6)[:,perm],
                              b[:,C.JOINT_PLAY:C.JOINT_POTION].reshape(1,10,6),atol=2e-6,rtol=1e-5)


def test_entity_auxiliary_target_uses_actual_enemy_hp(combat_factory):
    from rl.algorithms.combat_distribution import transition_features
    env=combat_factory(['STRIKE_RED']*5,combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    obs=env._obs();changed={k:v.copy() for k,v in obs.items()}
    changed['monster_scalars'][0,1]-=.1
    delta=transition_features(changed)-transition_features(obs)
    np.testing.assert_allclose(delta,[0,0,0,-.1,0,0],atol=1e-7)


def test_entity_pile_encoding_cannot_observe_hidden_card_order(combat_factory):
    env=combat_factory(['BASH','STRIKE_RED','DEFEND_RED','ANGER','CLEAVE']*3,
        combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    model=make_model(env);model.policy.set_training_mode(False)
    obs={k:torch.as_tensor(v[None]) for k,v in env._obs().items()}
    shuffled={k:v.clone() for k,v in obs.items()}
    for pile in ('draw','discard'):
        for suffix in ('ids','counts','upgrades','upgraded_counts'):
            key=pile+'_card_'+suffix
            shuffled[key]=shuffled[key].flip(1)
    with torch.no_grad():
        a,v=model.policy.core.combat_actor_critic(obs)
        b,w=model.policy.core.combat_actor_critic(shuffled)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(v,w,atol=1e-6,rtol=1e-5)


def test_real_loss_has_explicit_outcome(short_combat_env_type):
    env=short_combat_env_type(combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    try:
        env.reset(seed=42)
        env.thick.bc.apply_custom_state({'player':{'hp':1,'block':0}})
        env._invalidate_backend_cache()
        for _ in range(6):
            _,_,terminated,truncated,info=env.step(C.JOINT_END)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info['nosl_value_terminal_outcome']=='loss'
        assert info['nosl_value_terminal_target']==0
    finally:
        env.close()


def test_truncation_does_not_create_outcome_labels(short_combat_env_type):
    from gymnasium.wrappers import TimeLimit
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    env=TimeLimit(short_combat_env_type(combat_action_contract=C.JOINT_CONTRACT,card_language=True),1)
    model=AccumulatingMaskablePPO(EntityCombatPolicy,env,device='cpu',seed=42,
        n_steps=2,batch_size=2,nosl_is_az=True,nosl_is_az_cpp=True,nosl_is_az_inner_only=True,
        nosl_is_az_upper_strategy='rules',nosl_is_az_simulations=1,nosl_is_az_depth=1,
        policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)),
        nosl_is_az_replay_capacity=32,nosl_is_az_batch_size=2)
    try:
        model.learn(2)
        replay=getattr(model,'_nosl_is_az_replay',None)
        assert replay is None or replay.size==0
        assert not any(model._nosl_is_az_pending_fights)
    finally:
        env.close()


def test_fresh_only_distribution_training_updates_heads(short_combat_env_type):
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from gymnasium.wrappers import TimeLimit
    env=TimeLimit(short_combat_env_type(combat_action_contract=C.JOINT_CONTRACT,card_language=True),12)
    model=AccumulatingMaskablePPO(EntityCombatPolicy,env,device='cpu',seed=42,verbose=1,
        n_steps=16,batch_size=4,nosl_is_az=True,nosl_is_az_cpp=True,nosl_is_az_inner_only=True,
        nosl_is_az_upper_strategy='rules',nosl_is_az_simulations=2,nosl_is_az_depth=1,
        nosl_is_az_replay_enabled=False,nosl_is_az_batch_size=16,nosl_is_az_micro_batch_size=4,
        policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)))
    distribution=model.policy.core.combat.distribution
    before={k:v.clone() for k,v in distribution.named_parameters()}
    try:
        model.learn(32)
        assert model.policy.optimizer.state
        assert any(not torch.equal(before[k],v) for k,v in distribution.named_parameters())
        assert getattr(model,'_nosl_is_az_replay',None) is None
    finally:
        env.close()
