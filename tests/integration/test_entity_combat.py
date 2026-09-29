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
