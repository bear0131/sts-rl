import numpy as np
import pytest
import torch
import slaythespire as sts
from sb3_contrib import MaskablePPO

from env.core import native_actions as C
from rl.policies.split_policy import SplitStSPointerPolicy

pytestmark = pytest.mark.native


def test_joint_model_predicts_atomic_targeted_actions(combat_factory):
    env = combat_factory(['BASH','STRIKE_RED','DEFEND_RED','STRIKE_RED','DEFEND_RED'],
                         'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    model = MaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cpu')
    model.policy.set_training_mode(False)
    obs = env._obs()
    assert env.action_space.n == C.JOINT_SIZE
    assert obs['combat_legal'].shape == (C.JOINT_WIDTH,)
    mask = env.action_masks()
    assert not mask[:C.JOINT_END].any()
    predicted, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert mask[int(predicted)]
    pair = next((aid,bits) for aid,bits in env.thick.bc.joint_policy_actions()
                if bits >> 29 == 0 and env.thick.bc.is_targeted_action(bits))
    aid, bits = pair
    before = env.thick.bc.clone()
    before.execute(sts.Action(bits))
    result = env.step(aid)
    assert env._pending is None and not result[4].get('illegal')
    assert env.thick.bc.to_dict() == before.to_dict()


@pytest.mark.parametrize('multi',[False,True])
def test_native_complete_policy_survives_search_and_execution(combat_factory,multi):
    env = combat_factory(['PURITY' if multi else 'STRIKE_RED']*5, 'Two Louse',
                         combat_action_contract=C.JOINT_CONTRACT)
    if multi:
        env.thick.bc.apply_custom_state({'hand':['PURITY']+['STRIKE_RED']*9})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
        env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    logits = np.zeros(C.JOINT_SIZE)
    legal = np.flatnonzero(env.action_masks())
    assert len(legal) == (130 if multi else 11)
    favored = legal[-1] if multi else C.JOINT_PLAY+1
    logits[favored] = 12
    response = env.nosl_is_az_cpp_begin(logits,state_nonce=42,simulations=4,depth=1,
        leaf_batch_size=4,compact_observation=True)
    for _ in range(8):
        if response['done']:
            break
        obs = response['observation_batch']
        assert obs['combat_legal'].shape[1] == C.JOINT_WIDTH
        for row, leaf in enumerate(response['leaf_meta']):
            np.testing.assert_array_equal(leaf['action_mask'][C.JOINT_END:], obs['combat_legal'][row])
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, np.zeros(C.JOINT_SIZE)) for leaf in response['leaf_meta']],max_leaves=4)
    assert response['done']
    result = response['result']
    assert result['visit_policy'].shape == (C.JOINT_SIZE,)
    assert result['visit_policy'].sum() == pytest.approx(1)
    assert result['action'] == favored
    assert env.thick.bc.to_dict() == before
    env.step(result['action'])
    assert env.thick.bc.to_dict() != before


def test_cpp_and_python_complete_codec_match_public_entities(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE']*5, combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['SECRET_TECHNIQUE'],
        'draw':['STRIKE_RED']*12+['SHRUG_IT_OFF','DEFEND_RED']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
    desc=env.thick.bc.select_info()
    handles=[c['native_index'] for c in desc['candidates']]
    for aid,bits in env.thick.bc.joint_policy_actions():
        assert aid == C.joint_action_id(bits, targeted=env.thick.bc.is_targeted_action(bits),candidate_indices=handles)
        assert C.joint_native_bits(aid,candidate_indices=handles) == bits


def test_pending_target_views_share_the_source_conditional_distribution(combat_factory):
    from env.core import action_space as A
    from env.core import vocab as V
    env = combat_factory(['BASH','STRIKE_RED','DEFEND_RED','STRIKE_RED','DEFEND_RED'],
                         'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['BASH','STRIKE_RED','DEFEND_RED']})
    env._invalidate_backend_cache()
    model = MaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cpu')
    model.policy.set_training_mode(False)
    with torch.no_grad():
        model.policy.core.combat.complete_head.source_query.weight.normal_(0, .5)
    obs = env._obs()
    variants = [obs]
    for slot in (0,1):
        pending = {k:v.copy() for k,v in obs.items()}
        pending['arg_step'][0] = V.ARG_STEP_VOCAB_INDEX[A.PICK_TARGET]
        pending['pending_card_slot'][0] = slot+1
        variants.append(pending)
    tensors = {k:torch.as_tensor(np.stack([view[k] for view in variants])) for k in obs}
    with torch.no_grad():
        logits, values = model.policy.core.combat_actor_critic(tensors)
    for row, slot in enumerate((0,1), 1):
        root = logits[0,C.JOINT_PLAY+6*slot:C.JOINT_PLAY+6*slot+2].softmax(-1)
        pending = logits[row,A._TARGET:A._TARGET+2].softmax(-1)
        torch.testing.assert_close(root,pending,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(values[0],values[row])
    assert not torch.allclose(logits[1,A._TARGET:A._TARGET+2].softmax(-1),
                              logits[2,A._TARGET:A._TARGET+2].softmax(-1),atol=1e-6,rtol=1e-6)
