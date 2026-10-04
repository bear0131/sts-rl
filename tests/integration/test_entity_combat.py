import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from env.core import native_actions as C
from rl.policies.entity_combat import EntityCombatPolicy, disable_entity_transition_auxiliary
from rl.search.native import NativeSearchConfig, search

pytestmark = [pytest.mark.native,pytest.mark.timeout(90)]


def make_model(env):
    return MaskablePPO(EntityCombatPolicy,env,seed=42,n_steps=4,batch_size=4,device='cpu',
        policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4,quantiles=8)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('masked', [False, True])
def test_entity_attention_matches_pytorch_outputs_and_gradients(dtype, masked):
    from rl.policies.entity_combat import EntityEncoderLayer
    torch.manual_seed(23)
    kwargs = dict(d_model=32, nhead=4, dim_feedforward=128, dropout=0.,
                  activation='gelu', batch_first=True, norm_first=True, dtype=dtype)
    reference = torch.nn.TransformerEncoderLayer(**kwargs)
    actual = EntityEncoderLayer(**kwargs)
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(actual.state_dict())
    data = torch.randn(3, 11, 32, dtype=dtype)
    padding = torch.tensor([[False] * 11, [False, True] * 5 + [False],
                            [False] + [True] * 10]) if masked else None
    def run(layer):
        x = data.clone().requires_grad_()
        output = layer(x, src_key_padding_mask=padding)
        (output.sin().square().sum()).backward()
        return output, x.grad, {k: p.grad for k, p in layer.named_parameters()}
    expected, expected_dx, expected_grads = run(reference)
    observed, observed_dx, observed_grads = run(actual)
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(observed, expected, **tolerance)
    torch.testing.assert_close(observed_dx, expected_dx, **tolerance)
    for name in expected_grads:
        torch.testing.assert_close(observed_grads[name], expected_grads[name], **tolerance)


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('width,heads,length', [(32, 4, 11), (256, 8, 54)])
def test_entity_attention_matches_explicit_fp64_formula(optimized, masked, width, heads, length):
    """Judge both implementations against the formula, not against each other."""
    from rl.policies.entity_combat import EntityEncoderLayer
    torch.manual_seed(1729)
    layer_type = EntityEncoderLayer if optimized else torch.nn.TransformerEncoderLayer
    layer = layer_type(width, heads, 4*width, dropout=0., batch_first=True,
                       norm_first=True, dtype=torch.float64).train()
    data = torch.randn(3, length, width, dtype=torch.float64)
    padding = torch.zeros(3, length, dtype=torch.float64)
    if masked:
        padding[1, 1::2] = -torch.inf
        padding[2, 1:] = -torch.inf  # One visible key, as in a mostly empty state.
    upstream = torch.randn_like(data) / data.numel()
    attention = layer.self_attn
    parameters = (attention.in_proj_weight, attention.in_proj_bias,
                  attention.out_proj.weight, attention.out_proj.bias)

    def explicit_attention(x):
        # Independent per-head projections: no combined QKV reshape, MHA or SDPA.
        head_width = width // heads
        head_outputs = []
        for head in range(heads):
            projected = []
            for component in range(3):
                start = component * width + head * head_width
                projected.append(torch.nn.functional.linear(
                    x, attention.in_proj_weight[start:start+head_width],
                    attention.in_proj_bias[start:start+head_width]))
            query, key, value = projected
            scores = query @ key.transpose(-1, -2) / head_width**.5
            probabilities = torch.softmax(scores + padding[:, None, :], dim=-1)
            head_outputs.append(probabilities @ value)
        return torch.nn.functional.linear(torch.cat(head_outputs, dim=-1),
                                          attention.out_proj.weight, attention.out_proj.bias)

    x = data.clone().requires_grad_()
    expected = explicit_attention(x)
    expected_grads = torch.autograd.grad((expected*upstream).sum(), (x, *parameters))
    y = data.clone().requires_grad_()
    actual = layer._sa_block(y, None, padding)
    actual_grads = torch.autograd.grad((actual*upstream).sum(), (y, *parameters))
    # FP64 accumulation may reorder additions; this is far below FP32/BF16 noise.
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, reference, rtol=1e-11, atol=1e-12)


def test_entity_inference_keeps_reference_attention_layout():
    from rl.policies.entity_combat import EntityEncoderLayer
    torch.manual_seed(23)
    kwargs = dict(d_model=32, nhead=4, dim_feedforward=128, dropout=0.,
                  activation='gelu', batch_first=True, norm_first=True)
    reference = torch.nn.TransformerEncoderLayer(**kwargs).eval()
    actual = EntityEncoderLayer(**kwargs).eval()
    actual.load_state_dict(reference.state_dict())
    for count, length in ((2, 37), (4, 54), (3, 49)):
        x = torch.randn(count, length, 32)
        padding = torch.rand(count, length) > .5
        padding[:, 0] = False
        # Autograd enabled bypasses the fused inference fast path, exercising
        # the attention implementation used by AMP/compiled leaf forwards.
        torch.testing.assert_close(actual(x, src_key_padding_mask=padding),
                                   reference(x, src_key_padding_mask=padding),
                                   rtol=0, atol=0)


def test_card_table_lookup_preserves_padded_choice_gradients(combat_factory):
    env=combat_factory(['BASH','STRIKE_RED','DEFEND_RED','ANGER','CLEAVE'],
        combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    network=make_model(env).policy.core.combat
    observation=env._obs()
    ids=torch.as_tensor(np.repeat(observation['select_card_ids'][None],2,axis=0))
    upgrades=torch.as_tensor(np.repeat(observation['select_card_upgrades'][None],2,axis=0))
    ids[1,0]=1;upgrades[1,0]=1
    table=network._encode_card_table().detach()

    def run(reference):
        weight=table.clone().requires_grad_()
        network.zero_grad(set_to_none=True)
        if reference:
            vectors=weight[ids.long(),upgrades.gt(0).long()]
            output=network.card_projection(torch.cat((vectors,upgrades.float().unsqueeze(-1)),-1))
        else:
            output=network._card_tokens(weight,ids,upgrades)
        output.square().sum().backward()
        return output.detach(),weight.grad.detach(),network.card_projection.weight.grad.detach().clone()

    expected=run(True)
    actual=run(False)
    for observed,reference in zip(actual,expected,strict=True):
        torch.testing.assert_close(observed,reference,rtol=1e-5,atol=1e-5)


def test_shared_pile_attention_matches_independent_pools_and_gradients(combat_factory, monkeypatch):
    env=combat_factory(['BASH','STRIKE_RED','DEFEND_RED','ANGER','CLEAVE'],
        combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    network=make_model(env).policy.core.combat
    network.train()
    obs={k:torch.as_tensor(np.repeat(v[None],2,axis=0)) for k,v in env._obs().items()}
    # Different piles exercise all three count masks over the same card table.
    obs['draw_counts'][1].zero_()
    obs['discard_counts'][0].zero_()
    original_pool=network._pool

    def run(independent):
        network.zero_grad(set_to_none=True)
        with monkeypatch.context() as patch:
            if independent:
                patch.setattr(network,'_pool',lambda entities,counts,context,**kwargs:
                    original_pool(entities[None].expand(len(counts),-1,-1)
                                  if entities.ndim==2 else entities,counts,context))
            state,blocks=network.encode(obs)
            loss=state.square().sum()+blocks['pile'].square().sum()
            loss.backward()
        return (state.detach().clone(),blocks['pile'].detach().clone(),
                {name:p.grad.detach().clone() for name,p in network.named_parameters()
                 if p.grad is not None})

    old_state,old_piles,old_grads=run(True)
    new_state,new_piles,new_grads=run(False)
    torch.testing.assert_close(new_state,old_state,rtol=0,atol=0)
    torch.testing.assert_close(new_piles,old_piles,rtol=0,atol=0)
    assert new_grads.keys()==old_grads.keys()
    for name in new_grads:
        torch.testing.assert_close(new_grads[name],old_grads[name],rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('amp', [False, True])
def test_card_cache_reuses_constants_but_preserves_training_and_reload(combat_factory, tmp_path, amp, monkeypatch):
    env=combat_factory(['BASH','STRIKE_RED','DEFEND_RED','ANGER','CLEAVE'],
        combat_action_contract=C.JOINT_CONTRACT,card_language=True)
    model=make_model(env);model.policy.set_training_mode(False)
    network=model.policy.core.combat
    obs={k:torch.as_tensor(v[None]) for k,v in env._obs().items()}
    calls=[]
    hook=network.card.local_encoder.register_forward_hook(lambda *args:calls.append(1))
    try:
        with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
            for _ in range(3):
                cached=network(obs)
            assert len(calls)==1
            entry=next(iter(network._card_inference_tables.values()))[1]
            pointer=entry.data_ptr()
            with monkeypatch.context() as patch:
                patch.setattr(network, 'inference_card_table', network._encode_card_table)
                reference=network(obs)
            assert torch.equal(cached[0],reference[0])
            assert torch.equal(cached[1],reference[1])
        network.train()
        model.policy.optimizer.zero_grad()
        _,value,_=network(obs)
        value.sum().backward()
        assert network.card.local_encoder.weight.grad.abs().sum()>0
        model.policy.optimizer.step()
        network.eval()
        with torch.inference_mode(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
            after=network(obs)
            assert next(iter(network._card_inference_tables.values()))[1].data_ptr()==pointer
            assert not torch.equal(after[1],cached[1])
            with monkeypatch.context() as patch:
                patch.setattr(network, 'inference_card_table', network._encode_card_table)
                expected=network(obs)
            assert torch.equal(after[0],expected[0])
            assert torch.equal(after[1],expected[1])
            state={k:v.clone() for k,v in network.state_dict().items()}
            state['card.local_encoder.bias'].add_(.25)
            network.load_state_dict(state)
            after=network(obs)
            with monkeypatch.context() as patch:
                patch.setattr(network, 'inference_card_table', network._encode_card_table)
                expected=network(obs)
            assert torch.equal(after[0],expected[0])
        with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=not amp,cache_enabled=False):
            switched=network(obs)
            with monkeypatch.context() as patch:
                patch.setattr(network, 'inference_card_table', network._encode_card_table)
                expected=network(obs)
            assert torch.equal(switched[0],expected[0])
            assert torch.equal(switched[1],expected[1])
        assert len(network._card_inference_tables)==2
        assert not any('inference_table' in k for k in network.state_dict())
        path=tmp_path/'cache.zip';model.save(path)
        restored=MaskablePPO.load(path,device='cpu')
        assert not restored.policy.core.combat._card_inference_tables
    finally:
        hook.remove()


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


def test_disable_action_effect_head_keeps_value_and_checkpoint_compatible(combat_factory, tmp_path):
    env = combat_factory(['STRIKE_RED'] * 5, 'Two Louse',
                         combat_action_contract=C.JOINT_CONTRACT, card_language=True)
    legacy = MaskablePPO(EntityCombatPolicy, env, seed=42, n_steps=4, batch_size=4,
                        device='cpu', policy_kwargs=dict(entity_config=dict(
                            width=32, layers=1, heads=4, quantiles=8,
                            auxiliary_loss_coef=.1)))
    path = tmp_path / 'with-action-effect.zip'
    legacy.save(path)
    loaded = MaskablePPO.load(path, device='cpu')
    obs = {key: torch.as_tensor(value[None]) for key, value in env._obs().items()}
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        before_logits, before_value, before_output = loaded.policy.core.combat_training_outputs(
            obs, torch.tensor([0]))
    assert 'transition' in before_output
    disable_entity_transition_auxiliary(loaded)
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        after_logits, after_value, after_output = loaded.policy.core.combat_training_outputs(
            obs, torch.tensor([0]))
    torch.testing.assert_close(after_logits, before_logits)
    torch.testing.assert_close(after_value, before_value)
    assert 'transition' not in after_output
    assert not any(p.requires_grad for p in loaded.policy.core.combat.transition.parameters())
    switched = tmp_path / 'without-action-effect.zip'
    loaded.save(switched)
    restored = MaskablePPO.load(switched, device='cpu')
    assert restored.policy.entity_config['auxiliary_loss_coef'] == 0
    assert 'transition' not in restored.policy.core.combat_training_outputs(
        obs, torch.tensor([0]))[2]


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
    transition = model.policy.core.combat.transition
    unused = {k: v.clone() for k, v in transition.named_parameters()}
    try:
        model.learn(32)
        assert model.policy.optimizer.state
        assert any(not torch.equal(before[k],v) for k,v in distribution.named_parameters())
        assert all(torch.equal(unused[k], v) for k, v in transition.named_parameters())
        assert all(v not in model.policy.optimizer.state for v in transition.parameters())
        assert getattr(model,'_nosl_is_az_replay',None) is None
    finally:
        env.close()
