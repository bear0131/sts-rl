import numpy as np
import pytest
import torch
import env.core
import slaythespire as sts
from rl.policies.public_playout import PublicPlayoutPolicy,public_sketch
from rl.policies.small_playout import collate_features

pytestmark=pytest.mark.native


@pytest.mark.parametrize('per_state',[False,True])
def test_native_refresh_and_torch_observation_agree(combat_factory,per_state):
    torch.manual_seed(124)
    env=combat_factory(['BASH','CLEAVE','ANGER']*3,relics=['Anchor','Frozen Eye'])
    gc,bc=env.thick.gc,env.thick.bc
    bc.apply_custom_state({'hand':['BASH','CLEAVE','ANGER'],'draw':['DEFEND_RED'],'discard':['BASH']})
    net=PublicPlayoutPolicy(sts.small_playout_key_count())
    torch.nn.init.normal_(net.context.weight,std=.15)
    w=net.export()
    native=sts.SmallPlayoutNet(w[:-net.context.weight.numel()])
    adapter=sts.PublicPlayoutAdapter(w[-net.context.weight.numel():])
    cfg=sts.SmallPlayoutConfig()
    cfg.condition(adapter,gc,bc,per_state)
    root=cfg.public_context.copy()
    for changed in [False,True]:
        if changed:bc.apply_custom_state({'discard':['STRIKE_RED','WOUND']})
        f=sts.small_playout_public_features(gc,bc)
        table={k:np.concatenate([f['entities'][k],f['inventory'][k]]) for k in f['entities']}
        sketch=public_sketch(table,[len(table['ids'])])
        if not changed:np.testing.assert_allclose(root,sketch[0],atol=1e-6,rtol=1e-6)
        else: assert not np.array_equal(root,sketch[0])
        state,actions,keys,mask=collate_features([f])
        expected=net(state,actions,keys,torch.from_numpy(sketch if per_state else root[None,:]),mask)
        np.testing.assert_allclose(native.conditioned_logits(bc,cfg),expected[0].detach().numpy(),atol=2e-5,rtol=2e-5)
    cfg.simulations=12
    out=native.search(bc,cfg,42)
    assert sum(out['visits'])==12 and out['action'] in out['action_bits']


def test_live_entity_cache_rejects_old_weights_and_evicted_properties(combat_factory):
    torch.manual_seed(211)
    env=combat_factory(['BASH','ANGER','CLEAVE']*3)
    gc,bc=env.thick.gc,env.thick.bc
    bc.apply_custom_state({'hand':['BASH','ANGER','CLEAVE']})
    models=[PublicPlayoutPolicy(sts.small_playout_key_count()) for _ in range(2)]
    native=[]; configs=[]
    for model in models:
        torch.nn.init.normal_(model.context.weight,std=.2)
        weights=model.export()
        native.append(sts.SmallPlayoutNet(weights[:-65536]))
        adapter=sts.PublicPlayoutAdapter(weights[-65536:])
        cfg=sts.SmallPlayoutConfig()
        cfg.condition(adapter,gc,bc,True)
        configs.append(cfg)
    baseline=[np.asarray(n.conditioned_logits(bc,c)) for n,c in zip(native,configs)]
    assert not np.allclose(*baseline)
    # More distinct card rows than cache entries, changing only an instance attribute.
    other=bc.clone()
    for j in range(0,2200,20):
        other.apply_custom_state({'discard':[('SEARING_BLOW',j+k) for k in range(20)]})
        native[0].conditioned_logits(other,configs[0])
    for i in (1,0,1):
        np.testing.assert_array_equal(native[i].conditioned_logits(bc,configs[i]),baseline[i])


def test_turn_mode_refreshes_inside_each_simulation_without_reencoding_every_action(combat_factory):
    env=combat_factory(['STRIKE_RED']*5+['DEFEND_RED']*4+['BASH'])
    g,b=env.thick.gc,env.thick.bc
    net=PublicPlayoutPolicy(sts.small_playout_key_count())
    weights=net.export()
    native=sts.SmallPlayoutNet(weights[:-65536])
    adapter=sts.PublicPlayoutAdapter(weights[-65536:])
    counts=[]
    for mode in ('root','turn','state'):
        cfg=sts.SmallPlayoutConfig();cfg.simulations=30
        if mode=='turn':cfg.condition_turns(adapter,g,b)
        else:cfg.condition(adapter,g,b,mode=='state')
        result=native.search(b,cfg,11571)
        stats=result['stats']
        counts.append(stats.get('public_state_refreshes',0))
        if mode=='root':baseline=result
        else:
            # Zero initial adapter isolates the refresh schedule from game outcomes.
            assert (result['action'],result['visits'])==(baseline['action'],baseline['visits'])
        if mode=='state':assert counts[-1]==stats['policy_calls']
    assert 0==counts[0]<counts[1]<counts[2]
