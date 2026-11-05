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
