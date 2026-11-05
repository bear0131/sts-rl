import numpy as np
import torch

from rl.policies.public_playout import PublicPlayoutPolicy, public_sketch
from rl.policies.small_playout import SmallPlayoutPolicy


def test_frozen_migration_and_new_context_can_change_action_ranking():
    torch.manual_seed(84)
    old=SmallPlayoutPolicy(20)
    new=PublicPlayoutPolicy.from_mature(old)
    s=torch.randn(2,32)
    a=torch.randn(2,99,24)
    k=torch.randint(20,(2,99,10))
    mask=torch.ones(2,99,dtype=torch.bool)
    mask[0,3:]=False
    slots=mask.flatten().nonzero().flatten()
    t=dict(ids=np.array([1,6,8]),zones=np.array([1,2,3]),monsters=np.array([-1,-1,3]),numbers=np.zeros((3,16)))
    sketch=torch.from_numpy(public_sketch(t,[1,2]))
    pred=new.forward_packed(s,a[mask],k[mask],slots,sketch,mask)
    torch.testing.assert_close(pred,old.forward_packed(s,a[mask],k[mask],slots,mask),rtol=0,atol=0)
    torch.testing.assert_close(pred,new(s,a,k,sketch,mask))
    old_diff=pred[0,0]-pred[0,1]
    opt=torch.optim.SGD(new.context.parameters(),lr=.1)
    (-old_diff).backward();opt.step()
    logits=new(s,a,k,sketch,mask)
    assert logits[0,0]-logits[0,1]>old_diff
    assert np.allclose(public_sketch({k:v[[0,2,1]] for k,v in t.items()},[1,2]),sketch.numpy())
    t['numbers'][1,6]=.25
    assert not np.allclose(public_sketch(t,[1,2]),sketch.numpy())
