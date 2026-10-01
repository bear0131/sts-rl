import numpy as np
import pytest
import torch
from gymnasium import spaces

from env.core import native_actions as C
from rl.algorithms.action_groups import quotient_distribution
from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer


def test_loss_is_group_probability_without_uniform_alias_penalty():
    logits = torch.tensor([[.4, .1, .5]], dtype=torch.float64).log().float().requires_grad_()
    logp, target, mask = quotient_distribution(logits, torch.tensor([[0., 1., 0.]]),
        torch.ones(1, 3, dtype=torch.bool), torch.tensor([[0, 0, 2]]))
    loss = -(logp*target).sum()
    assert loss.item() == pytest.approx(np.log(2))
    assert mask.tolist() == [[True, False, True]]
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.4, -.1, .5]]))
    # Moving teacher mass between aliases must not change the objective.
    other = quotient_distribution(logits.detach(), torch.tensor([[.75, .25, 0.]]),
        torch.ones(1, 3, dtype=torch.bool), torch.tensor([[0, 0, 2]]))
    torch.testing.assert_close(-(other[0]*other[1]).sum(), loss.detach())


def test_extreme_logits_and_illegal_actions_have_finite_correct_gradients():
    logits = torch.tensor([[-1000., -1001., 0., 1e6]], requires_grad=True)
    masks = torch.tensor([[True, True, True, False]])
    logp, targets, _ = quotient_distribution(logits, torch.tensor([[1., 0., 0., 0.]]),
        masks, torch.tensor([[0, 0, 2, 3]]))
    loss = -(logp*targets).sum()
    loss.backward()
    assert loss.item() == pytest.approx(1000-np.log1p(np.exp(-1)))
    torch.testing.assert_close(logits.grad, torch.tensor([[-.7310586, -.2689414, 1., 0.]]))


def test_group_metadata_survives_replay_wrap_gather_and_restore():
    space = spaces.Dict(combat_legal=spaces.Box(0,1,(C.AUTOREGRESSIVE_SIZE,),bool),
                        player_scalars=spaces.Box(-np.inf,np.inf,(1,),np.float32))
    def buffer():
        return NOSLISAZReplayBuffer(capacity=2,observation_space=space,
            observation_keys=space.spaces,action_dim=C.AUTOREGRESSIVE_SIZE)
    def sample(i):
        masks = np.zeros(C.AUTOREGRESSIVE_SIZE,bool); masks[[0,1,7]]=True
        groups=np.arange(C.AUTOREGRESSIVE_SIZE,dtype=np.int16); groups[7]=1
        targets=np.zeros(C.AUTOREGRESSIVE_SIZE,np.float32); targets[7]=1
        return dict(observation=dict(combat_legal=masks,player_scalars=np.array([i],np.float32)),
            action_mask=masks,action_groups=groups,policy_target=targets,value_target=.3)
    replay=buffer(); replay.add_samples([sample(0),sample(1),sample(2)])
    state=replay.state_dict(); restored=buffer(); restored.load_state_dict(state)
    batch=restored.get_batch(np.array([1,0,1]),workers=2)
    np.testing.assert_array_equal(batch['action_groups'],np.stack([sample(0)['action_groups']]*3))
    np.testing.assert_array_equal(batch['observations']['player_scalars'][:,0],[2,1,2])
    state.pop('action_group_contract')
    with pytest.raises(ValueError,match='group contract'):
        buffer().load_state_dict(state)
    bad=sample(0); bad['action_groups'][7]=2
    with pytest.raises(ValueError,match='representative must be legal'):
        replay.add_samples([bad])
