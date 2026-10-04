"""Independent numerical cases for the TRAIN34 objective (raw HP labels stay raw)."""
import pytest
import torch

from env.core.boss_value import is_boss_room, win_hp_utility
from rl.algorithms.combat_distribution import distribution_loss, TARGET_WIDTH
from rl.policies.entity_combat import EntityCombatConfig


@pytest.mark.parametrize('win,hp,weight,expected', [
    (1, .05, .9, .905), (1, 1, .9, 1), (0, .5, .9, 0),
    (.8, .25, .9, .74), (.8, .25, 0, .2), (1, 0, .9, .9)])
def test_explicit_utilities(win, hp, weight, expected):
    assert win_hp_utility(win, hp, weight) == pytest.approx(expected)


def test_low_hp_safe_win_beats_riskier_high_hp_boss_win():
    assert win_hp_utility(.99, .05, .9) > win_hp_utility(.9, .4, .9)
    assert win_hp_utility(.99, .05, 0) < win_hp_utility(.9, .4, 0)


@pytest.mark.parametrize('bad', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_weights(bad):
    with pytest.raises(ValueError, match='boss_win_weight'):
        EntityCombatConfig(boss_win_weight=bad)


def test_boss_identity_uses_room_not_surviving_monsters():
    assert is_boss_room('Room.BOSS') and is_boss_room('MonsterRoomBoss')
    assert not is_boss_room('MONSTER') and not is_boss_room('ELITE')


def test_hp_pinball_and_win_labels_remain_raw_with_utility_mse():
    outputs = dict(win_logits=torch.zeros(3, 1, requires_grad=True),
                   hp_quantiles=torch.tensor([[.1,.3],[.1,.3],[.1,.3]], requires_grad=True),
                   boss_win_weight=torch.tensor([[.9],[.9],[0.]]))
    targets = torch.zeros(3, TARGET_WIDTH); targets[:, 0] = torch.tensor([1., 0., 1.])
    hp = torch.tensor([.1, 0., .1])
    loss, metrics = distribution_loss(outputs, targets, hp, auxiliary_coef=0)
    # Win probability=.5, mean HP=.2: boss prediction=.46, ordinary=.1.
    torch.testing.assert_close(metrics['utility_mse'], torch.tensor([(.46-.91)**2, .46**2, 0.]))
    assert metrics['hp_pinball'][0] == metrics['hp_pinball'][2]
    assert metrics['hp_pinball'][1] == 0
    loss.mean().backward()
    assert all(torch.isfinite(outputs[k].grad).all() for k in ('win_logits','hp_quantiles'))
    assert hp.tolist() == pytest.approx([.1,0,.1])


def test_microbatch_composition_does_not_reweight_boss_samples():
    logits = torch.tensor([[.4],[-.7],[1.1]], requires_grad=True)
    hp = torch.tensor([[.2,.5],[.1,.9],[.4,.8]], requires_grad=True)
    weights = torch.tensor([[.9],[0],[.9]])
    targets = torch.zeros(3,TARGET_WIDTH); targets[:,0]=torch.tensor([1,0,1])
    values = torch.tensor([.3,0,.7])
    full,_=distribution_loss(dict(win_logits=logits,hp_quantiles=hp,boss_win_weight=weights),targets,values,auxiliary_coef=0)
    expected=torch.autograd.grad(full.mean(),(logits,hp))
    pieces=[]
    for sl in (slice(0,1),slice(1,3)):
        loss,_=distribution_loss(dict(win_logits=logits[sl],hp_quantiles=hp[sl],boss_win_weight=weights[sl]),targets[sl],values[sl],auxiliary_coef=0)
        pieces.append(loss.sum()/3)
    actual=torch.autograd.grad(sum(pieces),(logits,hp))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
