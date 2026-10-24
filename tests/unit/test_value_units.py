"""External unit conversions preserve the raw objective and its gradients."""
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from rl.algorithms.combat_distribution import mean_outcome_loss

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_raw_unit_loss_and_gradients_include_death_mask_and_bce(dtype):
    raw = torch.tensor([.3, 1.1, .8, .4], dtype=dtype, requires_grad=True)
    logits = torch.tensor([.1, -.2, .7, -.8], dtype=dtype, requires_grad=True)
    targets = torch.zeros(4, 9); targets[:, 0] = torch.tensor([1., 1., 0., 1.])
    truth = torch.tensor([.6, 1.25, 0., .5])
    actual, metrics = mean_outcome_loss(dict(win_logits=logits, hp_mean=raw / 2),
                                        targets, truth / 2, value_unit_scale=2.)
    expected = F.binary_cross_entropy_with_logits(logits.float(), targets[:,0], reduction='none')
    expected = expected + (raw.float() - truth).square() * targets[:,0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, metrics['win_bce'] + metrics['hp_mse'], rtol=0, atol=0)
    a = torch.autograd.grad(actual.mean(), (raw, logits), retain_graph=True)
    b = torch.autograd.grad(expected.mean(), (raw, logits))
    for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert a[0][2] == 0 and (a[1] != 0).all()

def test_new_profile_changes_only_external_units():
    root=Path(__file__).resolve().parents[2]/'configs/train'
    a=json.loads((root/'combat_matched_curves_876_4x5090.json').read_text())
    b=json.loads((root/'combat_value_units_876_4x5090.json').read_text())
    for key in ('source','source_checkpoint_contract','source_components','simulator','training'):
        assert a[key] == b[key]
    assert b['contract']['external_value_unit_scale'] == 2.
    search=dict(b['search']); assert search.pop('gumbel_value_scale') == 2.
    old=dict(a['search']); assert old.pop('gumbel_value_scale') == 1.
    assert old == search
    assert a['contract']['curve_bank_identity'] == b['contract']['curve_bank_identity']
