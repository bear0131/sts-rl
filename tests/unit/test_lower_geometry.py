"""Lower-size experiments preserve the teacher and conditional-value semantics."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from env.core.finite_curves import UNSCALED_OBJECTIVE
from rl.policies.entity_combat import EntityCombatNetwork
from rl.policies.outer_return import LowerGeometry, OuterCombatNetwork

pytestmark = pytest.mark.native


@pytest.mark.parametrize('values', [dict(width=15,layers=1,heads=4),
    dict(width=16,layers=0,heads=2),dict(width=True,layers=1,heads=1)])
def test_invalid_geometry_fails_closed(values):
    with pytest.raises(ValueError, match='geometry'):
        LowerGeometry(**values)


@pytest.mark.parametrize('objective,profile_name', [
    (UNSCALED_OBJECTIVE,'combat14_small_conditional_876_2x5090.json'),
    ('combat-outcome-means-v1','combat15_small_manual_876_4x5090.json')])
def test_production_small_lower_is_at_most_one_quarter(frozen_upper_source, outer_return_core, objective, profile_name):
    source, source_space, _, _ = frozen_upper_source
    _, _, env = outer_return_core
    # Use production width/depth/head count, retaining real observation vocabularies.
    config = replace(source.combat.config, width=256, layers=4, heads=8)
    reference_source = EntityCombatNetwork(source_space, config,
        card_embedding=deepcopy(source.combat.card))
    before = {k:v.clone() for k,v in source.state_dict().items() if isinstance(v,torch.Tensor)}
    baseline = OuterCombatNetwork(reference_source, source_space, env.observation_space,
                                  objective=objective)
    profile = json.loads((Path(__file__).resolve().parents[2]/
        'configs/train'/profile_name).read_text())
    small = OuterCombatNetwork(reference_source, source_space, env.observation_space,
        objective=objective, lower_geometry=profile['lower_geometry'])
    count = lambda net: sum(p.numel() for p in net.parameters())
    assert count(small) <= count(baseline) * .25
    assert all(p.requires_grad for p in small.parameters())
    assert small.config.pile_queries == baseline.config.pile_queries
    assert small.card.embedding_dim == baseline.card.embedding_dim
    if objective == UNSCALED_OBJECTIVE:
        assert [type(m) for m in small.conditional.value] == [nn.Linear,nn.GELU,nn.Linear]
        with torch.no_grad():
            small.conditional.value[-1].weight.zero_()
            small.conditional.value[-1].bias.fill_(1.3)
            value, outputs = small.conditional(torch.zeros(2,small.config.width))
        assert torch.all(outputs['value_given_win'] > 1)
        torch.testing.assert_close(value,outputs['win_logits'].sigmoid()*outputs['value_given_win'],rtol=0,atol=0)
    else:
        assert hasattr(small,'means') and not hasattr(small,'conditional')
        assert small.config.mean_loss_coef == small.config.auxiliary_loss_coef == 0
        assert not hasattr(small,'distribution') and not hasattr(small,'transition')
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError,match='fresh'):
        OuterCombatNetwork(reference_source,source_space,env.observation_space,
            objective=objective, initialization='876',lower_geometry=profile['lower_geometry'])


def test_small_profile_keeps_combat13_training_and_target():
    root=Path(__file__).resolve().parents[2]/'configs/train'
    old=json.loads((root/'combat13_unscaled_conditional_876_2x5090.json').read_text())
    new=json.loads((root/'combat14_small_conditional_876_2x5090.json').read_text())
    for key in ('source','source_checkpoint_contract','simulator','training','search'):
        assert old[key]==new[key]
    assert {k for k in old['contract'] if old['contract'][k]!=new['contract'][k]} == {
        'experiment_number','comparison'}
    assert new['full_training_budget']['rounds'] is None
