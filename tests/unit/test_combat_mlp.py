"""Fixed input, phase separation and slot contracts for the combat MLP."""

import pytest
import torch

from env.core import action_space as A
from env.core import encode, vocab as V
from rl.policies.combat_mlp import CombatMLPActorCritic
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS, SplitStSPolicyCore


@pytest.fixture(params=[False, True])
def mlp_core(request):
    language = request.param
    space = encode.obs_space(global_map_view=True, card_language=language, combat_language=language)
    torch.manual_seed(42)
    core = SplitStSPolicyCore(space, d_model=32, n_layers=2, n_heads=4,
                              card_language=language, combat_language=language, combat_arch="mlp").eval()
    obs = {key: torch.zeros((2, *field.shape), dtype=torch.long if field.dtype.kind in "iu"
                           else torch.float32) for key, field in space.spaces.items()}
    obs["phase"].fill_(V.PHASE_VOCAB_INDEX["combat"])
    return core, obs


def test_small_attention_free_root_and_leaf_agree(mlp_core):
    core, obs = mlp_core
    assert isinstance(core.combat, CombatMLPActorCritic)
    assert sum(p.numel() for p in core.combat.parameters()) < 1_000_000
    assert not any(isinstance(module, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoder))
                   for module in core.combat.modules())
    root = core.actor_critic(obs)
    leaf = core.combat_actor_critic({key: value for key, value in obs.items()
                                    if key in COMBAT_OBSERVATION_KEYS}, active_relic_width=1)
    assert root[0].shape == (2, A.N_ACTIONS) and root[1].shape == (2, 1)
    for expected, actual in zip(root, leaf):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    # Combat losses do not consume the supposedly frozen semantic/upper modules.
    (root[0].square().sum() + root[1].square().sum()).backward()
    assert any(p.grad is not None for p in core.combat.parameters())
    assert all(p.grad is None for p in core.shared.parameters())
    assert all(p.grad is None for p in core.overworld.parameters())


def test_hand_and_monster_slots_are_not_compacted(mlp_core):
    core, obs = mlp_core
    for key, slot in (("hand_ids", 4), ("monster_id", 3)):
        left = {k: v.clone() for k, v in obs.items()}
        right = {k: v.clone() for k, v in obs.items()}
        left[key][:, 0] = 1
        right[key][:, slot] = 1
        a, b = core.actor_critic(left), core.actor_critic(right)
        assert not torch.equal(a[0], b[0])
        assert not torch.equal(a[1], b[1])


def test_overworld_fields_cannot_change_combat_outputs(mlp_core):
    core, obs = mlp_core
    before = core.actor_critic(obs)
    obs["floor"].fill_(0.8)
    obs["event_id"].fill_(1)
    obs["deck_ids"].fill_(2)
    after = core.actor_critic(obs)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_language_pool_is_pad_position_independent(mlp_core):
    core, obs = mlp_core
    if "player_status_language" not in obs:
        return
    left = {k: v.clone() for k, v in obs.items()}
    right = {k: v.clone() for k, v in obs.items()}
    left["player_status_language"][:, :2] = torch.tensor([1, 2])
    right["player_status_language"][:, 7:9] = torch.tensor([1, 2])
    for a, b in zip(core.actor_critic(left), core.actor_critic(right)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_unknown_combat_arch_rejected():
    with pytest.raises(ValueError, match="combat_arch"):
        SplitStSPolicyCore(encode.obs_space(), d_model=32, n_layers=2, n_heads=4, combat_arch="typo")
