"""Packing changes physical work, never legal slots or Adam sample membership."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from env.core import native_actions as C, vocab
from rl.algorithms import packed_ppo
from rl.algorithms.ppo import prepare_advantage_normalization
from rl.algorithms.updates import ppo_forward, optimizer_step
from rl.policies.entity_combat import EntityCombatConfig
from rl.policies.shared_wholerun import LearnedCardTable, ScalarCombatNetwork


def test_compacted_combat_preserves_pad_queries_slots_and_gradients(combat_factory):
    torch.manual_seed(20261109)
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH'],
        combat_action_contract='autoregressive-v2', card_language=True)
    obs = {k: torch.as_tensor(np.stack([v] * 4)) for k, v in env._obs().items()}
    obs['combat_legal'].zero_()
    obs['combat_legal'][:, 0] = 1
    # Legal source/target queries may have PAD identity, including hole slots.
    obs['hand_ids'][0, 7] = 0
    obs['monster_id'][0, 4] = 0
    obs['combat_legal'][0, 1 + 7 * 6 + 4] = 1
    obs['potions'][1, 3] = 0
    obs['combat_legal'][1, 1 + 13 * 6] = 1
    obs['hand_ids'][2, 8] = 0
    obs['select_hand_mask'][2, 8] = 1
    obs['combat_legal'][2, C.AUTOREGRESSIVE_STOP] = 1
    obs['hand_ids'][3, 6] = 0
    obs['combat_legal'][3, C.JOINT_MULTI - C.JOINT_END + 6] = 1
    obs['relics'][:, :4] = torch.tensor([0, 1, 0, 2])
    obs['relic_counters'][:, :4] = torch.tensor([0., 3., 0., 7.])
    cards = LearnedCardTable(int(env.observation_space['hand_ids'].high.max()) + 1)
    full = ScalarCombatNetwork(env.observation_space,
        EntityCombatConfig(width=32, layers=2, heads=4), cards)
    # Warm-start zeros would hide broken target/prefix query handling.
    with torch.no_grad():
        for p in full.complete_head.parameters():
            p.uniform_(-.2, .2)
    compact = deepcopy(full)
    compact.compact_training_tokens = True
    legal = obs['combat_legal'].bool()
    outputs = []
    for model in (full, compact):
        logits, value, _ = model(obs)
        outputs.append((logits[legal], value))
        (logits[legal].square().mean() + value.square().mean()).backward()
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    for (n, p), (m, q) in zip(full.named_parameters(), compact.named_parameters()):
        assert n == m and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=5e-5, atol=3e-6, msg=n)
    # Evaluation, including old checkpoints, retains the exact original path.
    full.eval(); compact.eval()
    with torch.no_grad():
        a, b = full(obs), compact(obs)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert set(full.state_dict()) == set(compact.state_dict())


@pytest.mark.parametrize('normalization', ['group256', 'rollout'])
@pytest.mark.parametrize('scoped_oom', [False, True])
def test_packed_batches_keep_original_normalization_and_adam(monkeypatch, normalization, scoped_oom):
    from rl.algorithms.ppo_metrics import update_moments, finalize_metrics
    torch.manual_seed(20261109)

    class Policy(torch.nn.Module):
        ppo_advantage_group_size = 4

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 3, dtype=torch.float64) * .1)
            self.core = SimpleNamespace(combat=SimpleNamespace(
                training_token_counts=lambda obs: obs['width']),
                context=SimpleNamespace(key_tensor=lambda obs: obs['row'][:, None]))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            self.fail_upper = False
            self.attempts = []

        def evaluate_actions(self, obs, actions, action_masks=None):
            is_combat = bool((obs['phase'] == vocab.PHASE_VOCAB_INDEX['combat']).all())
            self.attempts.append((is_combat, len(actions)))
            if self.fail_upper and not is_combat and len(actions) == 4:
                self.fail_upper = False
                raise torch.OutOfMemoryError('injected upper-only allocation failure')
            x = self.weight[obs['row']]
            logp = x.log_softmax(-1)
            return x[:, :1], logp.gather(1, actions[:, None]).flatten(), -(logp.exp() * logp).sum(-1)

    combat, upper = (vocab.PHASE_VOCAB_INDEX[k] for k in ('combat', 'map'))
    advantages = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    advantages = (advantages * torch.tensor([.1, 10., 1., 100.])[:, None]).flatten()
    batch = SimpleNamespace(observations=dict(row=torch.arange(16),
        phase=torch.tensor([combat, upper] * 8), width=torch.tensor([5, 2, 4, 1] * 4),
        select_card_ids=torch.zeros(16, 96)), actions=torch.arange(16) % 3,
        action_masks=torch.ones(16, 3, dtype=torch.bool), advantages=advantages,
        returns=advantages * .001, old_values=torch.zeros(16, dtype=torch.float64),
        old_log_prob=torch.full((16,), -1.1, dtype=torch.float64))
    model = SimpleNamespace(policy=Policy(), batch_size=4, gradient_accumulation_steps=2,
        device=torch.device('cpu'), action_space=spaces.Discrete(3), max_grad_norm=1.,
        normalize_advantage=True, ppo_advantage_normalization=normalization,
        hierarchical_combat_hp=False, ent_coef=.02, vf_coef=.5, autocast_context=nullcontext,
        ppo_update_loss_scale=1., num_timesteps=16, logger=SimpleNamespace(record=lambda *a: None))
    model.rollout_buffer = SimpleNamespace(buffer_size=16, n_envs=1, full=True,
        advantages=advantages.numpy(), normalization_generation=1)
    model.rollout_buffer.get = lambda size: (
        packed_ppo._select(batch, slice(i, i + size)) for i in range(0, 16, size))
    prepare_advantage_normalization(model)
    expected = deepcopy(model)
    prepare_advantage_normalization(expected)
    # Independent original loop: two four-row losses then one Adam boundary.
    for start in (0, 8):
        expected.policy.optimizer.zero_grad()
        for pos in (start, start + 4):
            _, loss = ppo_forward(expected, packed_ppo._select(batch, slice(pos, pos + 4)), .2, None)
            (loss / 2).backward()
        optimizer_step(expected)
    monkeypatch.setattr(packed_ppo, 'configure_packed_updates',
        lambda m: packed_ppo.PackedPPOConfig(token_budget=1000 if scoped_oom else 7,
            max_rows=4 if scoped_oom else 3, context_grouping=scoped_oom))
    monkeypatch.setattr(packed_ppo.SharedOverworldEncoder, 'merged_token_counts',
        staticmethod(lambda obs: obs['width']))
    if scoped_oom:
        # A legacy saved global cap must not throttle the new grouped path.
        model._ppo_microbatch_cap = 1
        model.policy.fail_upper = True
    result = packed_ppo.packed_ppo_epoch(model, .2, None)
    assert result['optimizer_steps'] == 2 and result['overflow_retries'] == 0
    assert model._ppo_update_metric_count == 16
    assert model._ppo_microbatch_stats['max_rows'] <= (4 if scoped_oom else 3)
    if scoped_oom:
        assert model._ppo_microbatch_stats['oom_retries'] == 1
        assert model._ppo_microbatch_caps == {'upper': 2}
        assert [n for combat, n in model.policy.attempts if combat] == [4, 4]
    torch.testing.assert_close(model.policy.weight, expected.policy.weight, rtol=1e-12, atol=1e-12)
    a, b = model.policy.optimizer.state[model.policy.weight], expected.policy.optimizer.state[expected.policy.weight]
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(a[key], b[key], rtol=1e-12, atol=1e-12)
    assert all(np.isfinite(x) for x in finalize_metrics(update_moments(model)).values())
