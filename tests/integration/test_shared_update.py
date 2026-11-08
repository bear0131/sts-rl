"""Training keeps valid slots, gradients and Adam while dropping only PAD."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.utils import obs_as_tensor

from env.core import native_actions as C
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.shared_wholerun import SharedWholeRunPolicy

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('layout', ['empty', 'holes', 'last'])
def test_training_relic_padding_preserves_outputs_gradients_and_adam(
        combat_factory, device, layout):
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    torch.manual_seed(20261109)
    torch.set_float32_matmul_precision('highest')
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED', 'ANGER'],
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True)
    model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
        n_steps=4, batch_size=4, n_epochs=1, device=device, gamma=1.,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4))
    core = model.policy.core
    core.train()
    obs = obs_as_tensor({k: np.repeat(v[None], 4, axis=0) for k, v in env._obs().items()},
                        model.device)
    obs['relics'].zero_()
    obs['relic_counters'].fill_(.7)  # Nonzero PAD metadata must remain masked.
    if layout == 'holes':
        obs['relics'][0, 3] = 1
        obs['relics'][1, 8] = 2
    if layout == 'last':
        obs['relics'][0, -1] = 1
    initial = deepcopy(core.state_dict())
    results = []
    widths = []
    hook = core.combat.transformer.register_forward_pre_hook(
        lambda module, args: widths.append(args[0].shape[1]))
    try:
        for reference in (True, False):
            core.load_state_dict(initial)
            optimizer = model.policy.optimizer_class(core.parameters(), lr=1e-4,
                                                       **model.policy.optimizer_kwargs)
            optimizer.zero_grad()
            kwargs = {'active_relic_width': obs['relics'].shape[1]} if reference else {}
            logits, values = core.combat_actor_critic(obs, **kwargs)
            # Probe every finite action score, including value and legal/choice
            # heads. Identical coefficients keep the objective independent of
            # the candidate's mask/shape implementation.
            finite = logits.abs() < 1e8
            coefficients = torch.linspace(-.3, .7, logits.shape[1], device=device)
            loss = (logits.masked_fill(~finite, 0) * coefficients).mean()
            loss = loss + values.square().mean()
            loss.backward()
            gradients = {k: v.grad.detach().clone() for k, v in core.named_parameters()
                         if v.grad is not None}
            torch.nn.utils.clip_grad_norm_(core.parameters(), model.max_grad_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            results.append((logits.detach(), values.detach(), gradients,
                            deepcopy(core.state_dict()), deepcopy(optimizer.state_dict())))
    finally:
        hook.remove()
    a, b = results
    for x, y in zip(a[:2], b[:2], strict=True):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
    assert a[2].keys() == b[2].keys()
    for key in a[2]:
        torch.testing.assert_close(a[2][key], b[2][key], rtol=3e-5, atol=3e-6)
    for key in a[3]:
        torch.testing.assert_close(a[3][key], b[3][key], rtol=2e-5, atol=2e-6)
    assert a[4]['param_groups'] == b[4]['param_groups']
    for key in a[4]['state']:
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[4]['state'][key][field], b[4]['state'][key][field],
                                       rtol=3e-5, atol=3e-6)
    assert (widths[1] < widths[0]) == (layout != 'last')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_map_keeps_phase_rows_outputs_and_gradients(device):
    from env.core.run_env import SlayTheSpireEnv
    from env.core import vocab
    from rl.policies.entity_joint import EntityJointCore
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    env = SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
                         event_option_observation=True, neow_observation=True,
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    try:
        observation, _ = env.reset(seed=20261109)
        frames = {}
        combat = vocab.PHASE_VOCAB_INDEX['combat']
        for _ in range(128):
            frames.setdefault(int(observation['phase'][0]), deepcopy(observation))
            if combat in frames and len(frames) >= 3:
                break
            action = int(np.flatnonzero(env.action_masks())[0])
            observation, _, ended, truncated, _ = env.step(action)
            if ended or truncated:
                observation, _ = env.reset()
        assert combat in frames and len(frames) >= 3
        rows = [frame for frame in frames.values()]
        rows = [rows[i] for i in (2, 0, 1, 2, 1, 0, 2)]
        obs = obs_as_tensor({key: np.stack([r[key] for r in rows]) for key in rows[0]},
                            torch.device(device))
        model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
            n_steps=4, batch_size=4, n_epochs=1, device=device, gamma=1.,
            reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4))
        core = model.policy.core
        core.train()
        results, calls = [], []
        hook = core.overworld.encoder.map_dag.register_forward_pre_hook(
            lambda module, args: calls.append(len(args[0])))
        try:
            for reference in (True, False):
                core.zero_grad(set_to_none=True)
                begin = len(calls)
                logits, values = (EntityJointCore.actor_critic(core, obs) if reference
                                  else core.actor_critic(obs))
                loss = logits.masked_fill(logits.abs() >= 1e8, 0).square().mean()
                loss = loss + values.square().mean()
                loss.backward()
                results.append((logits.detach(), values.detach(),
                    {key: p.grad.detach().clone() for key, p in core.named_parameters()
                     if p.grad is not None}, calls[begin:]))
        finally:
            hook.remove()
        a, b = results
        assert len(a[3]) >= 2 and len(b[3]) == 1
        assert sum(a[3]) == b[3][0]
        for x, y in zip(a[:2], b[:2], strict=True):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
        assert a[2].keys() == b[2].keys()
        for key in a[2]:
            torch.testing.assert_close(a[2][key], b[2][key], rtol=3e-5, atol=3e-6)
    finally:
        env.close()
