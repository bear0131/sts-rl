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
@pytest.mark.parametrize('layout', ['holes', 'empty_context', 'dense'])
def test_upper_compaction_preserves_pad_action_queries_and_map_choices(device, layout):
    from env.core.run_env import SlayTheSpireEnv
    from rl.policies.split_policy import OverworldEncoder
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    torch.manual_seed(20261109)
    torch.set_float32_matmul_precision('highest')
    env = SlayTheSpireEnv(card_language=True, global_map_view=True,
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    try:
        model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
            n_steps=4, batch_size=4, n_epochs=1, device=device, gamma=1.,
            reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
        encoder = model.policy.core.overworld.encoder
        encoder.train()
        initial = deepcopy(encoder.state_dict())
        source = torch.randn(3, 23, 32, device=device)
        spans = dict(deck=(1, 5), relic=(5, 8), potion=(8, 11), selected=(11, 14),
                     select=(14, 18), map_full=(18, 23))
        mask = torch.zeros(3, 23, dtype=torch.bool, device=device)
        if layout != 'dense':
            mask[:, [2, 4, 6, 11, 13, 18, 20, 22]] = True
            # A PAD candidate can be an executable skip. Potion queries also
            # reach their head, independently of their attention key mask.
            mask[:, 8:11] = True
            mask[:, 14:18] = True
        if layout == 'empty_context':
            mask[:, 1:8] = True
            mask[:, 11:14] = True
            mask[:, 18:23] = True
        map_features = torch.zeros(3, 5, 12, device=device)
        map_features[:, 4, 3] = 1  # Choice at PAD map slot 22 must survive.
        # Existing heads read CLS, all potion/select candidates and map choices.
        queries = [0, 8, 9, 10, 14, 15, 16, 17, 22]
        results, lengths = [], []
        hook = encoder.transformer.register_forward_pre_hook(
            lambda module, args: lengths.append(args[0].shape[1]))
        try:
            for reference in (True, False):
                encoder.load_state_dict(initial)
                optimizer = model.policy.optimizer_class(encoder.parameters(), lr=1e-4,
                                                           **model.policy.optimizer_kwargs)
                optimizer.zero_grad()
                x = source.detach().clone().requires_grad_()
                output = (OverworldEncoder._transform_tokens(encoder, x, mask, spans,
                          map_features=map_features) if reference else
                          encoder._transform_tokens(x, mask, spans, map_features=map_features))
                selected = output[:, queries]
                coefficients = torch.linspace(-.3, .7, 32, device=device)
                (selected.square().mean() + (selected * coefficients).mean()).backward()
                gradients = {k: p.grad.detach().clone() for k, p in encoder.named_parameters()
                             if p.grad is not None}
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), model.max_grad_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
                results.append((selected.detach(), x.grad, gradients,
                                deepcopy(encoder.state_dict()), deepcopy(optimizer.state_dict())))
        finally:
            hook.remove()
        a, b = results
        for left, right in zip(a[:2], b[:2], strict=True):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
        assert a[2].keys() == b[2].keys()
        for key in a[2]:
            torch.testing.assert_close(a[2][key], b[2][key], rtol=3e-5, atol=3e-6)
        for key in a[3]:
            torch.testing.assert_close(a[3][key], b[3][key], rtol=2e-5, atol=2e-6)
        for key in a[4]['state']:
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(a[4]['state'][key][field], b[4]['state'][key][field],
                                           rtol=3e-5, atol=3e-6)
        assert (lengths[1] < lengths[0]) == (layout != 'dense')
    finally:
        env.close()


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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('phase_subset', ['all', 'one'])
def test_upper_phase_batch_preserves_unused_parameters_and_adam(upper_phase_batch, device, phase_subset):
    from rl.policies.entity_joint import EntityJointCore
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    torch.set_float32_matmul_precision('highest')
    env, frames = upper_phase_batch
    frames = [frames[i] for i in ([4, 0, 6, 2, 7, 1, 3, 5, 2] if phase_subset == 'all' else [4, 4, 4])]
    obs = obs_as_tensor({key: np.stack([r[key] for r in frames]) for key in frames[0]},
                        torch.device(device))
    model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
        n_steps=4, batch_size=4, n_epochs=1, device=device, gamma=1.,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    core = model.policy.core
    core.train()
    initial = deepcopy(core.state_dict())
    results = []
    for reference in (True, False):
        core.load_state_dict(initial)
        optimizer = torch.optim.Adam(core.parameters(), lr=1e-4, eps=1e-5)
        # Existing momentum makes a spurious zero gradient observably different
        # from None, even when that head has no samples in this update.
        for parameter in core.parameters():
            parameter.grad = torch.full_like(parameter, .003)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logits, values = (EntityJointCore.actor_critic(core, obs) if reference
                          else core.actor_critic(obs))
        coefficients = torch.linspace(-.2, .4, logits.shape[1], device=device)
        loss = (logits.masked_fill(logits.abs() >= 1e8, 0) * coefficients).mean()
        (loss + values.square().mean()).backward()
        gradients = {key: None if p.grad is None else p.grad.detach().clone()
                     for key, p in core.named_parameters()}
        optimizer.step()
        results.append((logits.detach(), values.detach(), gradients,
                        deepcopy(core.state_dict()), deepcopy(optimizer.state_dict())))
    left, right = results
    for x, y in zip(left[:2], right[:2], strict=True):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
    for key, x in left[2].items():
        y = right[2][key]
        assert (x is None) == (y is None), key
        if x is not None:
            torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-6)
    for key in left[3]:
        torch.testing.assert_close(left[3][key], right[3][key], rtol=2e-5, atol=2e-6)
    assert left[4]['state'].keys() == right[4]['state'].keys()
    for key in left[4]['state']:
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(left[4]['state'][key][field], right[4]['state'][key][field],
                                       rtol=3e-5, atol=3e-6)
