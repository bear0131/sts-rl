"""Frozen actor/critic invariants under actual entity networks and native inputs."""
from copy import deepcopy
import io

import numpy as np
import pytest
import torch
from torch import nn

from env.core import action_space as A, encode, vocab as V
from rl.policies.frozen_upper import FrozenUpperTeacher

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(120)]


def test_output_equivalence_parent_train_amp_and_lower_updates(frozen_upper_source):
    core, space, env, obs = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    parent = nn.ModuleDict({'teacher': teacher, 'lower': core.combat})
    teacher_digest = teacher.assert_integrity()
    assert core.upper_tokenizer.card is core.combat.card
    assert teacher.upper_tokenizer.card.vectors.data_ptr() != core.combat.card.vectors.data_ptr()
    with torch.no_grad():
        reference = core.actor_critic(obs)
    parent.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = teacher(obs)
    for got, expected in zip(actual, reference):
        assert got.dtype == torch.float32 and not got.requires_grad
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    optimizer = torch.optim.Adam(core.combat.parameters(), lr=5e-5)
    for _ in range(2):
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in core.combat.parameters()
            if parameter.requires_grad).backward()
        optimizer.step()
    # Buffer isolation is essential even though the original card table is frozen.
    core.combat.card.vectors.add_(1)
    assert teacher.assert_integrity() == teacher_digest
    actions, values = teacher.decide(obs, env.action_masks()[None])
    expected_action = reference[0].masked_fill(~torch.as_tensor(env.action_masks()[None]), -torch.inf).argmax(-1)
    assert torch.equal(actions, expected_action)
    torch.testing.assert_close(values, reference[1], rtol=0, atol=0)
    assert all(not module.training for module in teacher.modules())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_save_restore_and_reject_tampered_teacher(frozen_upper_source):
    core, space, _, obs = frozen_upper_source
    parent = nn.ModuleDict({'teacher': FrozenUpperTeacher(core, space, source_sha256='1' * 64)})
    before = parent['teacher'].assert_integrity()
    buffer = io.BytesIO()
    torch.save(parent.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = nn.ModuleDict({'teacher': FrozenUpperTeacher(core, space, source_sha256='1' * 64)})
    restored.train()
    restored.load_state_dict(state)
    assert restored['teacher'].assert_integrity() == before
    for actual, expected in zip(restored['teacher'](obs), parent['teacher'](obs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='copy, not alias'):
        restored.load_state_dict(deepcopy(state), assign=True)
    changed = deepcopy(state)
    changed['teacher.upper_tokenizer.card.vectors'].flatten()[0] += 1
    with pytest.raises(ValueError, match='tensors or identity changed'):
        restored.load_state_dict(changed)
    assert restored['teacher'].assert_integrity() == before
    changed = deepcopy(state)
    changed['teacher._extra_state']['source_sha256'] = '2' * 64
    with pytest.raises(ValueError, match='tensors or identity changed'):
        restored.load_state_dict(changed)


def test_freeze_guards_and_input_scope(frozen_upper_source):
    core, space, _, obs = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    before = teacher.assert_integrity()
    with pytest.raises(ValueError, match='FP32'):
        teacher.bfloat16()
    with pytest.raises(ValueError, match='enable gradients'):
        teacher.requires_grad_(True)
    assert teacher.assert_integrity() == before
    wrong_shape = dict(obs, act=obs['act'].repeat(1, 2))
    with pytest.raises(ValueError, match='shape changed'):
        teacher(wrong_shape)
    combat = dict(obs, phase=torch.full_like(obs['phase'], V.PHASE_VOCAB_INDEX['combat']))
    with pytest.raises(ValueError, match='upper decision'):
        teacher(combat)
    no_key_inputs = dict(obs)
    del no_key_inputs['act']
    with pytest.raises(ValueError, match='fields missing'):
        teacher(no_key_inputs)
    # A generic parent requires_grad_ walks parameters directly; fail before use.
    nn.ModuleDict({'teacher': teacher}).requires_grad_(True)
    with pytest.raises(RuntimeError, match='made trainable'):
        teacher(obs)
    teacher.train()
    teacher.upper_tokenizer.card.vectors.add_(1)
    with pytest.raises(RuntimeError, match='parameter or buffer changed'):
        teacher.assert_integrity()


def test_cpu_projection_and_explicit_widths_match_teacher(frozen_upper_source):
    from rl.search.rollout_inference import RolloutInference
    from types import SimpleNamespace
    core, space, _, obs = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    raw = {key: value.numpy() for key, value in obs.items()}
    reference = teacher(obs)
    runtime = RolloutInference(SimpleNamespace(policy=SimpleNamespace(
        core=SimpleNamespace(teacher=teacher)), device='cpu'))
    actual = runtime.forward_upper(raw)
    for expected, got in zip(reference, actual):
        np.testing.assert_allclose(got.reshape(expected.shape), expected.numpy(), rtol=0, atol=0)
    again = runtime.forward_upper(raw)
    np.testing.assert_array_equal(actual[0], again[0])
    assert runtime.upper.stats['hits'] == len(raw['phase'])
    assert runtime.upper.stats['misses'] == len(raw['phase'])
    runtime.clear()
    assert not runtime.upper.entries
    teacher.assert_integrity()


def test_checkpoint_key_mask_is_phase_specific(frozen_upper_source):
    core, space, env, obs = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    # Reach a genuine upper reward state through the simulator.
    for _ in range(300):
        if env.thick.bc is not None:
            env.thick.bc.set_player_strength(999)
            env._invalidate_backend_cache()
        if env.thick.phase() == 'combat_reward':
            break
        legal = np.flatnonzero(env.action_masks())
        env.step(int(legal[0]))
    assert env.thick.phase() == 'combat_reward'
    obs = {k: torch.as_tensor(v[None]) for k, v in env._obs().items()}
    mask = env.action_masks()[None].copy()
    mask[0, A._RKEY] = True
    original_mask = mask.copy()
    actions, _ = teacher.decide(obs, mask)
    assert actions.item() != A._RKEY
    np.testing.assert_array_equal(mask, original_mask)
    invalid = dict(obs, act=torch.ones_like(obs['act']) * (4 / encode.ACT_SCALE))
    with pytest.raises(ValueError, match='act=4'):
        teacher.decide(invalid, mask)
