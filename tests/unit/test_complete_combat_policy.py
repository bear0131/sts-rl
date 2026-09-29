import numpy as np
import pytest
import torch

from env.core import action_space as A, native_actions as C
from rl.policies.complete_combat import complete_logits


def test_complete_codec_is_injective_and_round_trips_all_supported_actions():
    seen = set()
    choices = [91, 12, 0]
    native = [(4 << 29, False)]
    for kind, count in ((0, 10), (1, 5)):
        native.extend((kind << 29 | source | target << 16, True)
                      for source in range(count) for target in range(5))
    native.extend((2 << 29 | index, False) for index in choices)
    native.extend((3 << 29 | subset, False) for subset in range(1024))
    for bits, targeted in native:
        aid = C.joint_action_id(bits, targeted=targeted, candidate_indices=choices)
        assert aid not in seen
        seen.add(aid)
        assert C.joint_native_bits(aid, candidate_indices=choices) == bits
    for kind, count in ((0, 10), (1, 5)):
        for source in range(count):
            bits = kind << 29 | source
            aid = C.joint_action_id(bits, targeted=False)
            assert aid not in seen  # NONE and monster zero are different
            assert C.joint_native_bits(aid) == bits
    assert len(C.joint_mask(sorted(seen))) == 1392


def test_joint_cross_entropy_and_gradient_equal_conditional_factorization():
    torch.manual_seed(42)
    source = torch.randn(1, A.N_ACTIONS, requires_grad=True)
    target = torch.randn(1, 15, 6, requires_grad=True)
    subsets = torch.randn(1, 1024, requires_grad=True)
    legal = torch.zeros(1, C.JOINT_WIDTH, dtype=torch.bool)
    legal[:, 1:3] = True  # card zero -> two targets
    legal[:, 7:9] = True  # card one -> two targets
    scores = complete_logits(source, target, subsets, legal)[:, C.JOINT_END:]
    logp = scores.masked_fill(~legal, -1e9).log_softmax(-1)
    truth = torch.tensor([0.1, 0.2, 0.4, 0.3])
    joint_ce = -(logp[legal] * truth).sum()
    source_ce = -(source[:, A._PLAY:A._PLAY+2].log_softmax(-1) * torch.tensor([[.3, .7]])).sum()
    conditional_ce = -(target[:, :2, :2].log_softmax(-1) * truth.reshape(1,2,2)).sum()
    expected = source_ce + conditional_ce
    torch.testing.assert_close(joint_ce, expected, atol=2e-7, rtol=1e-6)
    actual_grad = torch.autograd.grad(joint_ce, (source, target), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (source, target))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)


def test_multisets_and_forced_targets_have_correct_probabilities():
    source = torch.zeros(2, A.N_ACTIONS)
    target = torch.randn(2, 15, 6, requires_grad=True)
    subset = torch.zeros(2, 1024)
    subset[1, 7] = 2
    legal = torch.zeros(2, C.JOINT_WIDTH, dtype=torch.bool)
    legal[0, 5] = True  # source zero, actual monster slot four
    legal[1, C.JOINT_MULTI-C.JOINT_END:] = True
    logits = complete_logits(source,target,subset,legal)[:,C.JOINT_END:]
    logp = logits.masked_fill(~legal, -1e9).log_softmax(-1)
    assert logp[0, 5].item() == 0
    assert torch.isfinite(logp).all()
    probs = logp[1].exp()
    assert probs[C.JOINT_MULTI-C.JOINT_END+7] > probs[C.JOINT_MULTI-C.JOINT_END]
    assert torch.count_nonzero(probs) == 1024
    logp[0,5].backward()
    assert torch.count_nonzero(target.grad) == 0
