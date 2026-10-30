"""Packed online inputs preserve legal rows, labels, losses and gradients."""
import copy

import numpy as np
import pytest
import torch

from rl.policies.small_playout import (
    PreparedPlayoutSamples, SmallPlayoutPolicy, collate_features, search_target,
)


def samples():
    rng = np.random.default_rng(723)
    rows = []
    for n in (2, 3, 7, 129):
        keys = np.full((n, 10), -1, np.int64)
        for i in range(n):
            count = 1 + i % 10
            keys[i, :count] = rng.integers(0, 785, count)
        rows.append(dict(state=rng.normal(size=32).astype(np.float32),
            actions=rng.normal(size=(n, 24)).astype(np.float32), keys=keys,
            action_bits=np.arange(n) + 13, visits=rng.integers(1, 20, n),
            recommended_action=14))
    return rows


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_packed_inputs_preserve_logits_targets_gradients_and_order(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA comparison requires a CUDA device')
    torch.manual_seed(45)
    rows = samples()
    frozen = copy.deepcopy(rows)
    prepared = PreparedPlayoutSamples(rows, .2)
    dense = SmallPlayoutPolicy(785).to(device)
    packed = copy.deepcopy(dense)
    for order in (np.array([3, 0, 2]), np.array([1]), np.array([2, 3, 1, 0])):
        batch = [rows[i] for i in order]
        inputs = collate_features(batch, device)
        compact, target, teacher = prepared.batch(order, device)
        assert compact[1].shape[0] == sum(len(r['visits']) for r in batch)
        assert compact[1].shape[-1] == 24
        expected = np.zeros(inputs[-1].shape, np.float32)
        for i, row in enumerate(batch):
            expected[i, :len(row['visits'])] = search_target(row, .2)
        np.testing.assert_array_equal(target.cpu(), expected)
        np.testing.assert_array_equal(teacher.cpu(), expected.argmax(-1))
        a, b = dense(*inputs), packed.forward_packed(*compact)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        losses = []
        for net, logits in ((dense, a), (packed, b)):
            net.zero_grad(set_to_none=True)
            loss = -(logits.log_softmax(-1).masked_fill(~inputs[-1], 0.) * target).sum()
            loss.backward()
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-5)
        for a, b in zip(dense.parameters(), packed.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    for before, after in zip(frozen, rows):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    targets = [search_target(row, .2) for row in rows]
    assert prepared.target_entropy == pytest.approx(sum(
        -float(np.sum(t * np.log(t))) for t in targets), abs=1e-7)


def test_preparation_rejects_empty_and_invalid_recommendation():
    with pytest.raises(ValueError, match='empty'):
        PreparedPlayoutSamples([], .2)
    bad = samples()[0]
    with pytest.raises(ValueError, match='exactly one legal'):
        PreparedPlayoutSamples([dict(bad, recommended_action=999)], .2)
    with pytest.raises(ValueError, match='live decision'):
        PreparedPlayoutSamples([dict(bad, action_bits=[])], .2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_adamw_preserves_action_constant_feature_cancellation(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA comparison requires a CUDA device')
    # These columns add the same logit offset to every action. Their near-zero
    # gradients are sensitive to reduction changes, especially with Adam's eps.
    torch.manual_seed(872)
    rows = samples()
    for row in rows:
        row['actions'][:, :3] = [1., 2., -.5]
    prepared = PreparedPlayoutSamples(rows, .2)
    dense = SmallPlayoutPolicy(785).to(device)
    packed = copy.deepcopy(dense)
    optimizers = [torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.0001)
                  for net in (dense, packed)]
    rng = np.random.default_rng(35)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        for _ in range(8):
            order = rng.permutation(len(rows))
            for start in (0, 3):
                indices = order[start:start + 3]
                features = collate_features([rows[i] for i in indices], device)
                inputs, target, _ = prepared.batch(indices, device)
                for net, optimizer, logits in (
                    (dense, optimizers[0], dense(*features)),
                    (packed, optimizers[1], packed.forward_packed(*inputs)),
                ):
                    optimizer.zero_grad(set_to_none=True)
                    loss = -(logits.log_softmax(-1).masked_fill(~features[-1], 0.) * target).sum() / len(indices)
                    loss.backward()
                    optimizer.step()
        for a, b in zip(dense.parameters(), packed.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=2e-7, rtol=2e-4)
    finally:
        torch.set_float32_matmul_precision(precision)
