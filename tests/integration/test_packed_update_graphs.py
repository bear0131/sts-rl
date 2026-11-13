"""Training graph ownership across allocator release, recapture and live weights."""
import gc

import pytest
import torch

from rl.policies.packed_update_graphs import _CapturedHead

pytestmark = [pytest.mark.training, pytest.mark.cuda]


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(1024, 1024)
        self.shared = torch.nn.Linear(1024, 1024)
        self.value = torch.nn.Linear(1024, 1)

    def forward(self, obs):
        hidden = torch.relu(self.input(obs['features']) + self.shared(obs['_run_context']))
        return self.shared(hidden), self.value(hidden), {}


@pytest.fixture
def graph_head():
    assert torch.cuda.is_available(), 'explicit CUDA training-graph qualification required'
    torch.manual_seed(20261112)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield _Head().cuda()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def _release_and_reuse_scratch():
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    # Exercise both unmapped expandable segments and reuse by unrelated tensors.
    return [torch.full((1024 * 1024,), float(i + 7), device='cuda') for i in range(16)]


def _compare(head, entry, rows):
    features = torch.randn(rows, 1024, device='cuda')
    context = torch.randn_like(features, requires_grad=True)
    actual = entry.replay(dict(features=features), context)
    expected = head(dict(features=features, _run_context=context))[:2]
    upstream = tuple(torch.randn_like(value) for value in actual)
    inputs = (context, *head.parameters())
    gradients = torch.autograd.grad(actual, inputs, upstream)
    reference = torch.autograd.grad(expected, inputs, upstream)
    torch.cuda.synchronize()
    for left, right in zip((*actual, *gradients), (*expected, *reference), strict=True):
        torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)


def test_training_graph_survives_workspace_release_and_live_weight_changes(graph_head):
    head = graph_head
    features = torch.randn(896, 1024, device='cuda')
    context = torch.randn_like(features, requires_grad=True)
    entry = _CapturedHead(head, dict(features=features), context, 896, 1)
    for iteration in range(20):
        junk = _release_and_reuse_scratch()
        with torch.no_grad():
            head.input.weight.add_(.0001)
        _compare(head, entry, (896, 640, 17)[iteration % 3])
        del junk


def test_training_graph_survives_sibling_capture_and_eviction(graph_head):
    head = graph_head
    features = torch.randn(896, 1024, device='cuda')
    context = torch.randn_like(features, requires_grad=True)
    survivor = _CapturedHead(head, dict(features=features), context, 896, 1)
    for rows in (384, 256, 192):
        sibling = _CapturedHead(head, dict(features=features[:rows]), context[:rows], rows, 1)
        _compare(head, sibling, rows)
        _compare(head, survivor, 896)
        del sibling
        junk = _release_and_reuse_scratch()
        _compare(head, survivor, 640)
        del junk
