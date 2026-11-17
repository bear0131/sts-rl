"""Exact frame-local sharing, independent of the game and model implementation."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.training, pytest.mark.timeout(120)]


@pytest.fixture
def shared_frame():
    assert torch.cuda.is_available(), 'shared policy qualification requires CUDA'

    def make(count=4):
        # Seventy entities exercise the second chunk, including its final channel.
        def ints(data):
            return torch.as_tensor(data, dtype=torch.int32, device='cuda')
        entities, actions, stride = 70, 3, 128
        p = SimpleNamespace(
            meta=ints([[actions, entities, row * actions, 1] for row in range(count)]),
            chunks=ints([[row * stride + chunk * 64 for chunk in range(16)]
                         for row in range(count)]),
            numbers=torch.zeros((count * stride, 64), device='cuda'),
            ids=ints(list(range(stride)) * count),
            zones=ints([1] * (count * stride)),
            monsters=ints([0] * (count * stride)),
            keys=ints([[action, *([-1] * 9)] for _ in range(count)
                       for action in (0, 1, 2)]),
            bits=ints([11, 22, 33] * count),
            allocator=ints([0] * 5),
            scores=torch.arange(count * actions, dtype=torch.float32, device='cuda'))
        w = dict(dispatch=ints([count]), legal=ints([[1] * 198] * count),
                 selected=torch.zeros((count, 10), device='cuda'), remaining=ints([0] * count),
                 # Deliberately force every row into one hash bucket.
                 shared_table=ints([0]), shared_hash=ints([0] * count),
                 shared_rep=ints([0] * count), shared_saved=ints([0] * count),
                 owners=ints([0] * (count * actions)), summary=ints([0] * 10),
                 shared_total=torch.zeros(1, dtype=torch.int64, device='cuda'))
        return p, w

    return make


@pytest.mark.parametrize('field', [
    'feature', 'signed_zero', 'slot', 'present', 'query', 'action_bits',
    'action_order', 'legal', 'selected', 'remaining', 'action_count',
    'entity_count', 'value_only',
])
def test_hash_collision_compares_all_deck_inputs(shared_frame, field):
    from rl.search.full480 import shared_policy

    p, w = shared_frame()
    last = 2 * 128 + 69
    if field == 'feature':
        p.numbers[last, 63] = 1
    elif field == 'signed_zero':
        p.numbers[last, 63] = -0.0
    elif field == 'slot':
        p.ids[last] = 70
    elif field == 'present':
        p.zones[last] = 2
    elif field == 'query':
        p.monsters[last] = 1
    elif field == 'action_bits':
        p.bits[8] = 34
    elif field == 'action_order':
        p.keys[6:9, 0] = torch.tensor([2, 1, 0], device='cuda')
    elif field == 'legal':
        w['legal'][2, 197] = 0
    elif field == 'selected':
        w['selected'][2, 9] = 1
    elif field == 'remaining':
        w['remaining'][2] = 1
    elif field == 'action_count':
        p.meta[2, 0] = 2
    elif field == 'entity_count':
        p.meta[2, 1] = 69
    elif field == 'value_only':
        p.meta[2, 3] = 2

    original_meta, original_scores = p.meta.clone(), p.scores.clone()
    shared_policy.prepare(p, w, 4)
    assert w['shared_rep'].tolist() == [-1, 0, -1, 0]
    w['summary'][9] = 2
    shared_policy.finish(p, w, 4)
    expected = original_scores.clone()
    expected[3:6] = expected[:3]
    expected[9:12] = expected[:3]
    torch.testing.assert_close(p.scores, expected, rtol=0, atol=0)
    torch.testing.assert_close(p.meta, original_meta, rtol=0, atol=0)
    assert w['shared_total'].item() == 2
    assert w['summary'][9].item() == 4


def test_failed_frame_restores_metadata_without_publishing_scores(shared_frame):
    from rl.search.full480 import shared_policy

    p, w = shared_frame()
    meta, scores = p.meta.clone(), p.scores.clone()
    shared_policy.prepare(p, w, 4)
    w['summary'][2] = 4  # Token capacity overflow after duplicate detection.
    shared_policy.finish(p, w, 4)
    torch.testing.assert_close(p.meta, meta, rtol=0, atol=0)
    torch.testing.assert_close(p.scores, scores, rtol=0, atol=0)
    assert w['shared_total'].item() == 0
    w['summary'].zero_()
    shared_policy.prepare(p, w, 4)
    shared_policy.finish(p, w, 4)
    torch.testing.assert_close(p.scores.view(4, 3), scores[:3].expand(4, 3), rtol=0, atol=0)
    torch.testing.assert_close(p.meta, meta, rtol=0, atol=0)
    assert w['shared_total'].item() == 3


def test_graph_replay_rechecks_inputs_after_storage_replacement(shared_frame):
    from rl.search.full480 import shared_policy

    for count in (4, 8):
        p, w = shared_frame(count)
        meta, scores = p.meta.clone(), p.scores.clone()

        def execute():
            p.meta.copy_(meta)
            p.scores.copy_(scores)
            shared_policy.prepare(p, w, count)
            shared_policy.finish(p, w, count)

        execute()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        for changed in (True, False, True):
            p.numbers[2 * 128 + 69, 63] = float(changed)
            graph.replay()
            expected = scores[:3].repeat(count)
            if changed:
                expected[6:9] = scores[6:9]
            torch.testing.assert_close(p.scores, expected, rtol=0, atol=0)
            torch.testing.assert_close(p.meta, meta, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph.reset()
        del graph, execute, p, w, meta, scores
        torch.cuda.empty_cache()
        temporary = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda').fill_(7)
        del temporary
