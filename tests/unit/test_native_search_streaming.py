from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np

from rl.search import native


class _FakeGroupedSearchStream:
    def __init__(self) -> None:
        self._pending = {}
        self._ready = deque()
        self.rounds = defaultdict(int)
        self.updates = defaultdict(list)

    @property
    def pending(self):
        return bool(self._pending)

    @staticmethod
    def _group(index):
        return int(index) // 2

    def submit(self, method_name, calls):
        grouped = defaultdict(dict)
        for index, call in calls.items():
            grouped[self._group(index)][int(index)] = call
        for group, group_calls in grouped.items():
            assert group not in self._pending
            self._pending[group] = (method_name, group_calls)
            self._ready.append(group)

    def receive(self, timeout=None):
        assert self._ready
        group = self._ready.popleft()
        method_name, calls = self._pending.pop(group)
        responses = {}
        for index, (args, kwargs) in calls.items():
            if method_name == "nosl_is_az_cpp_backprop_and_leaf_batch":
                assert kwargs == {"max_leaves": 1}
                updates = args[0]
                assert len(updates) == 1
                request_id, value, policy = updates[0]
                assert request_id == self.rounds[index]
                self.updates[index].append(
                    (request_id, value, np.asarray(policy).copy())
                )
                self.rounds[index] += 1
            else:
                assert method_name == "nosl_is_az_cpp_begin"
                assert self.rounds[index] == 0

            if self.rounds[index] == 2:
                responses[index] = {
                    "done": True,
                    "result": {"action": index, "rounds": 2},
                }
            else:
                request_id = self.rounds[index]
                responses[index] = {
                    "done": False,
                    "leaves": [{
                        "request_id": request_id,
                        "needs_policy": True,
                        "observation": {
                            "x": np.asarray([index, request_id], dtype=np.float32)
                        },
                    }],
                }
        return [SimpleNamespace(
            group_index=group,
            method_name=method_name,
            responses=responses,
        )]


def test_streamed_search_batches_across_trees_without_reordering_tree_requests(
    monkeypatch,
):
    inference_batches = []

    def fake_forward(_model, observations, **_kwargs):
        x = np.asarray(observations["x"])
        inference_batches.append(x.copy())
        logits = np.stack([
            np.asarray([row[0], row[1], row.sum()], dtype=np.float32)
            for row in x
        ])
        values = x[:, 0] * 0.1 + x[:, 1] * 0.01
        return logits, values

    monkeypatch.setattr(native, "forward_leaves", fake_forward)
    stream = _FakeGroupedSearchStream()
    begin_calls = {index: ((np.zeros(3),), {}) for index in range(4)}

    results, stats = native.complete_searches_streaming(
        object(),
        stream,
        begin_calls,
        leaf_batch_size=1,
        schedule=native.LeafInferenceSchedule(
            target_batch_size=2,
            max_wait_seconds=0.0,
        ),
        compact=True,
    )

    assert results == {
        index: {"action": index, "rounds": 2}
        for index in range(4)
    }
    assert stats["leaf_evaluations"] == 8
    assert stats["leaf_batches"] == 4
    assert stats["inference_batch_size_mean"] == 2.0
    assert stats["inference_batch_size_max"] == 2
    assert all(len(batch) == 2 for batch in inference_batches)
    for index in range(4):
        assert [item[0] for item in stream.updates[index]] == [0, 1]
        np.testing.assert_allclose(
            [item[1] for item in stream.updates[index]],
            [index * 0.1, index * 0.1 + 0.01],
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            stream.updates[index][0][2],
            [index, 0, index],
        )


def test_streamed_search_rejects_multiple_pending_leaves_per_tree():
    stream = _FakeGroupedSearchStream()
    try:
        native.complete_searches_streaming(
            object(),
            stream,
            {0: ((np.zeros(3),), {})},
            leaf_batch_size=2,
        )
    except ValueError as error:
        assert "leaf_batch_size=1" in str(error)
    else:
        raise AssertionError("streaming accepted multiple leaves per tree")
