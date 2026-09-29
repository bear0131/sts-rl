from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest

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
            elapsed_seconds=0.001,
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


@pytest.mark.parametrize('wait_seconds', [0.0, 0.001])
def test_fixed_schedule_clamps_target_to_active_tree_capacity(monkeypatch, wait_seconds):
    batch_sizes = []

    def fake_forward(_model, observations, **_kwargs):
        size = len(observations["x"])
        batch_sizes.append(size)
        return np.zeros((size, 3), dtype=np.float32), np.zeros(size)

    monkeypatch.setattr(native, "forward_leaves", fake_forward)
    stream = _FakeGroupedSearchStream()
    timeouts = []
    receive = stream.receive
    def tracked_receive(timeout=None):
        timeouts.append(timeout)
        return receive(timeout)
    stream.receive = tracked_receive
    results, stats = native.complete_searches_streaming(
        object(),
        stream,
        {index: ((np.zeros(3),), {}) for index in range(6)},
        leaf_batch_size=1,
        schedule=native.LeafInferenceSchedule(
            target_batch_size=128,
            max_wait_seconds=wait_seconds,
            mode="fixed_wait",
        ),
    )

    assert set(results) == set(range(6))
    assert batch_sizes == [6, 6]
    assert stats["inference_batch_size_max"] == 6
    assert stats["inference_watchdog_flushes"] == 0
    assert stats["inference_target_batch_sum"] == 12
    if wait_seconds == 0:
        assert 0.0 in timeouts
        assert all(value is None or value == 0.0 for value in timeouts)
    for index in range(6):
        assert [item[0] for item in stream.updates[index]] == [0, 1]


def test_adaptive_controller_moves_between_fixed_buckets_with_hysteresis():
    shrinking = native.AdaptiveLeafBatchController(
        256, initial_batch_size=128, window_batches=2, evidence_windows=2
    )
    for _ in range(4):
        shrinking.observe_batch(
            batch_size=64,
            target=128,
            active_capacity=384,
            ready_backlog=0,
            queue_delay_seconds=0.01,
            model_seconds=0.01,
            cycle_seconds=0.02,
        )
    assert shrinking.buckets == (64, 96, 128, 192, 256)
    assert shrinking.current_batch_size == 96
    assert shrinking.decreases == 1

    growing = native.AdaptiveLeafBatchController(
        256, initial_batch_size=128, window_batches=2, evidence_windows=2
    )
    for _ in range(4):
        growing.observe_batch(
            batch_size=128,
            target=128,
            active_capacity=384,
            ready_backlog=80,
            queue_delay_seconds=0.0,
            model_seconds=0.01,
            cycle_seconds=0.01,
        )
    assert growing.current_batch_size == 192
    assert growing.increases == 1
    assert growing.target(70) == 70
    for _ in range(2):
        growing.observe_batch(
            batch_size=192,
            target=192,
            active_capacity=384,
            ready_backlog=100,
            queue_delay_seconds=0.0,
            model_seconds=0.05,
            cycle_seconds=0.10,
        )
    assert growing.current_batch_size == 128
    assert growing.rollbacks == 1


def test_adaptive_streaming_uses_active_capacity_without_reordering_trees(
    monkeypatch,
):
    def fake_forward(_model, observations, **_kwargs):
        x = np.asarray(observations["x"])
        return np.zeros((len(x), 3), dtype=np.float32), x[:, 0] * 0.1

    monkeypatch.setattr(native, "forward_leaves", fake_forward)
    stream = _FakeGroupedSearchStream()
    controller = native.AdaptiveLeafBatchController(
        4, initial_batch_size=4, window_batches=100
    )
    results, stats = native.complete_searches_streaming(
        object(),
        stream,
        {index: ((np.zeros(3),), {}) for index in range(6)},
        leaf_batch_size=1,
        schedule=native.LeafInferenceSchedule(
            target_batch_size=4,
            max_wait_seconds=0.0,
            mode="adaptive",
        ),
        controller=controller,
    )

    assert set(results) == set(range(6))
    assert stats["leaf_evaluations"] == 12
    assert stats["inference_batch_target_current"] == 4
    assert stats["inference_full_flushes"] > 0
    assert stats["inference_ready_backlog_max"] > 0
    for index in range(6):
        assert [item[0] for item in stream.updates[index]] == [0, 1]
