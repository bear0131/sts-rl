from __future__ import annotations

from io import StringIO

import pytest

from rl.rules.vector_rule_stream import iter_jsonl_records, take_batch


def test_jsonl_stream_is_incremental_and_preserves_record_order():
    stream = StringIO('{"n": 0}\n\n{"n": 1}\n')
    records = iter_jsonl_records(stream)
    assert next(records)["n"] == 0
    assert next(records)["n"] == 1
    with pytest.raises(StopIteration):
        next(records)


def test_jsonl_stream_rejects_non_object_records():
    with pytest.raises(ValueError, match="must be an object"):
        list(iter_jsonl_records(StringIO('[1]\n')))


def test_take_batch_consumes_once_without_padding_or_replay():
    values = iter(range(5))
    assert take_batch(values, 2) == [0, 1]
    assert take_batch(values, 2) == [2, 3]
    assert take_batch(values, 2) == [4]
    assert take_batch(values, 2) == []
