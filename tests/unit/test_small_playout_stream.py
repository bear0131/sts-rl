"""Completion batching preserves per-slot ownership and controller-only commits."""
from types import SimpleNamespace
import threading

import pytest

from env.search.small_playout import SmallPlayoutStream


@pytest.fixture
def stream_factory():
    opened = []

    def create(count=3, blocked=()):
        gates = [threading.Event() for _ in range(count)]
        for i, gate in enumerate(gates):
            if i not in blocked:
                gate.set()
        accepted = []

        def search(index, config, nonce, prior):
            if not gates[index].wait(2):
                raise TimeoutError('test search was not released')
            return dict(action=nonce, index=index)

        def prepare(calls):
            inline, jobs = {}, []
            for index, (_, kw) in calls.items():
                if kw.get('inline'):
                    inline[index] = dict(action=kw['state_nonce'], index=index)
                else:
                    jobs.append((index, 'common', None, kw['state_nonce']))
            return inline, jobs

        def accept(job, result):
            assert job[0] == result['index']
            accepted.append((job[0], threading.get_ident()))
            return result

        batch = SimpleNamespace(envs=[SimpleNamespace(thick=SimpleNamespace(bc=i)) for i in range(count)],
                                net=SimpleNamespace(search=search), prior=None, prepare=prepare, accept=accept)
        stream = SmallPlayoutStream(batch, threads=count)
        opened.append((stream, gates))
        return stream, gates, accepted

    yield create
    for stream, gates in opened:
        for gate in gates:
            gate.set()
        stream.close()


def test_completions_are_consumed_once_and_only_by_controller(stream_factory):
    stream, _, accepted = stream_factory()
    calls = {i: ((), dict(state_nonce=20+i)) for i in range(3)}
    stream.submit(calls)
    with pytest.raises(RuntimeError, match='outstanding'):
        stream.submit(calls)
    rows = stream.receive(timeout=1, minimum=3)
    assert {i: row['action'] for i, row in rows.items()} == {0: 20, 1: 21, 2: 22}
    assert sorted(accepted) == [(i, threading.get_ident()) for i in range(3)]
    assert not stream.pending and stream.receive() == {}
    stream.submit({0: ((), dict(state_nonce=99))})
    assert stream.receive(timeout=1, minimum=10)[0]['action'] == 99
    assert not stream.pending


def test_batch_deadline_returns_partial_results_and_keeps_pending_slot(stream_factory):
    stream, gates, accepted = stream_factory(2, blocked=(1,))
    stream.submit({i: ((), dict(state_nonce=i)) for i in range(2)})
    assert set(stream.receive(timeout=.05, minimum=2)) == {0}
    assert stream.pending and accepted == [(0, threading.get_ident())]
    assert stream.receive(timeout=0, minimum=2) == {}
    gates[1].set()
    assert set(stream.receive(timeout=1, minimum=2)) == {1}
    assert not stream.pending


def test_inline_completion_counts_toward_requested_batch(stream_factory):
    stream, gates, accepted = stream_factory(2, blocked=(1,))
    stream.submit({0: ((), dict(state_nonce=7, inline=True)), 1: ((), dict(state_nonce=8))})
    assert set(stream.receive(timeout=1, minimum=1)) == {0}
    assert stream.pending and not accepted
    gates[1].set()
    assert set(stream.receive(timeout=1, minimum=2)) == {1}
    stream.close()
    assert not stream.pending
    with pytest.raises(RuntimeError, match='closed'):
        stream.receive()
    with pytest.raises(RuntimeError, match='closed'):
        stream.submit({})


@pytest.mark.parametrize('options', [dict(minimum=0), dict(minimum=True),
    dict(timeout=-1), dict(timeout=float('nan')), dict(timeout=float('inf'))])
def test_invalid_batch_request_is_rejected_without_consuming_work(stream_factory, options):
    stream, _, _ = stream_factory(1)
    stream.submit({0: ((), dict(state_nonce=42, inline=True))})
    with pytest.raises(ValueError, match='completion count'):
        stream.receive(**options)
    assert stream.receive()[0]['action'] == 42
