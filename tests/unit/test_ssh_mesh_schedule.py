import itertools

import pytest

from tools.benchmarks.benchmark_ssh_mesh import pair_rounds


@pytest.mark.parametrize('size', [2, 3, 4, 8, 9])
def test_every_pair_once_without_competing_for_a_host(size):
    rounds = pair_rounds(size)
    observed = []
    for pairs in rounds:
        participants = [host for pair in pairs for host in pair]
        assert len(participants) == len(set(participants))
        observed.extend(tuple(sorted(pair)) for pair in pairs)
    assert sorted(observed) == list(itertools.combinations(range(size), 2))


def test_single_host_cannot_measure_a_link():
    with pytest.raises(ValueError, match='two hosts'):
        pair_rounds(1)
