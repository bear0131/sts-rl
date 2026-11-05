"""Contracts for the finite validation-plateau protocol."""

from itertools import islice

import numpy as np
import pytest

from tools.analysis.lower_convergence import Plateau, batches


def test_small_improvements_accumulate_and_each_lr_gets_its_own_patience():
    p = Plateau(0.8, 0.001, 3)
    assert [p.observe(x) for x in (0.7996, 0.7993, 0.7988)] == ["continue"] * 3
    assert p.stale == 0
    for expected in ("reduce", "reduce", "converged"):
        assert [p.observe(0.7988) for _ in range(3)] == [
            "continue",
            "continue",
            expected,
        ]
    assert p.drops == 2
    with pytest.raises(ValueError, match="finite"):
        p.observe(float("nan"))


def test_training_pass_covers_only_allowed_decisions_including_short_batches():
    # IDs are intentionally sparse: excluded holdout games must stay excluded.
    sets = [np.array([4, 10, 18, 20, 31]), np.array([7, 8, 16, 55])]
    stream = batches(sets, 3, 211)
    expected = sorted((bank, int(i)) for bank, rows in enumerate(sets) for i in rows)
    passes = []
    for _ in range(2):
        ids = [(bank, int(i)) for bank, batch in islice(stream, 4) for i in batch]
        assert sorted(ids) == expected
        passes.append(ids)
    assert passes[0] != passes[1]
