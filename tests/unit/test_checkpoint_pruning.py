from pathlib import Path

import pytest

from rl.training.checkpoints import prune_bundles


def _generation(root: Path, step: int) -> Path:
    path = root / "rank0" / f"step-{step:012d}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    return path


def test_prune_bundles_can_leave_one_fallback_before_atomic_save(tmp_path):
    generations = [_generation(tmp_path, step) for step in (10, 20, 30)]

    prune_bundles(tmp_path, rank=0, keep=1)

    assert not generations[0].exists()
    assert not generations[1].exists()
    assert generations[2].is_dir()


def test_prune_bundles_rejects_removing_every_fallback(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        prune_bundles(tmp_path, rank=0, keep=0)
