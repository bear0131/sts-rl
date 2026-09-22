from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.training import checkpoints
from rl.training.checkpoints import ensure_bundle_disk_space, prune_bundles


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


def test_prune_bundles_bounds_incomplete_diagnostic_evidence(tmp_path):
    _generation(tmp_path, 30)
    stale = (
        tmp_path / "rank0"
        / ".incomplete-step-000000000010-00000000000000000000000000000000"
    )
    stale.mkdir()
    current = tmp_path / "rank0" / "step-000000000020"
    current.mkdir()
    unrelated = tmp_path / "rank0" / "step-manual-diagnostics"
    unrelated.mkdir()

    prune_bundles(tmp_path, rank=0, keep=1)

    assert not stale.exists()
    assert current.is_dir()
    assert unrelated.is_dir()


def test_bundle_space_check_uses_previous_generation_size(tmp_path, monkeypatch):
    generation = _generation(tmp_path, 10)
    (generation / "learner.pt").write_bytes(b"x" * 100)
    previous_bytes = sum(
        path.stat().st_size for path in generation.iterdir() if path.is_file()
    )
    monkeypatch.setattr(
        checkpoints.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=previous_bytes + 19),
    )

    with pytest.raises(RuntimeError, match=(
        rf"free_bytes={previous_bytes + 19} .*"
        rf"required_bytes={previous_bytes + 20} .*"
        rf"previous_bundle_bytes={previous_bytes}"
    )):
        ensure_bundle_disk_space(tmp_path, rank=0, reserve_bytes=20)


def test_bundle_space_check_reports_capacity_when_available(tmp_path, monkeypatch):
    generation = _generation(tmp_path, 10)
    (generation / "learner.pt").write_bytes(b"x" * 100)
    monkeypatch.setattr(
        checkpoints.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )

    capacity = ensure_bundle_disk_space(
        tmp_path, rank=0, reserve_bytes=20,
    )

    assert capacity["free_bytes"] == 1000
    assert capacity["required_bytes"] == capacity["previous_bundle_bytes"] + 20
