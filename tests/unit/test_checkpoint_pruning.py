from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

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


def _round_generation(root, round_value, *, legacy=False):
    # Deliberately not divisible by the round number or retention interval.
    path = _generation(root, round_value * 113 + 7)
    if legacy:
        with zipfile.ZipFile(path / 'model.zip', 'w') as archive:
            archive.writestr('data', json.dumps({'episode_rollouts_completed': round_value}))
    else:
        (path / 'manifest.json').write_text(json.dumps({'rollout_round': round_value}), encoding='utf-8')
    return path


def test_round_retention_keeps_latest_five_anchor_and_current(tmp_path):
    paths = {}
    for round_value, expected in [(179,{179}),(180,{180}),(181,{180,181}),
                                  (184,{180,184}),(185,{185}),(186,{185,186}),
                                  (189,{185,189}),(190,{190})]:
        paths[round_value] = _round_generation(tmp_path, round_value, legacy=round_value <= 180)
        prune_bundles(tmp_path, rank=0, keep=2, round_multiple=5)
        assert {r for r,p in paths.items() if p.exists()} == expected
    unrelated = tmp_path / 'rank0' / 'step-manual-diagnostics'
    unrelated.mkdir(); (unrelated / 'manifest.json').write_text('{}')
    prune_bundles(tmp_path, rank=0, keep=2, round_multiple=5)
    assert unrelated.is_dir()


def test_failed_new_bundle_preserves_round_anchor_and_current(tmp_path):
    anchor = _round_generation(tmp_path, 185)
    current = _round_generation(tmp_path, 189)
    def fail_save(path):
        Path(path).write_bytes(b'partial')
        raise OSError('simulated disk write failure')
    model = SimpleNamespace(num_timesteps=190*113+7, get_env=lambda: None, save=fail_save)
    with pytest.raises(OSError, match='simulated disk'):
        checkpoints.save_bundle(model,tmp_path,rank=0,world_size=1,run_id='fixture',wandb_id=None,config={})
    assert anchor.is_dir() and current.is_dir()
    prune_bundles(tmp_path,rank=0,keep=2,round_multiple=5)
    assert anchor.is_dir() and current.is_dir()


def test_unknown_or_nonmonotonic_rounds_fail_before_deletion(tmp_path):
    first = _round_generation(tmp_path, 185)
    second = _round_generation(tmp_path, 186)
    (second/'manifest.json').write_text(json.dumps({'rollout_round':184}))
    with pytest.raises(ValueError, match='must increase'):
        prune_bundles(tmp_path,rank=0,keep=2,round_multiple=5)
    assert first.is_dir() and second.is_dir()


@pytest.mark.parametrize('multiple', [-1, 1.5, True])
def test_invalid_round_multiple_rejected(tmp_path, multiple):
    with pytest.raises(ValueError,match='nonnegative integer'):
        prune_bundles(tmp_path,rank=0,keep=2,round_multiple=multiple)


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
        ensure_bundle_disk_space(
            tmp_path, rank=0, reserve_bytes=20, reclaim_timeout_seconds=0,
        )


def test_bundle_space_check_reports_capacity_when_available(tmp_path, monkeypatch):
    generation = _generation(tmp_path, 10)
    (generation / "learner.pt").write_bytes(b"x" * 100)
    monkeypatch.setattr(
        checkpoints.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )
    monkeypatch.setattr(
        checkpoints.time, "sleep", lambda _seconds: pytest.fail("unnecessary wait"),
    )

    capacity = ensure_bundle_disk_space(
        tmp_path, rank=0, reserve_bytes=20,
    )

    assert capacity["free_bytes"] == 1000
    assert capacity["required_bytes"] == capacity["previous_bundle_bytes"] + 20


@pytest.fixture
def reclaim_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(checkpoints.time, "sleep", sleep)
    return clock


def test_bundle_space_waits_for_observed_quota_release(
    tmp_path, monkeypatch, reclaim_clock,
):
    generation = _generation(tmp_path, 10)
    (generation / "learner.pt").write_bytes(b"x" * 100)
    previous = sum(p.stat().st_size for p in generation.iterdir())
    free = iter((previous + 1, previous + 19, previous + 20))
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(free)),
    )

    capacity = ensure_bundle_disk_space(tmp_path, rank=0, reserve_bytes=20)

    assert capacity["free_bytes"] == capacity["required_bytes"] == previous + 20
    assert 0 < reclaim_clock.now <= 5.0
    assert (generation / "learner.pt").read_bytes() == b"x" * 100


def test_bundle_space_still_rejects_shortfall_after_bounded_wait(
    tmp_path, monkeypatch, reclaim_clock,
):
    generation = _generation(tmp_path, 10)
    (generation / "learner.pt").write_bytes(b"x" * 100)
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1),
    )

    with pytest.raises(RuntimeError, match="insufficient disk space"):
        ensure_bundle_disk_space(
            tmp_path, rank=0, reserve_bytes=20, reclaim_timeout_seconds=0.25,
        )

    assert reclaim_clock.now == pytest.approx(0.25)
    assert generation.is_dir()
