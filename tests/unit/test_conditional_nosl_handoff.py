from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.launchers import conditional_nosl_handoff as handoff
from experiments.launchers.conditional_nosl_handoff import decide, floor_history, run_handoff


def config(tmp_path):
    root = tmp_path.resolve()
    for name in ("args", "override", "gate", "state", "events"):
        (root / name).write_text("{}")
    return dict(repo=str(root), source_run="source", source_wandb="source-wandb",
                target_run="target", target_wandb="target-wandb",
                deadline="2026-09-18T22:23:29+00:00", threshold=29.0,
                resume_step=1_622_016, rollout_size=49_152, timeout_sec=60,
                ports=[1, 2, 3, 4], source_pids=[11, 12, 13, 14], build_dirs=["a", "b", "c", "d"],
                args_file="args", overrides_file="override", gate_file="gate",
                state_file="state", events_file="events")


def inspections(values=(5.0, 27.1)):
    history = [(49_152 * (index + 1), value) for index, value in enumerate(values)]
    return [dict(rank=rank, identity_ok=True, alive=True, history=list(history)) for rank in range(4)]


def test_floor_history_requires_complete_fresh_sequence():
    text = ''.join(f'{{"global_timesteps": {step}, "game/floor_mean": {value}}}\n'
                   for step, value in inspections()[0]["history"])
    assert floor_history(text + '{"global_timesteps":', 49_152) == dict(inspections()[0]["history"])
    with pytest.raises(ValueError, match="fresh-run history"):
        floor_history(text.replace("49152", "98304", 1), 49_152)
    with pytest.raises(ValueError, match="duplicate"):
        floor_history(text + text.splitlines(keepends=True)[-1], 49_152)


def test_decision_is_strictly_greater_than_threshold_and_distributed():
    assert decide(inspections((29.0,)), 29.0) == (True, 29.0)
    assert decide(inspections((29.0001,)), 29.0) == (False, 29.0001)
    disagreement = inspections((5.0, 27.1))
    disagreement[3]["history"][-1] = (98_304, 27.2)
    with pytest.raises(ValueError, match="disagree"):
        decide(disagreement, 29.0)


def test_unknown_identity_fails_closed():
    rows = inspections()
    rows[0]["identity_ok"] = False
    with pytest.raises(ValueError, match="identity"):
        decide(rows, 29.0)


def test_separate_log_tail_writes_do_not_hide_a_threshold_crossing():
    rows = inspections((27.1,))
    rows[0]["history"].append((98_304, 29.1))
    assert decide(rows, 29.0) == (False, 29.1)


def test_stopped_source_is_not_a_valid_deadline_observation():
    rows = inspections()
    for row in rows: row["alive"] = False
    with pytest.raises(ValueError, match="identity"):
        decide(rows, 29.0)


def test_above_threshold_keeps_source_without_mutation(tmp_path):
    operations, events = [], []

    def call(operation):
        operations.append(operation)
        if operation == "validate-snapshot":
            return [dict(rank=i, global_step=1_622_016,
                         model_hashes={"policy.pth": "p", "policy.optimizer.pth": "a"}) for i in range(4)]
        return inspections((29.1,))

    result = run_handoff(config(tmp_path), call, lambda status, **fields: events.append(status),
                         now=lambda: datetime(2026, 9, 18, 22, 24, tzinfo=timezone.utc))
    assert result == "kept"
    assert operations == ["validate-snapshot", "inspect"]
    assert events == ["armed", "decision", "kept"]


def test_below_threshold_stops_pins_resumes_and_verifies(tmp_path):
    operations, events = [], []
    times = iter(datetime(2026, 9, 18, 22, 24, tzinfo=timezone.utc) + timedelta(seconds=i)
                 for i in range(20))

    def checked(step):
        return [dict(rank=i, global_step=step,
                     model_hashes={"policy.pth": "p", "policy.optimizer.pth": "a"}) for i in range(4)]

    def call(operation):
        operations.append(operation)
        if operation == "validate-snapshot": return checked(1_622_016)
        if operation == "inspect":
            rows = inspections()
            if "request-stop" in operations:
                for row in rows: row["alive"] = False
            return rows
        if operation in ("request-stop", "prepare-resume", "launch"):
            return [dict(rank=i) for i in range(4)]
        if operation == "pin-current": return checked(3_900_000)
        if operation == "inspect-target":
            return [dict(rank=i, alive=True, global_step=1_671_168) for i in range(4)]
        if operation == "verify-target": return checked(1_671_168)
        raise AssertionError(operation)

    result = run_handoff(config(tmp_path), call, lambda status, **fields: events.append(status),
                         now=lambda: next(times), sleep=lambda _seconds: None)
    assert result == "switched"
    assert operations == ["validate-snapshot", "inspect", "validate-snapshot", "request-stop",
                          "inspect", "pin-current", "prepare-resume", "launch", "inspect-target",
                          "verify-target"]
    assert events == ["armed", "decision", "stop_requested", "source_pinned", "target_prepared",
                      "target_launched", "switched"]


def test_bad_target_is_detected_before_stop(tmp_path):
    calls = 0

    def call(operation):
        nonlocal calls
        calls += 1
        if operation == "validate-snapshot" and calls == 3:
            raise ValueError("checksum")
        if operation == "validate-snapshot":
            return [dict(rank=i, global_step=1_622_016,
                         model_hashes={"policy.pth": "p", "policy.optimizer.pth": "a"}) for i in range(4)]
        return inspections()

    with pytest.raises(ValueError, match="checksum"):
        run_handoff(config(tmp_path), call, lambda *_args, **_kwargs: None,
                    now=lambda: datetime(2026, 9, 18, 22, 24, tzinfo=timezone.utc))


def test_forked_environment_children_are_not_extra_trainers(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    for pid, parent in ((7, 1), (8, 7), (9, 1)):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        argv = ["python", "-m", "rl.training.distributed_sb3", "--run-id", "source", "--rank", "0"]
        (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        (directory / "status").write_text(f"PPid:\t{parent}\n")
    monkeypatch.setattr(handoff, "Path", lambda path: proc if str(path) == "/proc" else Path(path))
    assert [pid for pid, _argv in handoff.trainers()] == [7, 9]
