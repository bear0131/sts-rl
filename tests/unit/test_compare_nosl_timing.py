import hashlib
import json

from tools.analysis.compare_nosl_timing import summarize_run


def _write_bundle(root, rank, step, run_id="run", wandb_id="wandb"):
    bundle = root / "recovery" / f"rank{rank}" / f"step-{step:012d}"
    bundle.mkdir(parents=True)
    model = bundle / "model.zip"
    learner = bundle / "learner.pt"
    model.write_bytes(f"model-{rank}-{step}".encode())
    learner.write_bytes(f"learner-{rank}-{step}".encode())
    manifest = {
        "schema_version": 2,
        "rank": rank,
        "world_size": 2,
        "global_step": step,
        "run_id": run_id,
        "wandb_id": wandb_id,
        "config": {"n_envs": 384, "n_steps": 32, "envs_per_worker": 16},
        "sha256": {
            "model.zip": hashlib.sha256(model.read_bytes()).hexdigest(),
            "learner.pt": hashlib.sha256(learner.read_bytes()).hexdigest(),
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _record(rank, step, wall):
    return {
        "rank": rank,
        "world_size": 2,
        "global_timesteps": step,
        "transitions_global": 49152,
        "iteration_wall_sec": wall,
        "rollout_sec": wall - 20,
        "backward_sec": 10,
        "gradient_pack_sec": 1,
        "sync_sec": 2,
        "optimizer_step_sec": 1,
        "checkpoint_sec": 6,
        "total_sec_rank": wall - 6,
        "optimizer_steps": 40,
        "nosl_is_az/updates": 40,
        "nosl_is_az/finite_fraction": 1,
        "nosl_is_az/replay_size": 524288,
        "nosl_is_az/replay_old_fraction": 0.9,
        "nosl_is_az/batch_size_used": 98304,
        "rollout_resources": {
            "sample_count": 3,
            "tree_cpu_percent_mean": 100,
            "tree_rss_peak_sampled_bytes": 10,
            "system_available_min_bytes": 20,
            "cgroup_memory_peak_sampled_bytes": 30,
            "gpu_devices": None,
            "errors": [],
        },
    }


def test_summarize_uses_slowest_rank_and_verifies_bundles(tmp_path):
    steps = [8159232, 8208384]
    for rank in range(2):
        rows = [_record(rank, step, 100 + rank * 5 + index) for index, step in enumerate(steps)]
        (tmp_path / f"rank{rank}.log").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        for step in steps:
            _write_bundle(tmp_path, rank, step)

    result = summarize_run(
        "candidate",
        tmp_path,
        steps=steps,
        world_size=2,
        require_manifests=True,
        expected_run_id="run",
        expected_wandb_id="wandb",
    )

    assert result["ok"] is True
    assert result["end_to_end_wall_sec"]["max"] == 106
    assert result["steps"][0]["wall_sec_slowest_rank"] == 105
    assert result["steps"][0]["manifests"]["ok"] is True


def test_summarize_rejects_nonfinite_or_missing_rank(tmp_path):
    row = _record(0, 8159232, 100)
    row["nosl_is_az/finite_fraction"] = 0
    (tmp_path / "rank0.log").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = summarize_run("bad", tmp_path, steps=[8159232], world_size=2)
    assert result["ok"] is False
    assert result["updates_complete"] == 0
    assert any("missing one or more ranks" in issue for issue in result["issues"])
