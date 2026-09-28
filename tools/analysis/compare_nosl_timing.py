"""Summarize distributed NOSL timing logs without touching a live run.

The production trainer writes one JSON record per rank and update.  This
utility intentionally treats ``iteration_wall_sec`` as the end-to-end clock:
the slowest rank gates the synchronized update.  Rollout/resource fields are
reported separately because ``rollout_resources`` covers collection only.

Example::

    python tools/analysis/compare_nosl_timing.py \
      --run baseline=runs/local/perf-20260922/baseline \
      --run candidate=runs/local/perf-20260922/candidate \
      --steps 8159232,8208384 --require-manifests

The input directories are expected to contain ``rank*.log`` and, when
manifest verification is requested, ``recovery/rankN/step-*`` bundles copied
from the remote host.  No trainer or network operation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TIMING_FIELDS = (
    "iteration_wall_sec",
    "rollout_sec",
    "backward_sec",
    "gradient_pack_sec",
    "sync_sec",
    "optimizer_step_sec",
    "checkpoint_sec",
    "total_sec_rank",
)
RESOURCE_FIELDS = (
    "tree_cpu_percent_mean",
    "tree_rss_peak_sampled_bytes",
    "system_available_min_bytes",
    "cgroup_memory_peak_sampled_bytes",
)
CONTRACT_FIELDS = (
    "nosl_is_az/updates",
    "nosl_is_az/finite_fraction",
    "nosl_is_az/replay_size",
    "nosl_is_az/replay_old_fraction",
    "nosl_is_az/batch_size_used",
    "optimizer_steps",
)
_RANK_RE = re.compile(r"^rank(?P<rank>[0-9]+)\.log$")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _records(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON update records, ignoring non-JSON log lines."""

    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and "global_timesteps" in value
            and "iteration_wall_sec" in value
        ):
            result.append(value)
    return result


def _group_logs(root: Path) -> tuple[dict[int, dict[int, dict[str, Any]]], list[str]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    issues: list[str] = []
    logs = sorted(root.glob("rank*.log"))
    if not logs:
        return grouped, [f"no rank*.log files under {root}"]
    for path in logs:
        match = _RANK_RE.match(path.name)
        if match is None:
            continue
        rank = int(match.group("rank"))
        for record in _records(path):
            try:
                step = int(record["global_timesteps"])
            except (TypeError, ValueError):
                issues.append(f"{path}: invalid global_timesteps")
                continue
            # A resumed log can contain an old copy of a step.  The final
            # complete JSON record is authoritative, while duplicate ranks
            # remain visible as a diagnostic rather than silently summed.
            if step in grouped and rank in grouped[step]:
                issues.append(f"duplicate rank={rank} step={step} in {path}")
            grouped.setdefault(step, {})[rank] = record
    return grouped, issues


def _numeric_summary(records: Iterable[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in records if _finite_number(row.get(field))]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_check(
    root: Path,
    *,
    step: int,
    world_size: int,
    expected_run_id: str | None,
    expected_wandb_id: str | None,
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    issues: list[str] = []
    for rank in range(world_size):
        bundle = root / "recovery" / f"rank{rank}" / f"step-{step:012d}"
        manifest_path = bundle / "manifest.json"
        item: dict[str, Any] = {"rank": rank, "path": str(bundle), "ok": False}
        if not manifest_path.is_file():
            issues.append(f"missing manifest rank={rank} step={step}: {manifest_path}")
            checked.append(item)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("global_step"), manifest.get("rank"), manifest.get("world_size")) != (
                step,
                rank,
                world_size,
            ):
                raise ValueError("manifest identity mismatch")
            if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
                raise ValueError("manifest run_id mismatch")
            if expected_wandb_id is not None and manifest.get("wandb_id") != expected_wandb_id:
                raise ValueError("manifest wandb_id mismatch")
            hashes = manifest.get("sha256", {})
            for name, expected in hashes.items():
                path = bundle / name
                if not path.is_file() or _sha256(path) != expected:
                    raise ValueError(f"checksum mismatch: {name}")
            config = manifest.get("config", {})
            item.update(
                schema_version=manifest.get("schema_version"),
                run_id=manifest.get("run_id"),
                wandb_id=manifest.get("wandb_id"),
                geometry={
                    key: config.get(key)
                    for key in ("n_envs", "n_steps", "envs_per_worker")
                    if key in config
                },
                ok=True,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid manifest rank={rank} step={step}: {exc}")
        checked.append(item)
    return {"ok": not issues, "bundles": checked, "issues": issues}


def summarize_run(
    label: str,
    root: Path,
    *,
    steps: list[int] | None = None,
    updates: int = 2,
    world_size: int = 4,
    require_manifests: bool = False,
    expected_run_id: str | None = None,
    expected_wandb_id: str | None = None,
) -> dict[str, Any]:
    grouped, issues = _group_logs(root)
    common_steps = sorted(
        step for step, rows in grouped.items()
        if len(rows) == world_size and set(rows) == set(range(world_size))
    )
    selected = list(steps) if steps else common_steps[-updates:]
    if not selected:
        issues.append("no complete common update steps")
    if steps:
        missing = [step for step in selected if step not in common_steps]
        issues.extend(f"step {step} is missing one or more ranks" for step in missing)

    per_step: list[dict[str, Any]] = []
    for step in selected:
        rows_by_rank = grouped.get(step, {})
        rows = [rows_by_rank[rank] for rank in range(world_size) if rank in rows_by_rank]
        row: dict[str, Any] = {"global_timesteps": step, "rank_count": len(rows)}
        if len(rows) != world_size:
            per_step.append(row)
            continue
        row["transitions_global"] = sorted({int(r["transitions_global"]) for r in rows if "transitions_global" in r})
        row["timing"] = {field: _numeric_summary(rows, field) for field in TIMING_FIELDS}
        row["wall_sec_slowest_rank"] = row["timing"]["iteration_wall_sec"]["max"]
        transitions = row["transitions_global"]
        if len(transitions) != 1:
            issues.append(f"step {step} has inconsistent transitions_global: {transitions}")
        if row["wall_sec_slowest_rank"] is not None and transitions:
            row["global_transitions_per_sec_slowest_rank"] = transitions[0] / row["wall_sec_slowest_rank"]
        contract: dict[str, Any] = {}
        for field in CONTRACT_FIELDS:
            contract[field] = _numeric_summary(rows, field)
        row["contract"] = contract
        finite = [r.get("nosl_is_az/finite_fraction") for r in rows]
        if any(not _finite_number(value) or float(value) != 1.0 for value in finite):
            issues.append(f"step {step} has non-finite NOSL fraction")
        resources = []
        for rank, record in sorted(rows_by_rank.items()):
            resource = record.get("rollout_resources")
            if not isinstance(resource, dict):
                issues.append(f"step {step} rank {rank} has no rollout_resources")
                continue
            resources.append({
                "rank": rank,
                **{field: resource.get(field) for field in RESOURCE_FIELDS},
                "sample_count": resource.get("sample_count"),
                "gpu_devices": resource.get("gpu_devices"),
                "errors": resource.get("errors", []),
            })
        row["resources"] = resources
        if require_manifests:
            row["manifests"] = _manifest_check(
                root,
                step=step,
                world_size=world_size,
                expected_run_id=expected_run_id,
                expected_wandb_id=expected_wandb_id,
            )
            issues.extend(row["manifests"]["issues"])
        per_step.append(row)

    complete = [row for row in per_step if row.get("rank_count") == world_size]
    end_to_end = [row["wall_sec_slowest_rank"] for row in complete if row.get("wall_sec_slowest_rank") is not None]
    result: dict[str, Any] = {
        "label": label,
        "root": str(root),
        "world_size": world_size,
        "selected_steps": selected,
        "common_steps_available": common_steps,
        "updates_complete": len(complete),
        "end_to_end_wall_sec": {
            "min": min(end_to_end) if end_to_end else None,
            "mean": sum(end_to_end) / len(end_to_end) if end_to_end else None,
            "max": max(end_to_end) if end_to_end else None,
        },
        "steps": per_step,
        "issues": issues,
        "ok": not issues and len(complete) == len(selected),
    }
    return result


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_run_spec)
    parser.add_argument("--steps", help="comma-separated global steps to select")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--require-manifests", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--wandb-id")
    args = parser.parse_args(argv)
    if args.updates < 1 or args.world_size < 1:
        parser.error("--updates and --world-size must be positive")
    steps = None
    if args.steps:
        try:
            steps = [int(value) for value in args.steps.split(",") if value]
        except ValueError as exc:
            parser.error(f"invalid --steps: {exc}")
    output = [
        summarize_run(
            label,
            path,
            steps=steps,
            updates=args.updates,
            world_size=args.world_size,
            require_manifests=args.require_manifests,
            expected_run_id=args.run_id,
            expected_wandb_id=args.wandb_id,
        )
        for label, path in args.run
    ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if all(item["ok"] for item in output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
