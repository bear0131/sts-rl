"""A bounded, key-only four-rank handoff, using the existing NOSL launcher.

Run the coordinator on rank0 independently of the trainer. Unknown history,
identity changes or invalid recovery/build files fail closed before STOP.
This is an operational entry: the ordinary launcher has no delayed condition,
history completeness check, graceful stop or snapshot-pinning operation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def under(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("operation path escaped its explicit repository")
    return path


def validate_config(config):
    for key in ("source_run", "source_wandb", "target_run", "target_wandb"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config[key]):
            raise ValueError(f"invalid {key}")
    if config["source_run"] == config["target_run"]:
        raise ValueError("source and target must differ")
    deadline = datetime.fromisoformat(config["deadline"])
    if deadline.tzinfo is None:
        raise ValueError("deadline requires a timezone")
    if len(config["ports"]) != 4 or len(config["source_pids"]) != 4 or len(config["build_dirs"]) != 4:
        raise ValueError("exactly four ranks required")
    if any(not 1 <= int(port) <= 65535 for port in config["ports"]):
        raise ValueError("invalid SSH port")
    if not math.isfinite(config["threshold"]) or config["resume_step"] <= 0:
        raise ValueError("invalid threshold or recovery step")
    if config["rollout_size"] <= 0 or config["timeout_sec"] <= 0:
        raise ValueError("invalid workload or operation timeout")
    for key in ("args_file", "overrides_file", "gate_file", "state_file", "events_file"):
        under(config["repo"], config[key])
    for build in config["build_dirs"]:
        under(config["repo"], build)
    return deadline


def floor_history(text, rollout_size):
    rows = {}
    # Ignore an unfinished trailing write, never an already completed bad row.
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n") or not line.startswith("{"):
            continue
        row = json.loads(line)
        if "global_timesteps" not in row:
            continue
        step, value = row["global_timesteps"], row.get("game/floor_mean")
        if type(step) is not int or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid or missing floor metric")
        if step in rows:
            raise ValueError("duplicate/resumed history is not a complete fresh-run history")
        rows[step] = value
    steps = list(rows)
    if not steps or steps != list(range(rollout_size, steps[-1] + 1, rollout_size)):
        raise ValueError("floor history is empty, unordered or incomplete")
    return rows


def decide(inspections, threshold):
    merged = {}
    if len(inspections) != 4:
        raise ValueError("missing rank history")
    for rank, item in enumerate(inspections):
        if item["rank"] != rank or not item["identity_ok"] or not item["alive"]:
            raise ValueError("trainer identity changed")
        for step, value in item["history"]:
            if step in merged and merged[step] != value:
                raise ValueError("distributed floor metrics disagree")
            merged[step] = value
    # Rank logs are written separately after the collective; a one-row tail lag
    # is expected, but every rank must retain its full contiguous fresh history.
    lengths = [len(item["history"]) for item in inspections]
    if max(lengths) - min(lengths) > 1:
        raise ValueError("rank history is stale")
    if not merged:
        raise ValueError("no floor observations")
    return max(merged.values()) <= threshold, max(merged.values())


def process_argv(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except FileNotFoundError:
        return []


def matches(argv, run, rank):
    def option(name):
        return argv[argv.index(name) + 1] if name in argv else None
    return ("rl.training.distributed_sb3" in argv and option("--run-id") == run
            and option("--rank") == str(rank))


def trainers():
    candidates = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        argv = process_argv(path.name)
        if "rl.training.distributed_sb3" not in argv:
            continue
        try:
            parent = next(int(line.split()[1]) for line in (path / "status").read_text().splitlines()
                          if line.startswith("PPid:"))
        except FileNotFoundError:
            continue
        candidates[int(path.name)] = (argv, parent)
    # Forked environment workers inherit the parent's command line.
    return [(pid, argv) for pid, (argv, parent) in candidates.items() if parent not in candidates]


def bundle(config, run, rank, path, *, checksums=False):
    manifest = json.loads((path / "manifest.json").read_text())
    expected_wandb = config["source_wandb"] if run == config["source_run"] else config["target_wandb"]
    if (manifest["schema_version"], manifest["rank"], manifest["world_size"],
            manifest["run_id"], manifest["wandb_id"]) != (1, rank, 4, run, expected_wandb):
        raise ValueError("recovery identity mismatch")
    if checksums:
        for name in ("model.zip", "learner.pt"):
            if sha(path / name) != manifest["sha256"][name]:
                raise ValueError(f"invalid {name} checksum")
    return manifest


def model_hashes(path):
    with zipfile.ZipFile(path / "model.zip") as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in ("policy.pth", "policy.optimizer.pth")}


def latest_bundle(config, run, rank):
    directory = under(config["repo"], f"runs/{run}/recovery/rank{rank}")
    generation = (directory / "latest").read_text().strip()
    if not re.fullmatch(r"step-\d{12}", generation):
        raise ValueError("invalid latest recovery pointer")
    return under(directory, generation)


def pin_bundle(config, run, rank, source):
    manifest = bundle(config, run, rank, source, checksums=True)
    destination = under(config["repo"],
                        f"runs/snapshots/{run}/step-{manifest['global_step']:012d}/rank{rank}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("model.zip", "learner.pt", "manifest.json"):
        target = destination / name
        if target.exists():
            if sha(target) != sha(source / name):
                raise ValueError("existing pinned snapshot differs")
        else:
            os.link(source / name, target)
    bundle(config, run, rank, destination, checksums=True)
    return dict(rank=rank, global_step=manifest["global_step"], pinned=str(destination),
                model_hashes=model_hashes(destination))


def validate_snapshot(config, rank):
    path = under(config["repo"], f"runs/snapshots/{config['target_run']}/"
                 f"step-{config['resume_step']:012d}/rank{rank}")
    manifest = bundle(config, config["target_run"], rank, path, checksums=True)
    if manifest["global_step"] != config["resume_step"]:
        raise ValueError("wrong snapshot step")
    original = manifest["config"]
    if (original["nosl_is_az_updates"], original["nosl_is_az_simulations"],
            original["nosl_is_az_paper_full_gumbel"]) != (40, 128, True):
        raise ValueError("unexpected old training contract")
    for name, expected in config["args_sha256"].items():
        if sha(under(config["repo"], name)) != expected:
            raise ValueError("launch configuration changed")
    gate = json.loads(under(config["repo"], config["gate_file"]).read_text())
    if gate["validation"] != "pass" or gate["binary_sha256"] != config["binary_sha256"]:
        raise ValueError("optimization gate is not accepted")
    modules = list(under(config["repo"], config["build_dirs"][rank]).glob("slaythespire*.so"))
    if len(modules) != 1 or sha(modules[0]) != config["binary_sha256"]:
        raise ValueError("candidate native build identity mismatch")
    return dict(rank=rank, global_step=manifest["global_step"], model_hashes=model_hashes(path))


def local_operation(config, rank, operation):
    repo = Path(config["repo"]).resolve()
    source, target = config["source_run"], config["target_run"]
    if operation == "inspect":
        pid = config["source_pids"][rank]
        argv = process_argv(pid)
        unexpected = [(p, a) for p, a in trainers() if p != pid]
        text = under(repo, f"runs/{source}/rank{rank}.log").read_text()
        history = floor_history(text, config["rollout_size"])
        identity_ok = (matches(argv, source, rank) if argv else under(
            repo, f"runs/{source}/STOP").exists()) and not unexpected
        return dict(rank=rank, alive=bool(argv), tree_alive=bool(argv) or bool(unexpected), identity_ok=identity_ok,
                    history=list(history.items()),
                    latest=bundle(config, source, rank, latest_bundle(config, source, rank))["global_step"])
    if operation == "validate-snapshot":
        return validate_snapshot(config, rank)
    if operation == "request-stop":
        item = local_operation(config, rank, "inspect")
        if not item["identity_ok"]:
            raise ValueError("refuse STOP for a changed trainer")
        under(repo, f"runs/{source}/STOP").touch(exist_ok=True)
        return dict(rank=rank, stop_requested=True)
    if operation == "pin-current":
        if trainers():
            raise ValueError("wait for every trainer to exit before pinning")
        return pin_bundle(config, source, rank, latest_bundle(config, source, rank))
    if operation == "prepare-resume":
        if trainers():
            raise ValueError("a trainer is still running")
        validate_snapshot(config, rank)
        snapshot = under(repo, f"runs/snapshots/{target}/step-{config['resume_step']:012d}/rank{rank}")
        recovery = under(repo, f"runs/{target}/recovery/rank{rank}/step-{config['resume_step']:012d}")
        recovery.mkdir(parents=True, exist_ok=True)
        for name in ("model.zip", "learner.pt", "manifest.json"):
            destination = recovery / name
            if destination.exists():
                if sha(destination) != sha(snapshot / name):
                    raise ValueError("target recovery differs from pinned snapshot")
            else:
                os.link(snapshot / name, destination)
        stop = under(repo, f"runs/{target}/STOP")
        if stop.exists():
            preserved = stop.with_name("STOP.completed-" + str(config["resume_step"]))
            if preserved.exists():
                raise ValueError("previous completed STOP marker already exists")
            stop.rename(preserved)
        return dict(rank=rank, prepared=True)
    if operation == "launch":
        active = trainers()
        if active:
            if len(active) == 1 and matches(active[0][1], target, rank):
                return dict(rank=rank, pid=active[0][0], already_running=True)
            raise ValueError("refuse concurrent training")
        validate_snapshot(config, rank)
        env = dict(os.environ, NOSL_RUN_ID=target, NOSL_WANDB_ID=config["target_wandb"],
                   NOSL_RUN_NAME=target, NOSL_ARGS_FILE=str(under(repo, config["args_file"])),
                   NOSL_OVERRIDES_FILE=str(under(repo, config["overrides_file"])),
                   NOSL_RESUME_STEP=str(config["resume_step"]), NOSL_MAX_CHECKPOINTS="2",
                   NOSL_PEER_PORTS=" ".join(map(str, config["ports"])), NOSL_RING_PORT="29789",
                   NOSL_REPO=str(repo), STS_LIGHTSPEED_BUILD_DIR=str(under(repo, config["build_dirs"][rank])),
                   NOSL_CHECKPOINT=str(under(repo, f"runs/snapshots/{target}/"
                                            f"step-{config['resume_step']:012d}/rank{rank}/model.zip")))
        completed = subprocess.run(["bash", "experiments/launchers/launch_nosl_replay.sh", str(rank)],
                                   cwd=repo, env=env, capture_output=True, text=True, timeout=30, check=True)
        pid = int(under(repo, f"runs/{target}/rank{rank}.pid").read_text())
        if not matches(process_argv(pid), target, rank):
            raise ValueError("launched trainer identity mismatch")
        return dict(rank=rank, pid=pid, output=completed.stdout.strip())
    if operation == "inspect-target":
        pid = int(under(repo, f"runs/{target}/rank{rank}.pid").read_text())
        path = latest_bundle(config, target, rank)
        manifest = bundle(config, target, rank, path)
        return dict(rank=rank, alive=matches(process_argv(pid), target, rank),
                    global_step=manifest["global_step"])
    if operation == "verify-target":
        path = latest_bundle(config, target, rank)
        manifest = bundle(config, target, rank, path, checksums=True)
        snapshot = under(repo, f"runs/snapshots/{target}/step-{config['resume_step']:012d}/rank{rank}")
        old = bundle(config, target, rank, snapshot)
        ignored = {"checkpoint", "resume_state", "updates", "wandb_step_base", "wandb_step_base_timesteps"}
        differences = [key for key in old["config"] if key not in ignored
                       and manifest["config"].get(key) != old["config"][key]]
        if differences:
            raise ValueError(f"resumed configuration changed: {differences}")
        remaining_rounds = math.ceil((old["config"]["total_steps"] - config["resume_step"])
                                    / config["rollout_size"])
        if manifest["config"]["updates"] != remaining_rounds:
            raise ValueError("resumed run changed its remaining transition budget")
        maps = (Path("/proc") / str(int(under(repo, f"runs/{target}/rank{rank}.pid").read_text())) / "maps").read_text()
        if str(under(repo, config["build_dirs"][rank])) not in maps:
            raise ValueError("trainer did not load accepted native build")
        import torch
        def adam_steps(directory):
            with zipfile.ZipFile(directory / "model.zip") as archive:
                state = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")),
                                   map_location="cpu", weights_only=True)
            return {int(item["step"]) for item in state["state"].values() if "step" in item}
        original_steps, resumed_steps = adam_steps(snapshot), adam_steps(path)
        rounds = (manifest["global_step"] - config["resume_step"]) // config["rollout_size"]
        if len(original_steps) != 1 or resumed_steps != {next(iter(original_steps)) + rounds * 40}:
            raise ValueError("Adam did not resume and perform40 updates per round")
        learner = torch.load(path / "learner.pt", map_location="cpu", weights_only=False)
        if learner["global_step"] != manifest["global_step"] or learner["replay"]["size"] <= 0:
            raise ValueError("invalid resumed replay or learner step")
        rng = {name: learner.get(name) is not None
               for name in ("numpy_rng", "python_rng", "torch_rng", "cuda_rng")}
        if not all(rng.values()):
            raise ValueError("resumed learner has incomplete RNG state")
        return dict(rank=rank, global_step=manifest["global_step"], model_hashes=model_hashes(path),
                    config_matches=True, native_matches=True, adam_steps=sorted(resumed_steps),
                    replay_size=learner["replay"]["size"], rng=rng)
    raise ValueError("unknown rank operation")


def check_common(items):
    if len({item["global_step"] for item in items}) != 1 or len(
            {json.dumps(item["model_hashes"], sort_keys=True) for item in items}) != 1:
        raise ValueError("four-rank checkpoint/policy/Adam state differs")


def run_handoff(config, call, emit, *, now=lambda: datetime.now(timezone.utc), sleep=time.sleep):
    deadline = validate_config(config)
    snapshots = call("validate-snapshot")
    check_common(snapshots)
    emit("armed", deadline=config["deadline"], snapshots=snapshots)
    while now() < deadline:
        inspections = call("inspect")
        _, maximum = decide(inspections, config["threshold"])
        emit("waiting", maximum=maximum, source_alive=[row["alive"] for row in inspections])
        sleep(min(60, max(0, (deadline - now()).total_seconds())))
    inspections = call("inspect")
    switch, maximum = decide(inspections, config["threshold"])
    emit("decision", switch=switch, maximum=maximum, histories=inspections)
    if not switch:
        emit("kept", maximum=maximum)
        return "kept"
    # Revalidate before stopping; a corrupt/missing target never shuts down a healthy source.
    check_common(call("validate-snapshot"))
    emit("stop_requested", receipts=call("request-stop"))
    timeout = now().timestamp() + config["timeout_sec"]
    while True:
        post_stop = call("inspect")
        if not any(row.get("tree_alive", row["alive"]) for row in post_stop):
            break
        if now().timestamp() >= timeout:
            raise TimeoutError("source has not stopped; no force-kill or target launch")
        sleep(15)
    pinned = call("pin-current")
    check_common(pinned)
    emit("source_pinned", snapshots=pinned)
    emit("target_prepared", receipts=call("prepare-resume"))
    emit("target_launched", receipts=call("launch"))
    target_step = config["resume_step"] + config["rollout_size"]
    timeout = now().timestamp() + config["timeout_sec"]
    while True:
        targets = call("inspect-target")
        if not all(row["alive"] for row in targets):
            raise RuntimeError("a resumed target trainer exited")
        if all(row["global_step"] >= target_step for row in targets):
            break
        if now().timestamp() >= timeout:
            raise TimeoutError("resumed target has not published its first complete round")
        sleep(15)
    verified = call("verify-target")
    check_common(verified)
    emit("switched", maximum=maximum, target_step=target_step, verification=verified)
    return "switched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operation", choices=("inspect", "validate-snapshot", "request-stop",
                        "pin-current", "prepare-resume", "launch", "inspect-target", "verify-target"))
    parser.add_argument("--rank", type=int, choices=range(4))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.operation:
        if args.rank is None:
            parser.error("rank operation requires --rank")
        print(json.dumps(local_operation(config, args.rank, args.operation)))
        return

    def emit(status, **fields):
        row = dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **fields)
        line = json.dumps(row, allow_nan=False)
        events = under(config["repo"], config["events_file"])
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        state = under(config["repo"], config["state_file"])
        temporary = state.with_suffix(".tmp")
        temporary.write_text(line + "\n", encoding="utf-8")
        temporary.replace(state)
        print(line, flush=True)

    def worker(rank, operation):
        command = [str(Path(config["repo"]) / ".venv/bin/python"), str(Path(__file__).resolve()),
                   "--config", str(args.config.resolve()), "--operation", operation, "--rank", str(rank)]
        remote = ["ssh", "-i", config["ssh_key"], "-o", "BatchMode=yes", "-o",
                  "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", "-p", str(config["ports"][rank]),
                  f"root@{config['host']}", shlex.join(command)]
        for attempt in range(3):
            try:
                completed = subprocess.run(remote, text=True, capture_output=True, timeout=180)
                if completed.returncode == 0:
                    return json.loads(completed.stdout)
                error = completed.stderr.strip()
            except subprocess.TimeoutExpired:
                error = "SSH operation timed out"
            if attempt < 2:
                time.sleep(3)
        raise RuntimeError(f"rank{rank} {operation} failed: {error}")

    def call(operation):
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(lambda rank: worker(rank, operation), range(4)))

    # A durable singleton excludes a second coordinator, including accidental relaunches.
    import fcntl
    lock_path = under(config["repo"], config["state_file"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = under(config["repo"], config["state_file"])
        if state_path.exists():
            raise RuntimeError("coordinator state already exists; explicit new job identity required")
        try:
            run_handoff(config, call, emit)
        except Exception as error:
            emit("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
