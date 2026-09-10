"""Backfill and live-stream a running v7 trainer log to Weights & Biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


TRAIN_RE = re.compile(
    r"step=(?P<step>\d+)/(?P<limit>\d+) loss=(?P<loss>[-+0-9.eE]+) "
    r"token=(?P<token>[-+0-9.eE]+) delta=(?P<delta>[-+0-9.eE]+) "
    r"acc=(?P<acc>[-+0-9.eE]+) seq_max=(?P<seq>\d+) "
    r"model_input_max=(?P<input>\d+) unknown=(?P<unknown>\d+) "
    r"samples=(?P<samples>\d+)"
)
VALIDATION_RE = re.compile(
    r"step=(?P<step>\d+)/(?P<limit>\d+) "
    r"validation_loss=(?P<loss>[-+0-9.eE]+) "
    r"validation_token=(?P<token>[-+0-9.eE]+) "
    r"validation_acc=(?P<acc>[-+0-9.eE]+)"
)


def _gpu_metrics() -> dict[str, float]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=3,
        ).strip().splitlines()[0]
        util, used, total = (float(value.strip()) for value in output.split(","))
        return {
            "system/gpu_utilization_pct": util,
            "system/gpu_memory_used_mb": used,
            "system/gpu_memory_total_mb": total,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def _alive(pid: int | None) -> bool:
    if pid is None:
        return True
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip())


def _payload(line: str):
    match = TRAIN_RE.search(line)
    if match:
        values = match.groupdict()
        return int(values["step"]), {
            "train/loss": float(values["loss"]),
            "train/token_loss": float(values["token"]),
            "train/delta_loss": float(values["delta"]),
            "train/token_accuracy": float(values["acc"]),
            "train/sequence_length_max": int(values["seq"]),
            "train/model_input_length_max": int(values["input"]),
            "train/unknown_tokens": int(values["unknown"]),
            "train/samples": int(values["samples"]),
        }
    match = VALIDATION_RE.search(line)
    if match:
        values = match.groupdict()
        return int(values["step"]), {
            "validation/loss": float(values["loss"]),
            "validation/token_loss": float(values["token"]),
            "validation/token_accuracy": float(values["acc"]),
        }
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--project", default="sts-rl-vector-v7")
    parser.add_argument("--entity")
    parser.add_argument("--name", default="vector-rule-v7-training-20260831")
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def main(args: argparse.Namespace) -> None:
    import wandb

    config = {"schema_version": 7, "log": str(args.log), "pid": args.pid}
    if args.out is not None:
        metadata_path = args.out / "metadata.json"
        model_config_path = args.out / "config.json"
        for path in (metadata_path, model_config_path):
            if path.exists():
                try:
                    config[path.stem] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
    run = wandb.init(
        project=args.project, entity=args.entity, name=args.name, id=args.run_id,
        resume="allow" if args.run_id else None, config=config,
    )
    offset = 0
    seen_steps: set[tuple[int, str]] = set()
    try:
        while True:
            with args.log.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                lines = stream.readlines()
                offset = stream.tell()
            for line in lines:
                parsed = _payload(line)
                if parsed is None:
                    continue
                step, payload = parsed
                key = (step, "validation" if "validation/loss" in payload else "train")
                if key in seen_steps:
                    continue
                seen_steps.add(key)
                payload.update(_gpu_metrics())
                wandb.log(payload, step=step)
            if args.pid is not None and not _alive(args.pid):
                # One final read catches a last checkpoint/log line flushed at exit.
                with args.log.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(offset)
                    for line in stream:
                        parsed = _payload(line)
                        if parsed is None:
                            continue
                        step, payload = parsed
                        key = (step, "validation" if "validation/loss" in payload else "train")
                        if key not in seen_steps:
                            seen_steps.add(key)
                            payload.update(_gpu_metrics())
                            wandb.log(payload, step=step)
                break
            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        run.finish()


if __name__ == "__main__":
    main(build_parser().parse_args())
