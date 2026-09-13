"""Compare a real checkpoint's bounded strict-search evaluation across checkouts.

Both children use the same interpreter and explicit STS_LIGHTSPEED_BUILD_DIR.
No training, remote access, or checkpoint mutation. A timed-out child is killed
by subprocess.run; this evaluator creates no worker processes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not os.environ.get("STS_LIGHTSPEED_BUILD_DIR"):
        parser.error("set an explicit shared STS_LIGHTSPEED_BUILD_DIR")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve(strict=True)
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-c",
               "import torch,runpy; torch.set_num_threads(1); "
               "torch.use_deterministic_algorithms(True); "
               "runpy.run_module('rl.eval_full_gumbel_bash_clash',run_name='__main__')",
               str(checkpoint), "--seeds", "219", "--repeats", "1", "--simulations", "8",
               "--depth", "0", "--leaf-batch", "1", "--modes", "full", "--paper-full-gumbel"]
    manifests = {}
    try:
        for name, repo in (("baseline", args.baseline_repo), ("candidate", args.candidate_repo)):
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command + ["--output", str(output / f"{name}.json")],
                               cwd=repo.resolve(strict=True), env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
            manifests[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except subprocess.TimeoutExpired:
        raise SystemExit(124)
    a, b = ((output / f"{name}.json").read_bytes() for name in ("baseline", "candidate"))
    manifests.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     output_sha256=hashlib.sha256(a).hexdigest(), equal=a == b,
                     native_build=os.environ["STS_LIGHTSPEED_BUILD_DIR"], device="cpu", threads=1)
    (output / "comparison.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    if a != b:
        raise SystemExit("evaluation JSON bytes differ; inspect both outputs")
    print(f"EVAL BITWISE PASS: {len(a)} identical JSON bytes; {output}")


if __name__ == "__main__":
    main()
