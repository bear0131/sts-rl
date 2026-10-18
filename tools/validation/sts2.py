"""Build/inspect the pinned STS2 backend. Never modifies the STS1 environment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from env.sts2.backend import PIN, load_backend, sha256


def command(args, cwd, timeout=60):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=timeout).strip()


def build(source: Path, output: Path, timeout: int, jobs: int):
    source, output = source.resolve(), output.resolve()
    if command(["git", "rev-parse", "HEAD"], source) != PIN["commit"]:
        raise RuntimeError("Wrong source commit; checkout the backend.json commit first")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], source):
        raise RuntimeError("Tracked upstream source is modified; refusing an unrecorded build")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ, RUSTUP_TOOLCHAIN=PIN["rust_version"],
               PYO3_PYTHON=sys.executable, CARGO_BUILD_JOBS=str(jobs))
    with (output / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(["cargo", "build", "--locked", "--release", "-p", PIN["crate"],
                        "--features", "pyo3/extension-module"],
                       cwd=source / "sts2-rs", env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", source / "sts2-rs/target")) / "release"
    built_name = "sts2_rs.dll" if sys.platform == "win32" else "libsts2_rs.dylib" if sys.platform == "darwin" else "libsts2_rs.so"
    binary = output / ("sts2_rs.pyd" if sys.platform == "win32" else "sts2_rs.abi3.so")
    shutil.copy2(target / built_name, binary)
    spec = importlib.util.spec_from_file_location("sts2_rs", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = dict(pin=PIN, binary=binary.name, binary_sha256=sha256(binary),
                    source_hash=module.src_hash(), build_info=module.build_info(),
                    python=sys.version, build_seconds=time.monotonic()-started)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--timeout", type=int, default=1200)
    builder.add_argument("--jobs", type=int, default=4)
    sub.add_parser("doctor")
    args = parser.parse_args()
    if args.command == "build":
        if args.timeout <= 0 or args.jobs <= 0:
            parser.error("timeout and jobs must be positive")
        build(args.source, args.output, args.timeout, args.jobs)
    else:
        module, manifest = load_backend()
        print(json.dumps(dict(manifest=manifest, loaded=module.__file__), indent=2))


if __name__ == "__main__":
    main()
