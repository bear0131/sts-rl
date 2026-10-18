"""Load only a hash-verified, explicitly selected native extension."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PIN = json.loads(Path(__file__).with_name("backend.json").read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_backend():
    manifest_path = os.environ.get("STS2_BACKEND_MANIFEST")
    if not manifest_path:
        raise RuntimeError("Set STS2_BACKEND_MANIFEST to the manifest made by tools.validation.sts2 build")
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("pin") != PIN:
        raise RuntimeError("STS2 backend pin mismatch; rebuild the frozen backend")
    binary = (path.parent / manifest["binary"]).resolve(strict=True)
    if sha256(binary) != manifest["binary_sha256"]:
        raise RuntimeError("STS2 extension SHA256 mismatch")
    module = sys.modules.get("sts2_rs")
    if module is not None and Path(module.__file__).resolve() != binary:
        raise RuntimeError("Another sts2_rs extension is already loaded; start a fresh process")
    if module is None:
        spec = importlib.util.spec_from_file_location("sts2_rs", binary)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load native extension {binary}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["sts2_rs"] = module
    if module.src_hash() != manifest["source_hash"]:
        raise RuntimeError("STS2 embedded source hash mismatch")
    return module, manifest
