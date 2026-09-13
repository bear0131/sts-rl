"""Guard dependency direction and old checkpoint/CLI entry points."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = json.loads((ROOT / "tools/validation/migrations.json").read_text(encoding="utf-8"))["modules"]


def test_runtime_does_not_depend_on_apps_tools_tests_or_legacy_modules():
    violations = []
    for package in ("rl", "env"):
        for path in (ROOT / package).rglob("*.py"):
            if "_compat" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    names.extend(f"{node.module}.{n.name}" for n in node.names)
                for name in names:
                    if name.split(".")[0] in {"apps", "tools", "tests"} or name in MIGRATION:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
                    if package == "env" and name.startswith("rl."):
                        violations.append(f"env imports learning runtime: {name}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("legacy", [
    "rl.accumulating_ppo", "rl.model", "rl.pointer_policy", "rl.split_policy",
    "rl.semantic_reasoning_policy", "rl.rule_model", "rl.rule_vector_model",
    "rl.nosl_is_az", "rl.nosl_replay", "rl.nosl_sampler", "rl.distributed_sb3",
    "rl.eval", "rl.dashboard", "rl.watch",
])
def test_legacy_import_is_same_module_object(legacy):
    assert importlib.import_module(legacy) is importlib.import_module(MIGRATION[legacy])


@pytest.mark.parametrize("module", ["rl.distributed_sb3", "rl.eval", "rl.dashboard"])
def test_legacy_cli_still_exposes_parser(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_source_and_test_files_stay_in_their_packages():
    assert not list((ROOT / "rl").glob("test_*.py"))
    assert not list((ROOT / "env").rglob("test_*.py"))
    assert {p.name for p in (ROOT / "rl").glob("*.py")} == {"__init__.py"}
    for legacy, canonical in MIGRATION.items():
        assert (ROOT / (canonical.replace(".", "/") + ".py")).is_file()
        assert (ROOT / "rl/_compat" / (legacy.split(".")[-1] + ".py")).is_file()


def test_historical_artifacts_keep_their_original_bytes():
    from tools.validation.locate_legacy import verify_archive
    assert verify_archive(ROOT) > 0
