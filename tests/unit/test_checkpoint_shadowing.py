"""An in-place deployment must not resurrect a retired flat implementation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_legacy_checkpoint_alias_precedes_stale_flat_file(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    shadow = tmp_path/'rl'
    shadow.mkdir()
    shutil.copyfile(repo/'rl/__init__.py',shadow/'__init__.py')
    shutil.copytree(repo/'rl/_compat',shadow/'_compat',ignore=shutil.ignore_patterns('__pycache__'))
    (shadow/'split_policy.py').write_text('raise RuntimeError("retired implementation loaded")\n')
    script = (
        'import rl; rl.__path__.append('+repr(str(repo/'rl'))+'); '
        'import rl.split_policy as legacy; import rl.policies.split_policy as canonical; '
        'assert legacy is canonical; '
        'assert legacy.SplitStSPointerPolicy is canonical.SplitStSPointerPolicy'
    )
    environment = {**os.environ,'PYTHONPATH':os.pathsep.join([str(tmp_path),str(repo)])}
    subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=environment,
                   capture_output=True,text=True,check=True,timeout=30)
