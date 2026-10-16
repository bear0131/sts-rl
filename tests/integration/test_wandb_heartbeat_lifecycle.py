"""OPS02 shared-core lifecycle proof; run with the training W&B SDK installed.

Can also run directly with Python (unittest) without the native test fixtures.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WandbHeartbeatLifecycleTest(unittest.TestCase):
    def test_attached_helper_exit_preserves_training_records_and_parent(self):
        repo = Path(__file__).resolve().parents[2]
        # Offline only: no API credentials or network run, no model or trainer.
        script = r'''
import os, sys, pathlib, json, subprocess
import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore
assert wandb.__version__ == '0.28.1', 'OPS02 supported SDK is 0.28.1'
r = wandb.init(project='ops02-offline-test', mode='offline', dir=sys.argv[1],
               settings=wandb.Settings(console='off', disable_code=True, x_disable_stats=True))
r.define_metric('legacy_step')
r.define_metric('train/*', step_metric='legacy_step')
r.log({'legacy_step': 677, 'global_step': 532414464, 'train/value': .123})
child = """
import sys, wandb
from tools.operations.wandb_heartbeat import Client
r = wandb.attach(sys.argv[1])
r.define_metric('ops/heartbeat_unix', hidden=True, summary='none')
client = object.__new__(Client)
client.attached = r
for _ in range(3): client.heartbeat()
# Deliberately no finish() on the attached client.
"""
p = subprocess.run([sys.executable, '-c', child, r.id], env=os.environ.copy(),
                   capture_output=True, text=True, timeout=30)
assert p.returncode == 0, p.stderr
# This succeeds only if child shutdown leaves the original service alive.
r.log({'legacy_step': 678, 'global_step': 533200896, 'train/value': .456})
summary = dict(r.summary)
r.finish()
assert summary['global_step'] == 533200896
assert summary['train/value'] == .456
assert 'ops/heartbeat_unix' not in summary
d = DataStore()
d.open_for_scan(str(pathlib.Path(r.dir).parent / f'run-{r.id}.wandb'))
history = []
while True:
    data = d.scan_data()
    if data is None: break
    record = pb.Record()
    record.ParseFromString(data)
    if record.HasField('history'):
        history.append({(i.key or '.'.join(i.nested_key)): json.loads(i.value_json)
                        for i in record.history.item})
training = [x for x in history if 'train/value' in x]
assert [x['legacy_step'] for x in training] == [677, 678]
assert [x['global_step'] for x in training] == [532414464, 533200896]
assert [x['train/value'] for x in training] == [.123, .456]
assert len(history) == 5
assert [x['_step'] for x in history] == list(range(5))
assert all('global_step' not in x and 'legacy_step' not in x
           for x in history if 'ops/heartbeat_unix' in x)
print('PASS: 2 intact training records, 3 health records, parent owns finish')
'''
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop('WANDB_SERVICE', None)
            env.pop('_WANDB_SERVICE', None)
            env['PYTHONPATH'] = str(repo)
            result = subprocess.run([sys.executable, '-c', script, directory], cwd=repo,
                                    env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS:', result.stdout)


if __name__ == '__main__':
    unittest.main()
