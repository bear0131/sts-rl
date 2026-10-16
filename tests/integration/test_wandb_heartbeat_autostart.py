"""Linux /proc + real offline W&B automatic startup/restart contract.

Runs two bounded logging-only processes, never a model or production run.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform.startswith('linux'), 'automatic discovery requires Linux /proc')
class HeartbeatAutostartTest(unittest.TestCase):
    def test_delayed_init_restart_and_trainer_exit(self):
        trainer_code = r'''
import json, pathlib, sys, time, wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore
root, health, result = map(pathlib.Path, sys.argv[1:4])
time.sleep(1)  # Launcher starts its helper before wandb-core exists.
r = wandb.init(id='ops02-offline', entity='offline-entity', project='offline-project',
               mode='offline', dir=str(root),
               settings=wandb.Settings(console='off', x_disable_stats=True, disable_code=True))
r.define_metric('legacy_step')
r.define_metric('train/*', step_metric='legacy_step')
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    if health.exists() and '"event": "heartbeat"' in health.read_text(): break
    time.sleep(.05)
else: raise RuntimeError('helper never attached')
for step in [1000, 2000]:
    r.log({'global_step':step, 'legacy_step':step // 1000, 'train/value':step / 1000})
    print(json.dumps({'global_timesteps':step, 'x/finite_fraction':1}), flush=True)
r.finish()
d = DataStore()
d.open_for_scan(str(pathlib.Path(r.dir).parent / f'run-{r.id}.wandb'))
rows = []
while True:
    data = d.scan_data()
    if data is None: break
    record = pb.Record(); record.ParseFromString(data)
    if record.HasField('history'):
        rows.append({(i.key or '.'.join(i.nested_key)):json.loads(i.value_json)
                     for i in record.history.item})
result.write_text(json.dumps(rows))
'''
        helper_code = r'''
import json
from types import SimpleNamespace
from tools.operations import wandb_heartbeat as hb
# Replace only cloud credentials/status. Core discovery, attachment, logging,
# startup/progress guards and process lifetime are the actual production code.
hb.netrc.netrc = lambda: SimpleNamespace(authenticators=lambda host: ('api', None, 'offline-placeholder'))
def offline_status(self):
    assert self.attached.settings.mode == 'offline'
    assert self.entity == 'offline-entity'
    return {'state':'running'}
hb.Client.status = offline_status
try: hb.main()
except hb.GuardError as error:
    print(json.dumps({'event':'stop', 'reason':str(error)}), flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer_script, helper_script = root / 'trainer.py', root / 'helper.py'
            trainer_script.write_text(trainer_code)
            helper_script.write_text(helper_code)
            log = root / 'rank0.log'
            log.write_text('{"global_timesteps":9000}\n')
            env = os.environ.copy()
            env.pop('WANDB_SERVICE', None)
            env.pop('_WANDB_SERVICE', None)
            env['PYTHONPATH'] = str(ROOT)
            pids = []
            for incarnation in range(2):
                health = root / f'health-{incarnation}.jsonl'
                result = root / f'result-{incarnation}.json'
                offset = log.stat().st_size
                with log.open('ab', buffering=0) as output, health.open('ab', buffering=0) as health_out:
                    trainer = subprocess.Popen([sys.executable, '-u', str(trainer_script),
                        str(root), str(health), str(result), '--wandb-id', 'ops02-offline',
                        '--wandb-project', 'offline-project'], stdout=output, stderr=output, env=env)
                    helper = subprocess.Popen([sys.executable, '-u', str(helper_script), '--auto',
                        '--pid', str(trainer.pid), '--wandb-id', 'ops02-offline',
                        '--project', 'offline-project', '--log', str(log), '--log-start', str(offset),
                        '--startup-timeout', '25', '--max-seconds', '35'],
                        stdout=health_out, stderr=health_out, env=env)
                    try:
                        self.assertEqual(trainer.wait(timeout=30), 0, log.read_text() + health.read_text())
                        self.assertEqual(helper.wait(timeout=5), 0, health.read_text())
                    finally:
                        for child in [helper, trainer]:
                            if child.poll() is None:
                                child.kill()
                            child.wait(timeout=5)
                pids.append(trainer.pid)
                records = [json.loads(line) for line in health.read_text().splitlines() if line.startswith('{')]
                beats = [x for x in records if x['event'] == 'heartbeat']
                self.assertEqual(len(beats), 1, records)
                self.assertEqual(beats[0]['step'], -1)  # Neither old 9000 nor previous 2000.
                self.assertEqual(records[-1]['event'], 'stop')
                rows = json.loads(result.read_text())
                self.assertEqual([x['global_step'] for x in rows if 'global_step' in x], [1000, 2000])
                self.assertEqual([x['legacy_step'] for x in rows if 'global_step' in x], [1, 2])
                health_rows = [x for x in rows if 'ops/heartbeat_unix' in x]
                self.assertEqual(len(health_rows), 1)
                self.assertNotIn('legacy_step', health_rows[0])
            self.assertNotEqual(*pids)


if __name__ == '__main__':
    unittest.main()
