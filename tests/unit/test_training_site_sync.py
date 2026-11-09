import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

from apps.training_site_sync import compact_rows, discover_training, public_snapshot, switch_monitor, upload, serve_state


class TrainingSiteSyncTests(unittest.TestCase):
    def test_forced_ssh_requests_expose_only_public_current_and_known_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text(json.dumps(dict(run_id='new', rows=[{'global_timesteps': 2}],
                ranks=[dict(rank=0, cursor={'offset': 99}, alive=True)],
                comparisons=[dict(run_id='old', rows=[{'global_timesteps': 1}])])), encoding='utf-8')
            current = json.loads(serve_state(path, 'current'))
            self.assertNotIn('cursor', current['ranks'][0])
            key = current['comparisons'][0]['history_key']
            self.assertEqual(json.loads(serve_state(path, 'history:' + key))['rows'], [{'global_timesteps': 1}])
            for request in ['echo should-not-run', '../state.json', 'history:' + '0' * 64]:
                with self.assertRaises(ValueError):
                    serve_state(path, request)

    def test_upload_sends_history_once_and_current_each_poll(self):
        class Opener:
            calls = []
            def open(self, request, timeout):
                self.calls.append((request.full_url, json.loads(request.data)))
                return BytesIO(b'{"ok":true,"received_at":123}')
        opener = Opener()
        state = dict(run_id='new', rows=[{'global_timesteps': 2}], ranks=[],
                     comparisons=[dict(run_id='old', rows=[{'global_timesteps': 1}])])
        cache = {}
        with patch('apps.training_site_sync.urllib.request.build_opener', return_value=opener):
            upload('https://monitor.example', 'test-only', state, cache)
            upload('https://monitor.example', 'test-only', state, cache)
        self.assertEqual([url.rsplit('/', 1)[1] for url, _ in opener.calls],
                         ['ingest-history', 'ingest', 'ingest'])
        historical = opener.calls[-1][1]['state']['comparisons'][0]
        self.assertEqual(historical['rows'], [])
        self.assertEqual(historical['row_count'], 1)
        self.assertEqual(len(historical['history_key']), 64)
        with self.assertRaises(ValueError):
            upload('http://monitor.example', 'test-only', state)

    def test_projection_preserves_chart_values_and_missing_data(self):
        row = {'global_timesteps': 10, 'pure_rl/value_loss': .125,
               'upper/value_mc_mse_pre_update': .25, 'upper/policy_loss_pre_update': 0.,
               'upper/kl_epoch_2': .01, 'upper/campfire_rest_choice_rate': None,
               'upper/reward_relic_pickup_rate': .8, 'debug_dump': ['private'],
               'secret_token': 'not-exported', 'rollout_resources': {'tree_cpu_percent_mean': 321}}
        state = {'run_id': 'c45', 'rows': [row], 'comparisons': [],
                 'ranks': [{'rank': 0, 'alive': True, 'cursor': {'offset': 5},
                            'connection_error': 'hostname and internal path'}]}
        out = public_snapshot(state)
        for key in ['pure_rl/value_loss', 'upper/value_mc_mse_pre_update',
                    'upper/policy_loss_pre_update', 'upper/kl_epoch_2',
                    'upper/campfire_rest_choice_rate', 'upper/reward_relic_pickup_rate']:
            self.assertEqual(out['rows'][0][key], row[key])
        self.assertNotIn('debug_dump', out['rows'][0])
        self.assertNotIn('secret_token', json.dumps(out))
        self.assertNotIn('cursor', out['ranks'][0])
        self.assertNotIn('hostname', json.dumps(out))
        self.assertEqual(out['resources']['rollout_resources']['tree_cpu_percent_mean'], 321)
        self.assertIsNone(compact_rows([{'pure_rl/value_loss': float('nan')}])[0]['pure_rl/value_loss'])

    def test_switch_preserves_history_and_resuming_restores_the_original_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            config = dict(run_id='old', root='/old', iteration_steps=10, ports=[1])
            path.write_text(json.dumps(dict(**config, rows=[{'global_timesteps': 10}], ranks=[],
                comparisons=[dict(run_id='first', rows=[{'global_timesteps': 5}])], label='old label')))
            new = dict(root='/new', run_id='new', iteration_steps=10, label='new label')
            monitor = switch_monitor(config, path, 30, new)
            self.assertEqual(monitor.state['rows'], [])
            self.assertEqual([h['run_id'] for h in monitor.state['comparisons']], ['first', 'old'])
            monitor.state['rows'] = [{'global_timesteps': 20}]
            path.write_text(json.dumps(monitor.state))
            restored = switch_monitor(config, path, 30, config)
            self.assertEqual(restored.state['rows'], [{'global_timesteps': 10}])
            self.assertEqual([h['run_id'] for h in restored.state['comparisons']], ['first', 'new'])

    def test_discovery_requires_a_matching_live_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = root / 'job'; job.mkdir()
            run = job / 'runs' / 'combat45'; run.mkdir(parents=True)
            proc = root / 'proc' / '123'; (proc / 'cwd').mkdir(parents=True)
            (run / 'rank0.pid').write_text('123')
            args = job / 'config.args'; args.write_text('--n-envs\n256\n--n-steps\n768\n')
            plan = dict(rank=0, continuous=True, run=str(run), cwd=str(proc),
                        argv=['python', '@' + str(args), '--world-size', '4'])
            (job / 'training-plan.json').write_text(json.dumps(plan))
            (proc / 'cmdline').write_bytes(b'python\0-m\0rl.training.distributed_sb3\0--run-id\0combat45\0')
            (proc / 'stat').write_text('123 (python) S ' + '0 ' * 18 + '999 0')
            target = discover_training(root, proc_root=root / 'proc')
            self.assertEqual(target['run_id'], 'combat45')
            self.assertEqual(target['iteration_steps'], 786432)
            (proc / 'cmdline').write_bytes(b'python\0unrelated-job\0')
            self.assertIsNone(discover_training(root, proc_root=root / 'proc'))
            (proc / 'cmdline').unlink()
            self.assertIsNone(discover_training(root, proc_root=root / 'proc'))


if __name__ == '__main__':
    unittest.main()
