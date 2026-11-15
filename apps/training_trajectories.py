"""Read-only checkpoint trajectories for the training monitor.

The HTTP process uses only the standard library. Journal replay and image
thumbnails run in a bounded local worker with the configured simulator Python.
No policy, optimizer, evaluation or training process is started here.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class TrajectoryFeed:
    def __init__(self, config):
        self.config = config or {}
        self.directory = Path(self.config['directory']) if self.config else None

    def snapshot(self, run_id):
        value = dict(run_id=run_id, rows=[], error=None, updated_at=None)
        if self.directory is None:
            value['error'] = '此监控尚未配置轨迹数据'
        else:
            try:
                saved = json.loads((self.directory / 'state.json').read_text(encoding='utf-8'))
                if saved['run_id'] == run_id:
                    value = saved
                else:
                    value['error'] = '当前训练已切换，正在等待对应轨迹'
            except (OSError, ValueError, KeyError):
                value['error'] = '正在读取当前训练的轨迹，首次加载需要一点时间'
            try:
                status = json.loads((self.directory / 'status.json').read_text(encoding='utf-8'))
                if status.get('run_id') == run_id:
                    value['refreshing'] = status.get('refreshing', False)
                    if status.get('error'):
                        value['error'] = status['error']
            except (OSError, ValueError):
                pass
        value['stale'] = time.time() - (value.get('captured_at') or 0) > 900
        return json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')

    def asset(self, name):
        # Only generated content-addressed thumbnails, never arbitrary paths.
        if self.directory is None or len(name) != 68 or not name.endswith('.png'):
            return None
        if any(c not in '0123456789abcdef' for c in name[:-4]):
            return None
        try:
            return (self.directory / 'assets' / name).read_bytes()
        except OSError:
            return None

    def watch(self, monitor):
        if not self.config.get('collect', False):
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        while not monitor.stop.is_set():
            config = dict(monitor.config, trajectories=self.config)
            status = dict(run_id=config['run_id'], refreshing=True, error=None)
            try:
                write_json(self.directory / 'status.json', status)
                command = [self.config['python'], '-c',
                    'import json,sys; from apps.training_trajectories import collect; '
                    'collect(json.load(sys.stdin))']
                environment = dict(os.environ, **self.config.get('environment', {}))
                environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1', PYTHONUTF8='1')
                child = subprocess.Popen(command, cwd=self.config['cwd'], env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    output, errors = child.communicate(json.dumps(config).encode(), timeout=240)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                       capture_output=True, timeout=15)
                    child.kill()
                    child.communicate(timeout=15)
                    raise RuntimeError('轨迹提取超过240秒，稍后自动重试') from None
                if child.returncode:
                    # Detailed diagnostics stay local, not in the public API.
                    (self.directory / 'worker-error.log').write_bytes(errors[-16000:] + output[-4000:])
                    raise RuntimeError('轨迹读取失败，保留上次快照并自动重试')
            except Exception as exc:
                status['error'] = str(exc) if isinstance(exc, RuntimeError) else '轨迹同步异常，稍后自动重试'
            finally:
                status.update(refreshing=False, checked_at=time.time())
                try:
                    write_json(self.directory / 'status.json', status)
                except OSError:
                    pass
            monitor.stop.wait(60)


def remote(config, rank, code):
    script = 'cd ' + shlex.quote(config['root']) + "\n.venv/bin/python - <<'PY_TRAJECTORIES'\n"
    script += 'from pathlib import Path\nimport json,sys\n'
    script += f"run=Path('runs')/{config['run_id']!r}\nrank={rank}\n" + code
    script += '\nPY_TRAJECTORIES\n'
    hosts = config.get('hosts') or [config['host']] * len(config['ports'])
    args = ['ssh', '-i', config['ssh_key'], '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=10']
    if config.get('ssh_known_hosts'):
        args += ['-o', 'UserKnownHostsFile=' + config['ssh_known_hosts']]
    args += ['-p', str(config['ports'][rank]), hosts[rank], 'bash -se']
    result = subprocess.run(args, input=script.encode(), capture_output=True, timeout=35)
    if result.returncode:
        raise RuntimeError(f'rank {rank}: snapshot SSH failed ({result.returncode})')
    return result.stdout


def replay_bundle(environment_path, manifest, observation_flags):
    """Use the original restore path, including its observation/mask hashes."""
    import torch
    from env.core.run_env import SlayTheSpireEnv

    torch.set_num_threads(1)
    raw = Path(environment_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']['environment.pt']:
        raise ValueError('environment checkpoint SHA mismatch')
    import io
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if len(payload['states']) != manifest['config']['n_envs']:
        raise ValueError('environment count mismatch')
    rows = []
    for index, wrapped in enumerate(payload['states']):
        saved = wrapped.get('base', wrapped)
        if saved.get('backend_identity') is not None:
            raise ValueError('special opening datasets need their matching backend')
        env = SlayTheSpireEnv(**observation_flags,
            ascension=manifest['config']['ascension'],
            combat_action_contract=saved['combat_action_contract'],
            combat_rules=saved.get('combat_rules', 'standard-v1'),
            combat_exit_anchor=saved.get('combat_exit_anchor', False),
            deck_latent_context=saved.get('deck_latent_context', False),
            reward_flow=saved.get('reward_flow', 'full'))
        try:
            env.restore_training_state(saved)
            canonical = env.thick.obs_canonical()
            state = canonical.get('global', canonical)
            # Preserve canonical IDs, upgrade counts, duplicates and relic order.
            row = {key: state.get(key) for key in ('act', 'floor', 'hp', 'max_hp', 'gold', 'deck', 'relics')}
            row.update(rank=manifest['rank'], slot=index,
                seed=str(saved['backend_seed']), phase=canonical.get('phase'),
                terminal=bool(saved.get('terminal')), actions=len(saved['actions']))
            if row['deck'] is None or row['relics'] is None:
                raise ValueError('canonical inventory missing')
            rows.append(row)
        finally:
            env.close()
    return rows


def collect(config):
    """Internal worker; download only environment journals from a common round."""
    settings = config['trajectories']
    directory = Path(settings['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def manifests(rank):
        data = json.loads(remote(config, rank,
            "out=[]\nfor p in (run/'recovery'/f'rank{rank}').glob('step-*/manifest.json'):\n"
            " try:\n  m=json.loads(p.read_text()); m['captured_at']=p.stat().st_mtime; out.append(m)\n"
            " except FileNotFoundError: pass\nprint(json.dumps(out))"))
        return {m['global_step']: m for m in data if m['run_id'] == config['run_id'] and m['rank'] == rank}
    with concurrent.futures.ThreadPoolExecutor(len(config['ports'])) as pool:
        available = list(pool.map(manifests, range(len(config['ports']))))
    common = set.intersection(*(set(m) for m in available))
    if not common:
        raise RuntimeError('no common complete recovery generation')
    step = max(common)
    old = directory / 'state.json'
    if old.exists():
        saved = json.loads(old.read_text(encoding='utf-8'))
        if saved['run_id'] == config['run_id'] and saved['global_step'] == step:
            return
    def download(rank):
        manifest = available[rank][step]
        raw = remote(config, rank,
            f"sys.stdout.buffer.write((run/'recovery'/f'rank{{rank}}'/'step-{step:012d}'/'environment.pt').read_bytes())")
        path = directory / f'environment-rank{rank}.pt'
        path.write_bytes(raw)
        return path, manifest
    with concurrent.futures.ThreadPoolExecutor(len(config['ports'])) as pool:
        bundles = list(pool.map(download, range(len(config['ports']))))
    rows = []
    for path, manifest in bundles:
        rows.extend(replay_bundle(path, manifest, settings['observation_flags']))
    from apps.trajectory_assets import build_catalog
    catalog = build_catalog(settings['game_jar'], directory / 'assets', rows)
    write_json(old, dict(schema_version=1, run_id=config['run_id'],
        label=config.get('label', config['run_id']), kind='checkpoint_live',
        global_step=step, rollout_round=bundles[0][1]['rollout_round'],
        captured_at=min(m['captured_at'] for _, m in bundles), updated_at=time.time(),
        extraction_seconds=time.monotonic()-started, rows=rows, catalog=catalog))
