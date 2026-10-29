"""Provisioning and bounded supervision behind the existing NOSL launchers.

This module has no CLI or training algorithm. The PowerShell manager calls
manage(); the shell launcher calls remote_start().
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def run_ssh_script(command, script, timeout):
    # Windows text-mode pipes translate LF to CRLF, which becomes literal CR
    # in POSIX paths and commands. Keep the wire protocol UTF-8/LF on all hosts.
    result = subprocess.run(command, input=script.replace('\r\n', '\n').encode('utf-8'),
                            capture_output=True, timeout=timeout)
    result.stdout = result.stdout.decode('utf-8', errors='replace')
    result.stderr = result.stderr.decode('utf-8', errors='replace')
    return result


def validate(config, *, connected=False):
    if config['kind'] != 'combat01-deployment-v1' or not re.fullmatch(r'[A-Za-z0-9_-]+', config['run_id']):
        raise ValueError('invalid COMBAT01 deployment identity')
    if config['profile'] not in ('configs/train/combat_outer_return_876_4x5090.json',
                                 'configs/train/combat_normalized_outer_return_876_4x5090.json',
                                 'configs/train/combat_mixed_raw_outer_return_876_4x5090.json',
                                 'configs/train/combat_actual_run_return_876_4x5090.json',
                                 'configs/train/combat_finite_curves_876_4x5090.json',
                                 'configs/train/combat_two_means_876_4x5090.json',
                                 'configs/train/combat_train47_context_876_4x5090.json'):
        raise ValueError('this launcher requires the COMBAT01 production profile')
    if not re.fullmatch(r'[0-9a-f]{64}', config['material_sha256']):
        raise ValueError('material SHA256 required')
    if config.get('resume_step', 0):
        raise ValueError('fresh deployment only; resume a verified common generation through the trainer')
    budget = config['budget']
    continuous = config.get('continuous', False)
    if type(continuous) is not bool:
        raise ValueError('continuous must be an explicit boolean')
    if continuous:
        if budget['rounds'] is not None or budget['max_seconds'] is not None:
            raise ValueError('continuous deployment requires null round and wall-time limits')
    elif any(type(budget[k]) is not int or budget[k] < 1 for k in ('rounds', 'max_seconds')):
        raise ValueError('explicit positive round and wall-time budgets are required')
    if type(budget['seed']) is not int or budget['seed'] < 0:
        raise ValueError('an explicit nonnegative seed is required')
    if config.get('minimum_free_disk_gib', 30) < 14:
        raise ValueError('retain at least 14 GiB for two generations plus an atomic save and reserve')
    if len(config['ranks']) != 4 or [r['rank'] for r in config['ranks']] != list(range(4)):
        raise ValueError('exactly ranks 0..3 are required')
    if config.get('wandb_id') and not re.fullmatch(r'[A-Za-z0-9_-]+', config['wandb_id']):
        raise ValueError('invalid W&B id')
    if connected:
        root = config.get('remote_repo') or ''
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', root) or '..' in root.split('/') or len(Path(root).parts) < 4:
            raise ValueError('remote_repo must be a dedicated new absolute directory')
        if not 1024 <= int(config.get('ring_port') or 0) <= 65535:
            raise ValueError('set the current ring_port')
        for rank in config['ranks']:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', rank.get('host') or '') or not 1 <= int(rank.get('port') or 0) <= 65535:
                raise ValueError('fill current machine host/port for every rank; no historical fallback')
        if len({(r['host'], r['port']) for r in config['ranks']}) != 4:
            raise ValueError('four distinct SSH endpoints are required')
    return config


def training_command(config, profile, rank):
    """Pure argument builder, exercised without SSH by deployment tests."""
    root = Path(config['remote_repo'])
    run = root/'runs'/config['run_id']
    t, budget = profile['training'], config['budget']
    cmd = [str(root/'.venv/bin/python'), '-u', '-m', 'rl.training.distributed_sb3',
        '--rank', str(rank), '--world-size', '4', '--run-id', config['run_id'],
        '--checkpoint', str(root/'material/initial.zip'), '--output', str(run/f'rank{rank}.zip'),
        '--updates', str(budget['rounds']), '--n-envs', str(t['n_envs_per_rank']), '--n-steps', str(t['n_steps']),
        '--seed', str(budget['seed']), '--device', 'cuda', '--vec', 'native',
        '--native-search-threads', str(t['native_search_threads']), '--learning-rate', str(t['learning_rate']),
        '--max-grad-norm', str(t['max_grad_norm']), '--gamma', '1', '--gae-lambda', '1',
        '--amp-mode', t['lower_amp'], '--float32-matmul-precision', t['float32_matmul_precision'],
        '--nosl-is-az-simulations', str(profile['search']['simulations']),
        '--nosl-is-az-batch-size', str(t['batch_size_per_rank']),
        '--nosl-is-az-micro-batch-size', str(t['micro_batch_size']),
        '--nosl-is-az-updates', str(t['optimizer_updates_per_round']), '--nosl-is-az-update-schedule', 'constant',
        '--nosl-is-az-policy-coef', str(t['policy_loss_coefficient']), '--nosl-is-az-value-coef', str(t['value_loss_coefficient']),
        '--nosl-is-az-replay-capacity', str(t['replay']['capacity_per_rank']),
        '--nosl-is-az-replay-min-size', str(t['replay']['min_size']),
        '--nosl-is-az-replay-max-old-fraction', str(t['replay']['max_old_fraction']),
        '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning',
        '--leaf-inference', 'cuda-graph', '--leaf-graph-batch-multiple', str(t['leaf_graph_batch_multiple']),
        '--rollout-resource-interval', str(t['resource_interval_seconds']),
        '--training-state-dir', str(run/'recovery'), '--save-every-updates', '1', '--max-checkpoints', '2',
        '--stop-file', str(run/'STOP'), '--remote-root', str(run/'transport'),
        '--ssh-control-dir', '/tmp/combat01-ssh', '--ssh-key', str(run/'ssh/id_ed25519'), '--barrier-timeout', '900',
        '--gradient-transport', 'ssh-ring', '--ring-port', str(config['ring_port'])]
    if t.get('leaf_amp') is not None:
        cmd += ['--leaf-amp-mode', t['leaf_amp']]
    if config.get('continuous', False):
        i = cmd.index('--updates')
        del cmd[i:i+2]
        cmd += ['--continuous']
    for peer in config['ranks']:
        if peer['rank'] != rank:
            cmd += ['--peer', f"{peer['rank']}=root@{peer['host']}:{peer['port']}"]
    if config.get('wandb_id'):
        cmd += ['--wandb', '--wandb-id', config['wandb_id'], '--wandb-name', config['run_id']]
    return cmd


def remote_preflight(config_path, rank):
    import torch
    import psutil
    import shutil
    from env.core import game_env  # initializes the explicit native path
    import slaythespire as sts
    from rl.policies.checkpoint import load_checkpoint
    config = validate(read(config_path), connected=True)
    root = Path(config['remote_repo'])
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
        raise ValueError('this production profile requires an RTX 5090')
    device = torch.cuda.get_device_properties(0)
    from rl.training.resource_monitor import ResourceMonitor
    limit = ResourceMonitor._cgroup_number('memory.max')
    used = ResourceMonitor._cgroup_number('memory.current')
    memory = psutil.virtual_memory()
    ram_total = min(memory.total, limit) if limit is not None else memory.total
    ram_free = min(memory.available, max(0, limit-used)) if limit is not None and used is not None else memory.available
    if device.total_memory < 30 * 1024**3 or ram_total < 82 * 1024**3:
        raise ValueError('insufficient advertised VRAM/RAM for 876 geometry')
    if ram_free < 65 * 1024**3 or torch.cuda.mem_get_info()[0] < 28 * 1024**3:
        raise ValueError('need 65 GiB free RAM and 28 GiB free VRAM before launch')
    if shutil.disk_usage(root).free < config.get('minimum_free_disk_gib', 30) * 1024**3:
        raise ValueError('insufficient free disk for retained generations plus an atomic save')
    model = load_checkpoint(root/'material/initial.zip', device='cpu', inference_only=True)
    if model.outer_profile_sha256 != digest(root/config['profile']) or model.num_timesteps != 0:
        raise ValueError('initial checkpoint must be step zero from the exact deployment profile')
    if model.policy.core.lower_initialization != 'random' or not all(
            p.requires_grad for name, p in model.policy.core.combat.named_parameters()
            if not (model._combat_distribution and not model.policy.core.config.auxiliary_loss_coef
                    and name.startswith('transition.'))):
        raise ValueError('the entire fresh lower network must be trainable')
    source = read(root/'deployment-source.json')
    report = dict(rank=rank, source_commit=source['commit'], config_sha256=digest(config_path),
                  objective_identity=model.outer_return_contract,
                  profile_sha256=digest(root/config['profile']), native_sha256=digest(sts.__file__),
                  initial_sha256=digest(root/'material/initial.zip'), teacher_sha256=model.policy.core.teacher.assert_integrity(),
                  lower_sha256=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                      for v in model.policy.core.combat.state_dict().values())).hexdigest(),
                  torch=torch.__version__, python=sys.version.split()[0], gpu=device.name,
                  vram_bytes=device.total_memory, available_ram=ram_free, ram_total=ram_total,
                  free_disk_bytes=shutil.disk_usage(root).free,
                  minimum_free_disk_gib=config.get('minimum_free_disk_gib', 30),
                  cgroup_memory_limit=limit,
                  runtime_files={p: digest(root/p) for p in source['files']})
    if report['runtime_files'] != source['files']:
        raise ValueError('deployed runtime differs from the pinned source')
    write(root/'runs'/config['run_id']/f'preflight-rank{rank}.json', report)
    print(f"rank {rank} preflight passed")


def agree(reports):
    if len(reports) != 4 or {r['rank'] for r in reports} != set(range(4)):
        raise ValueError('all four preflight reports are required')
    for key in ('source_commit', 'config_sha256', 'profile_sha256', 'native_sha256', 'teacher_sha256', 'objective_identity',
                'lower_sha256', 'torch', 'python', 'runtime_files'):
        if any(r[key] != reports[0][key] for r in reports[1:]):
            raise ValueError(f'rank preflight mismatch: {key}')
    # ZIP timestamps can differ; tensor identities, rather than ZIP bytes, must agree.
    return {str(r['rank']): r for r in reports}


def supervise(process, stop_file, budget):
    """Wait without a deadline when explicitly unbounded, otherwise enforce budget."""
    if budget is None:
        return process.wait()
    started = time.time()
    deadline = started + budget
    soft_stop = started + max(budget * .8, budget-120)
    terminate = deadline - min(10., budget * .1)
    signalled = False
    while process.poll() is None:
        now = time.time()
        if now >= soft_stop:
            Path(stop_file).touch()
        if now >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            return 124
        if now >= terminate and not signalled:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            signalled = True
        boundary = deadline if signalled else terminate if now >= soft_stop else soft_stop
        try:
            process.wait(timeout=max(.01, min(5., boundary-now)))
        except subprocess.TimeoutExpired:
            continue
    return 124 if signalled else process.returncode


def remote_status(config_path, rank, *, verify=False):
    """PID identity, production metric, and published generations; no inference."""
    import psutil
    config = validate(read(config_path), connected=True)
    run = Path(config['remote_repo'])/'runs'/config['run_id']
    state = dict(rank=rank, alive=False, exit=None, metric=None, generations=[])
    pid_file = run/f'rank{rank}.pid.json'
    if pid_file.is_file():
        pid = read(pid_file)
        try:
            process = psutil.Process(pid['trainer'])
            command = process.cmdline()
            state['alive'] = (process.create_time() == pid['created'] and
                              'rl.training.distributed_sb3' in command and
                              config['run_id'] in command)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    exit_file = run/f'rank{rank}.exit.json'
    if exit_file.is_file():
        state['exit'] = read(exit_file)
    log = run/f'rank{rank}.log'
    if log.is_file():
        # Metrics are a single JSON line emitted by the production entry point.
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-2*1024*1024))
            for line in stream.read().decode('utf-8', errors='replace').splitlines():
                if line.startswith('{'):
                    record = json.loads(line)
                    if 'global_timesteps' in record:
                        state['metric'] = record
    for path in sorted((run/'recovery'/f'rank{rank}').glob('step-*/manifest.json')):
        if verify:
            from rl.training.checkpoints import read_bundle
            manifest = read_bundle(path.parent, rank=rank, world_size=4)
        else:
            manifest = read(path)
        if manifest['run_id'] != config['run_id'] or manifest['rank'] != rank:
            raise ValueError('published recovery identity mismatch')
        if manifest['outer_return_contract'] != read(run/f'preflight-rank{rank}.json')['objective_identity']:
            raise ValueError('published recovery objective differs from preflight')
        state['generations'].append(manifest['global_step'])
    return state


def common_update(states):
    if len(states) != 4 or {s['rank'] for s in states} != set(range(4)):
        raise ValueError('all four rank statuses are required')
    common = set.intersection(*(set(s['generations']) for s in states))
    if not common:
        return None
    step = max(common)
    for state in states:
        metric = state['metric']
        if (not metric or metric['global_timesteps'] < step or metric['optimizer_steps'] < 1 or
                metric.get('frozen_upper/upper_updates', metric.get('outer_return/upper_updates')) != 0 or
                metric.get('nosl_is_az/completed_combat_roots', metric.get('outer_return/labelled_roots', 0)) < 1):
            return None
    return step


def remote_start(config_path, rank):
    config = validate(read(config_path), connected=True)
    root = Path(config['remote_repo'])
    run = root/'runs'/config['run_id']
    accepted = read(run/'agreement.json')[str(rank)]
    if accepted != read(run/f'preflight-rank{rank}.json') or accepted['config_sha256'] != digest(config_path):
        raise ValueError('configuration changed since four-rank agreement')
    source = read(root/'deployment-source.json')
    if any(digest(root/p) != sha for p, sha in source['files'].items()):
        raise ValueError('source changed after preflight')
    from env.core import game_env
    import slaythespire as sts
    if digest(sts.__file__) != accepted['native_sha256'] or digest(root/'material/initial.zip') != accepted['initial_sha256']:
        raise ValueError('native or initialization changed after preflight')
    if (run/'STOP').exists() or (run/f'rank{rank}.exit.json').exists():
        raise ValueError('run already ended/stopped; inspect and explicitly resume a common recovery generation')
    lock = run/f'rank{rank}.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    command = training_command(config, read(root/config['profile']), rank)
    write(run/f'command-rank{rank}.json', command)
    environment = os.environ.copy()
    # Match TRAIN47's allocator policy, using the current Torch variable name.
    environment['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    write(run/f'runtime-environment-rank{rank}.json',
          {'PYTORCH_ALLOC_CONF': environment['PYTORCH_ALLOC_CONF']})
    start, code = time.monotonic(), 1
    try:
        with (run/f'rank{rank}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            import psutil
            write(run/f'rank{rank}.pid.json', dict(supervisor=os.getpid(), trainer=process.pid,
                                                   created=psutil.Process(process.pid).create_time()))
            code = supervise(process, run/'STOP', config['budget']['max_seconds'])
    finally:
        write(run/f'rank{rank}.exit.json', dict(exit_code=code, seconds=time.monotonic()-start,
              budget=config['budget'], hard_timeout=code == 124,
              recovery='only completed common generations are resumable'))
        lock.unlink()
    return code


def manage(config_path, action):
    repo = Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    config = validate(read(config_path), connected=action != 'plan')
    profile = read(repo/config['profile'])
    if profile['contract']['lower_init'] != 'random' or profile['contract']['objective_id'] not in (
            'combat-outer-return-frozen-upper-v1', 'combat-outer-return-remaining-max-v1',
            'combat-outer-return-raw-trajectory-mix-v1', 'combat-actual-run-return-search-normalized-v1',
            'combat-finite-upper-curves-v1', 'combat-outcome-quantiles-v1', 'combat-outcome-means-v1'):
        raise ValueError('requires a wholly trainable fresh lower network with frozen upper')
    output = repo/'runs/local'/f"deployment-{config['run_id']}"
    output.mkdir(parents=True, exist_ok=True)
    if action == 'plan':
        plan = dict(config=config, unresolved=[f"rank{r['rank']} endpoint" for r in config['ranks'] if not r.get('host') or not r.get('port')],
                    global_slots=(None if config.get('continuous') else 4*profile['training']['n_envs_per_rank']*profile['training']['n_steps']*config['budget']['rounds']),
                    maximum_updates=(None if config.get('continuous') else profile['training']['optimizer_updates_per_round']*config['budget']['rounds']),
                    upper_updates=0, lower_trainable='all parameters from first update',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip())
        write(output/'plan.json', plan)
        print(json.dumps(plan, indent=2))
        return
    key = Path(config['ssh_key']).expanduser().resolve(strict=True)
    root, run_id = config['remote_repo'], config['run_id']
    run = f'{root}/runs/{run_id}'
    def ssh(rank, script, timeout=180):
        cmd = ['ssh', '-i', str(key), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
               '-o', 'ConnectTimeout=15', '-p', str(rank['port']), f"root@{rank['host']}", f'timeout {max(1, timeout-5)} bash -se']
        result = run_ssh_script(cmd, script, timeout)
        with (output/f"ssh-rank{rank['rank']}.log").open('a', encoding='utf-8') as log:
            log.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"rank {rank['rank']} failed; inspect {output}/ssh-rank{rank['rank']}.log")
        return result.stdout
    def scp(rank, path, target):
        subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
                        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4',
                        '-P',str(rank['port']),str(path),f"root@{rank['host']}:{target}"], check=True, timeout=600)
    prelude = f'cd {shlex.quote(root)}\nexport STS_LIGHTSPEED_BUILD_DIR={shlex.quote(root+"/sim/sts_lightspeed/build-combat01")}\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
    if action == 'preflight':
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=repo, text=True).strip():
            raise ValueError('commit the reviewed source before deployment')
        source_tar = output/'source.tar'
        roots = subprocess.check_output(['git','ls-tree','--name-only','HEAD'], cwd=repo, text=True).splitlines()
        roots = [p for p in roots if p not in ('runs','checkpoints')]
        subprocess.run(['git','archive','--format=tar','--output',str(source_tar),'HEAD','--',*roots], cwd=repo, check=True)
        files = {}
        with tarfile.open(source_tar) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    files[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        source = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(), files=files)
        write(output/'deployment-source.json', source)
        material = Path(config['material_directory']).expanduser().resolve(strict=True)
        if digest(material/'manifest.json') != config['material_sha256']:
            raise ValueError('material manifest SHA mismatch')
        bundle = output/'bundle.tar'
        with tarfile.open(bundle, 'w') as archive:
            archive.add(source_tar, arcname='source.tar')
            archive.add(output/'deployment-source.json', arcname='deployment-source.json')
            archive.add(config_path, arcname='deployment.json')
            for name in ('manifest.json','constructor.json','upper.pt','lower-init.pt','frozen-card-vectors.pt'):
                archive.add(material/name, arcname='material/'+name)
            if profile['contract']['objective_id'] == 'combat-finite-upper-curves-v1':
                if digest(material/'curve-bank.json') != profile['contract']['curve_bank_file_sha256']:
                    raise ValueError('finite curve material does not match the profile')
                archive.add(material/'curve-bank.json', arcname='material/curve-bank.json')
        def provision(rank):
            ssh(rank, f'test ! -e {shlex.quote(root)}\nmkdir -p {shlex.quote(root)}\n')
            scp(rank, bundle, root+'/bundle.tar')
            invoke = f"from experiments.launchers.combat_deployment import remote_preflight; remote_preflight('deployment.json', {rank['rank']})"
            setup = f"""cd {shlex.quote(root)}
exec > >(tee -a provisioning.log) 2>&1
tar xf bundle.tar
tar xf source.tar
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --index-url https://pypi.org/simple uv==0.13.4
uv sync --frozen --extra cuda --group test --python 3.11.15
export PATH="$PWD/.venv/bin:$PATH"
cmake -S sim/sts_lightspeed -B sim/sts_lightspeed/build-combat01 -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python" -DSTS_ENABLE_RULE_TRACE=ON
cmake --build sim/sts_lightspeed/build-combat01 --target slaythespire game-context-contract --parallel 8
{prelude}
mkdir -p {run}/ssh
ssh-keygen -q -t ed25519 -N '' -f {run}/ssh/id_ed25519
timeout 600 .venv/bin/python -m pytest tests/integration/test_outer_return_learner.py tests/integration/test_outer_return_policy.py tests/integration/test_native_outer_return.py tests/integration/test_outer_return_settlement.py tests/integration/test_event_callbacks.py tests/integration/test_frozen_upper_teacher.py -q --tb=short
timeout 180 .venv/bin/python -m rl.training.distributed_sb3 --rank 0 --world-size 1 --prepare-combat-outer {config['profile']} --material-directory material --material-sha256 {config['material_sha256']} --output material/initial.zip --device cpu --native-search-threads 2
timeout 180 .venv/bin/python -m rl.evaluation.eval material/initial.zip --resource-response --seed 9030000 --max-steps 256 --timeout 150 --device cpu --simulations 128 --output {run}/resource-response.json
timeout 600 .venv/bin/python -m pytest tests/integration/test_outer_resource_response.py::test_outer_graph_lifecycle -q --tb=short
timeout 120 .venv/bin/python -c {shlex.quote(invoke)}
"""
            ssh(rank, setup, timeout=2400)
            return json.loads(ssh(rank, f'cat {run}/preflight-rank{rank["rank"]}.json\n'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(provision, config['ranks']))
        write(output/'agreement.json', agree(reports))
        public_keys = [ssh(r, f'cat {run}/ssh/id_ed25519.pub\n').strip() for r in config['ranks']]
        for rank in config['ranks']:
            script = 'mkdir -p /root/.ssh\nchmod 700 /root/.ssh\n'
            for public in public_keys:
                script += f'grep -qxF {shlex.quote(public)} /root/.ssh/authorized_keys 2>/dev/null || printf "%s\\n" {shlex.quote(public)} >> /root/.ssh/authorized_keys\n'
            ssh(rank, script+'chmod 600 /root/.ssh/authorized_keys\n')
            scp(rank, output/'agreement.json', run+'/agreement.json')
        for rank in config['ranks']:
            for peer in config['ranks']:
                if rank['rank'] != peer['rank']:
                    ssh(rank, f'ssh -i {run}/ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p {peer["port"]} root@{peer["host"]} true\n')
        print('Four-rank source/model/native agreement accepted; no training started.')
    elif action == 'start':
        reports = [json.loads(ssh(r, f'cat {run}/preflight-rank{r["rank"]}.json\n')) for r in config['ranks']]
        agree(reports)
        if any(r['config_sha256'] != digest(config_path) for r in reports):
            raise ValueError('local deployment configuration differs from the preflighted one')
        def start(rank):
            return ssh(rank, prelude+f'COMBAT01_CONFIG=deployment.json bash experiments/launchers/launch_nosl_replay.sh {rank["rank"]}\n')
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for result in pool.map(start, config['ranks']):
                    print(result.strip())
        except BaseException:
            for rank in config['ranks']:
                try:
                    ssh(rank, f'touch {run}/STOP\n', timeout=30)
                except Exception:
                    pass  # Keep the original launch error; inspect every rank before retrying.
            raise
    elif action in ('status', 'wait-update'):
        # This is a startup acceptance deadline, never a trainer stop deadline.
        deadline = time.monotonic() + (1500 if config.get('continuous') else config['budget']['max_seconds'] + 60)
        def status(rank, verify=False):
            invoke = f"import json; from experiments.launchers.combat_deployment import remote_status; print(json.dumps(remote_status('deployment.json', {rank['rank']}, verify={verify})))"
            return json.loads(ssh(rank, prelude+f'.venv/bin/python -c {shlex.quote(invoke)}\n'))
        while True:
            with ThreadPoolExecutor(max_workers=4) as pool:
                states = list(pool.map(status, config['ranks']))
            write(output/'status.json', states)
            common = set.intersection(*(set(s['generations']) for s in states))
            if action == 'status':
                print(json.dumps(dict(common_steps=sorted(common), ranks=states), indent=2))
                break
            if any(s['exit'] is not None and s['exit']['exit_code'] != 0 for s in states):
                raise RuntimeError('a trainer failed; inspect deployment status.json')
            step = common_update(states)
            if step is not None:
                # Hash-check the actual per-rank bundles before accepting startup.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    verified = list(pool.map(lambda r: status(r, True), config['ranks']))
                write(output/'verified-update.json', verified)
                verified_step = common_update(verified)
                if verified_step is None:
                    raise RuntimeError('common generation changed during verification; retry wait-update')
                print(f'All ranks trained lower only and verified common generation {verified_step}.')
                break
            if any(s['exit'] is not None or not s['alive'] for s in states):
                raise RuntimeError('a rank exited before a common update; inspect deployment status.json')
            if time.monotonic() >= deadline:
                raise TimeoutError('no common update within the explicit training budget')
            time.sleep(15)
    else:
        raise ValueError(f'unsupported deployment action {action}')
