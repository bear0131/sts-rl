"""Log independent W&B health records for an existing, progressing Linux trainer.

Attach to its existing W&B core, which alone assigns history offsets. No
wandb.init/finish, training-metric writes, state overrides, or trainer signals.
Supported and lifecycle-tested with wandb 0.28.1 on Linux.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import fcntl
import json
import netrc
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class GuardError(ValueError):
    """A safe-to-log reason that sending further heartbeats is incorrect."""


@dataclass(frozen=True)
class Process:
    start: int
    cpu: int
    state: str


def read_process(pid: int, proc: Path = Path('/proc')) -> Process:
    fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return Process(int(fields[19]), int(fields[11]) + int(fields[12]), fields[0])


def read_progress(path: Path, start_offset: int = 0, default: int | None = None) -> int:
    # Bounded read, independent of the size of a multi-day training log.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        if stream.tell() < start_offset:
            raise GuardError('training log was truncated')
        stream.seek(max(start_offset, stream.tell() - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict) and 'global_timesteps' in record:
            for key, value in record.items():
                if key.endswith('/finite_fraction') and value != 1.0:
                    raise GuardError('non-finite training result')
            return int(record['global_timesteps'])
    if default is not None:
        return default
    raise GuardError('no completed training record in log tail')


class Guard:
    def __init__(self, trainer: Process, core: Process, step: int, now: float,
                 progress_age: float, max_progress_age: float, max_idle: float):
        self.trainer_start = trainer.start
        self.core_start = core.start
        self.cpu = trainer.cpu
        self.step = step
        self.last_cpu = now
        self.last_progress = now - progress_age
        self.max_progress_age = max_progress_age
        self.max_idle = max_idle

    def check(self, trainer: Process, core: Process, step: int, now: float) -> None:
        if trainer.start != self.trainer_start or core.start != self.core_start:
            raise GuardError('process identity changed')
        if trainer.state in ('Z', 'X', 'x', 'T', 't') or core.state in ('Z', 'X', 'x', 'T', 't'):
            raise GuardError('trainer or W&B core exited/stopped')
        if step < self.step or trainer.cpu < self.cpu:
            raise GuardError('training progress regressed')
        if trainer.cpu > self.cpu:
            self.last_cpu = now
        if step > self.step:
            self.last_progress = now
        self.cpu, self.step = trainer.cpu, step
        if now - self.last_progress >= self.max_progress_age:
            raise GuardError('no completed round within progress deadline')
        if now - self.last_cpu >= self.max_idle:
            raise GuardError('trainer CPU has not advanced within idle deadline')


def core_socket(pid: int, core_pid: int, proc: Path = Path('/proc')) -> str:
    # The startup port file is deleted after initialization. Discover the
    # existing listener from this core's own file descriptors, not a glob that
    # could select another run's service.
    inodes = set()
    for fd in (proc / str(core_pid) / 'fd').iterdir():
        try:
            target = str(fd.readlink())
        except FileNotFoundError:
            continue
        if target.startswith('socket:[') and target.endswith(']'):
            inodes.add(target[8:-1])
    candidates = []
    for line in (proc / 'net' / 'unix').read_text().splitlines()[1:]:
        fields = line.split(maxsplit=7)
        if (len(fields) == 8 and fields[6] in inodes and fields[3] == '00010000'
                and fields[4:6] == ['0001', '01']
                and Path(fields[7]).parent.name.startswith(f'wandb-{pid}-{core_pid}-')):
            candidates.append(fields[7])
    if len(candidates) != 1:
        raise GuardError('could not identify exactly one existing W&B core listener')
    return candidates[0]


class Client:
    def __init__(self, run: str | None, pid: int, core_pid: int,
                 *, wandb_id: str | None = None, project: str | None = None):
        # Auto mode learns the entity from the actual attached run, not a
        # deployment-specific account hardcoded in the launcher.
        if run is None:
            parts = [None, project, wandb_id]
        else:
            parts = run.split('/')
            if len(parts) != 3 or any(not x or x in ('.', '..') for x in parts):
                raise ValueError('--run must be entity/project/run_id')
        self.entity, self.project, self.run_id = parts
        auth = netrc.netrc().authenticators('api.wandb.ai')
        if not auth:
            raise ValueError('existing api.wandb.ai netrc credentials required')
        self.authorization = 'Basic ' + base64.b64encode(('api:' + auth[2]).encode()).decode()
        import wandb
        from wandb.sdk.lib.service.service_token import UnixServiceToken

        if wandb.__version__ != '0.28.1':
            raise GuardError('W&B SDK version has not been lifecycle-tested')
        token = UnixServiceToken(parent_pid=pid, path=core_socket(pid, core_pid))
        # Only sets this helper's environment; the shared core remains owned by
        # the trainer. Do not call init(), finish(), or change trainer settings.
        token.save_to_env()
        self.attached = wandb.attach(run_id=self.run_id)
        if (self.attached.project, self.attached.id) != (self.project, self.run_id) or (
                self.entity is not None and self.attached.entity != self.entity):
            raise GuardError('attached run identity does not match requested run')
        self.entity = self.attached.entity
        self.attached.define_metric('ops/heartbeat_unix', hidden=True, summary='none')

    def post(self, url: str, payload: dict) -> dict:
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
            'Content-Type': 'application/json', 'Authorization': self.authorization,
        }, method='POST')
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def status(self) -> dict:
        result = self.post('https://api.wandb.ai/graphql', {
            'query': 'query($entity:String!,$project:String!,$run:String!){'
                     'project(name:$project,entityName:$entity){run(name:$run){'
                     'state heartbeatAt historyLineCount}}}',
            'variables': {'entity': self.entity, 'project': self.project, 'run': self.run_id},
        })
        if result.get('errors'):
            raise ValueError('W&B status query failed')
        return result['data']['project']['run']

    def heartbeat(self) -> None:
        # No explicit step. The shared core serializes this alongside trainer
        # records, preserving global_timesteps/legacy_step and metric values.
        self.attached.log({'ops/heartbeat_unix': time.time()})


def pulse(client: Client, guard: Guard, pid: int, core_pid: int, log: Path,
          *, log_start: int = 0, initial_step: int | None = None) -> dict:
    def check():
        guard.check(read_process(pid), read_process(core_pid),
                    read_progress(log, log_start, initial_step), time.monotonic())

    check()
    status = client.status()
    if status['state'] not in ('running', 'crashed'):
        return {'event': 'stop', 'reason': 'server terminal state', **status}
    # Recheck after the network read; never signal the training process.
    check()
    client.heartbeat()
    return {'event': 'heartbeat', 'time': time.time(), 'step': guard.step, **status}


def command_option(command: list[bytes], option: bytes) -> bytes | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def find_core(pid: int, proc: Path = Path('/proc')) -> int | None:
    children = (proc / str(pid) / 'task' / str(pid) / 'children').read_text().split()
    matches = []
    for child in children:
        try:
            cmd = (proc / child / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if (Path(os.fsdecode(cmd[0])).name == 'wandb-core'
                and command_option(cmd, b'--pid') == str(pid).encode()):
            matches.append(int(child))
    if len(matches) > 1:
        raise GuardError('multiple W&B cores belong to trainer')
    return matches[0] if matches else None


def wait_for_client(pid: int, start: int, run_id: str, project: str,
                    deadline: float) -> tuple[Client, int]:
    while time.monotonic() < deadline:
        trainer = read_process(pid)
        if trainer.start != start or trainer.state in ('Z', 'X', 'x', 'T', 't'):
            raise GuardError('trainer exited or changed during W&B initialization')
        core = find_core(pid)
        if core is not None:
            try:
                core_socket(pid, core)  # Do not attach until the listener exists.
                return Client(None, pid, core, wandb_id=run_id, project=project), core
            except GuardError as error:
                if 'listener' not in str(error):
                    raise
            except Exception as error:
                # inform_attach can arrive before the trainer's wandb.init.
                print(json.dumps({'event': 'waiting_for_wandb',
                                  'error_type': type(error).__name__}), flush=True)
        time.sleep(2)
    raise GuardError('W&B initialization deadline exceeded')


def acquire_lock(log: Path, pid: int, start: int):
    lock = log.parent / f'.wandb-heartbeat-{pid}-{start}.lock'
    stream = lock.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise GuardError('a heartbeat already owns this trainer') from None
    return stream  # Keep open for process lifetime; never unlink a locked inode.


def wait_interval(seconds: float, pid: int, start: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            trainer = read_process(pid)
        except FileNotFoundError:
            raise GuardError('trainer exited') from None
        if trainer.start != start or trainer.state in ('Z', 'X', 'x', 'T', 't'):
            raise GuardError('trainer exited or changed')
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--core-pid', type=int)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--run')
    parser.add_argument('--auto', action='store_true', help='Wait for this launch to initialize W&B')
    parser.add_argument('--wandb-id')
    parser.add_argument('--project', default='sts-rl')
    parser.add_argument('--log-start', type=int)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--max-progress-age', type=float, default=1500)
    parser.add_argument('--max-idle', type=float, default=300)
    parser.add_argument('--max-seconds', type=float, help='Auto mode otherwise follows trainer lifetime')
    args = parser.parse_args()
    if (min(args.max_progress_age, args.max_idle, args.startup_timeout) <= 0
            or (args.max_seconds is not None and args.max_seconds <= 0) or args.interval < 30):
        parser.error('positive deadlines and interval >= 30 seconds required')
    if args.auto:
        if not args.wandb_id or args.log_start is None or args.log_start < 0 or args.run or args.core_pid:
            parser.error('--auto needs --wandb-id and --log-start, without --run/--core-pid')
    elif not args.run or not args.core_pid or args.max_seconds is None:
        parser.error('manual mode needs --run, --core-pid and --max-seconds')
    # Explicit identity check prevents accidentally keeping another run alive.
    command = Path(f'/proc/{args.pid}/cmdline').read_bytes().split(b'\0')
    run_id = args.wandb_id if args.auto else args.run.rsplit('/', 1)[-1]
    if command_option(command, b'--wandb-id') != run_id.encode():
        parser.error('trainer command does not match W&B run id')
    if args.auto and command_option(command, b'--wandb-project') != args.project.encode():
        parser.error('trainer command does not match W&B project')
    started = time.monotonic()
    trainer = read_process(args.pid)
    lock = acquire_lock(args.log, args.pid, trainer.start)
    if args.auto:
        print(json.dumps({'event': 'waiting_for_wandb', 'trainer_pid': args.pid}), flush=True)
        client, args.core_pid = wait_for_client(
            args.pid, trainer.start, run_id, args.project, started + args.startup_timeout)
    core_command = Path(f'/proc/{args.core_pid}/cmdline').read_bytes().split(b'\0')
    if command_option(core_command, b'--pid') != str(args.pid).encode():
        parser.error('W&B core does not belong to trainer')
    log_start = args.log_start or 0
    initial_step = -1 if args.auto else None
    now = time.monotonic()
    guard = Guard(trainer, read_process(args.core_pid), read_progress(args.log, log_start, initial_step),
                  now, now - started if args.auto else max(0, time.time() - args.log.stat().st_mtime),
                  args.max_progress_age, args.max_idle)
    guard.check(read_process(args.pid), read_process(args.core_pid),
                read_progress(args.log, log_start, initial_step), time.monotonic())
    if not args.auto:
        client = Client(args.run, args.pid, args.core_pid)
    while args.max_seconds is None or time.monotonic() - started < args.max_seconds:
        try:
            event = pulse(client, guard, args.pid, args.core_pid, args.log,
                          log_start=log_start, initial_step=initial_step)
            print(json.dumps(event), flush=True)
            if event['event'] == 'stop':
                return
        except GuardError as error:
            print(json.dumps({'event': 'stop', 'reason': str(error)}), flush=True)
            return
        except (urllib.error.URLError, TimeoutError) as error:
            # Retry without emitting a heartbeat or resetting progress clocks.
            print(json.dumps({'event': 'network_retry', 'error_type': type(error).__name__}), flush=True)
        except Exception as error:
            # Fail closed. Log type, not response text that might contain secrets.
            print(json.dumps({'event': 'stop', 'error_type': type(error).__name__}), flush=True)
            return
        remaining = args.interval if args.max_seconds is None else max(
            0, args.max_seconds - (time.monotonic() - started))
        wait_interval(min(args.interval, remaining), args.pid, trainer.start)
    print(json.dumps({'event': 'stop', 'reason': 'time budget reached'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except GuardError as error:
        print(json.dumps({'event': 'stop', 'reason': str(error)}), flush=True)
