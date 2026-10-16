"""Log independent W&B health records for an existing, progressing Linux trainer.

Attach to its existing W&B core, which alone assigns history offsets. No
wandb.init/finish, training-metric writes, state overrides, or trainer signals.
Supported and lifecycle-tested with wandb 0.28.1 on Linux.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import netrc
from pathlib import Path
import time
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


def read_progress(path: Path) -> int:
    # Bounded read, independent of the size of a multi-day training log.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 131072))
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
    def __init__(self, run: str, pid: int, core_pid: int):
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
        if (self.attached.entity, self.attached.project, self.attached.id) != tuple(parts):
            raise GuardError('attached run identity does not match requested run')
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


def pulse(client: Client, guard: Guard, pid: int, core_pid: int, log: Path) -> dict:
    def check():
        guard.check(read_process(pid), read_process(core_pid),
                    read_progress(log), time.monotonic())

    check()
    status = client.status()
    if status['state'] not in ('running', 'crashed'):
        return {'event': 'stop', 'reason': 'server terminal state', **status}
    # Recheck after the network read; never signal the training process.
    check()
    client.heartbeat()
    return {'event': 'heartbeat', 'time': time.time(), 'step': guard.step, **status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--core-pid', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--max-progress-age', type=float, default=1500)
    parser.add_argument('--max-idle', type=float, default=300)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    if min(args.max_progress_age, args.max_idle, args.max_seconds) <= 0 or args.interval < 30:
        parser.error('positive deadlines and interval >= 30 seconds required')
    # Explicit identity check prevents accidentally keeping another run alive.
    command = Path(f'/proc/{args.pid}/cmdline').read_bytes().split(b'\0')
    run_id = args.run.rsplit('/', 1)[-1].encode()
    if b'--wandb-id' not in command or command[command.index(b'--wandb-id') + 1] != run_id:
        parser.error('trainer command does not match W&B run id')
    core_command = Path(f'/proc/{args.core_pid}/cmdline').read_bytes().split(b'\0')
    if b'--pid' not in core_command or core_command[core_command.index(b'--pid') + 1] != str(args.pid).encode():
        parser.error('W&B core does not belong to trainer')
    started = time.monotonic()
    guard = Guard(read_process(args.pid), read_process(args.core_pid), read_progress(args.log),
                  started, max(0, time.time() - args.log.stat().st_mtime),
                  args.max_progress_age, args.max_idle)
    guard.check(read_process(args.pid), read_process(args.core_pid),
                read_progress(args.log), time.monotonic())
    client = Client(args.run, args.pid, args.core_pid)
    while time.monotonic() - started < args.max_seconds:
        try:
            event = pulse(client, guard, args.pid, args.core_pid, args.log)
            print(json.dumps(event), flush=True)
            if event['event'] == 'stop':
                return
        except GuardError as error:
            print(json.dumps({'event': 'stop', 'reason': str(error)}), flush=True)
            return
        except Exception as error:
            # Fail closed. Log type, not response text that might contain secrets.
            print(json.dumps({'event': 'stop', 'error_type': type(error).__name__}), flush=True)
            return
        time.sleep(min(args.interval, max(0, args.max_seconds - (time.monotonic() - started))))
    print(json.dumps({'event': 'stop', 'reason': 'time budget reached'}), flush=True)


if __name__ == '__main__':
    main()
