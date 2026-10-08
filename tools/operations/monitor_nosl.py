"""Inspect rank-0 training metrics, with an opt-in relic-collapse STOP guard."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from rl.training.monitoring import assess, assess_relic_collapse, RECOVERY_STEPS, RELIC_GUARD_KEYS


def read_records(path):
    records = []
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except json.JSONDecodeError:
                pass  # writer may be in the middle of its final line
    return records


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(path)


def relic_guard_check(log, state_path, stop_file, *, iteration_steps):
    """One check; never overwrite STOP or change any training parameter."""
    if stop_file.resolve().parent != log.resolve().parent or stop_file.name != 'STOP':
        raise ValueError('STOP must be in the same run directory as the log')
    identity = dict(log=str(log.resolve()), stop_file=str(stop_file.resolve()),
                    iteration_steps=iteration_steps, min_opportunities=400,
                    baseline_rounds=5, confirmation_rounds=3,
                    relative_threshold=.5, minimum_drop=.20)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'identity': identity, 'records': []}
    if state.get('identity') != identity:
        raise ValueError('guard state belongs to another run or configuration')
    if state.get('stop_requested') or stop_file.exists():
        return dict(status='intentional_stop', stop=True)
    rows = {}
    for record in state['records'] + read_records(log):
        if 'global_timesteps' in record:
            step = record['global_timesteps']
            if isinstance(step, (int, float)) and math.isfinite(step) and step > 0:
                rows[step] = {key: record[key] for key in RELIC_GUARD_KEYS if key in record}
    state['records'] = [rows[step] for step in sorted(rows)]
    result = assess_relic_collapse(state['records'], iteration_steps=iteration_steps)
    state.update(last_check=time.time(), result=result)
    if result['stop']:
        # The existing trainer averages stop_requested across the ring: one
        # rank's STOP requests a coordinated save and exit on all four ranks.
        reason = dict(reason='sustained_chest_relic_take_collapse', **result)
        try:
            with stop_file.open('x', encoding='utf-8') as f:
                f.write(json.dumps(reason, indent=2) + '\n')
        except FileExistsError:
            pass
        state['stop_requested'] = reason
    save_state(state_path, state)
    return result


def watch_relic_guard(args):
    if (args.record_change or not math.isfinite(args.watch_seconds)
            or not math.isfinite(args.poll_seconds)
            or args.watch_seconds < 0 or args.poll_seconds <= 0):
        raise ValueError('invalid guard arguments')
    if args.stop_file is None or args.iteration_steps is None:
        raise ValueError('relic guard requires --stop-file and --iteration-steps')
    if args.watch_seconds and args.pid_file is None:
        raise ValueError('bounded watch requires the active trainer --pid-file')
    deadline = time.monotonic() + args.watch_seconds
    while True:
        if args.pid_file is not None:
            pid = int(args.pid_file.read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print(json.dumps(dict(status='trainer_exited')), flush=True)
                return
            proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists():
                command = proc.read_bytes().split(b'\0')
                if (b'rl.training.distributed_sb3' not in command
                        or args.log.parent.name.encode() not in command):
                    raise ValueError('PID no longer belongs to this training run')
        result = relic_guard_check(args.log, args.state, args.stop_file,
                                   iteration_steps=args.iteration_steps)
        print(json.dumps(result), flush=True)
        if result['stop'] or args.watch_seconds == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(json.dumps(dict(status='watch_budget_exhausted')), flush=True)
            return
        time.sleep(min(args.poll_seconds, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--record-change', help='Parameter name, only after verified save/stop/resume')
    parser.add_argument('--at-step', type=int)
    parser.add_argument('--relic-guard', action='store_true')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--iteration-steps', type=int)
    parser.add_argument('--watch-seconds', type=float, default=0)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--pid-file', type=Path)
    args = parser.parse_args()
    if args.relic_guard:
        watch_relic_guard(args)
        return
    state = json.loads(args.state.read_text()) if args.state.exists() else {'interventions':[]}
    records = read_records(args.log)
    if args.record_change:
        index=len(state['interventions'])
        if index >= len(RECOVERY_STEPS) or args.record_change != RECOVERY_STEPS[index][0] or args.at_step is None:
            raise ValueError('intervention must be the next coordinate and have an explicit global step')
        state['interventions'].append(dict(parameter=args.record_change, value=RECOVERY_STEPS[index][1],
            global_step=args.at_step,time=time.time()))
    result = assess(records,state['interventions'],now=time.time())
    state.update(last_check=time.time(), result=result)
    save_state(args.state, state)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
