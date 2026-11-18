"""Read the native four-rank sts-rl-2 run and aggregate published rounds."""
from __future__ import annotations

import inspect
import shlex
from statistics import mean


def read_rank(root, run_id, rank, cursor):
    # This function also runs over SSH using the training host's Python.
    import hashlib
    import json
    import sys
    import time
    from pathlib import Path

    root = Path(root)
    directory = root / run_id
    process = json.loads((directory / 'process.json').read_text())
    if process['run'] != run_id:
        raise ValueError('process belongs to a different run')
    pid = process['pid']
    try:
        args = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
        alive = run_id.encode() in args and b'sts2.train' in args
    except FileNotFoundError:
        alive = False
    log = directory / f'rank{rank}.log'
    with log.open('rb') as stream:
        import os
        stat = os.fstat(stream.fileno())
        if cursor.get('inode') != stat.st_ino or cursor.get('offset', 0) > stat.st_size:
            cursor = {}
        offset = cursor.get('offset', 0)
        stream.seek(offset)
        data = stream.read(8 << 20)
    end = data.rfind(b'\n') + 1
    progress, errors, reports = cursor.get('progress'), list(cursor.get('errors', [])), []
    for line in data[:end].splitlines():
        if not line.startswith(b'{'):
            if any(word in line for word in (b'Traceback (', b'OutOfMemoryError', b'No space left')):
                errors.append(line.decode(errors='replace')[-200:])
            continue
        row = json.loads(line)
        if row.get('event') == 'start':
            progress, errors = None, []
        elif row.get('event') == 'sampling':
            progress = dict(vector_steps=row['rows'], target_steps=row['target'])
        elif 'round_seconds' in row:
            reports.append(row)
            progress = None
    exit_file = directory / 'exit.json'
    exit_code = json.loads(exit_file.read_text())['rc'] if exit_file.is_file() else None
    if exit_code:
        errors.append(f'training exited with code {exit_code}')
    manifests = sorted(directory.glob('round-*/manifest.json'),
                       key=lambda f: int(f.parent.name.split('-')[1]), reverse=True)
    step, saved_round, snapshot = None, None, cursor.get('snapshot')
    for path in manifests:
        try:
            meta = json.loads(path.read_text())
            if meta['format'] != 'sts2.checkpoint/1' or rank not in meta['ranks']:
                raise ValueError('invalid checkpoint manifest')
            step, saved_round = meta['global_step'], meta['round']
            if not snapshot or snapshot['step'] != step:
                # Read only the local rank on CPU, once per published checkpoint.
                import io
                sys.path.insert(0, str(root / 'runtime311/site-packages'))
                import torch
                torch.set_num_threads(1)
                member = f'rank{rank}.pt'
                raw = (path.parent / member).read_bytes()
                if hashlib.sha256(raw).hexdigest() != meta['files'][member]:
                    raise ValueError('rank checkpoint hash mismatch')
                env = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['environment']
                counts = dict(wins=0, games=0, floors=0)
                for slot in env['states']:
                    last = slot.get('last_completed_trajectory')
                    if not last or not last['terminated']:
                        continue
                    stats = last.get('run_stats') or {}
                    won = stats.get('outcome') == 'PLAYER_VICTORY' or (stats.get('act') or 0) > 3
                    if (won or stats.get('outcome') == 'PLAYER_LOSS') and last.get('floor') is not None:
                        counts['wins'] += int(won)
                        counts['games'] += 1
                        counts['floors'] += last['floor']
                snapshot = dict(step=step, **counts)
            break
        except FileNotFoundError:
            # A previous generation may retire between the directory scan and open.
            continue
    cursor = dict(inode=stat.st_ino, offset=offset + end, progress=progress,
                  errors=errors[-3:], snapshot=snapshot)
    return dict(rank=rank, pid=pid, alive=alive, step=step, saved_round=saved_round,
                progress=progress, errors=errors[-3:], log_mtime=stat.st_mtime, checked_at=time.time(),
                exit_code=exit_code, cursor=cursor, sts2_reports=reports)


def rank_command(config, rank, cursor):
    python = shlex.quote(config['monitor_python'])
    call = f"read_rank({config['root']!r}, {config['run_id']!r}, {rank!r}, {cursor!r})"
    return f"{python} - <<'PY_MONITOR'\n{inspect.getsource(read_rank)}\nimport json\nprint(json.dumps({call}))\nPY_MONITOR\n"


def completed_rows(state, ranks, world):
    # Persist partial rounds across slow ranks, failed polls and process restarts.
    cache = state.setdefault('sts2_rank_rounds', {})
    changed = set()
    for rank in ranks:
        identity = str(rank['rank'])
        for report in rank.pop('sts2_reports', []):
            step = str(report['global_step'])
            cache.setdefault(step, {}).setdefault(identity, {}).update(report)
            changed.add(step)
        sample = rank.get('cursor', {}).get('snapshot')
        if sample:
            step = str(sample['step'])
            bucket = cache.setdefault(step, {}).setdefault(identity, {})
            if bucket.get('outcomes') != sample:
                bucket['outcomes'] = sample
                changed.add(step)
    result = []
    for step in changed:
        group = cache[step]
        if len(group) != world or any('round_seconds' not in r for r in group.values()):
            continue
        records = list(group.values())
        first = group['0']
        for key in ('round', 'rows', 'adam_steps'):
            if any(r[key] != first[key] for r in records):
                raise ValueError(f'ranks disagree on {key} at {step}')
        row = dict(global_timesteps=int(step), rollout_round=first['round'],
                   transitions_global=first['rows'], optimizer_steps=first['adam_steps'],
                   iteration_wall_sec=first['round_seconds'], rollout_sec=first['collection_seconds'],
                   checkpoint_sec=first['save_seconds'])
        for name in ('round_seconds', 'collection_seconds', 'update_seconds', 'save_seconds'):
            times = sorted(r[name] for r in records)
            row['sts2/' + name] = times[max(0, world - 2)]
        row['sts2/episodes'] = sum(r['episodes'] for r in records)
        row['sts2/search_roots'] = sum(r['search_rows'] for r in records)
        row['sts2/total_loss'] = mean(v for r in records for v in r['losses'])
        outcomes = [r['outcomes'] for r in records if 'outcomes' in r]
        if len(outcomes) == world:
            n = sum(s['games'] for s in outcomes)
            row['sts2/slot_games'] = n
            if n:
                row['sts2/slot_win_rate'] = sum(s['wins'] for s in outcomes) / n
                row['sts2/slot_floor_mean'] = sum(s['floors'] for s in outcomes) / n
        result.append(row)
    return result
