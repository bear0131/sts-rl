"""Bounded validation subprocesses; phase decisions use one all-rank result."""
from pathlib import Path
import gc
import json
import subprocess
import sys

import torch

from rl.algorithms.deck_schedule import configure, check_due, advance, apply_stage


def at_collection_boundary(model, *, rank, world_size, local_root, ascension, average):
    # A fixed qualification checkpoint has no schedule. Its ephemeral weights
    # still belong to this run, never to the path serialized by the old run.
    model.deck_search_snapshot_dir = str(Path(local_root) / 'search-weights')
    schedule = configure(model)
    if schedule is None:
        return
    completed = model.ppo_rollouts_completed
    if schedule.validation_games % world_size:
        raise ValueError('fixed validation games must divide the training rank count')
    if check_due(model.deck_training_state, schedule, completed):
        folder = Path(local_root) / 'anchor-readiness'
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint, output = folder / 'model.zip', folder / f'round-{completed}.json'
        output.unlink(missing_ok=True)
        model.save(checkpoint)
        if model.device.type == 'cuda':
            # A child has its own allocator: the parent's unused update cache
            # is unavailable to it, even though the next parent rollout fits.
            # Keep live weights/Adam/graphs; only return unreferenced storage.
            torch.cuda.synchronize(model.device)
            gc.collect()
            with torch.cuda.device(model.device):
                torch.cuda.empty_cache()
        count = schedule.validation_games // world_size
        command = [sys.executable, '-m', 'rl.evaluation.eval', str(checkpoint), '--anchor-readiness',
            '--episodes', str(count), '--num-envs', str(min(8, count)), '--seed',
            str(schedule.validation_seed + rank * count), '--device', str(model.device),
            '--max-steps', str(schedule.validation_max_steps), '--timeout', str(schedule.validation_seconds-20),
            '--ascension', str(ascension), '--float32-matmul-precision', 'highest', '--output', str(output)]
        moments = dict.fromkeys(('natural', 'censored', 'anchors', 'target_sum', 'target_sq_sum',
            'error_sum', 'error_sq_sum', 'relic_offered', 'relic_taken', 'floor_sum', 'wins', 'timeout', 'failed'), 0.)
        try:
            with (folder / f'round-{completed}.log').open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=schedule.validation_seconds)
            if process.returncode:
                raise RuntimeError('anchor validation exited ' + str(process.returncode))
            data = json.loads(output.read_text())
            moments.update({k: data[k] for k in moments if k != 'failed'})
        except (subprocess.TimeoutExpired, RuntimeError, OSError, ValueError) as exc:
            moments.update(failed=1., censored=float(count))
            print('DECK_READINESS_FAILED ' + str(exc), flush=True)
        finally:
            checkpoint.unlink(missing_ok=True)
        # All ranks receive the same sums and make exactly the same transition.
        sums = {key: value * world_size for key, value in average(moments).items()}
        model.deck_training_state = advance(model.deck_training_state, schedule, completed, sums)
        (folder / f'round-{completed}-global.json').write_text(json.dumps(model.deck_training_state, indent=2))
        print('DECK_READINESS ' + json.dumps(model.deck_training_state), flush=True)
    apply_stage(model, schedule, completed, rank=rank)
    model.logger.record('deck_search/stage', ('warmup', 'anchor', 'search').index(model.deck_training_state['stage']))
    model.logger.record('deck_search/qualified', int(schedule.search_qualified))
    model.logger.record('deck_search/coefficient', (model.deck_search_config or {}).get('coefficient', 0.))
