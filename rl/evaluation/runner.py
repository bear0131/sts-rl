"""One bounded episode runner for CLI, GUI and arbitrary decision strategies."""
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np

from rl.evaluation.strategies import DecisionRequest, DecisionStrategy


class EvaluationCancelled(Exception):
    pass


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 100
    seed: int = 10000
    num_envs: int = 1
    max_steps: int = 20000
    timeout: float = 1800.0

    def __post_init__(self):
        if min(self.episodes, self.num_envs, self.max_steps) < 1:
            raise ValueError('episodes, num_envs and max_steps must be positive')
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout must be finite and positive')


def is_win(env):
    stats = env.unwrapped.thick.run_stats()
    return stats['act'] > 3 or stats['outcome'] == 'PLAYER_VICTORY'


def run_outcome(env, info, terminated, truncated):
    stats = env.unwrapped.thick.run_stats()
    win = is_win(env)
    result = ('win' if win else 'dead_screen' if truncated and info.get('dead_screen')
              else 'other_truncation' if truncated else 'death')
    return dict(result=result, floor=int(stats['floor']), act=int(stats['act']), outcome=stats['outcome'])


@dataclass(frozen=True)
class Scenario:
    name: str
    make_env: Callable
    outcome: Callable = run_outcome
    # A custom fixture may restore a fixed game state after the seeded reset.
    reset: Callable = lambda env, seed: env.reset(seed=seed)


def evaluate(strategy: DecisionStrategy, scenario: Scenario, config: EvaluationConfig,
             *, cancelled=lambda: False, progress=lambda event: None):
    """Yield per-episode rows; keep environments/model resident and batch decisions.

    Deadlines/cancellation are checked between steps and native leaf batches.
    They are cooperative; callers needing a hard native-code timeout must use
    a supervising process (as the validation harness does).
    """
    started = time.monotonic()
    active, environments = {}, []
    next_episode = 0

    def check():
        if cancelled():
            raise EvaluationCancelled('evaluation cancelled')
        if time.monotonic() - started >= config.timeout:
            raise TimeoutError('evaluation wall-time budget exceeded')

    def start(env):
        nonlocal next_episode
        if next_episode >= config.episodes:
            return None
        index = next_episode
        next_episode += 1
        observation, info = scenario.reset(env, config.seed + index)
        return dict(episode=index, seed=config.seed + index, observation=observation, info=info,
                    steps=0, raw_return=0.0, metrics={})

    try:
        for index in range(min(config.episodes, config.num_envs)):
            check()
            env = scenario.make_env()
            environments.append(env)
            active[index] = start(env)
        while active:
            check()
            indices = sorted(active)
            requests = [DecisionRequest(environments[i], active[i]['observation'], active[i]['info'],
                                        active[i]['seed'], active[i]['steps']) for i in indices]
            decisions = strategy.decide(requests, check=check)
            if len(decisions) != len(requests):
                raise ValueError('strategy returned a wrong decision count')
            for index, request, decision in zip(indices, requests, decisions):
                check()
                env, state = environments[index], active[index]
                mask = np.asarray(env.action_masks(), dtype=bool)
                action = decision.action
                if action < 0 or action >= len(mask) or not mask[action]:
                    raise ValueError(f'{decision.source} returned illegal action {action}')
                observation, reward, terminated, truncated, info = env.step(action)
                state['steps'] += 1
                state['raw_return'] += float(reward)
                state.update(observation=observation, info=info)
                for key, value in decision.diagnostics.items():
                    state['metrics'][key] = state['metrics'].get(key, 0) + value
                if state['steps'] == 1 or state['steps'] % 25 == 0:
                    progress(dict(current_episode=state['episode'] + 1, current_steps=state['steps'],
                                  current_floor=info.get('run_stats', {}).get('floor', 0),
                                  current_phase=info.get('phase'), current_action=action,
                                  current_action_source=decision.source, active_workers=len(active)))
                limited = state['steps'] >= config.max_steps and not (terminated or truncated)
                if terminated or truncated or limited:
                    outcome = scenario.outcome(env, info, terminated, truncated)
                    if limited:
                        outcome['result'] = 'step_limit'
                    row = {key: state[key] for key in ('episode', 'seed', 'steps', 'raw_return')}
                    row.update(outcome, **state['metrics'])
                    row['strategy'] = strategy.name
                    yield row
                    replacement = start(env)
                    if replacement is None:
                        del active[index]
                    else:
                        active[index] = replacement
    finally:
        for env in environments:
            env.close()


def summarize(rows):
    """Presentation-independent aggregate schema consumed by CLI and dashboard."""
    results, deaths, death_seeds, wins = {}, {}, {}, []
    for row in rows:
        result = row['result']
        results[result] = results.get(result, 0) + 1
        if result == 'win':
            wins.append(row['seed'])
        elif result == 'death':
            floor = row['floor']
            deaths[floor] = deaths.get(floor, 0) + 1
            death_seeds.setdefault(floor, []).append(row['seed'])
    stats = dict(completed=len(rows), results=results, death_floors=dict(sorted(deaths.items())),
                 death_seeds={key: sorted(value) for key, value in sorted(death_seeds.items())},
                 win_seeds=sorted(wins), floor_mean=float(np.mean([r['floor'] for r in rows])) if rows else 0.0,
                 floor_median=float(np.median([r['floor'] for r in rows])) if rows else 0.0,
                 steps_mean=float(np.mean([r['steps'] for r in rows])) if rows else 0.0)
    for key in ('search_decisions', 'search_simulations', 'search_leaf_evaluations', 'search_seconds'):
        stats[key] = sum(row.get(key, 0) for row in rows)
    return stats
