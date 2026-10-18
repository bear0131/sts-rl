"""STS2 candidate policies evaluated through the shared DecisionStrategy runner."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from env.sts2.game import EnvConfig, Sts2Env
from rl.evaluation.contracts import Decision
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate, summarize
from rl.policies.sts2 import CandidatePolicy, batch_observations, compact


class Sts2Strategy:
    name = 'sts2-policy'

    def __init__(self, checkpoint=None, seed=10000):
        self.rng = np.random.default_rng(seed)
        self.policy = None
        self.contract = None
        self.env_config = EnvConfig()
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['format'] != 'sts2-ppo-v1':
                raise ValueError('Unsupported checkpoint')
            self.policy = CandidatePolicy(payload['config']['hidden'])
            self.policy.load_state_dict(payload['policy'])
            self.policy.eval()
            self.contract = payload['contract']
            self.env_config = EnvConfig(**payload['env_config'])
        else:
            self.name = 'sts2-random'

    def decide(self, requests, *, check=lambda: None):
        check()
        if not requests:
            return []
        if self.contract is not None:
            for request in requests:
                # Evaluation can lengthen episodes; rule/codec settings cannot change.
                actual = dict(request.env.contract)
                actual['config'] = dict(actual['config'], max_steps=self.contract['config']['max_steps'])
                if actual != self.contract:
                    raise ValueError('Evaluation environment/backend does not match checkpoint')
        if self.policy is None:
            actions = [self.rng.choice(np.flatnonzero(r.env.action_masks())) for r in requests]
        else:
            with torch.no_grad():
                distribution, _ = self.policy(batch_observations([compact(r.observation) for r in requests], 'cpu'))
                actions = distribution.probs.argmax(dim=-1).tolist()
        return [Decision(int(action), self.name) for action in actions]


def outcome(env, info, terminated, truncated):
    return dict(result=info['outcome'], outcome=info['outcome'], floor=info['floor'], act=info['act'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('combat', 'run'))
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    torch.set_num_threads(1)
    strategy = Sts2Strategy(args.checkpoint, args.seed)
    config = replace(strategy.env_config, max_steps=args.max_steps)
    if args.task:
        config = replace(config, task=args.task)
    evaluation = EvaluationConfig(episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        rows = list(evaluate(strategy, Scenario('sts2', lambda: Sts2Env(config), outcome), evaluation))
        summary = summarize(rows)
        if config.task == 'combat':
            # A first-room combat win is not an act-three clear.
            summary = {key: value for key, value in summary.items() if not key.startswith('act')}
        result = dict(summary=summary, episodes=rows, task=config.task)
        json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
