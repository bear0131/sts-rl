"""Lower-only expert iteration: real battle roots -> MCTS visits -> small policy.

The collector intentionally owns no upper policy. Reproducible battle fixtures
exercise this architecture before whole-run PPO integration. Checkpoints include
unfinished battles as seed/action journals, Adam and the shuffle RNG.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import env.core
import slaythespire as sts

from rl.algorithms.updates import optimizer_step
from rl.policies.small_playout import CONTRACT, SmallPlayoutPolicy, collate_features
from rl.search.small_playout import SmallPlayoutSearch, SmallSearchConfig


@dataclass(frozen=True)
class BattleFixture:
    encounter: str = 'Jaw Worm'
    deck: tuple[str, ...] = ('STRIKE_RED',) * 5 + ('DEFEND_RED',) * 4 + ('BASH',)
    relics: tuple[str, ...] = ('Burning Blood',)
    ascension: int = 0

    def create(self, seed):
        game = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, self.ascension)
        game.floor_num = 1
        game.set_deck([(card, 0) for card in self.deck])
        game.set_relics(list(self.relics))
        return sts.create_battle(game, self.encounter)


class BattleCollector:
    def __init__(self, fixture=BattleFixture(), *, n_envs=1, seed=42, max_actions=6000):
        if n_envs < 1 or max_actions < 1:
            raise ValueError('invalid collector dimensions')
        self.fixture, self.seed, self.max_actions = fixture, seed, max_actions
        self.next_seed, self.steps = seed, 0
        self.journals, self.roots = [], []
        for _ in range(n_envs):
            self.journals.append(None)
            self.roots.append(None)
            self._reset(len(self.roots) - 1)

    def _reset(self, index):
        self.roots[index] = self.fixture.create(self.next_seed)
        self.journals[index] = dict(seed=self.next_seed, actions=[])
        self.next_seed += 1

    def collect(self, search, n_steps):
        if n_steps < 1:
            raise ValueError('collection steps must be positive')
        started = time.perf_counter()
        samples, battles, counts = [], [], {}
        for _ in range(n_steps):
            nonces = [self.seed * 1000000007 + self.steps + i for i in range(len(self.roots))]
            results = search.search(self.roots, nonces)
            for i, result in enumerate(results):
                samples.append({key: result[key] for key in ('state', 'actions', 'keys', 'action_bits', 'visits')})
                for key, value in result['stats'].items():
                    if key == 'playout_lengths':
                        counts.setdefault(key, np.zeros(201, np.int64))
                        counts[key] += value
                    else:
                        counts[key] = counts.get(key, 0) + value
                bits = result['action']
                self.roots[i].execute(sts.Action(bits))
                self.journals[i]['actions'].append(bits)
                root = self.roots[i]
                terminal = root.is_battle_over
                capped = len(self.journals[i]['actions']) >= self.max_actions
                if terminal or capped:
                    battles.append(dict(seed=self.journals[i]['seed'], outcome=int(root.outcome),
                                        actions=len(self.journals[i]['actions']), censored=not terminal,
                                        score=sts.small_playout_value(root)))
                    self._reset(i)
            self.steps += len(self.roots)
        counts['playout_lengths'] = counts['playout_lengths'].tolist()
        counts.update(real_transitions=len(samples), completed_battles=sum(not b['censored'] for b in battles),
                      censored_battles=sum(b['censored'] for b in battles),
                      collection_seconds=time.perf_counter() - started)
        return samples, dict(counts=counts, battles=battles)

    def state_dict(self):
        return dict(fixture=asdict(self.fixture), seed=self.seed, max_actions=self.max_actions,
                    next_seed=self.next_seed, steps=self.steps, journals=self.journals)

    @classmethod
    def restore(cls, state):
        collector = cls(BattleFixture(**state['fixture']), n_envs=len(state['journals']),
                        seed=state['seed'], max_actions=state['max_actions'])
        for i, journal in enumerate(state['journals']):
            root = collector.fixture.create(journal['seed'])
            for bits in journal['actions']:
                root.execute(sts.Action(bits))
            if root.is_battle_over:
                raise ValueError('saved live-battle journal ends at a terminal')
            collector.roots[i] = root
        collector.journals = state['journals']
        collector.next_seed, collector.steps = state['next_seed'], state['steps']
        return collector


class SmallPlayoutLearner:
    def __init__(self, *, seed=42, device='cpu', config=SmallSearchConfig(), threads=1, learning_rate=3e-4):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.policy = SmallPlayoutPolicy(sts.small_playout_key_count()).to(self.device)
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.max_grad_norm = 1.
        self.search = SmallPlayoutSearch(self.policy, config, threads)
        self.rng = np.random.default_rng(seed)
        self.updates, self.optimizer_steps = 0, 0

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def update(self, samples, *, batch_size=32, epochs=1):
        if not samples or batch_size < 1 or epochs < 1:
            raise ValueError('invalid compact update batch')
        self.synchronize()
        started = time.perf_counter()
        losses, norms = [], []
        before = self.policy.export()
        for _ in range(epochs):
            order = self.rng.permutation(len(samples))
            for start in range(0, len(samples), batch_size):
                batch = [samples[i] for i in order[start:start+batch_size]]
                tensors = collate_features(batch, self.device)
                target = np.zeros(tensors[-1].shape, np.float32)
                for i, sample in enumerate(batch):
                    visits = np.asarray(sample['visits'], np.float32)
                    if visits.sum() <= 0:
                        raise ValueError('search supervision needs positive visits')
                    target[i, :len(visits)] = visits / visits.sum()
                logp = torch.log_softmax(self.policy(*tensors), dim=-1)
                # Padding contributes exactly zero, avoiding 0 * -inf NaNs.
                logp = logp.masked_fill(~tensors[-1], 0.)
                loss = -(torch.as_tensor(target, device=self.device) * logp).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite compact policy loss')
                loss.backward()
                norms.append(optimizer_step(self))
                losses.append(loss.detach())
                self.optimizer_steps += 1
        self.search.refresh(self.policy)
        self.synchronize()
        self.updates += 1
        return dict(loss=float(torch.stack(losses).mean()), grad_norm=float(torch.stack(norms).mean()),
                    parameter_l2_change=float(np.linalg.norm(self.policy.export() - before)),
                    optimization_seconds=time.perf_counter() - started,
                    updates=self.updates, optimizer_steps=self.optimizer_steps, samples=len(samples))

    def save(self, path, collector):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(contract=CONTRACT, key_count=self.policy.key_count,
                       native_sha256=hashlib.sha256(Path(sts.__file__).read_bytes()).hexdigest(),
                       policy=self.policy.state_dict(), optimizer=self.policy.optimizer.state_dict(),
                       rng=self.rng.bit_generator.state, config=asdict(self.search.config),
                       collector=collector.state_dict(), updates=self.updates, optimizer_steps=self.optimizer_steps)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device='cpu', threads=1):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload['contract'] != CONTRACT or payload['key_count'] != sts.small_playout_key_count():
            raise ValueError('incompatible small policy checkpoint contract')
        if payload['native_sha256'] != hashlib.sha256(Path(sts.__file__).read_bytes()).hexdigest():
            raise ValueError('battle journal restore requires the same native build')
        learner = cls(device=device, config=SmallSearchConfig(**payload['config']), threads=threads)
        learner.policy.load_state_dict(payload['policy'])
        learner.policy.optimizer.load_state_dict(payload['optimizer'])
        learner.search.refresh(learner.policy)
        learner.rng.bit_generator.state = payload['rng']
        learner.updates, learner.optimizer_steps = payload['updates'], payload['optimizer_steps']
        return learner, BattleCollector.restore(payload['collector'])
