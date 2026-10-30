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


def update_online(model, *, rank=0, world_size=1, average=lambda values: values, synchronize=None):
    """Fresh search-visit expert iteration, with globally sample-weighted Adam.

    Every rank runs the same number of steps, including empty ranks/tails.
    Microbatches bound memory; only the effective batch is clipped and stepped.
    The outer trainer owns gradient transport and checkpoint/RNG restoration.
    """
    if not hasattr(model.policy, 'small_playout'):
        return {}
    from rl.algorithms.upper_ppo import train_phase
    from rl.policies.small_playout import OnlinePlayoutConfig, PreparedPlayoutSamples
    cfg = OnlinePlayoutConfig(**model.policy.small_playout_config)
    samples = getattr(model, '_small_playout_samples', [])
    counts = average({f'rank_{i}': float(len(samples) if i == rank else 0) for i in range(world_size)})
    sizes = [int(round(counts[f'rank_{i}'] * world_size)) for i in range(world_size)]
    out = dict(samples=sum(sizes), updates=0, loss=0., entropy=0., grad_norm=0., sync_seconds=0.,
               batch_size_per_rank=cfg.batch_size, batch_size_global=cfg.batch_size * world_size,
               micro_batch_size=cfg.micro_batch_size)
    if not out['samples']:
        return out
    net, optimizer = model.policy.small_playout, model.policy.small_playout_optimizer
    before = net.export()
    model._sync_device()
    started = time.perf_counter()
    totals = dict(loss=0., entropy=0., target_entropy=0., teacher_top1=0., uniform_ce=0.)
    prepared = PreparedPlayoutSamples(samples, cfg.target_mix) if samples else None
    if prepared is not None:
        totals.update(target_entropy=prepared.target_entropy * cfg.epochs,
                      uniform_ce=prepared.uniform_ce * cfg.epochs)
    device_totals = {key: [] for key in ('loss', 'entropy', 'teacher_top1')}
    norms = []
    model.policy.set_training_mode(True)
    with train_phase(model, small_playout=True):
        for _ in range(cfg.epochs):
            order = np.random.permutation(len(samples))
            for start in range(0, max(sizes), cfg.batch_size):
                mean_count = sum(min(cfg.batch_size, max(0, n-start)) for n in sizes) / world_size
                optimizer.zero_grad(set_to_none=True)
                indices = order[start:start+cfg.batch_size]
                for offset in range(0, len(indices), cfg.micro_batch_size):
                    tensors, target, teacher = prepared.batch(
                        indices[offset:offset+cfg.micro_batch_size], model.device)
                    # Compact learner is FP32 even when upper PPO uses BF16.
                    logp = torch.log_softmax(net.forward_packed(*tensors), -1).masked_fill(~tensors[-1], 0.)
                    loss = -(target * logp).sum()
                    entropy = -(logp.exp() * logp * tensors[-1]).sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite online compact loss')
                    (loss / mean_count).backward()
                    device_totals['loss'].append(loss.detach())
                    device_totals['entropy'].append(entropy.detach())
                    device_totals['teacher_top1'].append((
                        logp.masked_fill(~tensors[-1], -torch.inf).argmax(-1) == teacher).sum())
                if synchronize is not None:
                    stats = synchronize(out['updates'])
                    out['sync_seconds'] += float((stats or {}).get('sync_sec', 0.))
                norms.append(optimizer_step(model, optimizer=optimizer,
                    max_grad_norm=cfg.max_grad_norm if cfg.max_grad_norm else float('inf')))
                out['updates'] += 1
    for key, values in device_totals.items():
        totals[key] = float(torch.stack(values).double().sum()) if values else 0.
    totals = average(totals)
    mean_samples = out['samples'] / world_size * cfg.epochs
    out.update({key: value / mean_samples for key, value in totals.items()})
    out['target_kl'] = out['loss'] - out['target_entropy']
    out['grad_norm'] = float(torch.stack(norms).mean())
    out['parameter_l2_change'] = float(np.linalg.norm(net.export()-before))
    model.small_playout_optimizer_steps = getattr(model, 'small_playout_optimizer_steps', 0) + out['updates']
    out['optimizer_steps_total'] = model.small_playout_optimizer_steps
    model._small_playout_samples = []
    model._sync_device()
    out['seconds'] = time.perf_counter()-started
    for key, value in out.items():
        model.logger.record('small_playout_train/'+key, value)
    return out


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

    def reinforce(self, roots, *, nonces, k=8, batch_size=2048, entropy_coef=.01):
        """Independent on-policy playouts with a leave-one-out start baseline.

        This is a separate stage after visit distillation, not CE on MCTS actions
        weighted by outcomes. Censored groups provide no terminal supervision.
        """
        if k < 2 or len(roots) != len(nonces) or not roots:
            raise ValueError('REINFORCE needs nonempty starts and at least two playouts each')
        began = time.perf_counter()
        traces = sts.small_playout_collect_batch(self.search.pool, self.search.net,
            [root for root in roots for _ in range(k)], self.search.config.native(),
            [(int(nonce) + i) & ((1 << 64)-1) for nonce in nonces for i in range(k)])
        usable = [traces[i:i+k] for i in range(0, len(traces), k)
                  if not any(t['censored'] for t in traces[i:i+k])]
        stats = dict(playouts=len(traces), transitions=sum(t['transitions'] for t in traces),
                     censored=sum(t['censored'] for t in traces), accepted_groups=len(usable),
                     collection_seconds=time.perf_counter()-began)
        if not usable:
            return dict(stats, samples=0, loss=None, optimizer_steps=self.optimizer_steps)
        rewards = np.asarray([[t['value'] for t in group] for group in usable], np.float64)
        advantages = leave_one_out_advantages(rewards)
        samples = [dict(step, advantage=float(advantages[i, j]))
                   for i, group in enumerate(usable) for j, trace in enumerate(group)
                   for step in trace['steps'] if len(step['action_bits']) > 1]
        if not samples:
            return dict(stats, samples=0, loss=None, optimizer_steps=self.optimizer_steps)
        losses, norms = [], []
        before = self.policy.export()
        order = self.rng.permutation(len(samples))
        for start in range(0, len(samples), batch_size):
            rows = [samples[i] for i in order[start:start+batch_size]]
            tensors = collate_features(rows, self.device)
            logp = torch.log_softmax(self.policy(*tensors), dim=-1)
            choices = torch.as_tensor([r['choice'] for r in rows], device=self.device)
            advantage = torch.as_tensor([r['advantage'] for r in rows], device=self.device)
            chosen = logp.gather(1, choices[:, None]).squeeze(1)
            entropy = -(logp.exp() * logp.masked_fill(~tensors[-1], 0.)).sum(-1)
            loss = -(advantage * chosen).mean() - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite playout REINFORCE loss')
            loss.backward()
            norms.append(optimizer_step(self))
            losses.append(loss.detach())
            self.optimizer_steps += 1
        self.search.refresh(self.policy)
        self.synchronize()
        self.updates += 1
        return dict(stats, samples=len(samples), loss=float(torch.stack(losses).mean()),
                    grad_norm=float(torch.stack(norms).mean()), reward_mean=float(rewards.mean()),
                    reward_std=float(rewards.std()), parameter_l2_change=float(np.linalg.norm(self.policy.export()-before)),
                    optimizer_steps=self.optimizer_steps, seconds=time.perf_counter()-began)

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


def leave_one_out_advantages(rewards):
    rewards = np.asarray(rewards, np.float64)
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not np.isfinite(rewards).all():
        raise ValueError('leave-one-out baseline needs finite [starts, k>=2] rewards')
    advantage = rewards - (rewards.sum(1, keepdims=True)-rewards)/(rewards.shape[1]-1)
    return advantage / (advantage.std()+1e-6)


def save_policy_pair(path, policy, prior, *, provenance):
    """Freeze independent playout/PUCT parameters after their own pretraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(kind='sts1-small-playout-pair-v1', contract=CONTRACT,
                   key_count=policy.key_count,
                   policy={k: v.detach().cpu() for k, v in policy.state_dict().items()},
                   prior={k: v.detach().cpu() for k, v in prior.state_dict().items()},
                   provenance=provenance)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy_pair(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('kind') != 'sts1-small-playout-pair-v1' or payload['contract'] != CONTRACT
            or payload['key_count'] != sts.small_playout_key_count()):
        raise ValueError('incompatible compact policy pair')
    nets = [SmallPlayoutPolicy(payload['key_count']) for _ in range(2)]
    for net, key in zip(nets, ('policy', 'prior'), strict=True):
        net.load_state_dict(payload[key], strict=True)
        net.eval().requires_grad_(False)
        if not np.isfinite(net.export()).all():
            raise ValueError('nonfinite compact policy pair')
    return *nets, payload['provenance']
