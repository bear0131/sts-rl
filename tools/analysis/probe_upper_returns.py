"""Bounded paired continuations and a full-episode upper-return ledger.

Protected source models are never written. Optional value-only fitting restores
the loaded parameters afterwards; forced branch actions never become PPO data.
Reuses the evaluation runner, production rule search and macro-transition capture.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from env.core import action_space as A, native_actions as C
from env.core.run_env import SlayTheSpireEnv
from env.search.native_batch import NativeSearchBatch
from env.search.turn_planner import TurnPlannerConfig
from rl.algorithms import upper_ppo as U
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.strategies import Decision, PhaseStrategy
from rl.policies import model as M
from rl.policies.checkpoint import load_checkpoint


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def category(phase, action):
    if phase == 'chest':
        return 'open' if action == A._CHEST else 'skip_chest'
    if A._RRELIC <= action < A._RRELIC + A.REWARD_RELIC:
        return 'take_relic'
    if action == A._PROCEED:
        return 'proceed'
    return phase


def independent_reward(before, after, done, censored):
    if censored:
        return 0.
    floors = max(0, after['floor'] - before['floor'])
    bonus = sum(b for a, b in ((1, .1), (2, .2)) if before['act'] <= a < after['act'])
    won = done and (after['act'] > 3 or after['outcome'] == 'PLAYER_VICTORY')
    return floors / 500. + bonus + int(won)


def annotate_returns(records, complete):
    """Never call a censored prefix's bootstrap estimate a Monte Carlo return."""
    if not complete:
        return dict(complete=False, macros=len(records))
    assert records and records[-1]['terminal']
    boot = [r['bootstrap'] for r in records]
    a95, r95 = U.macro_advantages(records, boot, 1., .95)
    a1, r1 = U.macro_advantages(records, boot, 1., 1.)
    mc = np.cumsum([r['reward'] for r in records][::-1], dtype=np.float64)[::-1]
    error = float(np.max(np.abs(r1 - mc)))
    if error > 2e-6:
        raise AssertionError(f'lambda=1 does not telescope to terminal return: {error}')
    for i, row in enumerate(records):
        row.update(td=row['reward'] + boot[i] - row['value'], advantage95=float(a95[i]),
                   target95=float(r95[i]), mc_return=float(mc[i]), mc_advantage=float(a1[i]))
    # Production releases a macro when it completes, even if its entering action
    # preceded the collection boundary. Partition by completion, not start.
    for horizon in (32, 64, 128, 256, 512):
        groups = {}
        for i, row in enumerate(records):
            groups.setdefault((row['end']-1)//horizon, []).append(i)
        for indices in groups.values():
            adv, _ = U.macro_advantages([records[i] for i in indices], [boot[i] for i in indices], 1., .95)
            for i, value in zip(indices, adv, strict=True):
                records[i][f'advantage_n{horizon}'] = float(value)
    return dict(complete=True, macros=len(records), lambda1_max_error=error)


class EpisodeLedger:
    def __init__(self, env, output, case, save_training=False):
        self.env, self.output, self.case = env, output, case
        self.capture = SimpleNamespace(_upper_completed=[], _upper_censored=0)
        self.records, self.events = [], []
        self.steps, self.total, self.macro_reward = 0, 0., 0.
        self.last_done, self.last_censored = False, False
        self.waiting_hash = None
        self.start_stats = env.thick.run_stats()
        self.save_training = save_training
        self.training_rows = []

    def upper(self, value, log_prob):
        self.value, self.log_prob = float(value), float(log_prob)
        if self.records and not self.records[-1]['terminal']:
            assert self.waiting_hash == self.env._array_mapping_digest(
                {k: v for k, v in self.env._obs().items() if k in U.UPPER_KEYS})
            self.records[-1]['bootstrap'] = self.value

    def before_step(self, action, native_action):
        assert native_action is None  # Production fast-rule adapter returns thin actions.
        self.before_obs = {k: np.asarray(v)[None] for k, v in self.env._obs().items() if k in U.UPPER_KEYS}
        self.before = self.env.thick.run_stats()
        self.phase = self.env._phase()
        self.mask = self.env.action_masks()[None]
        if self.phase != 'combat':
            if hasattr(self.capture, '_upper_pending'):
                assert self.capture._upper_pending[0] is None
            self.pending_meta = dict(kind=category(self.phase, action), phase=self.phase,
                                     floor=int(self.before['floor']), act=int(self.before['act']), start=self.steps)
        else:
            self.value = self.log_prob = 0.

    def after_step(self, action, reward, terminated, truncated, info, native_action):
        self.steps += 1
        done = bool(terminated or truncated)
        censored = bool(truncated or info.get('dead_screen'))
        after = info['run_stats']
        expected = independent_reward(self.before, after, done, censored)
        self.total += expected
        if expected or done:
            self.events.append(dict(step=self.steps, before=self.before, after=after,
                                    reward=expected, done=done, censored=censored))
        nxt = {k: np.asarray(v)[None] for k, v in self.env._obs().items() if k in U.UPPER_KEYS}
        capture_info = dict(info)
        capture_info['TimeLimit.truncated'] = censored
        U.capture_step(self.capture, self.before_obs, [action], self.mask, [self.value], [self.log_prob],
                       nxt, [done], [capture_info])
        for row in self.capture._upper_completed:
            self.macro_reward += row['reward']
            self.waiting_hash = (None if row['terminal'] else
                                 self.env._array_mapping_digest(row['next_observation']))
            self.records.append({**self.pending_meta, **{k: row[k] for k in
                ('action', 'value', 'log_prob', 'reward', 'env', 'episode', 'terminal')},
                'end': self.steps, 'bootstrap': 0. if row['terminal'] else None})
            if self.save_training:
                self.training_rows.append(row)
        self.capture._upper_completed.clear()
        self.last_done, self.last_censored = done, censored

    def finish(self, result):
        complete = self.last_done and not self.last_censored
        audit = annotate_returns(self.records, complete)
        if complete and abs(self.total - self.macro_reward) > 1e-8:
            raise AssertionError('event ledger and production macro rewards disagree')
        data = dict(case=self.case, result=result, strategic_return=self.total,
                    start=self.start_stats, end=self.env.thick.run_stats(),
                    **audit, records=self.records, events=self.events)
        write(self.output, data)
        if self.save_training and complete:
            for row, scalar in zip(self.training_rows, self.records, strict=True):
                row.update(probe_end=scalar['end'], bootstrap=scalar['bootstrap'])
            with gzip.open(self.output.with_suffix('.pt.gz'), 'wb', compresslevel=1) as f:
                torch.save(dict(case=self.case, records=self.training_rows), f)
            if sum(p.stat().st_size for p in self.output.parent.iterdir() if p.is_file()) > 512*1024**2:
                raise RuntimeError('diagnostic output exceeded 512 MiB budget')
        return dict(path=str(self.output), strategic_return=self.total, **audit)


class SampledUpper:
    name = 'paired-stochastic-upper'

    def __init__(self, model):
        self.model = model

    def decide(self, requests, *, check):
        check()
        obs = {k: torch.as_tensor(np.stack([r.observation[k] for r in requests]), device=self.model.device)
               for k in requests[0].observation if k in U.UPPER_KEYS}
        masks = torch.as_tensor(np.stack([r.env.action_masks() for r in requests]), device=self.model.device)
        with torch.no_grad(), self.model.autocast_context():
            logits, values = self.model.policy.core.actor_critic(obs)
            probs = logits.float().masked_fill(~masks, -1e9).softmax(-1).cpu().numpy()
            values = values.flatten().float().cpu().numpy()
        result = []
        for r, p, value in zip(requests, probs, values, strict=True):
            # One independent categorical stream per continuation. Sharing a seed
            # across branches couples draws without consulting hidden game RNG.
            p = p.astype(np.float64); p /= p.sum()
            action = int(r.env.probe_rng.choice(len(p), p=p))
            r.env.probe_ledger.upper(value, np.log(p[action]))
            result.append(Decision(action, self.name))
        return result


class FastRules:
    name = 'production-v8-native-4threads'

    def __init__(self):
        self.config = TurnPlannerConfig(worlds=2, node_budget=8192, beam_width=8,
            max_depth=16, master_nonce=20260918, lookahead_turns=1, leaf_profile='legacy-v8')

    def bind_environments(self, envs):
        self.pool = NativeSearchBatch(envs, 4)
        self.indices = {id(e): i for i, e in enumerate(envs)}

    def unbind_environments(self):
        self.pool.close()

    def decide(self, requests, *, check):
        check()
        calls = {self.indices[id(r.env)]: ((), dict(asdict(self.config),
            state_nonce=(r.env.probe_nonce ^ (r.step * 0x9E3779B97F4A7C15)) & ((1 << 64)-1))) for r in requests}
        results = self.pool.rulebase_turn_decisions(calls)
        return [Decision(int(results[self.indices[id(r.env)]]['action']), self.name) for r in requests]


def new_env(model):
    return SlayTheSpireEnv(combat_action_contract=C.policy_contract(model.observation_space),
        global_map_view=M.uses_global_map_observation(model), card_language=M.uses_card_language(model),
        combat_language=M.uses_combat_language(model), deck_size_observation=M.uses_deck_size_observation(model),
        neow_observation=M.uses_neow_observation(model), event_option_observation=M.uses_event_option_observation(model))


def branch_states(model, panel, count):
    pairs, seen, exclusions = [], set(), []
    for i, row in enumerate(panel):
        if row['kind'] != 'relic' or row['state']['backend_seed'] in seen:
            continue
        env = new_env(model)
        try:
            env.restore_training_state(row['state'])
            for _ in range(4):
                gold = [int(a) for a in np.flatnonzero(env.action_masks()) if A._RGOLD <= a < A._RGOLD+A.REWARD_GOLD]
                if not gold:
                    break
                env.step(gold[0])
            common = env.export_training_state()
            take = next(int(a) for a in np.flatnonzero(env.action_masks()) if A._RRELIC <= a < A._RRELIC+A.REWARD_RELIC)
            branches = []
            for action in (take, A._PROCEED):
                env.restore_training_state(common)
                before = env.thick.run_stats()
                _, _, done, truncated, info = env.step(action)
                assert not done and not truncated
                assert U.strategic_reward(before['act'], before['floor'], info['run_stats'], done=False, censored=False) == 0
                for _ in range(8):
                    if env._phase() != 'combat_reward' or not env.action_masks()[A._PROCEED]:
                        break
                    env.step(A._PROCEED)
                branches.append((env._phase(), env.export_training_state(), env.thick.obs_canonical()['global']))
            if any(b[0] != 'map' for b in branches):
                exclusions.append(dict(panel_index=i, reason='requires additional non-reward choice'))
                continue
            pairs.append(dict(panel_index=i, seed=common['backend_seed'],
                relic=row['canon']['options']['relics'][0], common_steps=len(common['actions']),
                states=[b[1] for b in branches], globals=[b[2] for b in branches]))
            seen.add(common['backend_seed'])
            if len(pairs) == count:
                break
        finally:
            env.close()
    return pairs, exclusions


def audit_values(paths, panel_path, output, count):
    """CPU FP32 semantic/batch checks; fresh initialization is only auxiliary."""
    from env.core import vocab as V
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    model = load_checkpoint(paths[0], device='cpu')
    pairs, exclusions = branch_states(model, torch.load(panel_path, weights_only=False), count)
    env, observations, semantic = new_env(model), [], []
    for pair in pairs:
        branch_obs = []
        for state in pair['states']:
            obs, _ = env.restore_training_state(state)
            canonical = env.thick.obs_canonical()['global']
            actual = obs['relics'][obs['relics'] != 0].tolist()
            expected = [V.RELIC_VOCAB_INDEX.get(r, 0) for r in canonical['relics']]
            assert actual == expected and all(expected), 'owned relic ID mapping mismatch'
            observations.append({k: v for k, v in obs.items() if k in U.UPPER_KEYS})
            branch_obs.append(actual)
        assert len(branch_obs[0]) == len(branch_obs[1])+1
        semantic.append(dict(seed=pair['seed'], relic=pair['relic'], owned_take=branch_obs[0], owned_skip=branch_obs[1]))
    env.close()
    batch = {k: torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}
    report = dict(semantic=semantic, exclusions=exclusions, models=[])
    def probe(policy, name):
        policy.set_training_mode(False)
        with torch.no_grad():
            logits, values = policy.core.actor_critic(batch)
            singleton = torch.cat([policy.core.actor_critic({k: v[i:i+1] for k, v in batch.items()})[1]
                                   for i in range(len(observations))])
            error = float((singleton-values).abs().max())
            _, _, pad = policy.core.overworld.encoder._relic_block(batch, policy.core.upper_tokenizer,
                'relic', 'relics', batch['relic_counters'], trim_trailing_padding=True)
            assert torch.equal((~pad).sum(1), (batch['relics'] != 0).sum(1)), 'owned relic tokens masked out'
            delta = (values.flatten()[::2]-values.flatten()[1::2]).tolist()
        report['models'].append(dict(name=name, delta=delta, singleton_value_max_error=error))
        write(output, report)
    for path in paths:
        model = load_checkpoint(path, device='cpu')
        probe(model.policy, str(path))
    for seed in range(4):
        torch.manual_seed(seed)
        policy = model.policy_class(model.observation_space, model.action_space, model.lr_schedule, **model.policy_kwargs)
        probe(policy, f'fresh-initialization-{seed}')
        del policy
    return report


def fit_terminal_dataset(paths, directory, output):
    """Value-only label intervention, split by original state before fitting.

    All selected continuations must end; never select the first completed subset.
    The paired forced initial action is absent from these legitimate critic rows.
    """
    from stable_baselines3.common.logger import configure
    cfg = json.loads((directory/'manifest.json').read_text())
    assert (directory/'complete.json').exists()
    results = dict(models=[], split='even original pair train, odd pair validation')
    for model_index, path in enumerate(paths):
        model = load_checkpoint(path, device='cuda')
        model.configure_nosl_is_az_inner_only(); model.configure_rulebase_lower()
        model.set_logger(configure(str(output/'logs'), []))
        training, validation, probe = [], [], {}
        for file in sorted(directory.glob(f'trace-{model_index}-*.pt.gz')):
            with gzip.open(file, 'rb') as f:
                data = torch.load(f, weights_only=False)
            case = data['case']; rows = data['records']
            assert rows and rows[-1]['terminal']
            index = 2*(case['pair']*cfg['repeats']+case['repeat'])+case['branch']
            for row in rows:
                row['env'] = index
            (training if case['pair'] % 2 == 0 else validation).extend(rows)
            probe.setdefault((case['pair'], case['branch']), rows[0]['observation'])
        assert len(probe) == 2*cfg['states'] and training and validation
        assert len(list(directory.glob(f'trace-{model_index}-*.pt.gz'))) == 2*cfg['states']*cfg['repeats']

        def terminal_targets(rows):
            tails, ret = {}, np.zeros(len(rows), np.float32)
            for i in range(len(rows)-1, -1, -1):
                r = rows[i]; key = (r['env'], r['episode'])
                tails[key] = r['reward'] + (0. if r['terminal'] else tails[key])
                ret[i] = tails[key]
            return ret

        train_mc, valid_mc = terminal_targets(training), terminal_targets(validation)
        bootstrap = np.asarray([r['bootstrap'] for r in training], np.float32)
        original_adv, original_loss, original_step = U.macro_advantages, U.ppo_loss_components, U.optimizer_step
        groups = {}
        for i, row in enumerate(training):
            groups.setdefault((row['probe_end']-1)//128, []).append(i)
        a128 = np.zeros(len(training), np.float32)
        for indices in groups.values():
            a128[indices], _ = original_adv([training[i] for i in indices], bootstrap[indices], 1., .95)
        old_values = np.asarray([r['value'] for r in training], np.float32)
        targets = {'n128': a128+old_values, 'mc': train_mc}
        base = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        opt = copy.deepcopy(model.policy.upper_optimizer.state_dict())
        config = replace(model.policy.core.upper_config, micro_batch_size=1024, epochs=10)
        model.policy.core.upper_config = config
        model._upper_pending = []; model._upper_censored = 0

        def values(observations):
            out = []
            model.policy.set_training_mode(False)
            with torch.no_grad(), model.autocast_context():
                for start in range(0, len(observations), 128):
                    batch = observations[start:start+128]
                    obs = {k: torch.as_tensor(np.stack([r[k] for r in batch]), device=model.device) for k in batch[0]}
                    _, v = model.policy.core.actor_critic(obs)
                    out.extend(v.flatten().float().cpu().tolist())
            return np.asarray(out)

        def measure():
            v = values([probe[(p, b)] for p in range(cfg['states']) for b in range(2)])
            dv = v[::2]-v[1::2]
            vv = values([r['observation'] for r in validation])
            return dict(delta=dv.tolist(), train_delta=float(dv[::2].mean()), validation_delta=float(dv[1::2].mean()),
                validation_mc_mse=float(np.mean((vv-valid_mc)**2)))

        item = dict(source=str(path), train_samples=len(training), validation_samples=len(validation),
                    baseline=measure(), arms=[])
        arm = 'n128'
        try:
            U.ppo_loss_components = lambda **kw: replace(original_loss(**kw),
                policy_loss=kw['log_prob'].sum()*0, entropy_loss=kw['entropy'].sum()*0)
            U.macro_advantages = lambda records, bootstrap, gamma, gae_lambda: (targets[arm]-old_values, targets[arm])
            for arm in ('n128', 'mc'):
                model.policy.load_state_dict(base)
                model.policy.upper_optimizer.load_state_dict(copy.deepcopy(opt))
                model._upper_completed = training
                stats = U.update_upper(model)
                assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                item['arms'].append(dict(arm=arm, stats=stats, probe=measure()))
                print('VALUE_FIT', model_index, arm, json.dumps(item['arms'][-1]), flush=True)
        finally:
            U.macro_advantages, U.ppo_loss_components, U.optimizer_step = original_adv, original_loss, original_step
            model.policy.load_state_dict(base)
            model.policy.upper_optimizer.load_state_dict(opt)
        item['restored_exact'] = all(torch.equal(v.cpu(), base[k]) for k, v in model.policy.state_dict().items())
        results['models'].append(item)
        write(output/'fit.json', results)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=5400)
    parser.add_argument('--values-only', action='store_true')
    parser.add_argument('--save-training-data', action='store_true')
    parser.add_argument('--fit-dataset', type=Path)
    args = parser.parse_args()
    if not (1 <= args.states <= 8 and 1 <= args.repeats <= 4 and 0 < args.seconds <= 5400):
        raise ValueError('diagnostic budget exceeds protocol')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
    started = time.monotonic()
    panel_path = args.root/f'runs/upper06/rank{args.rank}/diagnosis/panel.pt'
    paths = [args.root/f'runs/{area}/rank{args.rank}/step-{step:012d}/model.zip'
             for area, step in [('perf20', 22806528), ('upper06', 33816576)]]
    before_sha = [sha(p) for p in paths]
    if args.fit_dataset:
        fit_terminal_dataset(paths, args.fit_dataset, args.output)
        assert [sha(p) for p in paths] == before_sha
        write(args.output/'complete.json', dict(source_unchanged=True, seconds=time.monotonic()-started))
        return
    if args.values_only:
        initial = args.root/'runs/initial-frozen-cardvec-n384-upper10/final.zip'
        audit_values([initial, *paths], panel_path, args.output/'values-audit.json', args.states)
        assert [sha(p) for p in paths] == before_sha
        write(args.output/'complete.json', dict(source_unchanged=True, seconds=time.monotonic()-started))
        return
    model = load_checkpoint(paths[0], device='cuda')
    pairs, exclusions = branch_states(model, torch.load(panel_path, weights_only=False), args.states)
    assert len(pairs) == args.states
    manifest = dict(rank=args.rank, source=[str(p) for p in paths], model_sha=before_sha,
        script_sha=sha(__file__), panel_sha=sha(panel_path), states=args.states, repeats=args.repeats,
        max_steps=20000, seconds=args.seconds, upper='stochastic categorical, independent per-case streams',
        lower=asdict(FastRules().config), nonce='paired continuation counter; not production scheduling replay',
        exclusions=exclusions, pairs=[{k: v for k, v in p.items() if k != 'states'} for p in pairs])
    write(args.output/'manifest.json', manifest)
    all_rows = []
    for model_index, path in enumerate(paths):
        if model_index:
            del model
            torch.cuda.empty_cache()
            model = load_checkpoint(path, device='cuda')
        values = []
        env = new_env(model)
        observations = []
        for p in pairs:
            for state in p['states']:
                obs, _ = env.restore_training_state(state)
                observations.append({k: v for k, v in obs.items() if k in U.UPPER_KEYS})
        env.close()
        with torch.no_grad(), model.autocast_context():
            obs = {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=model.device) for k in observations[0]}
            _, v = model.policy.core.actor_critic(obs)
            values = v.flatten().float().cpu().tolist()
        write(args.output/f'values-{model_index}.json', values)
        cases = [dict(model=model_index, pair=p, repeat=r, branch=b) for p in range(len(pairs))
                 for r in range(args.repeats) for b in range(2)]

        def reset(env, index):
            case = cases[index]; pair = pairs[case['pair']]
            result = env.restore_training_state(pair['states'][case['branch']])
            stream = np.random.SeedSequence([60609, args.rank, case['pair'], case['repeat']])
            env.probe_rng = np.random.default_rng(stream)
            env.probe_nonce = 20260918 ^ (case['pair'] << 24) ^ (case['repeat'] << 40) ^ pair['common_steps']
            return result

        def trace(env, index, seed):
            env.probe_ledger = EpisodeLedger(env, args.output/f'trace-{model_index}-{index:03d}.json', cases[index], args.save_training_data)
            return env.probe_ledger

        strategy = PhaseStrategy(SampledUpper(model), {'combat': FastRules()})
        cfg = EvaluationConfig(episodes=len(cases), seed=0, num_envs=16, max_steps=20000,
                               timeout=max(.001, args.seconds-(time.monotonic()-started)))
        for row in evaluate(strategy, Scenario('paired-relic-map', lambda: new_env(model), reset=reset), cfg,
                            cancelled=lambda: (args.output/'STOP').exists(), trace_factory=trace):
            row.update(cases[row['episode']]); all_rows.append(row)
            write(args.output/'rows.json', all_rows)
            print('FINISHED', model_index, len(all_rows), row['result'], round(time.monotonic()-started, 2), flush=True)
    assert [sha(p) for p in paths] == before_sha
    write(args.output/'complete.json', dict(rows=len(all_rows), seconds=time.monotonic()-started,
        source_unchanged=True, max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    main()
