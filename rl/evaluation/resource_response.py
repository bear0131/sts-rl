"""Paired COMBAT01 resource interventions; diagnostic, never a monotonicity gate."""
from dataclasses import asdict
import time

import numpy as np
import torch

from env.core import encode, native_actions as C
from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig, complete_searches_native, forward_root
from rl.evaluation.strategies import DecisionRequest, FrozenUpperStrategy


def audit(model, env, *, seed, max_steps, timeout, simulations=128):
    if not getattr(model.policy.core, 'frozen_upper', False):
        raise ValueError('resource response audit requires the frozen-upper objective')
    started = time.monotonic()
    def check():
        if time.monotonic() - started >= timeout:
            raise TimeoutError('resource audit budget exhausted')
    model.policy.set_training_mode(False)
    frozen = model.policy.core.teacher.assert_integrity()
    obs, info = env.reset(seed=seed)
    upper = FrozenUpperStrategy(model)
    for step in range(max_steps):
        check()
        if info['phase'] == 'combat':
            break
        decision = upper.decide([DecisionRequest(env, obs, info, seed, step)], check=check)[0]
        obs, _, done, truncated, info = env.step(decision.action)
        if done or truncated:
            raise RuntimeError('no live combat in the audit seed')
    else:
        raise RuntimeError('no combat before max_steps')
    root_before = env.thick.bc.to_dict()
    hp_max = int(env.thick.gc.max_hp)
    variants = [('hp', hp) for hp in sorted({max(1, hp_max//4), max(1, hp_max//2), hp_max})]
    variants += [('potions', count) for count in (0, 1, min(2, env.thick.gc.potion_capacity))]
    branches, rows = [], []
    config = NativeSearchConfig.from_model(model, simulations=simulations, gumbel_scale=0., leaf_batch_size=1)
    try:
        for axis, amount in variants:
            branch = env.fork_run()
            if axis == 'hp':
                branch.thick.gc.cur_hp = amount
                branch.thick.bc.apply_custom_state({'player': {'hp': amount}})
            else:
                potions = ['Fire Potion'] * amount
                branch.thick.gc.set_potion_state(potions)
                branch.thick.bc.apply_custom_state({'potions': potions})
            branch._invalidate_backend_cache()
            branches.append(branch)
            rows.append(dict(axis=axis, amount=amount))
        observations = {key: np.stack([b._obs()[key] for b in branches]) for key in obs}
        logits, values = forward_root(model, observations)
        calls = {i: ((logits[i],), dict(**asdict(config), root_value=float(values[i]),
                  state_nonce=seed, compact_observation=True)) for i in range(len(branches))}
        batch = NativeSearchBatch(branches, 2)
        try:
            check()
            results, stats = complete_searches_native(model, batch, calls)
            check()
        finally:
            batch.close()
        for i, row in enumerate(rows):
            mask = branches[i].action_masks()
            row.update(lower_value=float(values[i]), actor_action=int(np.where(mask, logits[i], -np.inf).argmax()),
                       search_action=int(results[i]['action']), search_value=float(results[i]['root_value']))
        # Obtain a genuine reward screen via the actual transition. Strength is
        # controlled only for this exit probe, never used for training labels.
        settled = env.fork_run(worldline=seed)
        branches.append(settled)
        settled.thick.bc.set_player_strength(999)
        for _ in range(max_steps):
            check()
            if settled._phase() != 'combat':
                break
            mask = settled.action_masks()
            plays = np.flatnonzero(mask[C.AUTOREGRESSIVE_PLAY:C.AUTOREGRESSIVE_POTION]) + C.AUTOREGRESSIVE_PLAY
            action = int(plays[0] if len(plays) else np.flatnonzero(mask)[0])
            _, _, done, truncated, _ = settled.step(action)
            if done or truncated:
                raise RuntimeError('reward probe did not reach a nonterminal exit')
        if settled._phase() != 'combat_reward':
            raise RuntimeError('controlled exit did not reach a reward screen')
        base = settled._obs()
        probes = []
        for axis, amounts in [('hp', [max(1, hp_max//4), max(1, hp_max//2), hp_max]),
                              ('wallet_gold', [0, 100, 300]), ('reward_gold', [0, 20, 100])]:
            for amount in amounts:
                observed = {key: value.copy() for key, value in base.items()}
                if axis == 'hp':
                    observed['run_hp'][:] = (amount/encode.HP_SCALE, hp_max/encode.HP_SCALE, amount/hp_max)
                elif axis == 'wallet_gold':
                    observed['gold'][0] = amount/encode.GOLD_SCALE
                else:
                    observed['reward_gold'][0] = amount/encode.GOLD_SCALE
                probes.append((axis, amount, observed))
        # Held potions are encoded through real simulator state, not hand-edited ids.
        for count in (0, 1, min(2, settled.thick.gc.potion_capacity)):
            settled.thick.gc.set_potion_state(['Fire Potion'] * count)
            settled._invalidate_backend_cache()
            probes.append(('potions', count, settled._obs()))
        tensor, _ = model.policy.obs_to_tensor({k: np.stack([o[k] for _, _, o in probes]) for k in base})
        with torch.inference_mode():
            _, values = model.policy.core.teacher(tensor)
        upper_rows = [dict(axis=axis, amount=amount, upper_value=float(value))
                      for (axis, amount, _), value in zip(probes, values.flatten().cpu())]
        assert env.thick.bc.to_dict() == root_before
        assert model.policy.core.teacher.assert_integrity() == frozen
        return dict(schema='combat-resource-response-v1', seed=seed, seconds=time.monotonic()-started,
                    objective_identity=getattr(model, 'outer_return_contract', None),
                    teacher_sha256=frozen, lower_initialization=model.policy.core.lower_initialization,
                    combat=rows, upper=upper_rows, search_simulations=simulations,
                    leaf_evaluations=stats['leaf_evaluations'],
                    scope={'combat': 'paired real HP/potion state, identical search nonce and zero Gumbel noise',
                           'upper': 'controlled native victory then single-field observation sensitivity; potion ids from native state',
                           'reward_gold': 'unclaimed offer; held wallet unchanged; no fabricated hidden future input to lower',
                           'interpretation': 'finite responses and isolation only; no monotonicity requirement for teacher or random lower'})
    finally:
        for branch in branches:
            branch.close()
