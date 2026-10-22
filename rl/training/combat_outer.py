"""Construct a fresh COMBAT01 learner from verified teacher material; no CLI."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from env.core.outer_return import OBJECTIVE, CONTEXT_SCHEMA, SCALAR_OBJECTIVES, FINITE_OBJECTIVE
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.outer_return import OuterReturnPolicy, load_initialization_material
from rl.algorithms.combat_distribution import CONTRACT as COMBAT_OBJECTIVE
from rl.training.environments import make_env, observation_flags


def construct(args, profile):
    material = dict(material_directory=str(args.material_directory.resolve()),
                    material_sha256=args.material_sha256, source_sha256=profile['source']['model_sha256'],
                    lower_initialization=profile['contract']['lower_init'],
                    objective=profile['contract']['objective_id'])
    if material['objective'] == FINITE_OBJECTIVE:
        bank_path = args.material_directory / 'curve-bank.json'
        if hashlib.sha256(bank_path.read_bytes()).hexdigest() != profile['contract']['curve_bank_file_sha256']:
            raise ValueError('finite curve bank file SHA256 mismatch')
        material['finite_curve_bank'] = json.loads(bank_path.read_text(encoding='utf-8'))
    source, space = load_initialization_material(directory=material['material_directory'],
        manifest_sha256=material['material_sha256'], source_sha256=material['source_sha256'])
    source.frozen_upper = True
    source.training_objective = material['objective']
    prototype = SimpleNamespace(policy=SimpleNamespace(core=source), observation_space=space,
                                _outer_return=material['objective'] in SCALAR_OBJECTIVES)
    # Preserve optional source observation fields; the lower extension is explicit.
    schema_env = SlayTheSpireEnv(combat_action_contract='autoregressive-v2', outer_return_context=True,
                                **observation_flags(prototype))
    prototype.observation_space = schema_env.observation_space
    schema_env.close()
    env = make_env(prototype, args)
    del source, prototype
    training, search = profile['training'], profile['search']
    search_kwargs = {'nosl_is_az_' + ('search_mode' if key == 'mode' else key): value
                     for key, value in search.items()}
    try:
        model = AccumulatingMaskablePPO(OuterReturnPolicy, env, policy_kwargs=material,
            seed=args.seed, device=args.device, gamma=1., gae_lambda=1., amp_mode='off',
            learning_rate=training['learning_rate'], max_grad_norm=training['max_grad_norm'],
            n_steps=args.n_steps, batch_size=min(256, args.n_envs * args.n_steps), n_epochs=1,
            nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
            nosl_is_az_upper_strategy='model', nosl_is_az_leaf_batch_size=1,
            nosl_is_az_batch_size=training['batch_size_per_rank'],
            nosl_is_az_micro_batch_size=training['micro_batch_size'],
            nosl_is_az_updates=training['optimizer_updates_per_round'],
            nosl_is_az_policy_coef=training['policy_loss_coefficient'],
            nosl_is_az_value_coef=training['value_loss_coefficient'],
            nosl_is_az_replay_capacity=training['replay']['capacity_per_rank'],
            nosl_is_az_replay_min_size=training['replay']['min_size'],
            nosl_is_az_replay_max_old_fraction=training['replay']['max_old_fraction'], **search_kwargs)
        return model
    except BaseException:
        env.close()
        raise


def prepare(args):
    """Existing distributed CLI preparation mode: create step zero, never train."""
    if args.rank != 0 or args.world_size != 1 or args.checkpoint or args.resume_state or args.fork_state:
        raise ValueError('preparation is a fresh single-rank material operation')
    if not args.output or not args.material_directory or not args.material_sha256:
        raise ValueError('preparation requires output, material directory and SHA256')
    if args.output.suffix != '.zip':
        raise ValueError('preparation output must end in .zip')
    if args.output.exists():
        raise FileExistsError(args.output)
    profile = json.loads(args.prepare_combat_outer.read_text(encoding='utf-8'))
    if (profile['contract']['objective_id'] not in (*SCALAR_OBJECTIVES, COMBAT_OBJECTIVE) or profile['contract']['lower_init'] != 'random' or
            profile['contract'].get('lower_context') != CONTEXT_SCHEMA):
        raise ValueError('production profile requires frozen upper with a fresh trainable lower network')
    args.n_envs, args.n_steps, args.seed = 1, 2, int(profile['training']['seed'])
    args.vec, args.ascension, args.interactive_reasoning = 'native', 0, False
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    torch.manual_seed(args.seed)
    model = construct(args, profile)
    try:
        model.outer_profile_sha256 = hashlib.sha256(args.prepare_combat_outer.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        report = dict(schema='combat-outer-initialization-v1', output=str(args.output),
            checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
            contract=model.outer_return_contract, profile_sha256=model.outer_profile_sha256,
            step=model.num_timesteps, replay=0, optimizer_states=len(model.policy.optimizer.state),
            lower_parameters=sum(p.numel() for p in model.policy.core.combat.parameters()),
            trainable_lower_parameters=sum(p.numel() for p in model.policy.core.combat.parameters() if p.requires_grad),
            teacher_sha256=model.policy.core.teacher.assert_integrity())
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        model.get_env().close()
