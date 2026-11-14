"""Configure compact CPU search for online learning or an explicit frozen backup."""
import hashlib
import json
from pathlib import Path
import time

from rl.algorithms.small_playout import load_policy_pair
from env.search.small_playout import SEARCH_CONTRACT


def fork_public_lower(model, mode, initializer=None):
    """Replace only the online lower module/Adam; model.save can reload the new family."""
    import torch
    from rl.algorithms.upper_probe import state_digest
    from rl.policies.public_playout import PublicPlayoutPolicy
    if (getattr(model,'training_scope',None)!='lower'
            or model.policy.small_playout_config.get('context_mode','none')!='none'):
        raise ValueError('public migration needs a lower-only fork of a compact v1 checkpoint')
    old=model.policy.small_playout
    # The zero adapter is deterministic; constructing temporary modules must not
    # advance collection RNG streams in a fork.
    with torch.random.fork_rng(devices=[model.device] if model.device.type=='cuda' else []):
        net=PublicPlayoutPolicy.from_mature(old)
    net.context_mode=mode
    cfg=dict(model.policy.small_playout_config,context_mode=mode)
    if mode not in ('root','turn','state') or cfg['optimizer']!='adamw':
        raise ValueError('public fork requires root/turn/state mode and an AdamW lower optimizer')
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    if initializer is not None:
        payload=torch.load(initializer,map_location=model.device,weights_only=True)
        if payload['parent_lower_digest']!=state_digest(old.state_dict()):
            raise ValueError('public initializer belongs to a different lower parent')
        net.load_state_dict(payload['policy'],strict=True)
        opt.load_state_dict(payload['optimizer'])
        cfg['learning_rate']=opt.param_groups[0]['lr']
        cfg['weight_decay']=opt.param_groups[0]['weight_decay']
    model.policy.small_playout=net
    model.policy.small_playout_optimizer=opt
    model.policy.small_playout_config=cfg
    model.policy_kwargs=dict(model.policy_kwargs,small_playout_config=dict(cfg))


def install_policy_pair(model, path):
    if hasattr(model.policy, 'small_playout'):
        raise ValueError('online random initialization and pretrained pair are mutually exclusive')
    if not getattr(model.policy.core, 'joint_upper', False):
        raise ValueError('small playout requires joint strategic PPO')
    if model.nosl_is_az_replay_enabled or getattr(model, '_frozen_upper', False):
        raise ValueError('small playout uses fresh strategic PPO without old combat replay/teacher')
    policy, prior, provenance = load_policy_pair(path)
    model.small_playout_pair = dict(
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        contract='sts1-small-playout-v1', search_contract=SEARCH_CONTRACT,
        policy=policy.export(), prior=prior.export(),
        provenance=provenance, common_simulations=250, elite_simulations=250,
        boss_simulations=1000, playout_depth=200, tree_depth=1024)


def configure_collection(model, env):
    if getattr(model.policy.core, 'model_contract', None) == 'deck-latent-combat-v1':
        from rl.search.paired_token_rollout import PairedTokenBatch
        from env.search.small_playout import PAIRED_CONTRACT
        from rl.algorithms.upper_probe import state_digest
        pair = dict(contract='deck-latent-combat-v1', search_contract=PAIRED_CONTRACT,
                    sha256=state_digest(model.policy.state_dict()), online=True,
                    common_simulations=250, elite_simulations=250, boss_simulations=1000,
                    playout_depth=200, tree_depth=1024)
        env.configure_small_playout(pair, batch_factory=lambda envs, pool, snapshot:
            PairedTokenBatch(envs, pool, snapshot, model.policy.core))
        batch = env.venv.small_playout_batch
        batch.model = model
        batch.observer = env.venv.native_root_observations
        if batch.observer is None:raise ValueError('full480 requires the existing native root encoder')
        if batch.inference.closed:
            from rl.search.rollout_inference import UpperPolicyInference
            batch.inference = UpperPolicyInference(model.policy.core, model.device)
        return
    if getattr(model.policy, 'small_playout_config', None) and model.policy.small_playout_config.get('context_mode') in ('token-root-64x2', 'token-budget-64x2'):
        from rl.policies.token_regret import OnlineTokenRegretPolicy, TOKEN_REGRET_CONTRACT
        from rl.search.paired_token_rollout import PairedTokenBatch
        from rl.policies.search_budget import AdaptiveTokenRegretPolicy, BUDGET_CONTRACT
        from rl.search.adaptive_budget import AdaptiveBudgetBatch
        net = model.policy.small_playout
        adaptive = isinstance(net, AdaptiveTokenRegretPolicy)
        expected = BUDGET_CONTRACT if adaptive else TOKEN_REGRET_CONTRACT
        factory = AdaptiveBudgetBatch if adaptive else PairedTokenBatch
        if adaptive:
            from rl.policies.search_budget import budget_manifest
            saved_budget = getattr(model, 'search_budget_config', None)
            if saved_budget != budget_manifest():
                if saved_budget is not None or model.num_timesteps != 0:
                    raise ValueError('saved learned budget configuration differs from this implementation')
                model.search_budget_config = budget_manifest()
        previous = getattr(model, 'small_playout_search_contract', None)
        if (not isinstance(net, OnlineTokenRegretPolicy) or previous != expected
                or not getattr(model, 'independent_rollout_config', None)
                or model.nosl_is_az_replay_enabled or getattr(model, '_frozen_upper', False)):
            raise ValueError('online token policy requires its own fresh independent paired-search contract')
        pair = net.collection_pair(collect_samples=True)
        model.small_playout_collection_sha256 = pair['sha256']
        env.configure_small_playout(pair, batch_factory=lambda envs, pool, snapshot:
                                   factory(envs, pool, snapshot, net))
        return
    saved_contract = getattr(model, 'small_playout_search_contract', None)
    if saved_contract != SEARCH_CONTRACT:
        if saved_contract is not None or model.num_timesteps != 0:
            raise ValueError('trained compact checkpoint has a different search objective; use its saved source')
        model.small_playout_search_contract = SEARCH_CONTRACT
    if hasattr(model.policy, 'small_playout'):
        import slaythespire as sts
        if (model.policy.small_playout.key_count != sts.small_playout_key_count()
                or model.nosl_is_az_replay_enabled or getattr(model, '_frozen_upper', False)
                or hasattr(model, 'small_playout_pair')):
            raise ValueError('online compact policy needs its own vocabulary and fresh whole-run objective')
        from rl.policies.small_playout import export_online_search_pair
        pair = export_online_search_pair(model.policy.small_playout, collect_samples=True)
        model.small_playout_collection_sha256 = pair['sha256']
    else:
        pair = getattr(model, 'small_playout_pair', None)
    if pair is None:
        raise ValueError('small-playout requires an online policy or an explicit frozen backup pair')
    if not hasattr(env, 'configure_small_playout'):
        raise ValueError('small-playout currently supports the native vector backend only')
    env.configure_small_playout(pair)


def _battle_bank(path, *, heldout=False):
    """Replay trusted simulator journals; never load historical model weights."""
    import torch
    from env.core.run_env import SlayTheSpireEnv
    from env.core.boss_value import enum_name, is_boss_room
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
        outer_return_context=True, combat_value_objective='combat-outcome-means-v1',
        deck_size_observation=True, neow_observation=True, event_option_observation=True)
    roots, tiers, ids = [], [], []
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        for index, saved in enumerate(state['states']):
            if (index % 5 == 0) != heldout:
                continue
            _, info = env.restore_training_state(saved['base'])
            if info['phase'] != 'combat' or saved['base']['multi'] is not None:
                continue
            if any(r['id'] == 'Runic Dome' for r in env.thick.gc.relic_state()):
                continue
            room = env.thick.gc.cur_room
            tiers.append('boss' if is_boss_room(room) else 'elite' if 'ELITE' in enum_name(room) else 'common')
            roots.append(env.thick.bc.clone())
            ids.append(index)
    finally:
        env.close()
    if not roots:
        raise ValueError('compact pretraining bank has no supported roots')
    return roots, tiers, ids


def pretrain(protocol_path):
    """Bounded collection shards or CE→REINFORCE preparation via the trainer CLI.

    Four hosts can collect independent nonce shards. One learner fits their
    union, then freezes a separate prior before outcome-only on-policy learning.
    This simulator-search prior replaces the reference's unavailable human BC.
    """
    import numpy as np
    import torch
    import slaythespire as sts
    from rl.algorithms.small_playout import SmallPlayoutLearner, save_policy_pair
    from rl.search.small_playout import SmallSearchConfig
    from rl.training.resource_monitor import ResourceMonitor
    cfg = json.loads(Path(protocol_path).read_text())
    if cfg['mode'] not in ('collect', 'fit') or int(cfg['seed']) < 0 or int(cfg['seconds']) <= 0:
        raise ValueError('invalid compact preparation protocol')
    torch.set_num_threads(1)
    start = time.monotonic()
    output = Path(cfg['output']); output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    def progress(row):
        records.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        output.with_suffix('.progress.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    def check_budget():
        if time.monotonic()-start > cfg['seconds']:
            raise TimeoutError('compact preparation exceeded its declared time budget')
    learner = SmallPlayoutLearner(seed=cfg['seed'], threads=cfg['threads'],
        device=cfg.get('device', 'cpu'), learning_rate=cfg.get('learning_rate', 3e-4))
    roots, tiers, ids = _battle_bank(cfg['bank'])
    provenance = dict(protocol=cfg, search_contract=SEARCH_CONTRACT,
        bank_sha256=hashlib.sha256(Path(cfg['bank']).read_bytes()).hexdigest(),
        bank_indices=ids, native_sha256=hashlib.sha256(Path(sts.__file__).read_bytes()).hexdigest(),
        prior_source='StS1 simulator search visits; no human BC or historical model weights',
        reference='tfriedel/spire-mentor@c60ce39e060e5180139eaf380e90b1dfcead4aee')
    progress(dict(stage='bank', roots=len(roots), seconds=time.monotonic()-start))
    with ResourceMonitor(interval=.5, gpu=True, torch_device=learner.device) as monitor:
        if cfg['mode'] == 'collect':
            if cfg['samples'] < 1 or cfg['batch_roots'] < 1:
                raise ValueError('collection dimensions must be positive')
            rng = np.random.default_rng(cfg['seed']+cfg['rank'])
            indices = rng.integers(len(roots), size=cfg['batch_roots'])
            active = [roots[i].clone() for i in indices]
            journals = [[] for _ in active]
            ages = np.zeros(len(active), np.int64)
            samples, work = [], {}
            while len(samples) < cfg['samples']:
                check_budget()
                count = min(len(active), cfg['samples']-len(samples))
                configs = [SmallSearchConfig(simulations=1000 if tiers[i]=='boss' else 250,
                                             boss=tiers[i]=='boss') for i in indices[:count]]
                nonces = [cfg['seed']*1000000007+cfg['rank']*100000000+len(samples)+i for i in range(count)]
                try:
                    results = learner.search.search(active[:count], nonces, configs)
                except RuntimeError:
                    for i in range(count):
                        try:
                            learner.search.search([active[i]], [nonces[i]], [configs[i]])
                        except RuntimeError as exc:
                            output.with_suffix('.failure.json').write_text(json.dumps(dict(
                                bank_index=ids[indices[i]], actions=journals[i], nonce=nonces[i],
                                simulations=configs[i].simulations, error=str(exc)), indent=2))
                            raise
                    raise
                for i, result in enumerate(results):
                    samples.append({key: result[key] for key in ('state','actions','keys','action_bits','visits')})
                    for key, value in result['stats'].items():
                        if key != 'playout_lengths': work[key] = work.get(key, 0)+value
                    active[i].execute(sts.Action(result['action'])); ages[i] += 1
                    journals[i].append(result['action'])
                    if active[i].is_battle_over or ages[i] >= 6000:
                        indices[i] = rng.integers(len(roots)); active[i] = roots[indices[i]].clone(); ages[i] = 0; journals[i] = []
                if len(samples) % 1024 < count or len(samples) == cfg['samples']:
                    progress(dict(stage='collect', samples=len(samples), seconds=time.monotonic()-start, work=work.copy()))
            torch.save(dict(samples=samples, provenance=provenance, work=work), output)
        else:
            samples, shards = [], []
            for name in cfg['shards']:
                shard = torch.load(name, map_location='cpu', weights_only=False)
                if (shard['provenance']['bank_sha256'] != provenance['bank_sha256']
                        or shard['provenance'].get('search_contract') != SEARCH_CONTRACT):
                    raise ValueError('pretraining shards use different source banks or search objectives')
                samples.extend(shard['samples']); shards.append(hashlib.sha256(Path(name).read_bytes()).hexdigest())
            prior = SmallPlayoutLearner(seed=cfg['seed']+1, device=learner.device, threads=1)
            for epoch in range(cfg['ce_epochs']):
                check_budget()
                progress(dict(stage='playout_ce', epoch=epoch, **learner.update(samples, batch_size=cfg['batch_size'])))
                progress(dict(stage='prior_ce', epoch=epoch, **prior.update(samples, batch_size=cfg['batch_size'])))
            hold, hold_tiers, hold_ids = _battle_bank(cfg['bank'], heldout=True)
            def evaluate():
                traces = []
                for i, (root, tier) in enumerate(zip(hold, hold_tiers, strict=True)):
                    traces.extend(sts.small_playout_collect_batch(learner.search.pool, learner.search.net,
                        [root]*8, SmallSearchConfig(boss=tier=='boss').native(),
                        [900000000+i*8+j for j in range(8)]))
                natural = [t for t in traces if not t['censored']]
                return dict(playouts=len(traces), censored=len(traces)-len(natural),
                    terminal_score_mean=float(np.mean([t['value'] for t in natural])) if natural else None)
            progress(dict(stage='heldout_before_rf', **evaluate()))
            rng = np.random.default_rng(cfg['seed'])
            for update in range(cfg['rf_updates']):
                check_budget()
                selected = rng.integers(len(roots), size=cfg['rf_starts'])
                result = learner.reinforce([roots[i] for i in selected],
                    configs=[SmallSearchConfig(boss=tiers[i]=='boss') for i in selected],
                    nonces=[1000000000+update*100000+i*100 for i in range(len(selected))],
                    k=cfg['rf_k'], batch_size=cfg['batch_size'], entropy_coef=cfg['entropy_coef'])
                progress(dict(stage='reinforce', update=update, **result))
                if (update+1) % 16 == 0:
                    save_policy_pair(output.with_suffix('.partial.pt'), learner.policy, prior.policy,
                                     provenance=dict(provenance, completed_rf_updates=update+1, shard_sha256=shards))
            progress(dict(stage='heldout_after_rf', **evaluate()))
            provenance.update(shard_sha256=shards, ce_samples=len(samples), holdout_indices=hold_ids,
                              completed_rf_updates=cfg['rf_updates'], records=records)
            save_policy_pair(output, learner.policy, prior.policy, provenance=provenance)
    output.with_suffix('.resources.json').write_text(json.dumps(monitor.summary, indent=2))
    progress(dict(stage='complete', seconds=time.monotonic()-start, sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
