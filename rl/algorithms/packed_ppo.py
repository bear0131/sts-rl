"""Shared PPO physical batches, preserving each original Adam sample group."""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch

from env.core import vocab
from rl.algorithms.ppo import model_normalized_advantages, normalize_ppo_advantages
from rl.algorithms.shared_buffer import SharedRolloutBuffer
from rl.algorithms.updates import optimizer_step, ppo_chunked_backward
from rl.policies.shared_wholerun import SharedWholeRunPolicy, SharedOverworldEncoder


@dataclass(frozen=True)
class PackedPPOConfig:
    token_budget: int
    max_rows: int
    precision: str = 'fp32'
    context_gradient_replay: bool = False
    context_grouping: bool = False
    upper_screen_max_rows: int | None = None
    max_contexts_per_chunk: int | None = None
    direct_masked_categorical: bool = False
    prefetch_groups: bool = False

    def __post_init__(self):
        if (type(self.token_budget) is not int or type(self.max_rows) is not int
                or min(self.token_budget, self.max_rows) < 1):
            raise ValueError('packed PPO token budget and max rows must be positive integers')
        if self.precision not in ('fp32', 'fp16'):
            raise ValueError('packed PPO precision must be fp32 or fp16')
        if type(self.context_gradient_replay) is not bool:
            raise ValueError('context_gradient_replay must be boolean')
        if self.upper_screen_max_rows is not None and (
                type(self.upper_screen_max_rows) is not int or self.upper_screen_max_rows < 1
                or not self.context_grouping):
            raise ValueError('upper screen max rows requires context grouping and a positive integer')
        if self.max_contexts_per_chunk is not None and (
                type(self.max_contexts_per_chunk) is not int or self.max_contexts_per_chunk < 1
                or self.upper_screen_max_rows is None):
            raise ValueError('context count limit requires upper screen grouping and a positive integer')
        if type(self.context_grouping) is not bool:
            raise ValueError('context_grouping must be boolean')
        if type(self.direct_masked_categorical) is not bool:
            raise ValueError('direct_masked_categorical must be boolean')
        if type(self.prefetch_groups) is not bool:
            raise ValueError('prefetch_groups must be boolean')


def configure_packed_updates(model, configuration=None):
    """Persist plain data; reconstruct encoder options on checkpoint loading."""
    if configuration == 'off':
        model.ppo_update_config = None
        model.ppo_update_loss_scale = 1.
        model.__dict__.pop('_ppo_microbatch_cap', None)
        model.__dict__.pop('_ppo_microbatch_caps', None)
    if configuration is None:
        configuration = getattr(model, 'ppo_update_config', None)
    if configuration is None or configuration == 'off':
        model.ppo_update_config = None
        if isinstance(model.policy, SharedWholeRunPolicy):
            model.policy.core.combat.compact_training_tokens = False
            if hasattr(model.policy.core.combat, 'transformer'):
                model.policy.core.combat.transformer.update_precision = 'fp32'
        return None
    config = (configuration if isinstance(configuration, PackedPPOConfig)
              else PackedPPOConfig(**configuration))
    if config.context_gradient_replay or config.context_grouping or config.direct_masked_categorical:
        from rl.policies.deck_latent import DeckLatentPolicy
        if not isinstance(model.policy, DeckLatentPolicy) or config.precision != 'fp32':
            raise ValueError('deck batch optimization requires FP32 deck-latent PPO')
    if (not isinstance(model.policy, SharedWholeRunPolicy)
            or model.nosl_is_az or model.hierarchical_combat_hp
            or getattr(model, 'reasoning_updates', 0) or model.target_kl is not None
            or model.amp_mode != 'off'):
        raise ValueError('packed updates require shared pure PPO without auxiliary updates, target KL or outer AMP')
    previous = getattr(model, 'ppo_update_config', None)
    model.ppo_update_config = asdict(config)
    if not config.direct_masked_categorical:
        model.ppo_update_config.pop('direct_masked_categorical')
    if not config.prefetch_groups:
        model.ppo_update_config.pop('prefetch_groups')
    if config.max_contexts_per_chunk is None:
        model.ppo_update_config.pop('max_contexts_per_chunk')
    if config.upper_screen_max_rows is None:
        # Keep the legacy configuration byte-for-byte compatible so an ordinary
        # resume preserves its learned OOM caps and old readers accept it.
        model.ppo_update_config.pop('upper_screen_max_rows')
    if (previous != model.ppo_update_config or not hasattr(model, 'ppo_update_loss_scale')):
        model.ppo_update_loss_scale = 65536. if config.precision == 'fp16' else 1.
        model.__dict__.pop('_ppo_microbatch_cap', None)
        model.__dict__.pop('_ppo_microbatch_caps', None)
    scale = model.ppo_update_loss_scale
    if not math.isfinite(scale) or scale < 1 or (config.precision == 'fp32' and scale != 1):
        raise ValueError('invalid saved packed PPO loss scale')
    model.policy.core.combat.compact_training_tokens = True
    if hasattr(model.policy.core.combat, 'transformer'):
        model.policy.core.combat.transformer.update_precision = config.precision
    elif config.precision != 'fp32':
        raise ValueError('deck latent updates currently require FP32')
    return config


def _select(batch, index):
    data = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
    return type(batch)(**{k: ({n: v[index] for n, v in x.items()}
        if isinstance(x, dict) else x[index]) for k, x in data.items()})


def packed_chunks(model, batch, config):
    """Normalize original statistical groups BEFORE any permutation/splitting."""
    if not model.normalize_advantage:
        advantages = batch.advantages
    elif model.ppo_advantage_normalization == 'group256':
        advantages = normalize_ppo_advantages(batch.advantages,
            min(model.batch_size, model.policy.ppo_advantage_group_size))
    else:
        advantages = model_normalized_advantages(model, batch.advantages)
    obs = batch.observations
    phase = obs['phase'].long().flatten()
    combat = phase == vocab.PHASE_VOCAB_INDEX['combat']
    selection = phase == vocab.PHASE_VOCAB_INDEX['card_select']
    width = SharedOverworldEncoder.merged_token_counts(obs)
    # Card selection retains its complete original candidate region. This is
    # a physical memory estimate, never an action-mask or observation filter.
    width = torch.where(selection, width + obs['select_card_ids'].shape[1], width)
    width = torch.where(combat, model.policy.core.combat.training_token_counts(obs), width)
    # A deck policy evaluates each upper screen separately. Keep that screen's
    # rows together before context sorting, preserving the original Adam batch.
    if config.upper_screen_max_rows is not None:
        kind = torch.where(combat, 0, phase + 1)
    else:
        kind = torch.where(combat, 0, torch.where(selection, 1, 2))
    host = torch.stack((kind, width)).cpu().numpy()
    context_boundaries = None
    if config.max_contexts_per_chunk is not None:
        # Count the same unmasked tokens that the context encoder pools.
        # Bucket by eight before sorting exact context IDs to reduce padding.
        context = obs['_deck_context_index'].flatten().cpu().numpy()
        valid_map = (obs['map_full_rooms'] != 0) | obs['map_full_features'].bool().any(-1)
        token_count = 1 + (obs['deck_ids'] != 0).sum(1) + (obs['relics'] != 0).sum(1) + valid_map.sum(1)
        order = np.lexsort((context, (token_count.cpu().numpy() + 7) // 8, host[0]))
        ids = context[order]
        context_boundaries = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    elif config.context_grouping:
        context = obs['_deck_context_index'].cpu().numpy()
        order = np.lexsort((host[1], context, host[0]))
    else:
        order = np.lexsort((host[1], host[0]))
    index = torch.as_tensor(order, device=model.device)
    ordered = _select(batch, index)
    normalized = advantages[index]
    lengths = host[:, order]
    start, total = 0, len(advantages)
    while start < total:
        group = int(lengths[0, start])
        limit = config.max_rows
        if config.upper_screen_max_rows is not None and group:
            limit = min(limit, config.upper_screen_max_rows)
        end = min(start + limit, total)
        if context_boundaries is not None:
            next_boundary = context_boundaries.searchsorted(start, side='right')
            last = min(next_boundary + config.max_contexts_per_chunk - 1, len(context_boundaries) - 1)
            end = min(end, int(context_boundaries[last]))
        other = np.flatnonzero(lengths[0, start:end] != lengths[0, start])
        if len(other):
            end = start + int(other[0])
        end = min(end, start + max(1, config.token_budget // int(lengths[1, start:end].max())))
        if config.upper_screen_max_rows is not None:
            oom_group = 'combat' if group == 0 else vocab.PHASE_VOCAB[group - 1]
        else:
            oom_group = ('combat', 'selection', 'upper')[group] if config.context_grouping else None
        yield _select(ordered, slice(start, end)), normalized[start:end], (end - start) / total, oom_group
        start = end


def _sync_device(model):
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)


def packed_ppo_epoch(model, clip_range, clip_range_vf, *, synchronize=None, average=None, schedule_chunks=None,
                     after_physical_chunk=None):
    """Same Adam boundaries for local and distributed learners; retry overflow.

    ``average`` must be the rank-wide scalar reduction whenever ``synchronize``
    exchanges gradients. All ranks agree to retry before sending any gradients.
    No optimizer steps or samples are silently skipped.
    """
    config = configure_packed_updates(model)
    if config is None:
        raise ValueError('packed PPO is not configured')
    if config.precision == 'fp16' and model.device.type != 'cuda':
        raise ValueError('packed FP16 updates require CUDA; CPU checkpoint inference is supported')
    if (synchronize is None) != (average is None):
        raise ValueError('distributed packed PPO needs both synchronization callbacks')
    effective = model.batch_size * model.gradient_accumulation_steps
    total = model.rollout_buffer.buffer_size * model.rollout_buffer.n_envs
    if total % effective:
        raise ValueError('packed PPO requires complete original optimizer batches')
    scale = model.ppo_update_loss_scale
    if average is not None:
        moments = average(dict(scale=scale, scale_square=scale * scale))
        if moments['scale'] != scale or moments['scale_square'] != scale * scale:
            raise ValueError('packed PPO loss scale differs across ranks')
    result = dict(losses=[],metrics=[],optimizer_steps=0,backward_sec=0.,optimizer_step_sec=0.,
                  sync_sec=0.,gradient_pack_sec=0.,sync_mbps=[],grad_bytes=0,overflow_retries=0,
                  context_input_rows=0,context_unique_rows=0,context_oom_retries=0)
    chunk_options = {} if after_physical_chunk is None else dict(after_physical_chunk=after_physical_chunk)
    if config.direct_masked_categorical:
        chunk_options['direct_masked_categorical'] = True
    if config.prefetch_groups:
        if not isinstance(model.rollout_buffer, SharedRolloutBuffer):
            raise ValueError('packed group prefetch requires SharedRolloutBuffer')
        batches = model.rollout_buffer.get(effective, prefetch=True)
    else:
        batches = model.rollout_buffer.get(effective)
    try:
        for batch in batches:
            from rl.algorithms.search_auxiliary import prepare_search_batch
            prepare_search_batch(model, batch, average)
            _sync_device(model)
            started = time.perf_counter()
            from rl.algorithms.deck_context import context_gradient_batch
            with context_gradient_batch(model, batch, config) as (context_batch, replay):
                chunks = list(packed_chunks(model, context_batch, config))
                previous = getattr(model, '_ppo_update_metric_sums', None)
                previous = None if previous is None else previous.clone()
                count = getattr(model, '_ppo_update_metric_count', 0)
                stats = dict(getattr(model, '_ppo_microbatch_stats', {}))
                for retry in range(17):
                    model.policy.optimizer.zero_grad()
                    if replay is not None:
                        replay.reset_gradient()
                    loss = 0.
                    # Optional experimental scheduler sees only complete physical chunks
                    # inside this Adam group. Its iterator resumes after each successful
                    # transactional backward and must drain before the finite collective.
                    scheduled = chunks if schedule_chunks is None else schedule_chunks(chunks)
                    for chunk, advantages, weight, oom_group in scheduled:
                        loss += weight * ppo_chunked_backward(model, chunk, clip_range, clip_range_vf,
                            pre_normalized_advantages=advantages, backward_scale=scale * weight,
                            oom_group=oom_group, **chunk_options)
                    if replay is not None:
                        replay.backward()
                    grads = [p.grad for p in model.policy.parameters() if p.grad is not None]
                    if scale != 1:
                        torch._foreach_mul_(grads, 1 / scale)
                    finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())
                    finite = finite and math.isfinite(loss)
                    bad = float(not finite)
                    if average is not None:
                        bad = average(dict(nonfinite=bad))['nonfinite']
                    if not bad:
                        break
                    # Failed attempts may have committed several chunks' diagnostics.
                    # Restore sample metrics along with gradients; retain OOM counters.
                    model._ppo_update_metric_sums = None if previous is None else previous.clone()
                    model._ppo_update_metric_count = count
                    current = model._ppo_microbatch_stats
                    for name in ('chunks', 'rows', 'min_rows', 'max_rows'):
                        current[name] = stats.get(name, len(batch.advantages) if name == 'min_rows' else 0)
                    model.policy.optimizer.zero_grad()
                    if config.precision != 'fp16' or scale <= 1 or retry == 16:
                        raise FloatingPointError('packed PPO gradients remain nonfinite; Adam was not advanced')
                    scale = max(1., scale / 2)
                    model.ppo_update_loss_scale = scale
                    result['overflow_retries'] += 1
            if replay is not None:
                result['context_input_rows'] += replay.input_rows
                result['context_unique_rows'] += replay.unique_rows
                result['context_oom_retries'] += replay.oom_retries
            elif config.context_grouping:
                result['context_input_rows'] += len(batch.advantages)
                result['context_unique_rows'] += int(context_batch.observations['_deck_context_index'].max()) + 1
            _sync_device(model)
            result['backward_sec'] += time.perf_counter() - started
            if synchronize is not None:
                sync = synchronize(result['optimizer_steps'])
                for name in ('sync_sec', 'gradient_pack_sec'):
                    result[name] += sync[name]
                result['sync_mbps'].append(sync['sync_mbps'])
                result['grad_bytes'] = int(sync['grad_bytes'])
            _sync_device(model)
            started = time.perf_counter()
            optimizer_step(model)
            _sync_device(model)
            result['optimizer_step_sec'] += time.perf_counter() - started
            sums = model._ppo_update_metric_sums
            delta = sums if previous is None else sums - previous
            result['metrics'].append((delta / (model._ppo_update_metric_count - count)).cpu().tolist())
            result['losses'].append(loss)
            result['optimizer_steps'] += 1
            del chunks, batch, context_batch, replay
    finally:
        if config.prefetch_groups:
            batches.close()
    return result
