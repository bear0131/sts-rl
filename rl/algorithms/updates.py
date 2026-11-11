"""Shared differentiable batches and optimizer boundary; orchestration stays outside."""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
import torch as th
from gymnasium import spaces
from rl.algorithms.ppo import masked_categorical_statistics, ppo_loss_components, model_normalized_advantages
from rl.search.native import observation_as_tensor


def ppo_forward(model, batch, clip_range, clip_range_vf, *, indices=None, after_forward=None,
                pre_normalized_advantages=None, direct_masked_categorical=False):
    if (pre_normalized_advantages is None and model.normalize_advantage
            and getattr(model, 'ppo_advantage_normalization', 'group256') == 'rollout'):
        pre_normalized_advantages = model_normalized_advantages(model, batch.advantages)
    def select(value):
        return value if indices is None else value.index_select(0, indices)
    observations = {key: select(value) for key, value in batch.observations.items()}
    actions = select(batch.actions)
    if isinstance(model.action_space, spaces.Discrete):
        actions = actions.long().flatten()
    categorical_valid = None
    with model.autocast_context():
        policy_obs = {k: v for k, v in observations.items() if not k.startswith('_search_')}
        masks = select(batch.action_masks)
        if direct_masked_categorical and getattr(model, 'deck_search_config', None) is None:
            logits, values = model.policy.core.actor_critic(policy_obs)
            log_prob, entropy, categorical_valid = masked_categorical_statistics(logits.float(), actions, masks)
        else:
            values, log_prob, entropy = model.policy.evaluate_actions(policy_obs, actions, action_masks=masks)
        search_loss = None
        if getattr(model, 'deck_search_config', None) is not None:
            from rl.algorithms.search_auxiliary import auxiliary_loss
            # evaluate_actions leaves the exact masked distribution; reuse its
            # graph rather than forwarding the large context encoder twice.
            search_loss = auxiliary_loss(model, observations,
                model.policy.action_dist.distribution.logits,
                select(batch.action_masks).bool())
    if after_forward is not None:
        after_forward(observations)
    components = ppo_loss_components(
        hierarchical=model.hierarchical_combat_hp,
        normalize_advantage=model.normalize_advantage and pre_normalized_advantages is None,
        observations=observations,
        advantages=select(batch.advantages if pre_normalized_advantages is None
                          else pre_normalized_advantages), returns=select(batch.returns),
        old_values=select(batch.old_values), old_log_prob=select(batch.old_log_prob),
        values=values.flatten(), log_prob=log_prob, entropy=entropy,
        clip_range=clip_range, clip_range_vf=clip_range_vf,
        skip_combat=bool(getattr(model, 'nosl_is_az', False)),
        normalization_group_size=getattr(model.policy, 'ppo_advantage_group_size', None))
    loss = components.policy_loss + model.ent_coef * components.entropy_loss + model.vf_coef * components.value_loss
    if search_loss is not None:
        loss = loss + search_loss
    if categorical_valid is not None:
        loss = loss.masked_fill(~categorical_valid, float('nan'))
    return components, loss


def ppo_chunked_backward(model, batch, clip_range, clip_range_vf, *,
                         pre_normalized_advantages=None, backward_scale=1.0, oom_group=None,
                         after_physical_chunk=None, direct_masked_categorical=False):
    """Fuse statistical groups; retry smaller chunks without corrupting accumulation.

    Only the shared whole-run policy uses this path (no dropout/running stats).
    A failed backward may already have written some gradients. Keep prior
    accumulation separate until an entire chunk succeeds, then add it once.
    The optional observer sees only these committed pieces, including OOM splits.
    """
    evaluate_options = dict(direct_masked_categorical=True) if direct_masked_categorical else {}
    parameters = list(model.policy.parameters())
    replay = getattr(getattr(model.policy, 'core', None), '_context_gradient_batch', None)
    if replay is not None:
        # The adjoint is transactional just like parameter gradients: a later
        # failed phase must not double-count earlier phases on an OOM retry.
        parameters.append(replay.latents)
    total = len(batch.advantages)
    group = 1 if pre_normalized_advantages is not None else model.policy.ppo_advantage_group_size
    stats = model.__dict__.setdefault('_ppo_microbatch_stats',
        dict(chunks=0, rows=0, oom_retries=0, min_rows=total, max_rows=0))
    losses = []
    def part(start, count):
        data = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
        return type(batch)(**{k: ({n: v[start:start+count] for n, v in x.items()}
            if isinstance(x, dict) else x[start:start+count]) for k, x in data.items()})
    def backward(start, count):
        # Packed phases have different memory costs. An expensive upper block
        # must not permanently shrink every subsequent combat block/round.
        caps = model.__dict__.setdefault('_ppo_microbatch_caps', {}) if oom_group is not None else None
        cap = (caps.get(oom_group, total) if caps is not None else
               getattr(model, '_ppo_microbatch_cap', total))
        if count > cap:
            backward(start, cap)
            backward(start + cap, count - cap)
            return
        previous = [p.grad for p in parameters]
        for p in parameters:
            p.grad = None
        components = loss = phase_loss = pending_metrics = phase_losses = None
        failed = False
        try:
            chunk = part(start, count)
            if ('phase' in getattr(chunk, 'observations', {})
                    and not getattr(model, 'hierarchical_combat_hp', False)):
                from env.core import vocab
                # Normalize BEFORE splitting combat/upper rows. Re-normalizing
                # either subset would change the chosen statistical objective.
                advantages = (pre_normalized_advantages[start:start + count]
                    if pre_normalized_advantages is not None else
                    model_normalized_advantages(model, chunk.advantages)
                    if model.normalize_advantage else chunk.advantages)
                combat = chunk.observations['phase'].long().flatten() == vocab.PHASE_VOCAB_INDEX['combat']
                pending_metrics, phase_losses = [], []
                for mask in (combat, ~combat):
                    indices = mask.nonzero().flatten()
                    rows = len(indices)
                    if not rows:
                        continue
                    components, phase_loss = ppo_forward(model, chunk, clip_range, clip_range_vf,
                        indices=None if rows == count else indices, pre_normalized_advantages=advantages,
                        **evaluate_options)
                    (phase_loss * (rows / total * backward_scale)).backward()
                    phase_losses.append(phase_loss.detach() * (rows / count))
                    # Retain only detached metrics until the entire chunk
                    # succeeds. A failed later phase must discard earlier
                    # gradients AND metrics before the existing OOM retry.
                    pending_metrics.append((replace(components, **{
                        key: value.detach() for key, value in vars(components).items()
                        if isinstance(value, th.Tensor)}), rows))
                    components = phase_loss = None
                loss = th.stack(phase_losses).sum()
            else:
                kwargs = ({} if pre_normalized_advantages is None else
                          dict(pre_normalized_advantages=pre_normalized_advantages[start:start + count]))
                components, loss = ppo_forward(model, chunk, clip_range, clip_range_vf, **kwargs, **evaluate_options)
                (loss * (count / total * backward_scale)).backward()
                pending_metrics = [(components, count)]
        except th.OutOfMemoryError:
            failed = True
        except BaseException:
            for p, old in zip(parameters, previous, strict=True):
                p.grad = old
            raise
        if failed:
            components = loss = phase_loss = pending_metrics = phase_losses = None
            for p, old in zip(parameters, previous, strict=True):
                p.grad = old
            if count <= group:
                raise th.OutOfMemoryError('PPO minimum statistical group does not fit device memory')
            # Leave the exception scope before clearing its failed autograd graph.
            import gc
            gc.collect()
            if model.device.type == 'cuda':
                th.cuda.empty_cache()
            cap = max(group, (count // group // 2) * group)
            if caps is None:
                model._ppo_microbatch_cap = cap
            else:
                caps[oom_group] = cap
            stats['oom_retries'] += 1
            backward(start, cap)
            backward(start + cap, count - cap)
            return
        old_grads, additions = [], []
        for p, old in zip(parameters, previous, strict=True):
            if old is not None:
                if p.grad is not None:
                    old_grads.append(old)
                    additions.append(p.grad)
                p.grad = old
        if old_grads:
            th._foreach_add_(old_grads, additions)
        losses.append(loss.detach() * (count / total))
        from rl.algorithms.ppo_metrics import accumulate_update_metrics
        for components, rows in pending_metrics:
            accumulate_update_metrics(model, components, rows)
        stats['chunks'] += 1
        stats['rows'] += count
        stats['min_rows'] = min(stats['min_rows'], count)
        stats['max_rows'] = max(stats['max_rows'], count)
        if after_physical_chunk is not None:
            after_physical_chunk(oom_group, count, total)
    backward(0, total)
    return float(th.stack(losses).sum().cpu())


def optimizer_step(model, micro_batches=1, *, value_only=False, optimizer=None, max_grad_norm=None,
                   clip_parameter_groups=None):
    """Average accumulation once, clip once, then Adam and clear gradients."""
    if value_only and getattr(model, '_outer_return', False):
        raise ValueError('frozen-upper objective trains the entire lower network from update one')
    if micro_batches < 1:
        return 0.0
    if value_only:
        # Ring unpack materializes zero gradients for unused heads. Adam's
        # existing momentum would still move them unless the gradient is None.
        for parameter in model.policy.core.policy_only_parameters():
            parameter.grad = None
    if micro_batches != 1:
        for parameter in model.policy.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(1.0 / micro_batches)
    limit = model.max_grad_norm if max_grad_norm is None else max_grad_norm
    if clip_parameter_groups is None:
        norm = th.nn.utils.clip_grad_norm_(model.policy.parameters(), limit, error_if_nonfinite=True)
    else:
        # Independent objectives must not shrink each other's update through a
        # shared norm, even when their forward graphs are detached.
        groups = [list(group) for group in clip_parameter_groups]
        ids = [id(p) for group in groups for p in group]
        active = {id(p) for p in model.policy.parameters() if p.grad is not None}
        if len(ids) != len(set(ids)) or not active.issubset(ids):
            raise ValueError('gradient clip groups must be disjoint and cover active parameters')
        norms = [th.nn.utils.clip_grad_norm_(group, limit, error_if_nonfinite=True) for group in groups]
        norm = th.linalg.vector_norm(th.stack(norms))
    optimizer = model.policy.optimizer if optimizer is None else optimizer
    optimizer.step()
    optimizer.zero_grad()
    if getattr(getattr(model.policy, 'core', None), 'frozen_upper', False):
        model.policy.core.teacher.assert_integrity()
    # Leave logging's device synchronization to callers that consume the norm.
    return norm.detach()


def reasoning_tensors(model, samples):
    """Pad on CPU, then transfer two token matrices instead of two copies per row."""
    from rl.policies.wholerun_reasoning import IGNORE_INDEX, VOCAB
    length = max(len(sample['inputs']) for sample in samples)
    inputs = np.full((len(samples), length), VOCAB.pad_id, dtype=np.int64)
    labels = np.full((len(samples), length), IGNORE_INDEX, dtype=np.int64)
    for row, sample in enumerate(samples):
        width = len(sample['inputs'])
        inputs[row, :width] = sample['inputs']
        labels[row, :width] = sample['labels']
    observations = observation_as_tensor({key: np.stack([s['observation'][key] for s in samples])
                                         for key in samples[0]['observation']}, model.device)
    actions = th.as_tensor([sample['action'] for sample in samples], dtype=th.long, device=model.device)
    return observations, actions, th.as_tensor(inputs, device=model.device), th.as_tensor(labels, device=model.device)


def reasoning_microbatch_backward(
    model, batch_size: int, loss_scale: float = 1.0, *, synchronize: bool = True,
) -> dict[str, float] | None:
    """Backpropagate one local reasoning batch without stepping the optimizer."""
    # Trace transcripts are compiled on a CPU helper thread during PPO
    # backward.  Join here at the last consumer so that compilation overlaps
    # the GPU-heavy phase without changing the sampled examples.
    compile_wait_started = time.perf_counter()
    model._wait_for_reasoning_samples()
    compile_wait_seconds = time.perf_counter() - compile_wait_started
    samples = model._reasoning_rollout_samples
    core = getattr(model.policy, "core", None)
    if (
        not samples
        or model.reasoning_coef <= 0.0
        or not hasattr(core, "reasoning_loss")
    ):
        return None
    # Long transcripts are valid supervision.  The decoder extends positions
    # beyond its initial learned table instead of dropping these samples.
    eligible = list(samples)
    if not eligible:
        return None
    count = min(int(batch_size), len(eligible))
    indices = np.random.choice(len(eligible), size=count, replace=False)
    batch = [eligible[int(index)] for index in indices]
    observations, actions, input_ids, labels = reasoning_tensors(model, batch)
    if synchronize and model.device.type == "cuda":
        th.cuda.synchronize(model.device)
    compute_started = time.perf_counter()
    with model.autocast_context():
        loss, stats = core.reasoning_loss(
            observations, actions, input_ids, labels
        )
        (model.reasoning_coef * loss * loss_scale).backward()
    if synchronize and model.device.type == "cuda":
        th.cuda.synchronize(model.device)
    compute_seconds = time.perf_counter() - compute_started
    return {
        "loss": float(loss.detach().cpu()),
        "token_accuracy": float(stats["token_accuracy"]),
        "supervised_tokens": float(stats["supervised_tokens"]),
        "sampled_samples": float(count),
        "samples": float(len(samples)),
        "eligible_samples": float(len(eligible)),
        "dropped_too_long": float(len(samples) - len(eligible)),
        "environment_tokens": float(sum(
            sample["environment_tokens"] for sample in batch
        )),
        "atomic_trace_rate": float(sum(
            sample["precision"] == "atomic" for sample in batch
        ) / len(batch)),
        "sequence_length_mean": float(np.mean(
            [len(sample["inputs"]) for sample in eligible]
        )),
        "sequence_length_max": float(max(
            len(sample["inputs"]) for sample in eligible
        )),
        "compile_wait_sec": float(compile_wait_seconds),
        "compute_sec": float(compute_seconds),
    }
