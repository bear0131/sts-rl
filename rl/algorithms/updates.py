"""Shared differentiable batches and optimizer boundary; orchestration stays outside."""
from __future__ import annotations
import time
import numpy as np
import torch as th
from gymnasium import spaces
from rl.algorithms.ppo import ppo_loss_components
from rl.search.native import observation_as_tensor


def ppo_forward(model, batch, clip_range, clip_range_vf, *, indices=None, after_forward=None):
    def select(value):
        return value if indices is None else value.index_select(0, indices)
    observations = {key: select(value) for key, value in batch.observations.items()}
    actions = select(batch.actions)
    if isinstance(model.action_space, spaces.Discrete):
        actions = actions.long().flatten()
    with model.autocast_context():
        values, log_prob, entropy = model.policy.evaluate_actions(
            observations, actions, action_masks=select(batch.action_masks))
    if after_forward is not None:
        after_forward(observations)
    components = ppo_loss_components(
        hierarchical=model.hierarchical_combat_hp,
        normalize_advantage=model.normalize_advantage, observations=observations,
        advantages=select(batch.advantages), returns=select(batch.returns),
        old_values=select(batch.old_values), old_log_prob=select(batch.old_log_prob),
        values=values.flatten(), log_prob=log_prob, entropy=entropy,
        clip_range=clip_range, clip_range_vf=clip_range_vf,
        skip_combat=bool(getattr(model, 'nosl_is_az', False)))
    loss = components.policy_loss + model.ent_coef * components.entropy_loss + model.vf_coef * components.value_loss
    return components, loss


def optimizer_step(model, micro_batches=1, *, value_only=False, optimizer=None, max_grad_norm=None):
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
    norm = th.nn.utils.clip_grad_norm_(
        model.policy.parameters(), model.max_grad_norm if max_grad_norm is None else max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer = model.policy.optimizer if optimizer is None else optimizer
    optimizer.step()
    optimizer.zero_grad()
    if getattr(model, '_outer_return', False):
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
