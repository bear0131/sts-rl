"""Observed fight outcomes and losses for the entity combat experiment.

Labels retain min(pre-settlement HP, 200)/200 and explicit win/loss. The
policy's optional boss weight transforms only the utility expectation/target;
conditional HP quantiles and win BCE retain their original interpretation.
No inferred win labels, censored fights, or imagined search leaves are accepted.
"""
import numpy as np
import torch
from torch.nn import functional as F
from env.core.finite_curves import FACTORED_OBJECTIVE, MATCHED_OBJECTIVE

CONTRACT = 'combat-outcome-quantiles-v1'
MEAN_CONTRACT = 'combat-outcome-means-v1'
OUTCOME_CONTRACTS = (CONTRACT, MEAN_CONTRACT, FACTORED_OBJECTIVE, MATCHED_OBJECTIVE)
AUX_WIDTH = 6
# win, executed complete action, auxiliary weight, six observed transition deltas
TARGET_WIDTH = 3 + AUX_WIDTH
METRICS = ('win_bce','hp_pinball','utility_mse','transition_loss',
           'win_probability','win_target','hp_given_win')
MEAN_METRICS = ('win_bce', 'hp_mse', 'win_probability', 'win_target', 'hp_given_win')
CONDITIONAL_METRICS = ('win_bce', 'conditional_value_mse', 'win_probability', 'win_target', 'value_given_win')


def metrics_for(contract):
    if contract == FACTORED_OBJECTIVE:
        return CONDITIONAL_METRICS
    return MEAN_METRICS if contract in (MEAN_CONTRACT, MATCHED_OBJECTIVE) else METRICS


def mean_outcome_loss(outputs, targets, values, *, value_unit_scale=1.):
    """BCE + survivor MSE in declared external units, with one batch denominator.

    Losses contain no information about HP conditional on victory. Their HP
    gradient is zero; do not label survivor HP as zero on a death.
    """
    return _conditional_mean_loss(outputs, targets, values, 'hp_mean', 'hp_mse', 'hp_given_win',
                                  value_unit_scale=value_unit_scale)


def conditional_value_loss(outputs, targets, values):
    """BCE plus win-masked curve-return MSE; no loss on the product."""
    return _conditional_mean_loss(outputs, targets, values,
                                  'value_given_win', 'conditional_value_mse', 'value_given_win')


def _conditional_mean_loss(outputs, targets, values, output_key, mse_key, mean_key, *, value_unit_scale=1.):
    win = targets[:, 0].float()
    logits = outputs['win_logits'].float().flatten()
    mean = outputs[output_key].float().flatten()
    bce = F.binary_cross_entropy_with_logits(logits, win, reduction='none')
    # Convert the residual to the objective's external units before squaring.
    # The metric is the actual MSE contribution, in those same units.
    mse = ((mean - values.float().flatten()) * value_unit_scale).square() * win
    return bce + mse, {'win_bce': bce, mse_key: mse,
                      'win_probability': logits.sigmoid(), 'win_target': win, mean_key: mean}


def transition_features(obs):
    """Public normalized HP/block/energy, enemy HP, strength, and hand size."""
    from env.core import vocab as V
    strength = V.PLAYER_BUFF_VOCAB_INDEX['Strength']
    player = np.asarray(obs['player_scalars'], dtype=np.float32)
    return np.asarray([player[0], player[2], player[3],
        np.asarray(obs['monster_scalars'])[:, 1].sum(),
        obs['player_buffs'][strength], np.count_nonzero(obs['hand_ids']) / 10], dtype=np.float32)


def outcome_targets(samples, outcome, *, include_transition=True, surviving_exit=False):
    accepted = ('win', 'loss', 'escape') if surviving_exit else ('win', 'loss')
    if outcome not in accepted:
        raise ValueError('distribution supervision requires an observed win or loss')
    for sample in samples:
        target = np.zeros(TARGET_WIDTH, dtype=np.float32)
        target[0] = float(outcome != 'loss')
        if include_transition and 'transition_target' in sample:
            target[1] = sample['executed_action']
            target[2] = 1
            target[3:] = sample['transition_target']
        sample['combat_targets'] = target


def validate_targets(targets, values, *, action_dim=1392):
    targets = np.asarray(targets)
    if targets.shape != (len(values), TARGET_WIDTH) or not np.isfinite(targets).all():
        raise ValueError('invalid combat distribution targets')
    win, action, auxiliary = targets[:, 0], targets[:, 1], targets[:, 2]
    if not np.isin(win, [0, 1]).all() or not np.isin(auxiliary, [0, 1]).all():
        raise ValueError('combat outcome and auxiliary masks must be explicit binary labels')
    if np.any((win == 0) & (np.asarray(values) != 0)):
        raise ValueError('a combat loss must have zero utility')
    from env.core.native_actions import JOINT_END, AUTOREGRESSIVE_SIZE
    minimum = 0 if action_dim == AUTOREGRESSIVE_SIZE else JOINT_END
    active = action[auxiliary > 0]
    if np.any((active < minimum) | (active >= action_dim) | (active != np.floor(active))):
        raise ValueError('auxiliary action is not a complete combat action')


def distribution_loss(outputs, targets, values, *, mean_coef=.25, auxiliary_coef=.1):
    """Per-row losses; all terms use the same logical-batch denominator.

    Win-only pinball loss is zero on losses. This weighting is deliberately
    unconditional (E[1_win * loss]), invariant to microbatch/rank composition.
    Quantiles model capped HP utility, so search averages utility, not raw HP
    followed by a cap. Auxiliary labels come only from the executed transition.
    """
    win = targets[:, 0].float()
    logits = outputs['win_logits'].float().flatten()
    quantiles = outputs['hp_quantiles'].float()
    tau = (torch.arange(quantiles.shape[1], device=quantiles.device) + .5) / quantiles.shape[1]
    error = values.float().flatten()[:, None] - quantiles
    pinball = torch.maximum(tau * error, (tau - 1) * error).mean(-1) * win
    bce = F.binary_cross_entropy_with_logits(logits, win, reduction='none')
    from env.core.boss_value import win_hp_utility
    weight = outputs.get("boss_win_weight")
    weight = 0.0 if weight is None else weight.float().flatten()
    expectation = win_hp_utility(torch.sigmoid(logits), quantiles.mean(-1), weight)
    utility_target = win_hp_utility(win, values.flatten().float(), weight)
    mean = (expectation - utility_target).square()
    if auxiliary_coef:
        aux = F.smooth_l1_loss(outputs['transition'].float(), targets[:, 3:].float(),
                              reduction='none').mean(-1) * targets[:, 2]
    else:
        aux = torch.zeros_like(win)
    total = bce + pinball + mean_coef * mean + auxiliary_coef * aux
    return total, dict(win_bce=bce, hp_pinball=pinball, utility_mse=mean,
                       transition_loss=aux, win_probability=torch.sigmoid(logits),
                       win_target=win, hp_given_win=quantiles.mean(-1))
