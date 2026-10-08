"""Conservative, inspectable proposals for the requested NOSL recovery experiment."""
import math


HP_ORDER_PREFIX = 'nosl_is_az/rollout_value_gt_current_hp_ge_future_hp'


def hp_order_fraction(record):
    """Global bad-event ratio from equally reduced numerator/denominator.

    Works with rank sums or rank means; averaging local ratios does not.
    No eligible observations means no estimate (None), not perfect calibration.
    """
    count = float(record[HP_ORDER_PREFIX + '_count'])
    matches = float(record[HP_ORDER_PREFIX + '_matches'])
    if not all(math.isfinite(x) for x in (count, matches)) or not 0 <= matches <= count:
        raise ValueError('invalid HP-order counts')
    return matches / count if count else None


# Move one coordinate at a time toward configs/train/nosl_paper_replay_fresh10_20260913.args.
RECOVERY_STEPS = (
    ('nosl_is_az_gumbel_scale', 1.0),
    ('nosl_is_az_gumbel_value_scale', 0.1),
    ('nosl_is_az_gumbel_interior_value_mode', 0),
    ('nosl_is_az_gumbel_interior_coverage_fraction', 0.0),
    ('nosl_is_az_paper_full_gumbel', True),
    ('nosl_is_az_end_turn_as_leaf', False),
    ('nosl_is_az_simulations', 128),
)


def assess(records, interventions=(), *, now):
    """Return a proposal only; the operator must save/stop/resume and record it."""
    rows = {}
    for record in records:
        if 'global_timesteps' not in record or 'game/floor_won/05' not in record:
            continue
        step, value = int(record['global_timesteps']), float(record['game/floor_won/05'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            return dict(status='invalid_metrics', proposal=None)
        if (float(record.get('nosl_is_az/metric_count', 1)) > 0
                and float(record.get('nosl_is_az/finite_fraction', 1)) < 1):
            return dict(status='nonfinite_training', proposal=None)
        rows[step] = value
    points = sorted(rows.items())
    if len(points) < 6:
        return dict(status='insufficient_history', points=len(points), proposal=None)
    values = [v for _,v in points]
    averages = [sum(values[i:i+3])/3 for i in range(len(values)-2)]
    best, recent = max(averages), averages[-1]
    result = dict(status='healthy', last_step=points[-1][0], latest=values[-1],
        recent_mean=recent, best_mean=best, points=len(points), proposal=None)
    last_change = interventions[-1] if interventions else None
    if last_change:
        fresh = [v for step,v in points if step > last_change['global_step']]
        if len(fresh) < 6 or now-last_change['time'] < 3600:
            return dict(result, status='intervention_grace')
    # A moving episode window is noisy: require a large relative AND absolute
    # loss plus an actual downward segment. After an intervention, also allow
    # a persistently low plateau without recovery after the full grace period.
    falling = values[-1] < values[-3] and recent < sum(values[-6:-3])/3
    plateau = bool(last_change and recent <= sum(fresh[:3])/3 * 1.1)
    if best - recent >= 0.08 and recent <= best * 0.5 and (falling or plateau):
        index = len(interventions)
        proposal = dict(zip(('parameter','value'), RECOVERY_STEPS[index])) if index < len(RECOVERY_STEPS) else None
        return dict(result, status='collapse' if proposal else 'strict_baseline_exhausted', proposal=proposal)
    return result


RELIC_GUARD_KEYS = (
    'global_timesteps', 'game/chest_relic_taken_window',
    'game/chest_relic_opportunities_window', 'game/chest_relic_take_rate',
)


def assess_relic_collapse(records, *, iteration_steps, min_opportunities=400):
    """A five-round reference and three *later*, consecutive low windows.

    Windows are training telemetry, not independent evaluation episodes. Missing
    or undersized windows break a streak; repeated steps never count twice.
    """
    if iteration_steps <= 0 or min_opportunities <= 0:
        raise ValueError('iteration steps and minimum opportunities must be positive')
    rows = {}
    for record in records:
        try:
            step = int(record['global_timesteps'])
            if step <= 0 or step != record['global_timesteps']:
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        rows[step] = None
        try:
            taken = float(record['game/chest_relic_taken_window'])
            count = float(record['game/chest_relic_opportunities_window'])
            rate = float(record['game/chest_relic_take_rate'])
        except (KeyError, TypeError, ValueError):
            continue
        if (all(math.isfinite(v) for v in (taken, count, rate))
                and 0 <= taken <= count and count >= min_opportunities
                and math.isclose(rate, taken / count, abs_tol=1e-9)):
            rows[step] = rate
    points = sorted(rows.items())
    result = dict(status='insufficient_history', stop=False, points=len(points),
                  latest=points[-1][1] if points else None,
                  last_step=points[-1][0] if points else None,
                  reference=None, threshold=None)

    def contiguous(window):
        return (all(rate is not None for _, rate in window)
                and all(b[0] - a[0] == iteration_steps for a, b in zip(window, window[1:])))

    if len(points) < 8:
        return result
    # Exclude the confirmation rounds from the historical reference.
    previous = points[:-3]
    windows = [previous[i:i + 5] for i in range(len(previous) - 4)]
    baselines = [sum(rate for _, rate in w) / 5 for w in windows if contiguous(w)]
    if not baselines:
        return result
    reference = max(baselines)
    threshold = min(reference * .5, reference - .20)
    result.update(reference=reference, threshold=threshold)
    if not contiguous(points[-3:]):
        return dict(result, status='incomplete_recent_windows')
    recent = [rate for _, rate in points[-3:]]
    collapsed = all(rate <= threshold + 1e-12 for rate in recent)
    return dict(result, status='collapse' if collapsed else 'healthy',
                stop=collapsed, recent=recent,
                confirmation_steps=[step for step, _ in points[-3:]])
