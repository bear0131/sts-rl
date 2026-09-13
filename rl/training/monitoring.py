"""Conservative, inspectable proposals for the requested NOSL recovery experiment."""
import math


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
