from rl.training.monitoring import assess


def records(values):
    return [dict(global_timesteps=(i+1)*12288, **{'game/floor_won/05':v}) for i,v in enumerate(values)]


def test_rising_noisy_and_single_bad_point_do_not_trigger():
    for values in ([.28,.51,.67,.6,.72,.7], [.7,.72,.69,.7,.68,.1], [.1]*8):
        assert assess(records(values),now=0)['proposal'] is None


def test_sustained_collapse_proposes_noise_first():
    result=assess(records([.2,.2,.19,.04,0,0]),now=0)
    assert result['status']=='collapse'
    assert result['proposal']==dict(parameter='nosl_is_az_gumbel_scale',value=1.0)


def test_interventions_wait_for_new_samples_and_time_then_move_one_coordinate():
    history=records([.7]*3+[.1]*10)
    changes=[dict(global_step=6*12288,time=0)]
    assert assess(history,changes,now=1800)['status']=='intervention_grace'
    assert assess(history[:9],changes,now=4000)['status']=='intervention_grace'
    result=assess(history,changes,now=4000)
    assert result['proposal']==dict(parameter='nosl_is_az_gumbel_value_scale',value=.1)


def test_invalid_metrics_are_not_search_parameter_evidence():
    assert assess(records([float('nan')]),now=0)['status']=='invalid_metrics'
