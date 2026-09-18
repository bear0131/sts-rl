"""Sampling preserves selected entities and never executes live hidden futures."""
import pytest
import slaythespire as sts

from env.core.native_actions import complete_actions

pytestmark = pytest.mark.native
END = 4 << 29


def draw_selection(combat_factory, draw, *, dark_embrace=False):
    env=combat_factory(['SECRET_WEAPON']*5)
    env.thick.bc.apply_custom_state({'hand':['SECRET_WEAPON'],'draw':draw,
        'discard':[],'exhaust':[],
        'player':{'powers':[('DARK_EMBRACE',int(dark_embrace))]}})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
    return env.thick.bc


def test_draw_remap_distinguishes_duplicate_entities_and_late_indices(combat_factory):
    source=draw_selection(combat_factory,['DEFEND_RED']*10+['RITUAL_DAGGER']*2)
    actions=complete_actions(source)
    assert [a & 0xffff for a in actions]==[10,11]
    source_before=source.to_dict()
    for nonce in range(8):
        sample=source.clone()
        sample.nosl_redeterminize(nonce)
        before=sample.to_dict()
        mapped=[sample.remap_draw_selection_from(source,a) for a in actions]
        assert len(set(mapped))==2  # same card name AND upgrade count, distinct entities
        assert set(mapped)==set(complete_actions(sample))
        assert [source.remap_draw_selection_from(sample,a) for a in mapped]==list(actions)
        assert sample.to_dict()==before and source.to_dict()==source_before
        for bits in mapped:
            branch=sample.clone()
            branch.execute(sts.Action(bits))
            assert branch.to_dict()['hand'][0][0]=='RITUAL_DAGGER'


def test_sample_before_selection_hides_queued_dark_embrace_draw(combat_factory):
    source=draw_selection(combat_factory,
        ['STRIKE_RED','BASH','DEFEND_RED','SHRUG_IT_OFF'],dark_embrace=True)
    chosen=complete_actions(source)[0]
    results=[]
    live_draws=[]
    for actual_future in range(8):
        live=source.clone()
        live.nosl_redeterminize(actual_future)
        live_bits=live.remap_draw_selection_from(source,chosen)
        # This reproduces why execute-before-sample is unsafe: selection pumps
        # the pending exhaust action, whose Dark Embrace draws from live future.
        old=live.clone()
        old.execute(sts.Action(live_bits))
        live_draws.append(tuple(card[0] for card in old.to_dict()['hand']))
        sampled=live.clone()
        sampled.nosl_redeterminize(999)
        mapped=sampled.remap_draw_selection_from(live,live_bits)
        sampled.execute(sts.Action(mapped))
        results.append(sampled.to_dict())
    assert len(set(live_draws))>1
    assert all(result==results[0] for result in results)
    assert len(results[0]['hand'])==2


def test_remap_rejects_wrong_context_and_filtered_source_index(combat_factory):
    source=draw_selection(combat_factory,['DEFEND_RED']*10+['STRIKE_RED','BASH'])
    sample=source.clone()
    sample.nosl_redeterminize(123)
    with pytest.raises(ValueError,match='legal source'):
        sample.remap_draw_selection_from(source,2<<29)  # non-attack source index0
    normal=combat_factory(['STRIKE_RED']*5).thick.bc
    with pytest.raises(ValueError,match='matching live'):
        normal.remap_draw_selection_from(source,2<<29 | 10)
    with pytest.raises(ValueError,match='legal source'):
        sample.remap_draw_selection_from(source,END)


@pytest.mark.parametrize('terminal',[False,True])
def test_end_turn_result_returns_boundary_or_terminal_without_mutation(combat_factory,terminal):
    env=combat_factory(['DEFEND_RED']*5)
    bc=env.thick.bc
    bc.apply_custom_state({'player':{'hp':1 if terminal else 80,'energy':0},
                          'monsters':[{'move':'JAW_WORM_CHOMP'}]})
    before=bc.to_dict()
    result=bc.preview_end_turn_result(END)
    assert bc.to_dict()==before
    legacy=bc.preview_end_turn_boundary(END)
    if terminal:
        assert legacy is None and result['outcome']==2 and result['player']['hp']<=0
        actual=bc.clone()
        actual.execute(sts.Action(END))
        assert actual.to_dict()==result
    else:
        assert result==legacy and result['post_enemy_turn_boundary']
    with pytest.raises(ValueError,match='legal live END_TURN'):
        bc.preview_end_turn_result(0)
