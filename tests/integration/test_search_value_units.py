"""Same oracle in raw vs normalized units, including native terminal transport."""
import numpy as np
import pytest
pytestmark=[pytest.mark.native, pytest.mark.timeout(120)]

@pytest.mark.parametrize('outlet', ['win','escape','loss','ongoing'])
@pytest.mark.parametrize('noise', [0.,1.])
def test_full_gumbel_unit_conversion_preserves_search(run_combat_factory,outlet,noise):
    import slaythespire as sts
    root=run_combat_factory(potions=['Smoke Bomb'] if outlet=='escape' else [])
    root.bc=sts.create_battle(root.gc,'Jaw Worm')
    battle=root.bc
    if outlet=='win': battle.set_player_strength(999)
    if outlet=='loss': battle.apply_custom_state(dict(player=dict(hp=1)))
    initial=battle.to_dict(); results=[]; request_traces=[]
    for divisor in (1.,2.):
        session=battle.nosl_neural_search_begin(np.linspace(-.4,.4,198), simulations=128,depth=0,
            nonce=9072400,root_value=.72/divisor,fixed_worldline=True,outer_return=True,
            root_search_mode=2,paper_full_gumbel=True,gumbel_candidates=16,gumbel_scale=noise,
            gumbel_value_scale=divisor,gumbel_rescale_q=False,gumbel_use_mixed_value=True)
        trace=[]
        for _ in range(2000):
            batch=session.next_batch_packed(1)
            if batch['done']:break
            exits=batch.get('combat_exits',{})
            for row,request in enumerate(batch['request_ids']):
                terminal=row in exits
                # Fixed oracle value in raw units; all consumers receive one unit system.
                value=1.2 if terminal and outlet=='win' else .3 if terminal and outlet=='escape' else 0. if terminal else .72
                trace.append((int(request),terminal,value))
                session.backprop(request,value/divisor,[] if terminal else np.linspace(-.4,.4,198))
        else: pytest.fail('bounded search did not complete')
        results.append(session.result()); request_traces.append(trace)
    assert request_traces[0] == request_traces[1]
    for key in ('action_bits','root_visits','total_root_visits'):
        assert results[0][key] == results[1][key],key
    np.testing.assert_array_equal(results[0]['root_policy'], results[1]['root_policy'])
    assert results[0]['root_value'] == pytest.approx(2*results[1]['root_value'],abs=1e-12)
    assert battle.to_dict() == initial
