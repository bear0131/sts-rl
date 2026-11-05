"""Public facts and actual entity references for a variable-action lower scorer."""
import numpy as np
import pytest
import env.core
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


def rows(features, kind, zone=None, group='entities'):
    layout = sts.small_playout_public_layout()
    table = features[group]
    mask = table['ids'] // layout['stride'] == layout['kinds'].index(kind)
    if zone is not None:
        mask &= table['zones'] == layout['zones'].index(zone)
    return np.flatnonzero(mask)


def get(env, battle=None):
    return sts.small_playout_public_features(env.thick.gc, env.thick.bc if battle is None else battle)


@pytest.mark.parametrize('frozen', [False, True])
def test_only_visible_draw_order_distinguishes_permutations(combat_factory, frozen):
    env = combat_factory(['STRIKE_RED'] * 5, relics=['Frozen Eye'] if frozen else [])
    bc = env.thick.bc
    bc.apply_custom_state({'hand':['BASH'], 'draw':['ANGER','CLEAVE','DEFEND_RED']})
    before = get(env)
    other = bc.clone()
    state = other.card_state()
    state['draw'].reverse()
    other.set_card_state(state)
    after = get(env, other)
    for field in ('state','actions','keys','action_bits','source_rows','target_slots'):
        np.testing.assert_array_equal(before[field], after[field])
    visible = rows(before,'card','known_draw' if frozen else 'hidden_draw')
    assert len(visible) == 3
    if frozen:
        assert list(before['entities']['ids'][visible]) == list(after['entities']['ids'][visible][::-1])
    else:
        for key in before['entities']:
            np.testing.assert_array_equal(before['entities'][key], after['entities'][key])


def test_placed_top_remains_visible_without_frozen_eye(combat_factory):
    env = combat_factory(['WARCRY']*5)
    bc = env.thick.bc
    bc.apply_custom_state({'hand':['WARCRY','BASH','DEFEND_RED'], 'draw':['ANGER','CLEAVE'], 'discard':[]})
    bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    bc.execute(next(a for a in bc.legal_actions() if int(a.action_type)==2))
    assert sts.small_playout_known_top(bc) == 1
    assert len(rows(get(env),'card','known_draw')) == 1


@pytest.mark.parametrize('dome', [False, True])
def test_concealed_current_intent_has_no_entity_side_channel(combat_factory, dome):
    env = combat_factory(['STRIKE_RED']*5, relics=['Runic Dome'] if dome else [])
    bc = env.thick.bc
    bc.apply_custom_state({'monsters':[{'move':'JAW_WORM_CHOMP'}]})
    a = get(env)
    bc.apply_custom_state({'monsters':[{'move':'JAW_WORM_BELLOW'}]})
    b = get(env)
    if dome:
        for key in a['entities']:
            np.testing.assert_array_equal(a['entities'][key],b['entities'][key])
        np.testing.assert_array_equal(a['state'],b['state'])
    else:
        assert not np.array_equal(a['entities']['ids'],b['entities']['ids'])


def test_source_and_target_reference_real_objects_across_empty_slot(combat_factory):
    env = combat_factory(['STRIKE_RED']*5,encounter='Three Darklings')
    bc = env.thick.bc
    bc.apply_custom_state({'monsters':[{'hp':0,'is_dead':True,'gone':True},{'hp':17},{'hp':29}],
                           'potions':['Fire Potion','Smoke Bomb']})
    x = get(env)
    assert set(x['entities']['monsters'][rows(x,'monster')]) == set(range(5))
    candidates = set()
    for bits, sources, target in zip(x['action_bits'],x['source_rows'],x['target_slots']):
        action = sts.Action(bits)
        if bc.is_targeted_action(bits):
            assert target == action.target_idx and target in (1,2)
            candidates.add(target)
        else:
            assert target == -1
        if action.action_type == sts.ActionType.POTION:
            assert sources[0] == rows(x,'potion','potion')[action.source_idx]
        if action.action_type == sts.ActionType.CARD:
            assert sources[0] == rows(x,'card','hand')[action.source_idx]
    assert candidates == {1,2}


@pytest.mark.parametrize('name,zone',[('HEADBUTT','discard'),('EXHUME','exhaust')])
def test_large_pile_sources_keep_instance_properties(combat_factory,name,zone):
    env = combat_factory([name]*5)
    bc = env.thick.bc
    bc.apply_custom_state({'hand':[name],zone:[('SEARING_BLOW',i) for i in range(100)]+[('SEARING_BLOW',0)]*72})
    bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    x = get(env)
    assert len(rows(x,'card',zone)) == 172
    assert x['source_rows'].shape == (100,10)
    assert list(x['action_bits']) == list(sts.small_playout_features(bc)['action_bits'])
    np.testing.assert_allclose(x['entities']['numbers'][x['source_rows'][:,0],0],np.arange(100)/10,rtol=1e-6)
    assert np.all(x['source_rows'][:,1:] == -1)


@pytest.mark.parametrize('potion',['Gamblers Brew','Elixir Potion','Attack Potion','Liquid Memories'])
def test_pending_potion_is_not_the_last_played_card(combat_factory,potion):
    env = combat_factory(['BASH']*5)
    bc = env.thick.bc
    bc.apply_custom_state({'hand':['BASH','STRIKE_RED','DEFEND_RED'],'potions':[potion],
                           'discard':['ANGER','CLEAVE','WOUND']})
    before = get(env)
    potion_id = before['entities']['ids'][rows(before,'potion','potion')[0]]
    bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    bc.execute(sts.Action(sts.ActionType.POTION,0,0))
    after = get(env)
    assert list(after['entities']['ids'][rows(after,'potion','pending')]) == [potion_id]
    assert not len(rows(after,'card','pending'))
    assert len(rows(after,'selection','pending')) == 1


def test_relic_gamble_and_card_selection_keep_distinct_origins(combat_factory):
    chip = combat_factory(['BASH']*5,relics=['Gambling Chip'])
    x = get(chip)
    owned = x['inventory']['ids'][rows(x,'relic','owned_relic','inventory')]
    assert list(x['entities']['ids'][rows(x,'relic','pending')]) == list(owned)
    env = combat_factory(['TRUE_GRIT']*5)
    bc = env.thick.bc
    bc.apply_custom_state({'hand':[('TRUE_GRIT',1),'BASH','DEFEND_RED']})
    before = get(env)
    source_id = before['entities']['ids'][rows(before,'card','hand')[0]]
    bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    after = get(env)
    assert list(after['entities']['ids'][rows(after,'card','pending')]) == [source_id]


def test_inventory_and_non_actionable_cards_are_visible(combat_factory):
    env = combat_factory(['BASH','STRIKE_RED','DEFEND_RED']*3,relics=['Burning Blood','Anchor'])
    bc = env.thick.bc
    bc.apply_custom_state({'hand':['BASH'], 'discard':['ANGER']})
    before = get(env)
    assert len(rows(before,'card','master_deck','inventory')) == 9
    assert len(rows(before,'relic','owned_relic','inventory')) == 2
    bc.apply_custom_state({'discard':['CLEAVE']})
    after = get(env)
    np.testing.assert_array_equal(before['state'],after['state'])
    np.testing.assert_array_equal(before['actions'],after['actions'])
    assert before['entities']['ids'][rows(before,'card','discard')[0]] != after['entities']['ids'][rows(after,'card','discard')[0]]
