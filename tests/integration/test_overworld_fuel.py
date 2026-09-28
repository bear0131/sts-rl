"""Real stacked exhaust block motivating the optional upper fuel prior."""
import pytest
import slaythespire as sts


@pytest.mark.parametrize('stacks,expected_block', [(1, 8), (2, 11), (3, 14)])
def test_feel_no_pain_copies_stack_block_when_corruption_exhausts_skill(combat_factory, stacks, expected_block):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({'player': {'energy': 10, 'block': 0},
        'hand': ['FEEL_NO_PAIN'] * stacks + [('CORRUPTION', 1), 'DEFEND_RED'],
        'draw': [], 'discard': [], 'exhaust': []})
    for name in ['FEEL_NO_PAIN'] * stacks + ['CORRUPTION', 'DEFEND_RED']:
        index = next(i for i, card in enumerate(battle.to_dict()['hand']) if card[0] == name)
        action = next(a for a in battle.rule_actions()
            if a.action_type == sts.ActionType.CARD and a.source_idx == index)
        battle.execute(action)
    state = battle.to_dict()
    assert state['player']['block'] == expected_block
    assert dict(state['player']['powers'])['Feel No Pain'] == 3 * stacks
    assert [c[0] for c in state['exhaust']] == ['DEFEND_RED']
