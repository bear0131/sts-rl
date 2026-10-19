"""BRIDGE31: recorded client indices must resolve to native entities."""
from copy import deepcopy

import pytest
import slaythespire as sts

from bridge import replay
from schemas.combat_state import normalize_sim

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _frame(battle, *, corpse=False, reverse=False):
    # Synthetic CommunicationMod monster records, with native-only holes absent.
    monsters = []
    for m in normalize_sim(battle.to_dict())['monsters']:
        if m['gone']:
            continue
        monsters.append(dict(id=m['id'], current_hp=m['hp'], max_hp=m['max_hp'],
            block=m['block'], is_gone=False, move_base_damage=m['base_damage'],
            move_adjusted_damage=m['adjusted_damage'], move_hits=m['hits'],
            powers=[dict(id=k, amount=v) for k, v in m['powers'].items()]))
    if reverse:
        monsters.reverse()
    if corpse:
        monsters.insert(0, dict(monsters[0], current_hp=0, is_gone=True))
    return {'combat_state': {'monsters': monsters}}


def _execute(path, env, command, pre):
    if path == 'isolated':
        replay._combat_cmd(env.thick.bc, command, pre=pre)
    else:
        driver = replay.Driver(env.thick.gc)
        driver.bc = env.thick.bc
        driver.combat_cmd(command, pre=pre)


@pytest.mark.parametrize('path', ['isolated', 'natural'])
@pytest.mark.parametrize('encounter,boss', [('Gremlin Leader', 'GREMLIN_LEADER'), ('Reptomancer', 'REPTOMANCER')])
@pytest.mark.parametrize('shape', ['compact', 'corpse', 'reordered'])
def test_recorded_attack_hits_native_entity(combat_factory, path, encounter, boss, shape):
    env = combat_factory(['BASH'] * 5, encounter)
    battle = env.thick.bc
    pre = _frame(battle, corpse=shape == 'corpse', reverse=shape == 'reordered')
    native = next(i for i, m in enumerate(battle.to_dict()['monsters']) if m['id'] == boss)
    real_id = normalize_sim(battle.to_dict())['monsters'][native]['id']
    wire = next(i for i, m in enumerate(pre['combat_state']['monsters']) if m['id'] == real_id and not m['is_gone'])
    control = battle.clone()
    control.execute(sts.Action(sts.ActionType.CARD, 0, native))
    _execute(path, env, f'PLAY 1 {wire}', pre)
    assert battle.to_dict() == control.to_dict()


@pytest.mark.parametrize('path', ['isolated', 'natural'])
def test_split_parent_in_real_list_does_not_shift_target(combat_factory, path):
    env = combat_factory(['BASH'] * 5, 'Slime Boss')
    battle = env.thick.bc
    battle.apply_custom_state({'monsters': [{'hp': 60, 'move': 'SLIME_BOSS_SPLIT'}]})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    pre = _frame(battle, corpse=True)
    pre['combat_state']['monsters'][0]['id'] = 'SlimeBoss'
    # Pick the last split child; the real frame retains an extra dead parent.
    wire = len(pre['combat_state']['monsters']) - 1
    target_id = pre['combat_state']['monsters'][wire]['id']
    native = next(i for i, m in enumerate(normalize_sim(battle.to_dict())['monsters'])
                  if m['id'] == target_id and not m['gone'])
    control = battle.clone()
    control.execute(sts.Action(sts.ActionType.CARD, 0, native))
    _execute(path, env, f'PLAY 1 {wire}', pre)
    assert battle.to_dict() == control.to_dict()


@pytest.mark.parametrize('path', ['isolated', 'natural'])
def test_recorded_targeted_potion_is_executed(combat_factory, path):
    # Fixed public shop entry, not a natural-run fixture/search.
    env = combat_factory(['BASH'] * 5)
    env.reset(seed=16)
    gc = env.thick.gc
    gc.act, gc.floor_num, gc.cur_map_node_y, gc.gold = 2, 18, 1, 999
    gc.transition_to_map_node(5)
    assert gc.shop_contents()['potions'][1]['id'] == 'Fire Potion'
    gc.shop_buy_potion(1)
    env.thick.bc = sts.create_battle(gc, 'Gremlin Leader')
    battle = env.thick.bc
    pre = _frame(battle)
    wire = next(i for i, m in enumerate(pre['combat_state']['monsters']) if m['id'] == 'GremlinLeader')
    native = next(i for i, m in enumerate(battle.to_dict()['monsters']) if m['id'] == 'GREMLIN_LEADER')
    control = battle.clone()
    control.execute(sts.Action(sts.ActionType.POTION, 0, native))
    _execute(path, env, f'POTION use 0 {wire}', pre)
    assert battle.to_dict() == control.to_dict()
    assert list(battle.potions) == list(control.potions)


@pytest.mark.parametrize('path', ['isolated', 'natural'])
@pytest.mark.parametrize('problem', ['missing', 'dead', 'negative', 'outside', 'desync', 'ambiguous'])
def test_unresolved_target_does_not_advance(combat_factory, path, problem):
    env = combat_factory(['BASH'] * 5, 'Gremlin Leader' if problem != 'ambiguous' else 'Two Fungi Beasts')
    battle = env.thick.bc
    if problem == 'ambiguous':
        battle.apply_custom_state({'monsters': [{'hp': 20, 'max_hp': 20, 'move': 'FUNGI_BEAST_BITE'}] * 2})
    pre = _frame(battle, corpse=problem == 'dead')
    wire = 0
    if problem == 'missing': pre = None
    if problem == 'negative': wire = -1
    if problem == 'outside': wire = 99
    if problem == 'desync': pre['combat_state']['monsters'][0]['current_hp'] -= 1
    before = deepcopy(battle.to_dict())
    with pytest.raises(ValueError):
        _execute(path, env, f'PLAY 1 {wire}', pre)
    assert battle.to_dict() == before


def test_encounter_map_keeps_reserved_slot_encounters(combat_factory, monkeypatch):
    env = combat_factory(['BASH'] * 5)
    names = ['Gremlin Leader', 'Reptomancer', 'Automaton', 'Collector']
    monkeypatch.setattr(replay.sts, 'encounter_names', lambda: names)
    mapping = replay.build_encounter_map(env.thick.gc)
    assert set(mapping.values()) == set(names)


@pytest.mark.parametrize('path', ['isolated', 'natural'])
@pytest.mark.parametrize('command', ['PLAY 0 0', 'PLAY 65537 0', 'CHOOSE -1', 'CHOOSE 65536'])
def test_invalid_source_index_cannot_wrap_to_a_legal_action(combat_factory, path, command):
    env = combat_factory(['BASH'] * 5)
    if command.startswith('CHOOSE'):
        env.thick.bc.apply_custom_state({'hand': ['SECRET_TECHNIQUE'],
            'draw': ['DEFEND_RED', 'SHRUG_IT_OFF']})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    before = deepcopy(env.thick.bc.to_dict())
    with pytest.raises(ValueError):
        _execute(path, env, command, _frame(env.thick.bc))
    assert env.thick.bc.to_dict() == before


def _full_frame(battle):
    frame = _frame(battle)
    canonical = normalize_sim(battle.to_dict())
    p = canonical['player']
    def cards(pile):
        return [dict(id=c['id'], upgrades=c['up'], cost=c.get('cost', 0)) for c in canonical[pile]]
    frame.update(seed=42, floor=1, ascension_level=0, screen_type='NONE')
    frame['combat_state'].update(turn=canonical['turn'],
        player=dict(current_hp=p['hp'], max_hp=p['max_hp'], block=p['block'], energy=p['energy'],
                    powers=[dict(id=k, amount=v) for k, v in p['powers'].items()]),
        hand=cards('hand'), draw_pile=cards('draw'), discard_pile=cards('discard'), exhaust_pile=cards('exhaust'))
    return frame


@pytest.mark.parametrize('entry', ['validate_combats', 'run_natural'])
def test_replay_loop_uses_each_commands_pre_frame(combat_factory, monkeypatch, capsys, entry):
    env = combat_factory(['BASH'] * 5, 'Gremlin Leader')
    env.thick.bc.set_player_energy(6)
    battle = env.thick.bc
    initial = _full_frame(battle)
    control = battle.clone()
    steps = []
    for _ in range(2):
        control.execute(sts.Action(sts.ActionType.CARD, 0, 3))
        steps.append(('PLAY 1 2', _full_frame(control)))
    # Isolate already-tested master-deck/map reconstruction. Both replay loops,
    # their frame progression, action execution and state diffs remain real.
    monkeypatch.setattr(replay, 'build_steps', lambda events: (initial, steps))
    if entry == 'validate_combats':
        key = tuple(sorted(m['id'] for m in initial['combat_state']['monsters']))
        monkeypatch.setattr(replay, 'build_encounter_map', lambda gc: {key: 'Gremlin Leader'})
        monkeypatch.setattr(replay, '_sync_gc', lambda *args: env.thick.gc)
        monkeypatch.setattr(replay.sts, 'create_battle', lambda *args: battle)
    else:
        driver = replay.Driver(env.thick.gc)
        driver.bc = battle
        monkeypatch.setattr(replay, 'Driver', lambda gc: driver)
    getattr(replay, entry)([], False)
    output = capsys.readouterr().out
    assert 'DESYNC' not in output and '[DIFF]' not in output and 'error on' not in output
    assert battle.to_dict() == control.to_dict()
