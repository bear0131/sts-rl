"""BRIDGE06: native event semantics must not be UI button ordinals.

Fixtures describe public option contracts observed in BRIDGE05 real frames.
No live game or checkpoint is needed for this static adapter audit.
"""
import pytest


@pytest.mark.parametrize('label', ['[喝水] 移除所有诅咒牌。', '[Drink] Remove all Curses from your deck.'])
@pytest.mark.parametrize('empty', [False, True])
def test_fountain_removes_exact_eligible_curses(event_adapter, label, empty):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('Fountain of Cleansing', [(label, False), ('[Leave]', False)])
    fixed = [dict(id=i, type='CURSE', upgrades=0) for i in ('AscendersBane', 'CurseOfTheBell', 'Necronomicurse')]
    fixed += [dict(id='Strike_R', type='ATTACK', upgrades=0)]
    curses = [] if empty else [dict(id=i, type='CURSE', upgrades=0) for i in ('Regret', 'Regret', 'Parasite')]
    a.gs.update(deck=fixed + curses, relics=[], potions=[], gold=50, max_hp=80, current_hp=70)
    assert a.command_for_semantic(('event', 0)) == 'CHOOSE 0'
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 0', 'policy', semantic=('event', 0)), 0)
    assert ack.waiting(a.frame, 1)
    after = deepcopy(a.frame)
    after['game_state']['deck'] = fixed
    assert ack.waiting(after, 2)  # Card deltas alone do not complete the command.
    after['game_state']['screen_state']['options'] = [dict(text='[Leave]', choice_index=0)]
    bad = deepcopy(after)
    bad['game_state']['deck'] = fixed[1:]
    with pytest.raises(RuntimeError, match='wrong card'):
        ack.waiting(bad, 3)
    if not empty:
        partial = deepcopy(after)
        partial['game_state']['deck'] = fixed + curses[:1]
        assert ack.waiting(partial, 3)
        after['game_state']['max_hp'] = 77  # Parasite removal side effect.
    assert not ack.waiting(after, 4)


@pytest.mark.parametrize('label', ['[搜索] #g找到一些药水！', '[Search] #gFind #gsome #gPotions!'])
@pytest.mark.parametrize('ascension,count', [(0, 3), (15, 2)])
def test_lab_is_reward_transport_with_exact_type_and_count(event_adapter, label, ascension, count):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('Lab', [(label, False)])
    a.gs['ascension_level'] = ascension
    assert a.auto_command() == ('CHOOSE 0', 'event_dialogue_transport')
    ack = EventEffectAck()
    decision = RealDecision('CHOOSE 0', 'event_dialogue_transport')
    ack.arm(a.frame, decision, 0)
    assert ack.waiting(a.frame, 1)
    with pytest.raises(RuntimeError, match='still unverified'):
        ack.arm(a.frame, decision, 1)
    after = deepcopy(a.frame)
    after['game_state'].update(screen_type='COMBAT_REWARD', screen_state={
        'rewards': [dict(reward_type='POTION', potion=dict(id='Dexterity Potion')) for _ in range(count)]})
    bad = deepcopy(after)
    bad['game_state']['screen_state']['rewards'][0]['reward_type'] = 'CARD'
    with pytest.raises(RuntimeError, match='type/count'):
        ack.waiting(bad, 2)
    bad = deepcopy(after)
    bad['game_state']['screen_state']['rewards'].pop()
    with pytest.raises(RuntimeError, match='type/count'):
        ack.waiting(bad, 2)
    assert not ack.waiting(after, 3)


@pytest.mark.parametrize('label', ['[放手吧] #r失去 #r0 #r金币。', '[Leave It] #rLose #r0 #rGold.'])
def test_goop_zero_gold_loss_requires_result_dialogue(event_adapter, label):
    from bridge.event_contract import EventEffectAck
    from types import SimpleNamespace
    from copy import deepcopy
    a = event_adapter('World of Goop', [('[Gather Gold]', False), (label, False)])
    a.gs.update(gold=0, current_hp=40, max_hp=80, deck=[], relics=[], potions=[])
    assert a.command_for_semantic(('event', 1)) == 'CHOOSE 1'
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 1)), 0)
    assert ack.waiting(a.frame, 1)
    after = deepcopy(a.frame)
    after['game_state']['screen_state']['options'] = [dict(text='[Leave]', choice_index=0)]
    assert not ack.waiting(after, 2)


@pytest.mark.parametrize('blocked', [False, True])
def test_goop_gold_ack_rejects_wrong_reward_and_waits_for_dialogue(event_adapter, blocked):
    from bridge.event_contract import EventEffectAck
    from types import SimpleNamespace
    from copy import deepcopy
    a = event_adapter('World of Goop', [('[收集金币]', False), ('[放手吧] 失去20金币。', False)])
    a.gs.update(gold=100, current_hp=40, max_hp=80, deck=[], potions=[],
                relics=[dict(id='Ectoplasm')] if blocked else [])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    after = deepcopy(a.frame)
    after['game_state'].update(gold=100 if blocked else 175, current_hp=29)
    assert ack.waiting(after, 1)
    bad = deepcopy(after)
    bad['game_state']['gold'] += 1
    with pytest.raises(RuntimeError, match='amount mismatch'):
        ack.waiting(bad, 2)
    after['game_state']['screen_state']['options'] = [dict(text='[Leave]', choice_index=0)]
    assert not ack.waiting(after, 3)


@pytest.mark.parametrize('bad_index', [None, -1, True, 0.5, '0'])
def test_event_wire_index_must_be_nonnegative_integer(event_adapter, bad_index):
    from bridge.event_contract import options, transport_only
    a = event_adapter('Big Fish', [('[Banana]', False)])
    a.gs['screen_state']['options'][0]['choice_index'] = bad_index
    with pytest.raises(ValueError, match='choice_index'):
        options(a.gs)
    a.gs['screen_state']['options'][0]['text'] = '[Continue]'
    with pytest.raises(ValueError, match='choice_index'):
        transport_only(a.gs)


def test_distinct_event_semantics_cannot_share_wire_index(event_adapter):
    from bridge.event_contract import options, transport_only
    a = event_adapter('Big Fish', [('[Banana]', False), ('[Donut]', False)])
    a.gs['screen_state']['options'][1]['choice_index'] = 0
    for decode in (options, transport_only):
        with pytest.raises(ValueError, match='duplicate'):
            decode(a.gs)
    # Disabled source options do not have a selectable wire index.
    a.gs['screen_state']['options'][1]['disabled'] = True
    assert options(a.gs) == {0: (0, 'heal')}


@pytest.mark.parametrize('label', ['[攻击] #b???', '[Attack] #b???', '[睡眠]', '[Sleep]', '[走向门前]', '[Approach Door]'])
def test_heart_ending_transport_is_not_terminal(event_adapter, label):
    a = event_adapter('Spire Heart', [(label, False)])
    assert a.auto_command() == ('CHOOSE 0', 'event_dialogue_transport')
    assert a.gs['screen_type'] == 'EVENT'
    from bridge.event_contract import transport_only
    a.gs['combat_state'] = {'turn': 1}
    assert transport_only(a.gs) is None


@pytest.mark.parametrize('label', ['……', '...'])
def test_winding_intro_is_transport_only(event_adapter, label):
    a = event_adapter('Winding Halls', [(label, False)])
    assert a.auto_command() == ('CHOOSE 0', 'event_dialogue_transport')
    a = event_adapter('Knowing Skull', [(label, False)])
    with pytest.raises(ValueError, match='unrecognized'):
        a._legal_event_actions()


@pytest.mark.parametrize('labels', [('献祭', '亵渎'), ('Sacrifice', 'Desecrate')])
def test_altar_disabled_idol_preserves_native_id_and_both_effects(event_adapter, labels):
    from bridge.event_contract import EventEffectAck
    from types import SimpleNamespace
    from copy import deepcopy
    a = event_adapter('Forgotten Altar', [('[Locked]', True), (f'[{labels[0]}]', False), (f'[{labels[1]}]', False)])
    assert a.command_for_semantic(('event', 1)) == 'CHOOSE 0'
    assert a.command_for_semantic(('event', 2)) == 'CHOOSE 1'
    a.gs.update(max_hp=92, current_hp=89, deck=[], relics=[])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 1)), 0)
    after = deepcopy(a.frame)
    after['game_state'].update(max_hp=97, current_hp=94)
    assert ack.waiting(after, 1)  # Max HP alone is not the complete effect.
    after['game_state']['current_hp'] = 71
    assert not ack.waiting(after, 2)
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 3)
    after = deepcopy(a.frame)
    after['game_state']['deck'] = [dict(id='Regret', upgrades=0)]
    with pytest.raises(RuntimeError, match='curse identity'):
        ack.waiting(after, 4)
    after['game_state']['deck'] = [dict(id='Decay', upgrades=0)]
    assert not ack.waiting(after, 5)


@pytest.mark.parametrize('label', ['给金币', 'Offer Gold'])
def test_beggar_paid_removal_two_dialogues(event_adapter, label):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('Beggar', [('[Leave]', False), (f'[{label}] 75 Gold', False)])
    assert a.command_for_semantic(('event', 0)) == 'CHOOSE 1'
    a.gs.update(gold=100, deck=[dict(id='Strike_R', upgrades=0)])
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 1', 'policy', semantic=('event', 0)), 0)
    assert ack.waiting(a.frame, 1)
    paid = deepcopy(a.frame)
    paid['game_state'].update(gold=25, screen_state={'event_id':'Beggar', 'options':[dict(choice_index=0,text='[Continue]')]})
    assert not ack.waiting(paid, 2) and ack.pending
    ack.arm(paid, RealDecision('CHOOSE 0','event_dialogue_transport'), 2)
    assert ack.waiting(paid, 3)  # Cannot repeat the second dialogue.
    paid['game_state']['screen_type']='GRID'
    assert not ack.waiting(paid, 4) and ack.pending
    bad=deepcopy(paid)
    bad['game_state']['gold']=0
    with pytest.raises(RuntimeError,match='mismatch'):
        ack.waiting(bad,5)
    paid['game_state']['deck']=[]
    assert not ack.waiting(paid,6) and ack.pending is None


@pytest.mark.parametrize('label', ['治疗', 'Heal'])
def test_cleric_heal_requires_payment_and_healing(event_adapter, label):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('The Cleric', [(f'[{label}]', False), ('[Leave]', False)])
    a.gs.update(gold=100, current_hp=40, max_hp=86)
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 0', 'policy', semantic=('event', 0)), 0)
    after = deepcopy(a.frame)
    after['game_state']['gold'] = 65
    assert ack.waiting(after, 1)
    after['game_state']['current_hp'] = 61
    assert not ack.waiting(after, 2)


@pytest.mark.parametrize('ascension,cost', [(0,50), (15,75)])
def test_cleric_removal_checks_cost_and_count(event_adapter, ascension, cost):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('The Cleric', [('[离开]', False), ('[净化]', False), ('[治疗]', True)])
    assert a.command_for_semantic(('event', 1)) == 'CHOOSE 1'
    a.gs.update(gold=100, ascension_level=ascension, deck=[dict(id='Strike_R', upgrades=0)] * 2)
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 1', 'policy', semantic=('event', 1)), 0)
    after = deepcopy(a.frame)
    after['game_state'].update(gold=100-cost, screen_type='GRID')
    assert not ack.waiting(after, 1) and ack.pending
    bad = deepcopy(after)
    bad['game_state']['deck'] = []
    with pytest.raises(RuntimeError, match='mismatch'):
        ack.waiting(bad, 2)
    after['game_state']['deck'].pop()
    assert not ack.waiting(after, 3) and ack.pending is None


def test_empty_event_load_wait_is_bounded(monkeypatch):
    import time
    a = RealGameAdapter()
    f = {'in_game': True, 'ready_for_command': True, 'available_commands': ['state'],
         'game_state': {'seed': 200206, 'floor': 11, 'screen_type': 'EVENT',
                        'screen_state': {'event_id': 'Mushrooms', 'options': []}}}
    monkeypatch.setattr(time, 'monotonic', lambda: 0)
    a.update(f)
    assert a.auto_command() == ('STATE', 'await_event_options')
    monkeypatch.setattr(time, 'monotonic', lambda: 11)
    with pytest.raises(RuntimeError, match='did not load'):
        a.auto_command()


@pytest.mark.parametrize('event,labels', [
    ('Mushrooms', ('[吃下] 回复21点生命。被诅咒——寄生。', '[踩扁] 激怒蘑菇们。')),
    ('Mysterious Sphere', ('[Leave]', '[Open Sphere] Fight. Reward: Rare Relic.')),
    ('Mysterious Sphere', ('[离开]', '[打开圆球] 战斗。奖励：稀有遗物。')),
])
def test_mushrooms_reordered_options_and_two_stage_combat(event_adapter, event, labels):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter(event, [(label, False) for label in labels])
    assert a.command_for_semantic(('event', 0)) == 'CHOOSE 1'
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 1', 'policy', semantic=('event', 0)), 0)
    assert ack.waiting(a.frame, 1)
    fight = deepcopy(a.frame)
    fight['game_state']['screen_state']['options'] = [dict(choice_index=0, text='[战斗]')]
    assert not ack.waiting(fight, 2)
    ack.arm(fight, RealDecision('CHOOSE 0', 'event_dialogue_transport'), 2)
    assert ack.waiting(fight, 3)  # No duplicate confirmation.
    fight['game_state']['room_phase'] = 'COMBAT'
    assert not ack.waiting(fight, 4)


@pytest.mark.parametrize('blocked', [False, True])
def test_mushrooms_heal_and_curse_both_required(event_adapter, blocked):
    from bridge.event_contract import EventEffectAck
    from bridge.real_game_env import RealDecision
    from copy import deepcopy
    a = event_adapter('Mushrooms', [('[Eat] Heal 20 HP. Become Cursed - Parasite.', False)])
    if blocked:
        a.gs['relics'] = [dict(id='Omamori', counter=2)]
    ack = EventEffectAck()
    ack.arm(a.frame, RealDecision('CHOOSE 0', 'policy', semantic=('event', 1)), 0)
    after = deepcopy(a.frame)
    after['game_state']['current_hp'] = 80
    assert ack.waiting(after, 1)
    if blocked:
        after['game_state']['relics'][0]['counter'] = 1
    else:
        after['game_state']['deck'] = [dict(id='Parasite', upgrades=0)]
    assert not ack.waiting(after, 2)


def test_complete_screen_is_transport_not_terminal_receipt():
    from bridge.campaign import SeedCampaign
    a = RealGameAdapter()
    frame = {'in_game': True, 'ready_for_command': True, 'available_commands': ['proceed', 'state'],
             'game_state': {'screen_type': 'COMPLETE', 'seed': 200205, 'floor': 51,
                            'screen_state': {}, 'choice_list': []}}
    a.update(frame)
    assert a.auto_command() == ('PROCEED', 'complete_run_transition')
    campaign = SeedCampaign(['4NF5'])
    assert campaign.observe(frame) is None
    assert not campaign.completed
    frame['available_commands'] = ['state']
    a.update(frame)
    assert a.auto_command() == ('STATE', 'await_complete_run_transition')


@pytest.mark.parametrize('failure', ['empty_mask', 'inner_limit'])
def test_policy_failures_never_choose_transport_fallback(monkeypatch, failure):
    import numpy as np
    from bridge.client import agent_orchestrator as o
    a = RealGameAdapter()
    monkeypatch.setattr(a, 'auto_command', lambda **kw: None)
    monkeypatch.setattr(a, 'observation', lambda **kw: {})
    monkeypatch.setattr(a, 'action_mask', lambda **kw: np.array([failure != 'empty_mask']))
    monkeypatch.setattr(a, 'legal_semantic_actions', lambda: [('event', 0)])
    monkeypatch.setattr(a, 'select_descriptor', lambda: None)
    monkeypatch.setattr(a, 'describe_action_id', lambda *args: 'prefix')
    monkeypatch.setattr(a, 'fallback_command', lambda: pytest.fail('unsafe fallback invoked'))
    monkeypatch.setattr(o.A, 'resolve', lambda *args: (None, ('prefix', 0)))
    agent = o.RealPolicyAgent(None, a, stochastic=False, topk=0, max_inner_decisions=1,
        auto_enter_shop=False, auto_start=False, start_seed=None, ascension=0, character='IRONCLAD')
    monkeypatch.setattr(agent, '_predict', lambda *args, **kw: (0, ()))
    with pytest.raises(RuntimeError, match='refusing fallback'):
        agent.decide({'in_game': True, 'available_commands': ['choose', 'end', 'proceed'],
            'game_state': {'screen_type': 'EVENT', 'choice_list': ['first']}})


def test_unknown_event_cannot_be_encoded_as_padding(event_adapter):
    a = event_adapter('Unknown Mod Event', [('[祈祷]', False)])
    with pytest.raises(ValueError, match='unknown event identity'):
        a._options('event')


def test_supported_event_contracts_have_checkpoint_identity():
    from bridge.event_contract import CONTRACTS
    from bridge.real_game_env import _normalize_event
    for event in [*CONTRACTS, 'SensoryStone', 'Falling']:
        assert _normalize_event(event) != 'PAD', event
    assert _normalize_event('MindBloom') == 'Mindbloom'


def test_mindbloom_real_rich_label_and_reordered_semantics(event_adapter):
    from bridge.event_contract import options
    a = event_adapter('MindBloom', [('[我必凯旋] 与第一阶段Boss战斗。', False),
        ('[我已觉醒] 升级所有牌。你无法再回复生命。', False),
        ('[我将富有] 获得999金币。被诅咒——2张凡庸。', False)])
    assert options(a.gs) == {0:(0,'combat'),1:(1,'relic'),2:(2,'gold')}
    assert a._options('event')['event'] == 'Mindbloom'


def test_falling_maps_by_type_label_not_ui_order_and_checks_exact_loss():
    from copy import deepcopy
    from types import SimpleNamespace
    from bridge.event_contract import options, EventEffectAck, falling_choices
    deck = [dict(id='Defend_R', name='防御', type='SKILL', upgrades=0),
            dict(id='Demon Form', name='恶魔形态', type='POWER', upgrades=0),
            dict(id='Iron Wave', name='铁斩波+', type='ATTACK', upgrades=1)]
    gs = dict(deck=deck, screen_type='EVENT', screen_state=dict(event_id='Falling', options=[
        dict(choice_index=0,text='[猛击] 失去 铁斩波+'),
        dict(choice_index=1,text='[落地] 失去 防御'),
        dict(choice_index=2,text='[释放] 失去 恶魔形态')]))
    assert options(gs) == {2:(0,'falling_remove'),0:(1,'falling_remove'),1:(2,'falling_remove')}
    assert falling_choices(gs)[0]['card']['upgrades'] == 1
    ack=EventEffectAck();ack.arm({'game_state':gs},SimpleNamespace(semantic=('event',2)),0)
    assert ack.waiting({'game_state':gs},1)
    bad=deepcopy(gs);bad['deck']=bad['deck'][1:]
    with pytest.raises(RuntimeError,match='identity/count'):
        ack.waiting({'game_state':bad},2)
    good=deepcopy(gs);good['deck']=good['deck'][:2]
    assert not ack.waiting({'game_state':good},3)
    bad=deepcopy(gs);bad['deck'][2]['name']='铁斩波'
    with pytest.raises(ValueError,match='identity'):
        options(bad)

from bridge.real_game_env import RealGameAdapter
from env.core import game_actions as ga


@pytest.fixture
def event_adapter():
    def make(event, options, floor=19):
        adapter = RealGameAdapter()
        adapter.update({
            "in_game": True, "ready_for_command": True,
            "available_commands": ["choose", "state"],
            "game_state": {
                "seed": "200205", "floor": floor, "act": 2,
                "screen_type": "EVENT", "room_phase": "EVENT",
                "choice_list": [o[0] for o in options if not o[1]],
                "screen_state": {"event_id": event, "options": [
                    {"text": text, "label": text, "disabled": disabled,
                     "choice_index": None if disabled else sum(not x[1] for x in options[:i])}
                    for i, (text, disabled) in enumerate(options)
                ]},
                "deck": [], "relics": [], "potions": [],
                "current_hp": 60, "max_hp": 80, "gold": 14,
            },
        })
        return adapter
    return make


def test_skull_potion_semantic_targets_potion_button(event_adapter):
    a = event_adapter("Knowing Skull", [("[来点喝的？] 获得一瓶药水。", False),
        ("[财富？] 获得90金币。", False), ("[成功？] 获得一张无色牌。", False),
        ("[我要怎么离开？] 失去6生命。", False)])
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 0"


def test_vampires_without_vial_has_accept_and_refuse_native_ids(event_adapter):
    a = event_adapter("Vampires", [("[接受] 失去24最大生命。", False), ("[拒绝]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(1)) == "CHOOSE 0"


def test_red_mask_disabled_option_keeps_native_ids(event_adapter):
    a = event_adapter("Tomb of Lord Red Mask", [("[锁定] 需要红面具。", True),
        ("[献上14金币] 获得遗物。", False), ("[离开]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 1"


def test_late_mind_bloom_heal_is_not_wealth(event_adapter):
    a = event_adapter("MindBloom", [("[我必凯旋] 战斗。", False),
        ("[我已觉醒] 升级所有牌。", False), ("[我无伤痛] 回复所有生命。", False)], floor=45)
    assert a._legal_event_actions() == [ga.event(0), ga.event(1), ga.event(3)]
    assert a.command_for_semantic(ga.event(3)) == "CHOOSE 2"


def test_unknown_event_does_not_fall_back_to_number(event_adapter):
    a = event_adapter("Cursed Tome", [("[继续阅读]", False)])
    with pytest.raises(ValueError, match="unsupported event"):
        a._legal_event_actions()


def test_sensory_intro_is_transport_not_reward(event_adapter):
    a = event_adapter('SensoryStone', [('[交互]', False)])
    assert a.auto_command() == ('CHOOSE 0', 'event_dialogue_transport')


@pytest.mark.parametrize('english', [False, True])
def test_sensory_cost_and_count_map_to_native(event_adapter, english):
    labels = ([f'[Recall] Add {n} Colorless cards to your deck.' +
               (f' Lose {(n-1)*5} HP.' if n > 1 else '') for n in (1, 2, 3)]
              if english else [f'[回忆] 在你的牌组中加入 {n} 张无色牌。' +
               (f' 失去 {(n-1)*5} 点生命。' if n > 1 else '') for n in (1, 2, 3)])
    a = event_adapter('SensoryStone', [(s, False) for s in reversed(labels)])
    assert a._options('event')['event'] == 'Sensory Stone'
    assert a.command_for_semantic(ga.event(0)) == 'CHOOSE 2'
    assert a.command_for_semantic(ga.event(2)) == 'CHOOSE 0'


def test_sensory_ack_checks_offered_rewards_not_card_gain(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('SensoryStone', [(f'[回忆] 在你的牌组中加入 {n} 张无色牌。' +
        (f' 失去 {(n-1)*5} 点生命。' if n > 1 else ''), False) for n in (1, 2, 3)])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 0)
    assert ack.waiting(a.frame, 1)
    reward = deepcopy(a.frame)
    reward['game_state'].update(screen_type='COMBAT_REWARD', screen_state={
        'rewards': [{'reward_type': 'CARD'}] * 3})
    assert not ack.waiting(reward, 2)
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 3)
    reward['game_state']['screen_state']['rewards'].pop()
    with pytest.raises(RuntimeError, match='type/count mismatch'):
        ack.waiting(reward, 4)


def test_multiple_reward_skip_advances_without_reopening_same_group():
    from copy import deepcopy
    a = RealGameAdapter()
    outer = {'in_game': True, 'ready_for_command': True, 'available_commands': ['choose', 'proceed', 'state'],
             'game_state': {'screen_type': 'COMBAT_REWARD', 'choice_list': ['card', 'card', 'card'],
                            'screen_state': {'rewards': [{'reward_type': 'CARD'}] * 3}}}
    inner = deepcopy(outer)
    inner['available_commands'] = ['choose', 'skip', 'state']
    inner['game_state'].update(screen_type='CARD_REWARD', choice_list=['Strike'])
    for index in (2, 1, 0):
        a.update(outer)
        assert a.auto_command() == (f'CHOOSE {index}', 'open_card_reward')
        a.update(inner)
        assert a.command_for_semantic(ga.skip_cards()) == 'SKIP'
    a.update(outer)
    assert a.auto_command() is None
    assert ga.reward_card(0) not in a._legal_reward_actions()


def test_singing_bowl_skip_uses_bowl_and_requires_hp_ack():
    from bridge.real_game_env import RealDecision
    from bridge.client.agent_orchestrator import RewardCommandAck
    from copy import deepcopy
    a = RealGameAdapter()
    frame = {'game_state': {'screen_type': 'CARD_REWARD', 'choice_list': ['Strike', 'bowl'], 'max_hp': 80}}
    a.update(frame)
    a._combat_reward_card_open = True
    command = a.command_for_semantic(ga.skip_cards())
    assert command == 'CHOOSE 1'
    assert a._skipped_reward_card_count == 0  # Bowl really consumes the reward.
    ack = RewardCommandAck()
    ack.arm(frame, RealDecision(command, 'policy', semantic=ga.skip_cards()), 0)
    after = deepcopy(frame)
    after['game_state']['screen_type'] = 'COMBAT_REWARD'
    assert ack.waiting(after, 1)
    after['game_state']['max_hp'] = 82
    assert not ack.waiting(after, 2)


def test_deferred_reward_preserves_claim_for_ack():
    from bridge.client.agent_orchestrator import RealPolicyAgent
    a = RealGameAdapter()
    a._deferred_combat_reward_action = ga.claim('gold', 0)
    agent = RealPolicyAgent(None, a, stochastic=False, topk=0, max_inner_decisions=1,
        auto_enter_shop=False, auto_start=False, start_seed=None, ascension=0, character='IRONCLAD')
    d = agent.decide({'in_game': True, 'ready_for_command': True,
        'available_commands': ['choose', 'state'], 'game_state': {
            'screen_type': 'COMBAT_REWARD', 'choice_list': ['gold'],
            'screen_state': {'rewards': [{'reward_type': 'GOLD', 'gold': 30}]}}})
    assert d.command == 'CHOOSE 0'
    assert d.semantic == ga.claim('gold', 0)


def test_settings_pause_excludes_only_explicit_menu_duration():
    from bridge.client.agent_orchestrator import UiPauseClock
    clock = UiPauseClock()
    active = {'game_state': {'screen_name': 'NONE', 'is_screen_up': False}}
    menu = {'game_state': {'screen_name': 'SETTINGS', 'is_screen_up': True}}
    assert clock.update(active, 100) == (100, False)
    assert clock.update(menu, 104) == (104, True)
    assert clock.update(menu, 125) == (104, True)
    assert clock.update(active, 130) == (104, False)
    assert clock.update(active, 137) == (111, False)  # real timeout still works


def test_settings_pause_keeps_reward_ack_and_does_not_acknowledge_it():
    from bridge.client.agent_orchestrator import UiPauseClock, RewardCommandAck
    from bridge.real_game_env import RealDecision
    clock, ack = UiPauseClock(), RewardCommandAck()
    f = {'game_state': {'screen_type': 'COMBAT_REWARD', 'screen_name': 'NONE',
         'screen_state': {}, 'is_screen_up': False}}
    now, _ = clock.update(f, 0)
    ack.arm(f, RealDecision('PROCEED', 'policy'), now)
    clock.update({'game_state': {'screen_name': 'SETTINGS', 'is_screen_up': True}}, 3)
    now, paused = clock.update({'game_state': {'screen_name': 'SETTINGS', 'is_screen_up': True}}, 63)
    assert paused and now == 3 and ack.pending is not None
    now, paused = clock.update(f, 65)
    assert not paused and ack.waiting(f, now)
    now, _ = clock.update(f, 73)
    with pytest.raises(RuntimeError, match='not acknowledged'):
        ack.waiting(f, now)


def test_manual_campaign_never_starts_after_returning_to_menu():
    from bridge.client.agent_orchestrator import RealPolicyAgent
    from bridge.campaign import SeedCampaign
    a = RealGameAdapter()
    campaign = SeedCampaign(['4NF5'])
    agent = RealPolicyAgent(None, a, stochastic=False, topk=0, max_inner_decisions=1,
        auto_enter_shop=False, auto_start=False, start_seed=None, ascension=0,
        character='IRONCLAD', campaign=campaign)
    menu = {'in_game': False, 'ready_for_command': True, 'available_commands': ['start', 'state']}
    for _ in range(3):
        campaign.observe({'in_game': True, 'game_state': {'seed': 200205, 'floor': 23}})
        assert agent.decide(menu).command == 'STATE'
        assert not campaign.start_sent


def test_saved_search_index_restore_is_scoped():
    from types import SimpleNamespace
    from bridge.client.agent_orchestrator import restore_attached_search_index
    args = SimpleNamespace(attach_search_index=204, attach_existing_seed=200205, attach_existing_floor=23)
    search = SimpleNamespace(decision_index=0, run_decisions=0)
    restore_attached_search_index(args, search, {'seed': 200205, 'floor': 23})
    assert search.decision_index == 204
    assert search.run_decisions == 0  # identity restoration is not actual search work
    for gs in [{'seed': 200206, 'floor': 23}, {'seed': 200205, 'floor': 24}]:
        with pytest.raises(RuntimeError, match='matching save'):
            restore_attached_search_index(args, search, gs)
    args.attach_search_index = -1
    with pytest.raises(RuntimeError, match='matching save'):
        restore_attached_search_index(args, search, {'seed': 200205, 'floor': 23})


def test_repeated_sl_rebuilds_search_and_adapter_without_start(monkeypatch):
    import io, json
    from types import SimpleNamespace
    from bridge.client import agent_orchestrator as o
    from bridge import search_bridge
    searches, adapters, observed, sent = [], [], [], []
    class Search:
        def __init__(self, *a, **kw):
            self.decision_index = 0
            self.closed = False
            searches.append(self)
        def close(self): self.closed = True
        def note_error(self): pytest.fail('unexpected error in save reload')
    class Agent:
        def __init__(self, model, adapter, **kw):
            self.adapter, self.combat_search = adapter, kw['combat_search']
        def decide(self, frame):
            observed.append((self.adapter, self.combat_search, self.combat_search.decision_index))
            self.combat_search.decision_index += 20
            return o.RealDecision('STATE', 'test')
    def adapter_factory(*a):
        adapter = SimpleNamespace()
        adapters.append(adapter)
        return adapter
    monkeypatch.setattr(o, '_load_model', lambda *a, **kw: SimpleNamespace(set_random_seed=lambda s: None))
    monkeypatch.setattr(o, '_adapter_for_model', adapter_factory)
    monkeypatch.setattr(search_bridge, 'RealCombatSearch', Search)
    monkeypatch.setattr(o, 'RealPolicyAgent', Agent)
    monkeypatch.setattr(o, '_log_decision', lambda *a: None)
    monkeypatch.setattr(o, 'send', sent.append)
    monkeypatch.setattr(o, 'send_decision', lambda f, d: sent.append(d.command))
    monkeypatch.setattr(o.time, 'sleep', lambda s: None)
    monkeypatch.setattr(o.sys, 'argv', ['agent', 'model.zip', '--combat-search',
        '--attach-existing-seed', '200205', '--attach-existing-floor', '29', '--attach-search-index', '255'])
    menu = {'in_game': False}
    live = {'in_game': True, 'game_state': {'seed': 200205, 'floor': 29, 'screen_type': 'NONE'}}
    monkeypatch.setattr(o.sys, 'stdin', io.StringIO('\n'.join(json.dumps(f) for f in [menu,live,menu,menu,live,menu,live])))
    o.main()
    assert [r[2] for r in observed] == [255,255,255]
    assert len(searches) == len(adapters) == 3
    assert all(s.closed for s in searches)
    assert all(cmd in ['ready','STATE'] for cmd in sent)


@pytest.mark.parametrize('label', ['[伸手进去] 失去3生命。25%:找到一件遗物。',
                                  '[Reach Inside] #rLose #r3 #rHP. #g25%: #gFind #ga #gRelic.'])
def test_scrap_failure_waits_for_updated_offer(event_adapter, label):
    from bridge.event_contract import EventEffectAck, options
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('Scrap Ooze', [(label, False), ('[离开]', False)])
    assert options(a.gs) == {0: (0, 'scrap_search'), 1: (1, 'leave')}
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    after = deepcopy(a.frame)
    after['game_state']['current_hp'] -= 3
    assert ack.waiting(after, 1)
    after['game_state']['screen_state']['options'][0]['text'] = '[接着往里伸] 失去4生命。35%:找到一件遗物。'
    assert not ack.waiting(after, 2)


def test_scrap_rejects_wrong_failure_increment(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('Scrap Ooze', [('[Reach Inside] Lose 3 HP. 25%: Find a Relic.', False), ('[Leave]', False)])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    after = deepcopy(a.frame)
    after['game_state']['current_hp'] -= 3
    after['game_state']['screen_state']['options'][0]['text'] = '[Deeper] Lose 3 HP. 35%: Find a Relic.'
    with pytest.raises(RuntimeError, match='cost/chance mismatch'):
        ack.waiting(after, 1)


def test_scrap_success_requires_relic_and_dialogue(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('Scrap Ooze', [('[Deeper] Lose 4 HP. 35%: Find a Relic.', False), ('[Leave]', False)])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    after = deepcopy(a.frame)
    after['game_state']['screen_state']['options'] = [{'text': '[Leave]', 'choice_index': 0}]
    assert ack.waiting(after, 1)
    after['game_state'].setdefault('relics', []).append({'id': 'Blood Vial'})
    assert not ack.waiting(after, 2)


@pytest.mark.parametrize('labels', [('[触碰] 失去8生命，获得75金币。','[交易] 50%：获得好脸。50%：获得坏脸。','[离开]'),
                                   ('[Touch] Lose 8 HP, gain 75 Gold.','[Trade] 50%: Good Face. 50%: Bad Face.','[Leave]')])
def test_face_trader_real_identity_and_choices(event_adapter, labels):
    a = event_adapter('FaceTrader', [(s, False) for s in labels])
    assert a._options('event')['event'] == 'Face Trader'
    assert a._options('event')['event_options'] == [0, 1, 2]


@pytest.mark.parametrize('trade', [False, True])
def test_face_trader_ack_waits_for_completed_dialogue(event_adapter, trade):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('FaceTrader', [('[Touch]', False), ('[Trade]', False), ('[Leave]', False)])
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', int(trade))), 0)
    after = deepcopy(a.frame)
    if trade:
        after['game_state'].setdefault('relics', []).append({'id': 'CultistMask'})
    else:
        after['game_state']['gold'] += 75
        after['game_state']['current_hp'] -= 8
    assert ack.waiting(after, 1)
    after['game_state']['screen_state']['options'] = [{'text':'[Leave]', 'choice_index':0}]
    assert not ack.waiting(after, 2)


@pytest.mark.parametrize('real,canonical', [('FaceTrader','Face Trader'), ('SecretPortal','Secret Portal'),
    ('Match and Keep!','Match and Keep'), ("N'loth",'Nloth'), ('NoteForYourself','Note For Yourself')])
def test_game_jar_event_ids_match_frozen_vocabulary(real, canonical):
    from bridge.real_game_env import _normalize_event
    assert _normalize_event(real) == canonical


def test_nloth_encodes_exact_displayed_relics_and_ack(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter("N'loth", [('[交出： 冰淇淋] 失去这件遗物。', False),
                                  ('[Offer: Anchor] Lose this relic.', False), ('[Leave]', False)])
    a.gs['relics'] = [{'id':'Ice Cream','name':'冰淇淋'}, {'id':'Anchor','name':'Anchor'}]
    assert a._options('event')['event_choices'] == [
        {'index':0,'kind':'lose_relic','relic':'Ice Cream'},
        {'index':1,'kind':'lose_relic','relic':'Anchor'}, {'index':2,'kind':'leave'}]
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    bad = deepcopy(a.frame)
    bad['game_state']['relics'] = [a.gs['relics'][0], {'id':"Nloth's Gift"}]
    with pytest.raises(RuntimeError, match='identity/count'):
        ack.waiting(bad, 1)
    good = deepcopy(a.frame)
    good['game_state']['relics'] = [a.gs['relics'][1], {'id':"Nloth's Gift"}]
    assert ack.waiting(good, 2)
    good['game_state']['screen_state']['options'] = [{'text':'[Leave]', 'choice_index':0}]
    assert not ack.waiting(good, 3)


@pytest.mark.parametrize('real_id', '''Falling
MindBloom
The Moai Head
Mysterious Sphere
SecretPortal
SensoryStone
Tomb of Lord Red Mask
Winding Halls
Addict
Back to Basics
Beggar
Colosseum
Cursed Tome
Drug Dealer
Forgotten Altar
Ghosts
Knowing Skull
Masked Bandits
Nest
The Joust
The Library
The Mausoleum
Vampires
Big Fish
The Cleric
Dead Adventurer
Golden Idol
Golden Wing
World of Goop
Living Wall
Mushrooms
Scrap Ooze
Shining Light
Liars Game
Accursed Blacksmith
Bonfire Elementals
Designer
Duplicator
FaceTrader
Fountain of Cleansing
Golden Shrine
Match and Keep!
Wheel of Change
Lab
N'loth
NoteForYourself
Purifier
Transmorgrifier
Upgrade Shrine
WeMeetAgain
The Woman in Blue'''.splitlines())
def test_all_game_jar_policy_event_ids_are_known(real_id):
    # Public ID constants extracted from local desktop-1.0.jar, not native enum
    # names. Spire Heart is intentionally covered by ending transport tests.
    from bridge.real_game_env import _normalize_event
    assert _normalize_event(real_id) != 'PAD'


def test_continuous_testing_requires_explicit_flag():
    from bridge.client.agent_orchestrator import build_parser
    args = build_parser().parse_args(['model.zip', '--auto-start'])
    assert args.auto_start and not args.continuous_test
    args = build_parser().parse_args(['model.zip', '--auto-start', '--continuous-test'])
    assert args.auto_start and args.continuous_test


def test_dialogue_bypasses_policy_without_claiming_resource_effect(event_adapter):
    a = event_adapter("Knowing Skull", [("[继续]", False)])
    assert a.auto_command() == ("CHOOSE 0", "event_dialogue_transport")


def test_unknown_label_is_not_guessed(event_adapter):
    a = event_adapter("Knowing Skull", [("[未知选项]", False)])
    with pytest.raises(ValueError, match="unrecognized"):
        a._legal_event_actions()


def test_event_potion_effect_rejects_card_gain(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter("Knowing Skull", [("[来点喝的？] 获得一瓶药水。", False)])
    a.gs['potions'] = [{'id': 'Potion Slot'}]
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 0)
    assert ack.waiting(a.frame, 1)
    bad = deepcopy(a.frame)
    bad['game_state']['deck'] = [{'id': 'Mind Blast', 'upgrades': 0}]
    bad['game_state']['current_hp'] -= 6
    with pytest.raises(RuntimeError, match="effect type mismatch"):
        ack.waiting(bad, 2)
    good = deepcopy(a.frame)
    good['game_state']['potions'] = [{'id': 'Strength Potion'}]
    good['game_state']['current_hp'] -= 6
    assert not ack.waiting(good, 2)
    assert ack.pending is None


def test_event_effect_timeout_and_grid_wait(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter("Purifier", [("[祈祷]", False)])
    a.gs['deck'] = [{'id': 'Strike_R', 'upgrades': 0}]
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    grid = deepcopy(a.frame)
    grid['game_state']['screen_type'] = 'GRID'
    assert not ack.waiting(grid, 1)
    assert ack.pending is not None
    with pytest.raises(RuntimeError, match="timeout"):
        ack.waiting(a.frame, 31)


@pytest.mark.parametrize('sozu', [False, True])
def test_skull_blocked_potion_requires_no_gain_and_completed_effect(event_adapter, sozu):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('Knowing Skull', [('[来点喝的？] 获得一瓶药水。失去 9 生命。', False)])
    a.gs['potions'] = [{'id': 'Potion Slot' if sozu else 'Strength Potion'}]
    a.gs['relics'] = [{'id': 'Sozu'}] if sozu else []
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 0)
    assert ack.waiting(a.frame, 1)
    after = deepcopy(a.frame)
    after['game_state']['current_hp'] -= 9
    assert ack.waiting(after, 2)  # HP-only intermediate frame is not completion.
    after['game_state']['screen_state']['options'][0]['text'] = '[来点喝的？] 失去 10 生命。'
    assert not ack.waiting(after, 3)
