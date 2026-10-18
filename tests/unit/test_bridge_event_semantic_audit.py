"""BRIDGE06: native event semantics must not be UI button ordinals.

Fixtures describe public option contracts observed in BRIDGE05 real frames.
No live game or checkpoint is needed for this static adapter audit.
"""
import pytest


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
