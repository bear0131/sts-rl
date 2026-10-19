"""Fail-closed event semantic/transport contracts (BRIDGE06).

Only audited public labels are accepted. Unlisted events/variants require an
explicit contract, never positional fallback. Native IDs remain checkpoint ABI.
"""
import re
from collections import Counter
from copy import deepcopy

from bridge.schemas.entity_ids import (
    CARD_ALIASES,
    DYNAMIC_EVENT_OPTION_KINDS,
    EVENT_ALIASES,
)


# native semantic ID, public label aliases, expected effect type
def first_two_match_slots(available_slots):
    """User-authorized fixed helper; takes public slot availability, never faces.

    Slot numbers retain board order even when matched cards disappear. The
    caller must wait for the previous pair's ACK before requesting another pair.
    This is not a model action and does not enable the unsupported Mod transport.
    """
    slots = list(available_slots)
    if any(type(slot) is not int or not 0 <= slot < 12 for slot in slots):
        raise ValueError('Match slots must be integer board positions 0..11')
    if len(set(slots)) != len(slots):
        raise ValueError('Duplicate Match slot')
    if len(slots) < 2:
        raise ValueError('Match pair requires two available slots')
    return tuple(sorted(slots)[:2])


CONTRACTS = {
    # Each semantic ID below is the native GameAction event index, not the
    # current CommunicationMod ordinal.  Multi-page events deliberately keep
    # their sparse native IDs (Cursed Tome 0..6; Golden Idol trap 2..4).
    "Accursed Blacksmith": [(0, '锻造|Forge', 'pending_blacksmith_upgrade'),
                            (1, '翻查|Rummage', 'pending_blacksmith_relic_curse'),
                            (2, '离开|Leave', 'leave')],
    "The Woman in Blue": [(0, '买1瓶药水|Buy 1 Potion', 'woman_potions_1'),
                          (1, '买2瓶药水|Buy 2 Potions', 'woman_potions_2'),
                          (2, '买3瓶药水|Buy 3 Potions', 'woman_potions_3'),
                          (3, '离开|Leave', 'woman_leave')],
    "Back to Basics": [(0, '简约|Elegance', 'basics_remove'),
                       (1, '质朴|Simplicity', 'basics_upgrade')],
    "Bonfire Elementals": [(0, '献上|Offer', 'pending_bonfire_offer')],
    "Colosseum": [(0, '战斗|Fight', 'pending_colosseum_first_combat'),
                   (0, '怂了|Cowardice', 'leave'),
                   (1, '战斗到底|Victory', 'pending_colosseum_second_combat')],
    "Cursed Tome": [(0, '阅读|Read', 'tome_read'),
                     (1, '离开|Leave', 'leave'),
                     (2, '继续|Continue', 'tome_continue_1'),
                     (3, '继续|Continue', 'tome_continue_2'),
                     (4, '继续|Continue', 'tome_continue_3'),
                     (5, '拿走|Take', 'tome_take'),
                     (6, '停止|Stop', 'tome_stop')],
    "Dead Adventurer": [(0, '搜索|继续|Search|Continue', 'pending_dead_search'),
                        (1, '离开|Leave', 'leave')],
    "Drug Dealer": [(0, '试一下J.A.X.|Test J.A.X.', 'augmenter_jax'),
                     (1, '当一下实验对象|Become Test Subject', 'augmenter_transform'),
                     (2, '喝一下突变剂|Ingest Mutagens', 'augmenter_relic')],
    "SecretPortal": [(0, '进入传送门|Enter the Portal', 'portal_boss'),
                     (1, '离开|Leave', 'leave')],
    "Fountain of Cleansing": [(0, '喝水|Drink', 'remove_curses'),
                             (1, '离开|Leave', 'leave')],
    "Mysterious Sphere": [(0, '打开圆球|Open Sphere', 'sphere_combat'),
                         (1, '离开|Leave', 'leave')],
    "World of Goop": [(0, '收集金币|Gather Gold', 'goop_gold'),
                      (1, '放手吧|Leave It', 'goop_loss')],
    "FaceTrader": [(0, '触碰|Touch', 'face_touch'), (1, '交易|Trade', 'face_trade'),
                   (2, '离开|Leave', 'leave')],
    "Scrap Ooze": [(0, '伸手进去|接着往里伸|Reach Inside|Deeper', 'scrap_search'),
                   (1, '离开|Leave', 'leave')],
    "Forgotten Altar": [(0, '献上|献上：金神像|Offer|Offer: Golden Idol', 'altar_idol'), (1, '献祭|Sacrifice', 'altar_sacrifice'),
                        (2, '亵渎|Desecrate', 'altar_decay')],
    "Beggar": [(0, '给金币|Offer Gold', 'beggar_remove'), (1, '离开|Leave', 'leave')],
    "The Cleric": [(0, '治疗|Heal', 'cleric_heal'), (1, '净化|Purify', 'cleric_remove'),
                   (2, '离开|Leave', 'leave')],
    "Mushrooms": [(0, '踩扁|Stomp', 'mushroom_combat'), (1, '吃下|Eat', 'mushroom_eat')],
    "Knowing Skull": [(0, '财富？|Riches?', 'gold'), (1, '成功？|Success?', 'card'),
                      (2, '来点喝的？|A Pick Me Up?', 'potion'), (3, '我要怎么离开？|How do I leave?', 'hp_loss')],
    "Vampires": [(0, '接受|Accept|失去鲜血药瓶|Lose Blood Vial', 'vial_bites'),
                 (1, '接受|Accept', 'maxhp_loss'), (2, '拒绝|Refuse', 'leave')],
    "Tomb of Lord Red Mask": [(0, '戴上|戴上红面具|Don|Don the Red Mask', 'gold'),
                              (1, '献上|Offer', 'relic'), (2, '离开|Leave', 'leave')],
    "MindBloom": [(0, '我必凯旋|I am War', 'combat'), (1, '我已觉醒|I am Awake', 'relic'),
                  (2, '我将富有|我很富有|I am Rich', 'gold'), (3, '我无伤痛|I am Healthy', 'card')],
    "Big Fish": [(0, '香蕉|Banana', 'heal'), (1, '甜甜圈|Donut', 'maxhp_gain'), (2, '盒子|Box', 'relic')],
    "Living Wall": [(0, '遗忘|Forget', 'remove'), (1, '改变|Change', 'transform'), (2, '成长|Grow', 'upgrade')],
    "Transmorgrifier": [(0, '祈祷|Pray', 'transform'), (1, '离开|Leave', 'leave')],
    "Purifier": [(0, '祈祷|Pray', 'remove'), (1, '离开|Leave', 'leave')],
    "Upgrade Shrine": [(0, '祈祷|Pray', 'upgrade'), (1, '离开|Leave', 'leave')],
    "Duplicator": [(0, '祈祷|Pray', 'card'), (1, '离开|Leave', 'leave')],
    "Ghosts": [(0, '接受|Accept', 'maxhp_loss'), (1, '拒绝|Refuse', 'leave')],
    "Masked Bandits": [(0, '付钱|Pay', 'gold_loss'), (1, '战斗|战斗！|Fight|Fight!', 'combat')],
    "Golden Shrine": [(0, '祈祷|Pray', 'gold'), (1, '亵渎|Desecrate', 'gold'), (2, '离开|Leave', 'leave')],
    "Golden Idol": [(0, '拿走|Take', 'pending_idol_take'),
                    (1, '离开|Leave', 'leave'),
                    (2, '逃跑|Outrun', 'pending_idol_curse'),
                    (3, '砸烂|Smash', 'pending_idol_damage'),
                    (4, '躲藏|Hide', 'pending_idol_maxhp')],
    "Golden Wing": [(0, '祈祷|Pray', 'pending_wing_remove'),
                    (1, '摧毁|Destroy', 'pending_wing_gold'),
                    (2, '离开|Leave', 'leave')],
    "Lab": [(0, '搜索|Search', 'lab_rewards')],
    "Liars Game": [(0, '同意|Agree', 'pending_serpent_gold_curse'),
                   (1, '反对|Disagree', 'leave')],
    "Nest": [(0, '抢了钱就跑|Smash and Grab', 'gold'), (1, '留在队伍中|Stay in Line', 'card')],
    "The Library": [(0, '阅读|Read', 'card'), (1, '睡觉|Sleep', 'heal')],
    "Shining Light": [(0, '走进|Enter', 'pending_shining_upgrade_damage'),
                      (1, '离开|Leave', 'leave')],
    "The Joust": [(0, '凶手|Murderer', 'pending_joust_murderer'),
                  (1, '主人|Owner', 'pending_joust_owner')],
    "The Mausoleum": [(0, '打开棺材|Open Coffin', 'pending_mausoleum_relic_curse'),
                      (1, '离开|Leave', 'leave')],
    "The Moai Head": [(0, '跳进去|Jump Inside', 'pending_moai_heal_maxhp'),
                      (1, '献上：金神像|Offer: Golden Idol', 'pending_moai_idol_gold'),
                      (2, '离开|Leave', 'leave')],
    "Addict": [(0, '给他金币|Offer Gold', 'relic'), (1, '抢夺|Rob', 'relic'), (2, '离开|Leave', 'leave')],
    "Winding Halls": [(0, '拥抱疯狂|Embrace Madness', 'card'), (1, '集中精神|Focus', 'card'), (2, '寻找来路|Retrace|Retrace Your Steps', 'maxhp_loss')],
    "Wheel of Change": [(0, '玩小游戏|Play', 'pending_wheel_random')],
}

# Exact Java execution phases, not inferred from HP/gold deltas or translated
# button ordinals. These fields are transport-only and never policy features.
EVENT_PROGRESS_CONTRACTS = {
    'Accursed Blacksmith': ('shrines.AccursedBlacksmith', {'0', '2'}),
    'Bonfire Elementals': ('shrines.Bonfire', {'INTRO', 'CHOOSE', 'COMPLETE'}),
    'Colosseum': ('city.Colosseum', {'INTRO', 'FIGHT', 'POST_COMBAT', 'LEAVE'}),
    'Dead Adventurer': ('exordium.DeadAdventurer', {'INTRO', 'FAIL', 'SUCCESS', 'ESCAPE'}),
    'Golden Idol': ('exordium.GoldenIdolEvent', {'0', '1', '2'}),
    'Golden Wing': ('exordium.GoldenWing', {'INTRO', 'PURGE', 'MAP'}),
    'Liars Game': ('exordium.Sssserpent', {'INTRO', 'AGREE', 'DISAGREE', 'COMPLETE'}),
    'Shining Light': ('exordium.ShiningLight', {'INTRO', 'COMPLETE'}),
    'The Joust': ('city.TheJoust', {'HALT', 'EXPLANATION', 'PRE_JOUST', 'JOUST', 'COMPLETE'}),
    'The Mausoleum': ('city.TheMausoleum', {'INTRO', 'RESULT'}),
    'The Moai Head': ('beyond.MoaiHead', {'0', '1'}),
    'Wheel of Change': ('shrines.GremlinWheelGame', {'INTRO', 'COMPLETE', 'LEAVE'}),
}

# (event ID, valid source phase, completion phase); pending_* is the existing
# stable effect keyword. Only the explicitly audited entries below are enabled.
STAGED_EFFECTS = {
    'pending_blacksmith_upgrade': ('Accursed Blacksmith', '0', '2'),
    'pending_blacksmith_relic_curse': ('Accursed Blacksmith', '0', '2'),
    'pending_bonfire_offer': ('Bonfire Elementals', 'CHOOSE', 'COMPLETE'),
    'pending_colosseum_first_combat': ('Colosseum', 'FIGHT', None),
    'pending_colosseum_second_combat': ('Colosseum', 'POST_COMBAT', None),
    'pending_dead_search': ('Dead Adventurer', 'INTRO', None),
    'pending_idol_take': ('Golden Idol', '0', '1'),
    'pending_idol_curse': ('Golden Idol', '1', '2'),
    'pending_idol_damage': ('Golden Idol', '1', '2'),
    'pending_idol_maxhp': ('Golden Idol', '1', '2'),
    'pending_wing_remove': ('Golden Wing', 'INTRO', 'MAP'),
    'pending_wing_gold': ('Golden Wing', 'INTRO', 'MAP'),
    'pending_serpent_gold_curse': ('Liars Game', 'INTRO', 'COMPLETE'),
    'pending_shining_upgrade_damage': ('Shining Light', 'INTRO', 'COMPLETE'),
    'pending_joust_murderer': ('The Joust', 'EXPLANATION', 'COMPLETE'),
    'pending_joust_owner': ('The Joust', 'EXPLANATION', 'COMPLETE'),
    'pending_mausoleum_relic_curse': ('The Mausoleum', 'INTRO', 'RESULT'),
    'pending_moai_heal_maxhp': ('The Moai Head', '0', '1'),
    'pending_moai_idol_gold': ('The Moai Head', '0', '1'),
    'pending_wheel_random': ('Wheel of Change', 'INTRO', 'LEAVE'),
}


def event_progress(gs, event=None):
    event = event or gs.get('screen_state', {}).get('event_id')
    if event not in EVENT_PROGRESS_CONTRACTS:
        raise ValueError(f'no audited event progress source: {event}')
    source, phases = EVENT_PROGRESS_CONTRACTS[event]
    progress = gs.get('simulator_event_progress')
    if (not isinstance(progress, dict)
            or progress.get('source') != 'com.megacrit.cardcrawl.events.' + source
            or progress.get('phase') not in phases):
        raise ValueError(f'missing/invalid actual event progress: {event}')
    if type(gs.get('simulator_obtain_pending')) is not bool:
        raise ValueError('missing actual delayed card-obtain state')
    if event == 'Dead Adventurer' and (type(progress.get('rewards_taken')) is not int
                                     or not 0 <= progress['rewards_taken'] <= 3):
        raise ValueError('invalid Dead Adventurer observed reward count')
    if event == 'Wheel of Change' and any(type(progress.get(key)) is not bool for key in
                                        ('startSpin', 'buttonPressed', 'purgeResult')):
        raise ValueError('missing actual Wheel execution flags')
    return progress

# Contracts whose semantic identity comes from public per-instance records
# rather than a static label: exact card, relic, service variant, or board slot.
# They still belong to the complete event word table and are validated beside
# CONTRACTS; none may fall through to a UI ordinal.
STRUCTURED_CONTRACTS = frozenset({
    "Falling", "N'loth", "SensoryStone", "Match and Keep!",
    *DYNAMIC_EVENT_OPTION_KINDS,
})


def canonical_contract_events():
    """Canonical policy-event identities with an explicit option disposition."""
    return {
        EVENT_ALIASES.get(event, event)
        for event in {*CONTRACTS, *STRUCTURED_CONTRACTS}
    } | {"NEOW"}


def _text(text):
    return re.sub(r'\s+|#[a-zA-Z]', '', str(text)).casefold()


def _rule_matches(event, text, labels, effect):
    """Match a whole button label, then explicitly disambiguate reused labels."""
    match = re.match(r'^\[([^\]]+)\]', text)
    if match is None:
        return False
    if event == 'Tomb of Lord Red Mask' and effect == 'relic':
        return re.fullmatch(r'(?:献上[:：]?\d+金币|offer:\d+gold)', match[1]) is not None
    if match[1] not in {_text(label) for label in labels.split('|')}:
        return False
    if event == 'Cursed Tome' and effect.startswith('tome_continue_'):
        cost = int(effect.rsplit('_', 1)[1])
        return re.fullmatch(
            r'\[(?:继续|continue)\](?:失去|lose)' + str(cost) + r'(?:生命。|hp\.)',
            text,
        ) is not None
    if event == 'Vampires' and effect in {'vial_bites', 'maxhp_loss'}:
        vial = any(word in text for word in ('失去鲜血药瓶', 'losebloodvial'))
        maxhp = re.search(r'(?:失去\d+最大生命|lose\d+maxhp)', text) is not None
        return (vial and not maxhp) if effect == 'vial_bites' else (maxhp and not vial)
    return True


def scrap_offer(gs):
    """Public damage/chance, not a prediction of the hidden random roll."""
    for o in (gs.get('screen_state') or {}).get('options', []):
        text = _text(o.get('text', ''))
        if o.get('disabled') or not re.match(r'\[(伸手进去|接着往里伸|reachinside|deeper)\]', text):
            continue
        m = re.search(r'(?:失去|lose)(\d+)(?:生命。|hp\.)(\d+)%', text)
        if not m:
            raise ValueError('Scrap Ooze public damage/chance missing')
        return int(m[1]), int(m[2])
    raise ValueError('Scrap Ooze search option missing')


def _validate_choice_indices(gs):
    """Reject malformed wire indices before policy or transport can send CHOOSE."""
    seen = set()
    for option in (gs.get('screen_state') or {}).get('options', []):
        if option.get('disabled'):
            continue
        index = option.get('choice_index')
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('invalid/duplicate event choice_index')
        seen.add(index)


def transport_only(gs):
    """An exact one-button dialogue, not a resource-bearing policy option."""
    _validate_choice_indices(gs)
    event = gs.get('screen_state', {}).get('event_id')
    if event in EVENT_PROGRESS_CONTRACTS:
        progress = event_progress(gs)
        phase = progress['phase']
        # Stock ChoiceScreenUtils exposes a special spin handle while the
        # dialogue is hidden. It is not an option in screen_state.options.
        if (event == 'Wheel of Change' and phase == 'INTRO'
                and progress['startSpin'] and not progress['buttonPressed']
                and gs.get('choice_list') == ['spin']):
            return 0
    opts = [o for o in (gs.get('screen_state') or {}).get('options', []) if not o.get('disabled')]
    if len(opts) != 1:
        return None
    o = opts[0]
    stages = {
        'Bonfire Elementals': {'INTRO', 'CHOOSE'}, 'Colosseum': {'INTRO'},
        'The Joust': {'HALT', 'PRE_JOUST', 'JOUST'},
        'Golden Wing': {'PURGE'}, 'Liars Game': {'AGREE'},
        'Dead Adventurer': {'FAIL'}, 'Wheel of Change': {'COMPLETE'},
    }
    if event in stages and phase in stages[event]:
        return o['choice_index']
    text = _text(o.get('text', '')).strip('[]')
    if ((gs.get('screen_state') or {}).get('event_id') == 'Lab'
            and not gs.get('combat_state')
            and text in {'搜索]找到一些药水！', 'search]findsomepotions!'}):
        # Native enters rewards directly; this is not a policy event action.
        return o.get('choice_index')
    if ((gs.get('screen_state') or {}).get('event_id') == 'Spire Heart'
            and not gs.get('combat_state')
            and text in {'攻击]???', 'attack]???', '睡眠', 'sleep', '走向门前', 'approachdoor'}):
        # Act-three ending dialogue, not a combat action or victory receipt.
        return o.get('choice_index')
    if ((gs.get('screen_state') or {}).get('event_id') == 'Winding Halls'
            and text in {'……', '...'}):
        return o.get('choice_index')
    if ((gs.get('screen_state') or {}).get('event_id') == 'SensoryStone'
            and text in {'交互', 'interact'}):
        return o.get('choice_index')
    if ((gs.get('screen_state') or {}).get('event_id') in {'Mushrooms', 'Mysterious Sphere'}
            and text in {'战斗', 'fight'}):
        return o.get('choice_index')
    if text in {'继续', '离开', 'continue', 'leave', 'proceed'}:
        return o.get('choice_index')
    return None


def falling_choices(gs):
    result = []
    rules = [('落地|land', 0, 'SKILL'), ('释放|channel', 1, 'POWER'), ('猛击|strike', 2, 'ATTACK')]
    for o in gs.get('screen_state', {}).get('options', []):
        if o.get('disabled'):
            continue
        text = _text(o.get('text', ''))
        matches = [(i, typ, re.fullmatch(r'\[(?:'+labels+r')\](?:失去|lose)(.+)', text)) for labels,i,typ in rules]
        matches = [(i,typ,m[1]) for i,typ,m in matches if m]
        if len(matches) != 1 or o.get('choice_index') is None:
            raise ValueError(f'unrecognized Falling option: {text}')
        i,typ,name = matches[0]
        cards = {(c['id'], c.get('upgrades', 0)): c for c in gs.get('deck', [])
                 if c.get('type') == typ and _text(c.get('name', '')) == name}
        if len(cards) != 1:
            raise ValueError(f'Falling displayed card identity ambiguous/missing: {name}')
        result.append(dict(index=i, ui=int(o['choice_index']), kind='lose_card', card=next(iter(cards.values()))))
    if not result or len({d['index'] for d in result}) != len(result):
        raise ValueError('invalid Falling option set')
    return result


def nloth_choices(gs):
    result = []
    for o in (gs.get('screen_state') or {}).get('options', []):
        if o.get('disabled'):
            continue
        text = _text(o.get('text', ''))
        index = o.get('choice_index')
        if text in {'[离开]', '[leave]'} and index == 2:
            result.append(dict(index=2, ui=2, kind='leave'))
            continue
        match = re.match(r'\[(?:交出：|offer:)([^\]]+)\]', text)
        if not match or index not in (0, 1):
            raise ValueError('Nloth option identity/position mismatch')
        relics = [r for r in gs.get('relics', []) if _text(r.get('name', '')) == match[1]]
        if len(relics) != 1:
            raise ValueError('Nloth displayed relic ambiguous/missing')
        result.append(dict(index=index, ui=index, kind='lose_relic', relic=relics[0]))
    if sorted(d['index'] for d in result) != [0, 1, 2]:
        raise ValueError('Nloth incomplete choices')
    return result


def options(gs):
    _validate_choice_indices(gs)
    ss = gs.get('screen_state') or {}
    event = ss.get('event_id')
    if event == 'Match and Keep!':
        raise ValueError(
            'Match and Keep requires the fixed board-slot transport contract'
        )
    if event == "N'loth":
        return {d['index']: (d['ui'], 'nloth_trade' if d['kind'] == 'lose_relic' else 'leave')
                for d in nloth_choices(gs)}
    if event == 'Falling':
        return {d['index']: (d['ui'], 'falling_remove') for d in falling_choices(gs)}
    if event == 'SensoryStone':
        result = {}
        for o in ss.get('options', []):
            if o.get('disabled'):
                continue
            text = _text(o.get('text', ''))
            match = re.fullmatch(r'\[回忆\]在你的牌组中加入([123])张无色牌。(?:失去(5|10)点生命。)?', text)
            if match is None:
                match = re.fullmatch(r'\[recall\]add([123])colorlesscards?toyourdeck\.(?:lose(5|10)hp\.)?', text)
            if not match or o.get('choice_index') is None:
                raise ValueError(f'unrecognized SensoryStone option: {text}')
            count, cost = int(match[1]), int(match[2] or 0)
            if cost != (count - 1) * 5 or count - 1 in result:
                raise ValueError('SensoryStone count/cost mismatch')
            result[count - 1] = (int(o['choice_index']), f'colorless_rewards_{count}')
        if set(result) != {0, 1, 2}:
            raise ValueError('SensoryStone incomplete choices')
        return result
    rules = CONTRACTS.get(event)
    if rules is None:
        raise ValueError(f'unsupported event contract: {event}')
    result = {}
    for o in ss.get('options', []):
        if o.get('disabled'):
            continue
        text = _text(o.get('text', ''))
        matches = [(idx, effect) for idx, pattern, effect in rules
                   if _rule_matches(event, text, pattern, effect)]
        if len(matches) != 1 or o.get('choice_index') is None:
            raise ValueError(f'unrecognized/ambiguous event option: {event}: {text}')
        idx, effect = matches[0]
        if idx in result:
            raise ValueError(f'duplicate event semantic ID: {event}/{idx}')
        result[idx] = (int(o['choice_index']), effect)
    if not result:
        raise ValueError(f'empty event contract: {event}')
    return result


class EventEffectAck:
    """Check observable effect types; permit selection UI, never stale retries."""
    def __init__(self):
        self.pending = None

    def _arm_staged_command(self, gs, decision):
        if decision.command.upper().split()[0] in {'STATE', 'WAIT', 'POTION'}:
            return
        old, effect, started = self.pending
        event = STAGED_EFFECTS[effect][0]
        progress = event_progress(gs, event)
        if gs.get('screen_type') == 'GRID':
            parts = decision.command.upper().split()
            if parts[:1] == ['CHOOSE']:
                if self._staged_pick is not None:
                    raise RuntimeError('event already has an uncompleted selection')
                from bridge.schemas.card_state import real_selection_card_states
                cards, selected = real_selection_card_states(
                    gs['screen_state']['cards'], gs['screen_state']['selected_cards'])
                index = int(parts[1])
                if len(parts) != 2 or selected or not 0 <= index < len(cards):
                    raise ValueError('invalid event grid selection command')
                self._staged_pick = cards[index]['uuid']
            elif parts != ['CONFIRM']:
                raise ValueError('unexpected command during event card selection')
        else:
            ui = transport_only(gs)
            if ui is None or decision.command != f'CHOOSE {ui}':
                raise RuntimeError('event result pending: no new policy/leave command allowed')
            self._staged_transport_sent = deepcopy(progress)
        self.pending = (old, effect, started)

    def _waiting_staged(self, old, gs, effect, ready):
        event, initial, final = STAGED_EFFECTS[effect]
        for key in ('seed', 'act', 'floor'):
            if key not in old or key not in gs or old[key] != gs[key]:
                raise RuntimeError(f'{event} changed/missing {key} before receipt')
        if not ready:
            return True
        if gs.get('screen_type') == 'GAME_OVER' and gs['current_hp'] <= 0:
            self.pending = None
            return False
        if effect in {'pending_colosseum_first_combat', 'pending_colosseum_second_combat', 'pending_dead_search'}:
            if gs.get('room_phase') == 'COMBAT' and gs.get('combat_state'):
                self.pending = None
                return False
        progress = event_progress(gs, event)
        phase = progress['phase']
        if self._staged_transport_sent == progress:
            return True  # Sent this intermediate page already; never retry it.
        self._staged_transport_sent = None
        if gs.get('screen_type') == 'GRID':
            contexts = {'pending_blacksmith_upgrade': 'UPGRADE',
                        'pending_bonfire_offer': 'BONFIRE_SPIRITS',
                        'pending_wing_remove': 'REMOVE', 'pending_wheel_random': 'REMOVE'}
            ss = gs['screen_state']
            if (effect not in contexts or ss.get('selection_context') != contexts[effect]
                    or ss.get('selection_source') != progress['source'] or ss.get('num_cards') != 1):
                raise RuntimeError(f'{event} unexpected selection source/context/count')
            if self._staged_pick is not None:
                selected = [c['uuid'] for c in ss['selected_cards']]
                if selected and selected != [self._staged_pick]:
                    raise RuntimeError('event acknowledged a different selected instance')
                return selected != [self._staged_pick]
            return False  # The policy still chooses the actual card.
        if effect == 'pending_wheel_random' and gs.get('screen_type') == 'COMBAT_REWARD':
            rows = gs['screen_state']['rewards']
            if phase != 'LEAVE' or len(rows) != 1 or rows[0].get('reward_type') != 'RELIC':
                raise RuntimeError('Wheel unexpected relic reward receipt')
            self.pending = None
            return False  # Claim/skip belongs to the ordinary reward policy.
        if gs.get('screen_type') != 'EVENT' or gs['screen_state'].get('event_id') != event:
            raise RuntimeError(f'{event} unexpected result screen')
        if effect == 'pending_dead_search':
            before = event_progress(old, event)['rewards_taken']
            if phase == 'FAIL':
                return transport_only(gs) is None  # Explicit Fight confirmation.
            if progress['rewards_taken'] == before:
                return True
            if progress['rewards_taken'] != before + 1 or phase not in {'INTRO', 'SUCCESS'}:
                raise RuntimeError('Dead Adventurer reward progress mismatch')
            # All three randomReward outcomes apply synchronously, including
            # NOTHING. A zero resource delta is not a failed command.
            self.pending = None
            return False
        intermediate = {
            'pending_wing_remove': {'PURGE'}, 'pending_serpent_gold_curse': {'AGREE'},
            'pending_joust_murderer': {'PRE_JOUST', 'JOUST'},
            'pending_joust_owner': {'PRE_JOUST', 'JOUST'},
            'pending_wheel_random': {'INTRO', 'COMPLETE'},
        }
        if phase in intermediate.get(effect, set()):
            return transport_only(gs) is None
        if phase != final:
            if phase == initial:
                return True
            raise RuntimeError(f'{event} unexpected phase after {effect}: {phase}')
        if gs['simulator_obtain_pending']:
            return True
        if effect == 'pending_wheel_random' and progress['purgeResult']:
            return True
        # Selection effects run in the event update after GRID closes. Verify
        # the chosen instance, not merely a new dialogue or a shorter deck.
        before = {c['uuid']: c for c in old['deck']}
        after = {c['uuid']: c for c in gs['deck']}
        if effect == 'pending_blacksmith_upgrade':
            pick = self._staged_pick
            if pick is None or pick not in after or after[pick]['upgrades'] <= before[pick]['upgrades']:
                return True
        if effect in {'pending_bonfire_offer', 'pending_wing_remove', 'pending_wheel_random'}:
            if self._staged_pick is not None and self._staged_pick in after:
                return True
            if effect == 'pending_wing_remove' and self._staged_pick is None:
                return True
        if effect in {'pending_blacksmith_relic_curse', 'pending_idol_curse', 'pending_serpent_gold_curse'}:
            curse = {'pending_blacksmith_relic_curse': 'Pain', 'pending_idol_curse': 'Injury',
                     'pending_serpent_gold_curse': 'Doubt'}[effect]
            prior_relics = {r['id']: r['counter'] for r in old['relics']}
            current_relics = {r['id']: r['counter'] for r in gs['relics']}
            blocked = prior_relics.get('Omamori', 0) > 0
            gained = Counter(c['id'] for u, c in after.items() if u not in before)
            expected = Counter() if blocked else Counter({curse: 1})
            if gained != expected or (blocked and current_relics.get('Omamori') != prior_relics['Omamori'] - 1):
                raise RuntimeError(f'{event} delayed curse receipt mismatch')
        # Other audited terminal pages are set after synchronous damage,
        # healing, gold, upgrades and relic obtain. Use the real result without
        # reimplementing relic interactions or requiring nonzero net changes.
        self.pending = None
        return False

    def _waiting_tome(self, old, gs, effect, ready):
        for key in ('seed', 'act', 'floor'):
            if key not in old or key not in gs or old[key] != gs[key]:
                raise RuntimeError(f'Cursed Tome changed/missing {key} before settlement')
        if not ready:
            return True
        if gs.get('screen_type') == 'GAME_OVER' and effect != 'tome_read' and gs['current_hp'] <= 0:
            self.pending = None
            return False
        if effect == 'tome_take' and gs.get('screen_type') == 'COMBAT_REWARD':
            owned = {r['id'] for r in old['relics']}
            eligible = {'Necronomicon', 'Enchiridion', "Nilry's Codex"} - owned
            if not eligible:
                eligible = {'Circlet'}
            rewards = gs['screen_state']['rewards']
            if (len(rewards) != 1 or rewards[0].get('reward_type') != 'RELIC'
                    or rewards[0].get('relic', {}).get('id') not in eligible):
                raise RuntimeError('Cursed Tome book reward identity/count mismatch')
            # randomBook opens rewards; it does not obtain the relic. The model
            # still owns the later claim/skip decision through RewardCommandAck.
            self.pending = None
            return False
        if (gs.get('screen_type') != 'EVENT'
                or gs.get('screen_state', {}).get('event_id') != 'Cursed Tome'):
            raise RuntimeError('Cursed Tome unexpected result screen')
        before, after = set(options(old)), set(options(gs))
        if after == before:
            return True  # Do not resend while observing the previous page.
        expected = {
            'tome_read': {2}, 'tome_continue_1': {3},
            'tome_continue_2': {4}, 'tome_continue_3': {5, 6},
            'tome_stop': {1},
        }
        if after != expected.get(effect):
            raise RuntimeError(f'Cursed Tome wrong next page after {effect}: {after}')
        # Java buttonEffect applies HP_LOSS synchronously before changing the
        # page. Read back actual HP/revival state rather than equating net HP
        # loss with nominal damage (Tungsten Rod/Fairy/Lizard Tail can alter it).
        self.pending = None
        return False

    @staticmethod
    def _augmenter_deck(gs):
        from bridge.schemas.card_state import real_card_state
        records = gs.get('deck')
        if not isinstance(records, list):
            raise ValueError('Augmenter requires an explicit master deck')
        cards = []
        for record in records:
            card = real_card_state(record)
            if record.get('type') not in {'ATTACK', 'SKILL', 'POWER', 'CURSE'}:
                raise ValueError('Augmenter master card type is missing/unsupported')
            cards.append(dict(card, type=record['type']))
        if len({c['uuid'] for c in cards}) != len(cards):
            raise ValueError('Augmenter master deck has duplicate UUIDs')
        return cards

    @staticmethod
    def _augmenter_relics(gs):
        rows = gs.get('relics')
        if (not isinstance(rows, list) or any(not isinstance(r, dict)
                or not isinstance(r.get('id'), str) or type(r.get('counter')) is not int for r in rows)):
            raise ValueError('Augmenter requires explicit relic IDs/counters')
        result = {r['id']: r['counter'] for r in rows}
        if len(result) != len(rows):
            raise ValueError('Augmenter has duplicate relic IDs')
        return result

    def _augmenter_grid(self, gs):
        from bridge.schemas.card_state import real_selection_card_states
        from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
        ss = gs.get('screen_state') or {}
        if (ss.get('simulator_parity_export') != SIMULATOR_PARITY_EXPORT_VERSION
                or ss.get('selection_source') != 'com.megacrit.cardcrawl.events.city.DrugDealer'
                or ss.get('selection_context') != 'TRANSFORM'
                or type(ss.get('num_cards')) is not int or ss['num_cards'] != 2
                or any(ss.get(key) is not False for key in
                       ('any_number', 'for_upgrade', 'for_purge', 'for_transform'))):
            raise ValueError('Augmenter GRID source/count/flags mismatch')
        cards, chosen = real_selection_card_states(ss.get('cards'), ss.get('selected_cards'))
        expected = {c['uuid']: {k: v for k, v in c.items() if k != 'type'}
                    for c in self._augmenter_original
                    if c['id'] not in {'AscendersBane', 'CurseOfTheBell', 'Necronomicurse'}}
        if {c['uuid']: c for c in cards} != expected:
            raise RuntimeError('Augmenter GRID candidates differ from purgeable master instances')
        if len(chosen) > 2:
            raise RuntimeError('Augmenter GRID selected more than two cards')
        return cards, [c['uuid'] for c in chosen]

    def _arm_augmenter_grid(self, gs, decision):
        cards, selected = self._augmenter_grid(gs)
        if selected != self._augmenter_picks:
            raise RuntimeError('Augmenter previous selection is still unacknowledged')
        command = decision.command.upper().split()
        if command == ['CONFIRM']:
            if len(selected) != 2 or self._augmenter_confirm_sent:
                raise ValueError('Augmenter premature/repeated confirmation')
            self._augmenter_confirm_sent = True
        elif len(command) == 2 and command[0] == 'CHOOSE':
            if len(selected) >= 2:
                raise ValueError('Augmenter already selected two cards')
            index = int(command[1])
            if not 0 <= index < len(cards) or cards[index]['uuid'] in selected:
                raise ValueError('Augmenter invalid/repeated physical card selection')
            remaining = [i for i, c in enumerate(cards) if c['uuid'] not in selected]
            semantic = decision.semantic
            if (not semantic or semantic[0] != 'select' or len(semantic) != 2
                    or type(semantic[1]) is not int or not 0 <= semantic[1] < len(remaining)
                    or index != remaining[semantic[1]]):
                raise ValueError('Augmenter policy candidate and UI command disagree')
            self._augmenter_picks.append(cards[index]['uuid'])
        elif command[:1] not in (['STATE'], ['WAIT']):
            raise ValueError('Augmenter unexpected command during selection')

    def _waiting_augmenter(self, old, gs, effect):
        for key in ('seed', 'floor', 'act', 'potions'):
            if key not in old or key not in gs or gs[key] != old[key]:
                raise RuntimeError(f'Augmenter changed/missing {key} before settlement')
        current = self._augmenter_deck(gs)
        relics = self._augmenter_relics(gs)
        before = {c['uuid']: c for c in self._augmenter_original}
        after = {c['uuid']: c for c in current}
        lost = set(before) - set(after)
        if (lost - set(self._augmenter_picks)
                or any(after[u] != before[u] for u in set(before) & set(after))):
            raise RuntimeError('Augmenter removed or changed an unselected master instance')
        if effect != 'augmenter_transform':
            if gs.get('screen_type') != 'EVENT' or gs.get('screen_state', {}).get('event_id') != 'Drug Dealer':
                raise RuntimeError('Augmenter reward left its event before verification')
            gained = [c for c in current if c['uuid'] not in before]
            expected_relics = dict(self._augmenter_initial_relics)
            gold = old['gold']
            if effect == 'augmenter_jax':
                up = int('Toxic Egg 2' in expected_relics)
                if len(gained) > 1 or any(c['id'] != CARD_ALIASES['J.A.X.'] or c['up'] != up for c in gained):
                    raise RuntimeError('Augmenter JAX identity/upgrade mismatch')
                if len(gained) != 1:
                    return True
                if 'CeramicFish' in expected_relics and 'Ectoplasm' not in expected_relics:
                    gold += 9
            else:
                if gained:
                    raise RuntimeError('Augmenter relic choice unexpectedly added cards')
                if 'MutagenicStrength' in expected_relics:
                    expected_relics['Circlet'] = expected_relics.get('Circlet', 0) + 1
                else:
                    expected_relics['MutagenicStrength'] = -1
            if (transport_only(gs) is not None and relics == expected_relics
                    and (gs['gold'], gs['current_hp'], gs['max_hp']) ==
                    (gold, old['current_hp'], old['max_hp'])):
                self.pending = None
                return False
            return True
        if gs.get('screen_type') == 'GRID':
            if current != self._augmenter_original or relics != self._augmenter_initial_relics:
                raise RuntimeError('Augmenter mutated resources before selection closed')
            _, selected = self._augmenter_grid(gs)
            if selected != self._augmenter_picks[:len(selected)]:
                raise RuntimeError('Augmenter acknowledged a different physical selection')
            return (selected != self._augmenter_picks or self._augmenter_confirm_sent)
        complete = (gs.get('screen_type') == 'EVENT'
                    and gs.get('screen_state', {}).get('event_id') == 'Drug Dealer'
                    and transport_only(gs) is not None)
        if gs.get('screen_type') != 'EVENT' or gs.get('screen_state', {}).get('event_id') != 'Drug Dealer':
            raise RuntimeError('Augmenter left its event before effect verification')
        if len(self._augmenter_picks) != 2:
            if current != self._augmenter_original:
                raise RuntimeError('Augmenter changed deck before two requested picks')
            return True  # Stale event frame before the GRID opens, never click Leave.
        picked = [before[u] for u in self._augmenter_picks]
        initial_relics = self._augmenter_initial_relics
        curse_count = sum(c['type'] == 'CURSE' for c in picked)
        blocked = min(curse_count, max(0, initial_relics.get('Omamori', 0)))
        gained = [c for c in current if c['uuid'] not in before]
        if len(gained) > 2 - blocked:
            raise RuntimeError('Augmenter obtained too many replacement cards')
        expected_relics = dict(initial_relics)
        if blocked:
            expected_relics['Omamori'] -= blocked
        if 'Du-Vu Doll' in expected_relics:
            expected_relics['Du-Vu Doll'] = sum(c['type'] == 'CURSE' for c in current)
        eggs = {'ATTACK': 'Molten Egg 2', 'SKILL': 'Toxic Egg 2', 'POWER': 'Frozen Egg 2'}
        for card in gained:
            upgrades = int(eggs.get(card['type']) in initial_relics)
            if card['up'] != upgrades or any(card[k] for k in
                    ('bottled_flame', 'bottled_lightning', 'bottled_tornado')):
                raise RuntimeError('Augmenter replacement upgrade/bottle state mismatch')
        if lost != set(self._augmenter_picks) or len(gained) != 2 - blocked:
            return True  # Removal and delayed ShowCardAndObtainEffect are separate frames.
        if sum(c['type'] == 'CURSE' for c in gained) != curse_count - blocked:
            raise RuntimeError('Augmenter replacement curse count mismatch')
        hp, max_hp = old['current_hp'], old['max_hp']
        for card in picked:
            if card['id'] == 'Parasite':
                max_hp = max(1, max_hp - 3)
                hp = min(hp, max_hp)
        if 'Darkstone Periapt' in initial_relics:
            bonus = 6 * (curse_count - blocked)
            max_hp += bonus
            if 'Mark of the Bloom' not in initial_relics:
                hp = min(hp + bonus, max_hp)
        gold = old['gold'] + (9 * len(gained) if 'CeramicFish' in initial_relics
                              and 'Ectoplasm' not in initial_relics else 0)
        if complete and (relics, gs['current_hp'], gs['max_hp'], gs['gold']) == (expected_relics, hp, max_hp, gold):
            self.pending = None
            return False
        return True

    def arm(self, frame, decision, now):
        gs = frame.get('game_state') or {}
        if self.pending and self.pending[1] in STAGED_EFFECTS:
            self._arm_staged_command(gs, decision)
            return
        # Native setupEvent opens Bonfire's selection directly. Both Java
        # dialogue buttons are transport, not an extra upper-policy action.
        if (not self.pending and gs.get('screen_type') == 'EVENT'
                and gs.get('screen_state', {}).get('event_id') == 'Bonfire Elementals'
                and event_progress(gs)['phase'] == 'CHOOSE'
                and decision.command == f'CHOOSE {transport_only(gs)}'):
            self.pending = (deepcopy(gs), 'pending_bonfire_offer', now)
            self._staged_transport_sent = None
            self._staged_pick = None
            return
        if self.pending and self.pending[1].startswith('augmenter_'):
            if self.pending[1] == 'augmenter_transform' and gs.get('screen_type') == 'GRID':
                self._arm_augmenter_grid(gs, decision)
                return
            raise RuntimeError('Augmenter result still unverified; cannot leave or retry')
        if (gs.get('screen_type') == 'EVENT' and gs.get('screen_state', {}).get('event_id') == 'Lab'
                and transport_only(gs) is not None
                and decision.command == f'CHOOSE {transport_only(gs)}'):
            if self.pending:
                raise RuntimeError('Lab reward transition still unverified')
            self.pending = (deepcopy(gs), 'lab_rewards', now)
            return
        if (self.pending and self.pending[1] == 'beggar_remove'
                and gs.get('screen_state', {}).get('event_id') == 'Beggar'
                and transport_only(gs) is not None
                and decision.command == f'CHOOSE {transport_only(gs)}'):
            old, _, started = self.pending
            self.pending = (old, 'beggar_remove_confirmed', started)
            return
        if (self.pending and self.pending[1] in {'mushroom_combat', 'sphere_combat', 'portal_boss'}
                and gs.get('screen_state', {}).get('event_id') == self.pending[0].get('screen_state', {}).get('event_id')
                and transport_only(gs) is not None
                and decision.command == f'CHOOSE {transport_only(gs)}'):
            old, effect, started = self.pending
            self.pending = (old, effect + '_confirmed', started)
            return
        if not decision.semantic or decision.semantic[0] != 'event':
            return
        if self.pending:
            raise RuntimeError('event effect still unverified')
        gs = frame.get('game_state') or {}
        _, effect = options(gs)[decision.semantic[1]]
        if effect.startswith('pending_') and effect not in STAGED_EFFECTS:
            raise ValueError(
                f'recognized event option has no audited effect receipt: {effect}'
            )
        if effect in STAGED_EFFECTS:
            event, initial, _ = STAGED_EFFECTS[effect]
            if (gs['screen_state']['event_id'] != event or event_progress(gs)['phase'] != initial
                    or decision.command != f'CHOOSE {options(gs)[decision.semantic[1]][0]}'):
                raise ValueError('staged event source/semantic/command mismatch')
            self._staged_transport_sent = None
            self._staged_pick = None
        if effect.startswith('tome_'):
            if decision.command != f'CHOOSE {options(gs)[decision.semantic[1]][0]}':
                raise ValueError('Cursed Tome semantic action and UI command disagree')
        if effect.startswith('augmenter_'):
            if decision.command != f'CHOOSE {options(gs)[decision.semantic[1]][0]}':
                raise ValueError('Augmenter event action and UI command disagree')
            self._augmenter_original = self._augmenter_deck(gs)
            self._augmenter_initial_relics = self._augmenter_relics(gs)
            if effect == 'augmenter_transform' and sum(
                    c['id'] not in {'AscendersBane', 'CurseOfTheBell', 'Necronomicurse'}
                    for c in self._augmenter_original) < 2:
                raise ValueError('Augmenter requires two purgeable cards')
            for key in ('seed', 'floor', 'act', 'potions', 'current_hp', 'max_hp', 'gold'):
                if key not in gs:
                    raise ValueError(f'Augmenter missing {key}')
            if (any(type(gs[k]) is not int or gs[k] < 0 for k in ('floor', 'act', 'current_hp', 'max_hp', 'gold'))
                    or not 0 < gs['current_hp'] <= gs['max_hp'] or not isinstance(gs['potions'], list)):
                raise ValueError('Augmenter invalid public resource state')
            if 'Omamori' in self._augmenter_initial_relics and not 0 <= self._augmenter_initial_relics['Omamori'] <= 2:
                raise ValueError('Augmenter invalid Omamori charge count')
            self._augmenter_picks = []
            self._augmenter_confirm_sent = False
        if effect == 'woman_leave' and gs.get('ascension_level', 0) >= 15:
            raise ValueError('Woman in Blue A15 leave damage variants not yet audited')
        if effect.startswith('woman_potions_'):
            count = int(effect.rsplit('_', 1)[1])
            ui, _ = options(gs)[decision.semantic[1]]
            text = next(_text(o['text']) for o in gs['screen_state']['options']
                        if not o.get('disabled') and o['choice_index'] == ui)
            match = re.fullmatch(r'\[(?:买\d瓶药水|buy\dpotions?)\](\d+)(?:金币。|gold\.)', text)
            if not match or int(match[1]) != 10 + 10 * count:
                raise ValueError('Woman in Blue displayed price mismatch')
        if effect == 'goop_loss':
            ui, _ = options(gs)[1]
            text = next(_text(o['text']) for o in gs['screen_state']['options']
                        if not o.get('disabled') and o['choice_index'] == ui)
            match = re.fullmatch(r'\[(?:放手吧|leaveit)\](?:失去|lose)(\d+)(?:金币。|gold\.)', text)
            if not match or int(match[1]) > gs.get('gold', 0):
                raise ValueError('World of Goop public gold loss missing/invalid')
            self.goop_loss = int(match[1])
        if effect == 'nloth_trade':
            self.nloth_target = next(d['relic']['id'] for d in nloth_choices(gs) if d['index'] == decision.semantic[1])
            if any(r['id'] == "Nloth's Gift" for r in gs.get('relics', [])):
                raise RuntimeError('Nloth duplicate gift variant not yet verified')
        if effect == 'scrap_search':
            scrap_offer(gs)  # Refuse unparsed variants before sending anything.
        if effect == 'falling_remove':
            card = next(d['card'] for d in falling_choices(gs) if d['index'] == decision.semantic[1])
            self.falling_target = (card['id'], card.get('upgrades', 0))
        if effect == 'potion' and (not any(p.get('id') == 'Potion Slot' for p in gs.get('potions', []))
                                   or any(r.get('id') == 'Sozu' for r in gs.get('relics', []))):
            # Skull pays HP even when obtainPotion cannot insert into inventory.
            # Keep the simulator's legal action; verify this precise no-gain case.
            if gs.get('screen_state', {}).get('event_id') != 'Knowing Skull':
                raise RuntimeError('unsupported blocked event potion reward')
            effect = 'blocked_skull_potion'
        self.pending = (deepcopy(gs), effect, now)

    def waiting(self, frame, now):
        if not self.pending:
            return False
        old, effect, started = self.pending
        if frame.get('error'):
            raise RuntimeError(f'event command rejected: {frame["error"]}')
        g = frame.get('game_state') or {}
        if now - started > 30:
            raise RuntimeError(f'event effect timeout/mismatch: expected {effect}')
        if not g:
            return True
        if effect in STAGED_EFFECTS:
            return self._waiting_staged(old, g, effect, frame.get('ready_for_command') is True)
        if effect.startswith('tome_'):
            return self._waiting_tome(old, g, effect, frame.get('ready_for_command') is True)
        if effect.startswith('augmenter_'):
            return self._waiting_augmenter(old, g, effect)
        if effect in {'portal_boss', 'portal_boss_confirmed'}:
            if g.get('combat_state') or g.get('room_phase') == 'COMBAT':
                if g.get('room_type') != 'MonsterRoomBoss':
                    raise RuntimeError('SecretPortal expected boss combat')
                self.pending = None
                return False
            if any(g.get(k) != old.get(k) for k in ('gold', 'current_hp', 'max_hp', 'deck', 'relics', 'potions')):
                raise RuntimeError('SecretPortal unexpected pre-combat resource effect')
            if (effect == 'portal_boss'
                    and g.get('screen_type') == 'EVENT'
                    and g.get('screen_state', {}).get('event_id') == 'SecretPortal'
                    and transport_only(g) is not None):
                return False  # One explicit confirmation; not completion.
            return True
        if effect == 'lab_rewards':
            if g.get('screen_type') != 'COMBAT_REWARD':
                return True
            rewards = (g.get('screen_state') or {}).get('rewards', [])
            count = 2 if old.get('ascension_level', 0) >= 15 else 3
            if len(rewards) != count or any(r.get('reward_type') != 'POTION' for r in rewards):
                raise RuntimeError('Lab reward type/count mismatch')
            self.pending = None
            return False
        if effect in {'mushroom_combat', 'mushroom_combat_confirmed', 'sphere_combat', 'sphere_combat_confirmed'}:
            if g.get('combat_state') or g.get('room_phase') == 'COMBAT':
                self.pending = None
                return False
            if (effect in {'mushroom_combat', 'sphere_combat'}
                    and g.get('screen_state', {}).get('event_id') == old.get('screen_state', {}).get('event_id')
                    and transport_only(g) is not None
                    and any(_text(o.get('text', '')).strip('[]') in {'战斗', 'fight'}
                            for o in g['screen_state'].get('options', []) if not o.get('disabled'))):
                return False  # Explicit second dialogue; arm prevents repeating it.
            return True
        if effect.startswith('colorless_rewards_'):
            if g.get('screen_type') != 'COMBAT_REWARD':
                return True
            count = int(effect.rsplit('_', 1)[1])
            rewards = (g.get('screen_state') or {}).get('rewards', [])
            if len(rewards) != count or any(r.get('reward_type') != 'CARD' for r in rewards):
                raise RuntimeError('SensoryStone reward type/count mismatch')
            # Cards are optional rewards, not guaranteed permanent deck gains.
            self.pending = None
            return False
        if effect == 'falling_remove':
            # arm stores the exact selected card; do not accept any deck loss.
            target = self.falling_target
            count = lambda s: Counter((c['id'], c.get('upgrades', 0)) for c in s.get('deck', []))
            before, after = count(old), count(g)
            if before == after:
                return True
            if before - after != Counter({target: 1}) or after - before:
                raise RuntimeError('Falling removed card identity/count mismatch')
            self.pending = None
            return False
        deck = lambda s: Counter((c['id'], c.get('upgrades', 0)) for c in s.get('deck', []))
        gained, lost = deck(g) - deck(old), deck(old) - deck(g)
        delta = lambda k: g.get(k, 0) - old.get(k, 0)
        potion_changed = [p.get('id') for p in g.get('potions', [])] != [p.get('id') for p in old.get('potions', [])]
        relic_changed = [r.get('id') for r in g.get('relics', [])] != [r.get('id') for r in old.get('relics', [])]
        if effect.startswith('woman_potions_') or effect == 'woman_leave':
            count = 0 if effect == 'woman_leave' else int(effect.rsplit('_', 1)[1])
            cost = min(old.get('gold', 0), 10 + 10 * count) if count else 0
            if (gained or lost or relic_changed or potion_changed or delta('current_hp')
                    or delta('max_hp') or delta('gold') not in (0, -cost)):
                raise RuntimeError('Woman in Blue payment/resource mismatch')
            if count and g.get('screen_type') == 'COMBAT_REWARD':
                rewards = g.get('screen_state', {}).get('rewards', [])
                if len(rewards) != count or any(r.get('reward_type') != 'POTION' for r in rewards):
                    raise RuntimeError('Woman in Blue reward type/count mismatch')
                if delta('gold') != -cost:
                    return True
                self.pending = None
                return False
            if (not count and g.get('screen_state', {}).get('event_id') == 'The Woman in Blue'
                    and transport_only(g) is not None):
                self.pending = None
                return False
            return True
        if effect in {'basics_remove', 'basics_upgrade'}:
            complete = (g.get('screen_state', {}).get('event_id') == 'Back to Basics'
                        and transport_only(g) is not None)
            if relic_changed or potion_changed or delta('gold'):
                raise RuntimeError('Back to Basics unexpected resource effect')
            if effect == 'basics_remove':
                permanent = {'AscendersBane', 'CurseOfTheBell', 'Necronomicurse'}
                if gained or sum(lost.values()) > 1 or any(cid in permanent for cid, _ in lost):
                    raise RuntimeError('Back to Basics invalid removal')
                if sum(lost.values()) == 1 and complete:
                    self.pending = None
                    return False
                return g.get('screen_type') != 'GRID'
            # Java uses STARTER_STRIKE/STARTER_DEFEND, not substring "Strike".
            basic = {'Strike_R', 'Strike_G', 'Strike_B', 'Strike_P',
                     'Defend_R', 'Defend_G', 'Defend_B', 'Defend_P'}
            expected = Counter((c['id'], 1 if c['id'] in basic and not c.get('upgrades', 0)
                                else c.get('upgrades', 0)) for c in old.get('deck', []))
            if (lost - (deck(old) - expected) or gained - (expected - deck(old))
                    or delta('current_hp') or delta('max_hp')):
                raise RuntimeError('Back to Basics upgraded wrong cards')
            if deck(g) == expected and complete:
                self.pending = None
                return False
            return True
        if effect == 'remove_curses':
            permanent = {'AscendersBane', 'CurseOfTheBell', 'Necronomicurse'}
            expected = Counter((c['id'], c.get('upgrades', 0)) for c in old.get('deck', [])
                               if c.get('type') == 'CURSE' and c['id'] not in permanent)
            if gained or lost - expected or relic_changed or potion_changed or delta('gold') != 0:
                raise RuntimeError('Fountain removed wrong card or changed reward type')
            complete = (g.get('screen_state', {}).get('event_id') == 'Fountain of Cleansing'
                        and transport_only(g) is not None)
            # Multiple purges may be observed separately, including Parasite's
            # max-HP loss. An empty eligible set still requires result dialogue.
            if lost == expected and complete:
                self.pending = None
                return False
            return True
        if effect in {'goop_gold', 'goop_loss'}:
            if effect == 'goop_gold' and g.get('screen_type') == 'GAME_OVER' and g.get('current_hp', 1) <= 0:
                self.pending = None
                return False
            reward = 0 if any(r['id'] == 'Ectoplasm' for r in old.get('relics', [])) else 75
            expected = reward if effect == 'goop_gold' else -self.goop_loss
            if gained or lost or relic_changed or potion_changed or delta('max_hp') != 0 or delta('gold') not in (0, expected):
                raise RuntimeError('World of Goop effect type/amount mismatch')
            complete = (g.get('screen_state', {}).get('event_id') == 'World of Goop'
                        and transport_only(g) is not None)
            # Loss can be zero; gaining gold may heal through Bloody Idol.
            # Require the result dialogue, not just a resource delta.
            if complete and delta('gold') == expected:
                if effect == 'goop_loss' and delta('current_hp') != 0:
                    raise RuntimeError('World of Goop leave unexpectedly changed HP')
                self.pending = None
                return False
            return True
        if effect == 'nloth_trade':
            before = Counter(r['id'] for r in old.get('relics', []))
            after = Counter(r['id'] for r in g.get('relics', []))
            removed, added = before - after, after - before
            if (removed and removed != Counter({self.nloth_target: 1})
                    or added and added != Counter({"Nloth's Gift": 1})):
                raise RuntimeError('Nloth relic exchange identity/count mismatch')
            if removed and added and transport_only(g) is not None:
                self.pending = None
                return False
            return True
        if effect in {'face_touch', 'face_trade'}:
            if g.get('screen_type') == 'GAME_OVER' and g.get('current_hp', 1) <= 0:
                self.pending = None
                return False
            before = Counter(r['id'] for r in old.get('relics', []))
            after = Counter(r['id'] for r in g.get('relics', []))
            complete = (g.get('screen_state', {}).get('event_id') == 'FaceTrader'
                        and transport_only(g) is not None)
            if effect == 'face_trade':
                faces = {'CultistMask', 'FaceOfCleric', 'GremlinMask', 'NlothsMask', 'SsserpentHead'}
                added = after - before
                if before - after or any(r not in faces for r in added) or sum(added.values()) > 1:
                    raise RuntimeError('FaceTrader relic identity/count mismatch')
                done = complete and sum(added.values()) == 1
            else:
                reward = 0 if 'Ectoplasm' in before else (50 if old.get('ascension_level', 0) >= 15 else 75)
                if gained or lost or relic_changed or delta('gold') not in (0, reward):
                    raise RuntimeError('FaceTrader touch effect mismatch')
                # Gold-triggered healing can offset damage; don't require net HP
                # loss. The resolved dialogue is required in addition to gold.
                done = complete and delta('gold') == reward and g != old
            if done:
                self.pending = None
                return False
            return True
        if effect == 'scrap_search':
            if g.get('screen_type') == 'GAME_OVER' and g.get('current_hp', 1) <= 0:
                self.pending = None
                return False
            if g.get('screen_state', {}).get('event_id') != 'Scrap Ooze':
                raise RuntimeError('Scrap Ooze unexpected result screen')
            # Relic acquisition can heal or otherwise change resources. Require
            # the acquired relic AND completed dialogue, not HP loss alone.
            if transport_only(g) is not None:
                before = Counter(r['id'] for r in old.get('relics', []))
                after = Counter(r['id'] for r in g.get('relics', []))
                if sum((after - before).values()) == 1 and not before - after:
                    self.pending = None
                    return False
                return True
            previous, current = scrap_offer(old), scrap_offer(g)
            if current == previous:
                return True  # Includes HP-only animation/intermediate frames.
            if current != (previous[0] + 1, previous[1] + 10):
                raise RuntimeError('Scrap Ooze failure cost/chance mismatch')
            if relic_changed or gained or lost or potion_changed or delta('gold'):
                raise RuntimeError('Scrap Ooze failure effect type mismatch')
            if delta('current_hp') >= 0:
                return True
            self.pending = None
            return False
        if effect.startswith('altar_'):
            before = {r['id']: r for r in old.get('relics', [])}
            after = {r['id']: r for r in g.get('relics', [])}
            if effect == 'altar_sacrifice':
                # Verify both effect types, not max-HP gain alone. Damage relic
                # interactions are not claimed to have exact numeric coverage.
                if gained or lost or delta('max_hp') not in (0, 5):
                    raise RuntimeError('Altar sacrifice effect mismatch')
                done = delta('max_hp') == 5 and delta('current_hp') < 0
            elif effect == 'altar_decay':
                blocked = before.get('Omamori', {}).get('counter', 0) > 0
                expected = Counter() if blocked else Counter({('Decay', 0): 1})
                if gained and gained != expected or lost:
                    raise RuntimeError('Altar curse identity/count mismatch')
                done = (after.get('Omamori', {}).get('counter') == before['Omamori']['counter'] - 1
                        if blocked else gained == expected)
            else:
                if 'Golden Idol' not in before or 'Bloody Idol' in before:
                    raise RuntimeError('Altar idol variant not verified')
                done = set(before) - set(after) == {'Golden Idol'} and set(after) - set(before) == {'Bloody Idol'}
            if done:
                self.pending = None
                return False
            return True
        if effect in {'cleric_heal', 'cleric_remove', 'beggar_remove', 'beggar_remove_confirmed'}:
            cost = (75 if effect.startswith('beggar_') else
                    35 if effect == 'cleric_heal' else (75 if old.get('ascension_level', 0) >= 15 else 50))
            if delta('gold') not in (0, -cost) or gained or sum(lost.values()) > (effect != 'cleric_heal'):
                raise RuntimeError('Paid event payment/card effect mismatch')
            if effect == 'cleric_heal':
                blocked = any(r['id'] == 'Mark of the Bloom' for r in old.get('relics', []))
                heal = 0 if blocked else min(old['max_hp'] // 4, old['max_hp'] - old['current_hp'])
                done = delta('gold') == -cost and delta('current_hp') == heal
            else:
                done = delta('gold') == -cost and sum(lost.values()) == 1
            if done:
                self.pending = None
                return False
            if effect != 'cleric_heal' and g.get('screen_type') == 'GRID' and delta('gold') == -cost:
                return False  # Selection is required, payment alone is not completion.
            if (effect == 'beggar_remove' and delta('gold') == -cost
                    and g.get('screen_state', {}).get('event_id') == 'Beggar'
                    and transport_only(g) is not None
                    and any(_text(o.get('text', '')).strip('[]') in {'继续', 'continue'}
                            for o in g['screen_state'].get('options', []) if not o.get('disabled'))):
                return False  # One explicit confirmation before the removal grid.
            return True
        if effect == 'mushroom_eat':
            relics = {r['id']: r for r in old.get('relics', [])}
            after_relics = {r['id']: r for r in g.get('relics', [])}
            blocked = relics.get('Omamori', {}).get('counter', 0) > 0
            expected = Counter() if blocked else Counter({('Parasite', 0): 1})
            heal = 0 if 'Mark of the Bloom' in relics else min(old['max_hp'] // 4, old['max_hp'] - old['current_hp'])
            if gained and gained != expected or lost:
                raise RuntimeError('Mushrooms curse identity/count mismatch')
            curse_done = (after_relics.get('Omamori', {}).get('counter') == relics['Omamori']['counter'] - 1
                          if blocked else gained == expected)
            if curse_done and delta('current_hp') == heal:
                self.pending = None
                return False
            return True
        if old.get('screen_state', {}).get('event_id') == 'Knowing Skull':
            # A card-vs-potion mismatch must never pass merely because HP fell.
            if (effect != 'card' and gained) or (effect != 'potion' and potion_changed) or (effect != 'gold' and delta('gold') > 0):
                raise RuntimeError(f'event effect type mismatch: expected {effect}, card={bool(gained)}, potion={potion_changed}, gold={delta("gold")}')
        if old.get('screen_state', {}).get('event_id') == 'Big Fish' and effect == 'heal':
            # Banana heals maxHP/3; Mark of the Bloom can legitimately make it
            # zero even below full health. Neither a stale frame nor a wrong
            # reward type is evidence that the selected option completed.
            blocked = any(r.get('id') == 'Mark of the Bloom' for r in old.get('relics', []))
            amount = 0 if blocked else min(old['max_hp'] // 3, old['max_hp'] - old['current_hp'])
            if gained or lost or relic_changed or potion_changed or delta('gold') or delta('max_hp'):
                raise RuntimeError('Big Fish banana unexpected resource effect')
            if delta('current_hp') not in (0, amount):
                raise RuntimeError('Big Fish banana heal amount mismatch')
            done = (g.get('screen_type') == 'EVENT'
                    and g.get('screen_state', {}).get('event_id') == 'Big Fish'
                    and transport_only(g) is not None)
            if done and delta('current_hp') == amount:
                self.pending = None
                return False
            return True
        success = {
            'gold': delta('gold') > 0, 'gold_loss': delta('gold') < 0,
            'card': bool(gained), 'remove': bool(lost) and not gained,
            'transform': bool(gained) and bool(lost),
            'upgrade': bool(gained) and bool(lost) and Counter(c['id'] for c in g.get('deck', [])) == Counter(c['id'] for c in old.get('deck', [])),
            'potion': potion_changed, 'relic': relic_changed,
            'blocked_skull_potion': (delta('current_hp') < 0 and not potion_changed
                                     and not gained and not lost and not relic_changed
                                     and delta('gold') == 0 and delta('max_hp') == 0
                                     and g.get('screen_state') != old.get('screen_state')),
            'heal': delta('current_hp') > 0 or (old.get('current_hp') == old.get('max_hp') and g.get('screen_state') != old.get('screen_state')),
            'maxhp_gain': delta('max_hp') > 0, 'maxhp_loss': delta('max_hp') < 0,
            'vial_bites': relic_changed and bool(gained), 'hp_loss': delta('current_hp') < 0,
            'combat': bool(g.get('combat_state')) or g.get('room_phase') == 'COMBAT',
            'leave': g.get('screen_state') != old.get('screen_state') or g.get('screen_type') != 'EVENT',
        }[effect]
        if success:
            self.pending = None
            return False
        if g.get('screen_type') in {'GRID', 'CARD_REWARD', 'HAND_SELECT'}:
            return False
        return True
