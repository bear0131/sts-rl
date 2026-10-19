"""Fail-closed event semantic/transport contracts (BRIDGE06).

Only audited public labels are accepted. Unlisted events/variants require an
explicit contract, never positional fallback. Native IDs remain checkpoint ABI.
"""
import re
from collections import Counter
from copy import deepcopy

from bridge.schemas.entity_ids import (
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
    "Cursed Tome": [(0, '阅读|Read', 'pending_tome_read'),
                     (1, '离开|Leave', 'leave'),
                     (2, '继续|Continue', 'pending_tome_continue_1'),
                     (3, '继续|Continue', 'pending_tome_continue_2'),
                     (4, '继续|Continue', 'pending_tome_continue_3'),
                     (5, '拿走|Take', 'pending_tome_take'),
                     (6, '停止|Stop', 'pending_tome_stop')],
    "Dead Adventurer": [(0, '搜索|继续|Search|Continue', 'pending_dead_search'),
                        (1, '离开|Leave', 'leave')],
    "Drug Dealer": [(0, '试一下J.A.X.|Test J.A.X.', 'pending_augmenter_jax'),
                     (1, '当一下实验对象|Become Test Subject', 'pending_augmenter_transform'),
                     (2, '喝一下突变剂|Ingest Mutagens', 'pending_augmenter_relic')],
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
    if event == 'Cursed Tome' and effect.startswith('pending_tome_continue_'):
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
    opts = [o for o in (gs.get('screen_state') or {}).get('options', []) if not o.get('disabled')]
    if len(opts) != 1:
        return None
    o = opts[0]
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

    def arm(self, frame, decision, now):
        gs = frame.get('game_state') or {}
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
        if effect.startswith('pending_'):
            raise ValueError(
                f'recognized event option has no audited effect receipt: {effect}'
            )
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
