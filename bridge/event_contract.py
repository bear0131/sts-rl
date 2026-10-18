"""Fail-closed event semantic/transport contracts (BRIDGE06).

Only audited public labels are accepted. Unlisted events/variants require an
explicit contract, never positional fallback. Native IDs remain checkpoint ABI.
"""
import re
from collections import Counter
from copy import deepcopy


# native semantic ID, public label aliases, expected effect type
CONTRACTS = {
    "Mushrooms": [(0, '踩扁|Stomp', 'mushroom_combat'), (1, '吃下|Eat', 'mushroom_eat')],
    "Knowing Skull": [(0, '财富？|Riches?', 'gold'), (1, '成功？|Success?', 'card'),
                      (2, '来点喝的？|A Pick Me Up?', 'potion'), (3, '我要怎么离开？|How do I leave?', 'hp_loss')],
    "Vampires": [(0, '失去鲜血药瓶|Lose Blood Vial', 'vial_bites'),
                 (1, '失去.*最大生命|Lose.*Max HP', 'maxhp_loss'), (2, '拒绝|Refuse', 'leave')],
    "Tomb of Lord Red Mask": [(0, '戴上|Don', 'gold'), (1, '献上|Offer', 'relic'), (2, '离开|Leave', 'leave')],
    "MindBloom": [(0, '我必凯旋|I am War', 'combat'), (1, '我已觉醒|I am Awake', 'relic'),
                  (2, '我将富有|我很富有|I am Rich', 'gold'), (3, '我无伤痛|I am Healthy', 'card')],
    "Big Fish": [(0, '香蕉|Banana', 'heal'), (1, '甜甜圈|Donut', 'maxhp_gain'), (2, '盒子|Box', 'relic')],
    "Living Wall": [(0, '遗忘|Forget', 'remove'), (1, '改变|Change', 'transform'), (2, '成长|Grow', 'upgrade')],
    "Transmorgrifier": [(0, '祈祷|Pray', 'transform'), (1, '离开|Leave', 'leave')],
    "Purifier": [(0, '祈祷|Pray', 'remove'), (1, '离开|Leave', 'leave')],
    "Upgrade Shrine": [(0, '祈祷|Pray', 'upgrade'), (1, '离开|Leave', 'leave')],
    "Duplicator": [(0, '祈祷|Pray', 'card'), (1, '离开|Leave', 'leave')],
    "Ghosts": [(0, '接受|Accept', 'maxhp_loss'), (1, '拒绝|Refuse', 'leave')],
    "Masked Bandits": [(0, '付钱|Pay', 'gold_loss'), (1, '战斗|Fight', 'combat')],
    "Golden Shrine": [(0, '祈祷|Pray', 'gold'), (1, '亵渎|Desecrate', 'gold'), (2, '离开|Leave', 'leave')],
    "Nest": [(0, '抢了钱就跑|Smash and Grab', 'gold'), (1, '留在队伍中|Stay in Line', 'card')],
    "The Library": [(0, '阅读|Read', 'card'), (1, '睡觉|Sleep', 'heal')],
    "Addict": [(0, '给他金币|Offer Gold', 'relic'), (1, '抢夺|Rob', 'relic'), (2, '离开|Leave', 'leave')],
    "Winding Halls": [(0, '拥抱疯狂|Embrace Madness', 'card'), (1, '集中精神|Focus', 'card'), (2, '寻找来路|Retrace', 'maxhp_loss')],
}


def _text(text):
    return re.sub(r'\s+|#[a-zA-Z]', '', str(text)).casefold()


def transport_only(gs):
    """An exact one-button dialogue, not a resource-bearing policy option."""
    opts = [o for o in (gs.get('screen_state') or {}).get('options', []) if not o.get('disabled')]
    if len(opts) != 1:
        return None
    o = opts[0]
    text = _text(o.get('text', '')).strip('[]')
    if ((gs.get('screen_state') or {}).get('event_id') == 'SensoryStone'
            and text in {'交互', 'interact'}):
        return o.get('choice_index')
    if ((gs.get('screen_state') or {}).get('event_id') == 'Mushrooms'
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


def options(gs):
    ss = gs.get('screen_state') or {}
    event = ss.get('event_id')
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
                   if any(re.search(_text(p).replace('?', r'\?'), text) for p in pattern.split('|'))]
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
        if (self.pending and self.pending[1] == 'mushroom_combat'
                and gs.get('screen_state', {}).get('event_id') == 'Mushrooms'
                and transport_only(gs) is not None
                and decision.command == f'CHOOSE {transport_only(gs)}'):
            old, _, started = self.pending
            self.pending = (old, 'mushroom_combat_confirmed', started)
            return
        if not decision.semantic or decision.semantic[0] != 'event':
            return
        if self.pending:
            raise RuntimeError('event effect still unverified')
        gs = frame.get('game_state') or {}
        _, effect = options(gs)[decision.semantic[1]]
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
        if effect in {'mushroom_combat', 'mushroom_combat_confirmed'}:
            if g.get('combat_state') or g.get('room_phase') == 'COMBAT':
                self.pending = None
                return False
            if (effect == 'mushroom_combat' and g.get('screen_state', {}).get('event_id') == 'Mushrooms'
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
