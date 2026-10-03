"""Sparse, human-readable run-history collection for whole-run evaluation.

The collector deliberately records decision boundaries and resource deltas, not
combat actions.  It consumes the backend's canonical state so that event/shop
metadata remains typed and action-aligned; missing native metadata is retained
as an option index/label instead of being guessed from localized text.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from html import escape
import json
from pathlib import Path
import re
import time
import numpy as np


# The native card table is used by the model and contains a few historical
# translation mistakes.  Reports have their own display vocabulary so that a
# log can never turn Combust (the HP-loss power) into Inflame (the Strength
# power).  Unknown IDs deliberately fall back to the canonical English ID;
# inventing a Chinese name is worse than showing an unambiguous ID.
CARD_NAME_ZH = {
    "Bash": "痛击", "Defend_R": "防御", "Strike_R": "打击", "Anger": "愤怒",
    "Armaments": "武装", "Body Slam": "全身撞击", "Clash": "交锋", "Cleave": "顺劈斩",
    "Clothesline": "金刚臂", "Flex": "活动肌肉", "Havoc": "破灭", "Headbutt": "头槌",
    "Heavy Blade": "重刃", "Iron Wave": "铁斩波", "Perfected Strike": "完美打击",
    "Pommel Strike": "剑柄打击", "Shrug It Off": "耸肩无视", "Sword Boomerang": "飞剑回旋镖",
    "Thunderclap": "闪电霹雳", "True Grit": "坚毅", "Twin Strike": "双重打击",
    "Warcry": "战吼", "Wild Strike": "狂野打击", "Battle Trance": "战斗专注",
    "Blood For Blood": "以血还血", "Bloodletting": "放血", "Burning Pact": "燃烧契约",
    "Carnage": "残杀", "Combust": "自燃", "Inflame": "燃烧", "Dark Embrace": "黑暗之拥",
    "Disarm": "缴械", "Dropkick": "飞身踢", "Dual Wield": "双持", "Entrench": "巩固",
    "Evolve": "进化", "Feel No Pain": "无痛", "Fire Breathing": "火焰吐息",
    "Flame Barrier": "火焰屏障", "Ghostly Armor": "幽灵铠甲", "Hemokinesis": "御血术",
    "Infernal Blade": "地狱之刃", "Intimidate": "威吓", "Metallicize": "金属化",
    "Power Through": "硬撑", "Pummel": "连续拳", "Rage": "狂怒", "Rampage": "暴走",
    "Reckless Charge": "鲁莽冲锋", "Rupture": "撕裂", "Searing Blow": "灼热攻击",
    "Second Wind": "重振精神", "Seeing Red": "盛怒", "Sentinel": "哨卫",
    "Sever Soul": "断魂斩", "Shockwave": "震荡波", "Spot Weakness": "观察弱点",
    "Uppercut": "上勾拳", "Whirlwind": "旋风斩", "Barricade": "壁垒", "Berserk": "狂暴",
    "Bludgeon": "重锤", "Brutality": "残暴", "Corruption": "腐化", "Demon Form": "恶魔形态",
    "Double Tap": "双发", "Exhume": "发掘", "Feed": "吞噬", "Fiend Fire": "恶魔之焰",
    "Immolate": "焚灭", "Impervious": "岿然不动", "Juggernaut": "势不可挡",
    "Limit Break": "突破极限", "Offering": "祭品", "Reaper": "死灵之镰",
}
RELIC_NAME_ZH = {
    "Burning Blood": "燃烧之血", "Toy Ornithopter": "玩具扑翼机", "WingedGreaves": "飞鞋",
    "Winged Greaves": "飞鞋", "Lizard Tail": "蜥蜴尾巴", "Eternal Feather": "永恒羽毛",
    "Oddly Smooth Stone": "奇怪的平滑石", "Incense Burner": "香炉", "Black Blood": "黑血",
}
POTION_NAME_ZH = {
    "Fire Potion": "火焰药水", "Weak Potion": "虚弱药水", "Skill Potion": "技能药水",
    "Speed Potion": "速度药水", "Power Potion": "能力药水", "Colorless Potion": "无色药水",
    "Block Potion": "格挡药水", "Explosive Potion": "爆炸药水", "Blessing Of The Forge": "锻造祝福",
    "Swift Potion": "迅捷药水", "EMPTY_POTION_SLOT": "空药水槽",
}
EVENT_NAME_ZH = {
    "NEOW": "涅奥", "Golden Wing": "金色翅膀", "Golden Idol": "黄金神像",
    "Back to Basics": "返璞归真", "The Mausoleum": "陵墓", "The Library": "图书馆",
    "Mindbloom": "心灵绽放",
}
ROOM_NAME_ZH = {
    "MONSTER": "普通战斗", "ELITE": "精英战斗", "BOSS": "首领战", "REST": "营火",
    "SHOP": "商店", "TREASURE": "宝箱", "EVENT": "问号事件", "INVALID": "未知节点",
}
ENEMY_NAME_ZH = {
    "AcidSlime_L": "大型酸液史莱姆", "AcidSlime_M": "中型酸液史莱姆", "AcidSlime_S": "小型酸液史莱姆",
    "AwakenedOne": "觉醒者", "BanditBear": "盗贼熊", "SlaverBlue": "蓝衣大盗",
    "BookOfStabbing": "刺人书", "BronzeAutomaton": "铜制机械人", "BronzeOrb": "铜球",
    "Byrd": "鸟形怪", "Centurion": "百夫长", "Chosen": "被选中的", "CorruptHeart": "腐化之心",
    "Cultist": "邪教徒", "Dagger": "匕首", "Darkling": "黑暗之影", "Deca": "德卡",
    "Donu": "多努", "Exploder": "爆炸机器人", "GremlinFat": "胖地精", "FungiBeast": "真菌兽",
    "GiantHead": "巨大头颅", "FuzzyLouseNormal": "攻击型毛毛虫", "FuzzyLouseDefensive": "防御型毛毛虫",
    "GremlinLeader": "地精首领", "GremlinNob": "地精大君", "GremlinWizard": "地精法师",
    "Hexaghost": "六火亡魂", "JawWorm": "大颚虫", "Jaw Worm": "大颚虫", "Lagavulin": "拉格瓦林",
    "Looter": "掠夺者", "GremlinWarrior": "地精战士", "Mugger": "小偷", "Healer": "治疗者",
    "Nemesis": "涅墨西斯", "Orb Walker": "球状行者", "OrbWalker": "球状行者",
    "SlaverRed": "红衣大盗", "Reptomancer": "蛇发女妖", "Repulsor": "排斥者", "Sentry": "哨卫",
    "Shelled Parasite": "带壳寄生物", "Snecko": "蛇怪", "SphericGuardian": "球形守护者",
    "Spheric Guardian": "球形守护者", "Spiker": "刺猬", "SpikeSlime_L": "大型尖刺史莱姆",
    "SpikeSlime_M": "中型尖刺史莱姆", "SpikeSlime_S": "小型尖刺史莱姆", "SpireGrowth": "尖塔生长者",
    "SpireShield": "尖塔之盾", "SpireSpear": "尖塔之矛", "SlaverBoss": "奴隶贩子首领",
    "Champ": "冠军", "TheCollector": "收藏家", "TheGuardian": "守护者", "TimeEater": "时间吞噬者",
    "TorchHead": "火把头", "Transient": "瞬变体", "WrithingMass": "扭曲团块",
    "Acid Slime": "酸液史莱姆", "Blue Slaver": "蓝衣大盗", "Red Slaver": "红衣大盗",
    "Gremlin Nob": "地精大君", "Gremlin Leader": "地精首领", "The Guardian": "守护者",
    "Hexaghost": "六火亡魂", "Shell Parasite": "带壳寄生物", "Shelled Parasite": "带壳寄生物",
    "GremlinThief": "地精小偷", "GremlinFat": "胖地精", "GremlinWarrior": "地精战士",
    "Giant Head": "巨大头颅", "TheCollector": "收藏家", "The Collector": "收藏家",
}

COLOR = {
    "hp": "#d64545", "gold": "#c58a00", "gain": "#2e8b57", "loss": "#777777",
    "card": "#2878b5", "relic": "#8a4fba", "potion": "#b35c00", "muted": "#777777",
}


def _colored(text, color):
    return f'<span style="color:{color}">{escape(str(text))}</span>'


def display_name(value, kind=None):
    """Return a stable Chinese display name for a card/relic/potion/event ID."""
    raw = str(value)
    if kind == "card" or raw in CARD_NAME_ZH:
        return CARD_NAME_ZH.get(raw, raw)
    if kind == "relic" or raw in RELIC_NAME_ZH:
        return RELIC_NAME_ZH.get(raw, raw)
    if kind == "potion" or raw in POTION_NAME_ZH:
        return POTION_NAME_ZH.get(raw, raw)
    return EVENT_NAME_ZH.get(raw, raw)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _snapshot(env):
    """Copy the backend canonical state without retaining native objects."""
    thick = getattr(getattr(env, "unwrapped", env), "thick", None)
    if thick is None:
        return {}
    return _jsonable(copy.deepcopy(thick.obs_canonical()))


def restore_combat_snapshot(env, path, *, expected_sha256=None):
    """Restore an opt-in evaluation artifact into an identically configured env.

    Seed and committed actions rebuild native callbacks through normal gameplay;
    the environment verifies its observation and mask before returning. These
    offline artifacts contain hidden state and must never be strategy inputs.
    """
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('combat snapshot checksum mismatch')
    artifact = json.loads(data)
    if artifact.get('schema_version') != 1 or artifact.get('kind') != 'evaluation-combat-start':
        raise ValueError('unsupported combat snapshot schema')
    state = artifact['environment_state']
    # Schema-1 thin pending decisions are tuples; JSON has only arrays. Native
    # pending actions and journal tuples are normalized by restore_training_state.
    if state['pending'] is not None:
        state['pending'] = tuple(state['pending'])
    return env.unwrapped.restore_training_state(state)


def _global(state):
    return state.get("global", state)


def _counter_delta(before, after):
    left = Counter(json.dumps(item, sort_keys=True) for item in (before or []))
    right = Counter(json.dumps(item, sort_keys=True) for item in (after or []))
    gained, lost = [], []
    for item, count in (right - left).items():
        gained.extend([json.loads(item)] * count)
    for item, count in (left - right).items():
        lost.extend([json.loads(item)] * count)
    return gained, lost


def state_diff(before, after):
    """Return typed resource changes used by event/combat/shop summaries."""
    bg, ag = _global(before), _global(after)
    result = {}
    for key in ("hp", "max_hp", "gold"):
        if bg.get(key) != ag.get(key):
            result[key] = {"before": bg.get(key), "after": ag.get(key),
                           "delta": (ag.get(key) or 0) - (bg.get(key) or 0)}
    for key in ("deck", "relics", "potions"):
        gained, lost = _counter_delta(bg.get(key, []), ag.get(key, []))
        if gained or lost:
            result[key] = {"gained": gained, "lost": lost}
    return result


def _phase(state):
    return state.get("phase")


def _position(state):
    current = _global(state)
    return {"act": current.get("act"), "floor": current.get("floor", state.get("floor"))}


def _persistent_state(state):
    current = _global(state)
    return {"phase": state.get("phase"), **{
        key: copy.deepcopy(current.get(key))
        for key in ("act", "floor", "hp", "max_hp", "gold", "deck", "relics", "potions")
    }}


def _action_label(env, action, native_action):
    if native_action is not None:
        return "native combat action"
    try:
        return str(env.describe_action(int(action)))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return f"action {action}"


def _event_index(label):
    match = re.search(r"event option (\d+)", label)
    if match:
        return int(match.group(1))
    match = re.search(r"Neow option (\d+)", label)
    return int(match.group(1)) - 1 if match else None


class EpisodeTraceCollector:
    """Collect one sparse run history.

    ``before_step`` and ``after_step`` are called by :func:`runner.evaluate`.
    A new combat/event/shop record is opened lazily from canonical phase state,
    which also makes this work with native complete combat actions.
    """

    def __init__(self, env, episode: int, seed: int, *, combat_snapshot_output=None):
        self.env = env
        self.episode = int(episode)
        self.seed = int(seed)
        self.combat_snapshot_output = (None if combat_snapshot_output is None
                                       else Path(combat_snapshot_output))
        self.map_path = []
        self.events = []
        self.combats = []
        self.shops = []
        self.rewards = []
        self.chests = []
        self.boss_relics = []
        self.actions = []
        self.timeline = []
        self._active_combat = None
        self._last_before = None
        self._last_after = None
        self._last_entry = None
        self._pending_shop_remove = None
        self._started = _snapshot(env)

    def before_step(self, action, native_action=None):
        state = _snapshot(self.env)
        self._last_before = state
        phase = _phase(state)
        label = _action_label(self.env, action, native_action)
        entry = {"phase": phase, "action": label, **_position(state)}
        if action is not None:
            entry["action_id"] = int(action)
        if native_action is not None:
            entry["native"] = True
        if phase == "map":
            node = _map_node(state, label)
            if node is not None:
                node.update(_position(state))
                self.map_path.append(node)
                entry["node"] = node
        elif phase == "event":
            entry.update(_event_entry(state, label))
        elif phase == "shop":
            entry["item"] = _shop_item(state, label)
        elif phase == "combat_reward":
            entry["reward"] = _reward_item(state, label)
        elif phase == "boss_relic":
            entry["candidates"] = list(_global(state).get("boss_relics", []))
            index = _boss_index(label)
            if index is not None and index < len(entry["candidates"]):
                entry["selected"] = entry["candidates"][index]
        self._last_entry = entry
        # Combat details are represented by start/end snapshots only.
        if phase != "combat":
            self.actions.append(entry)

        if phase == "combat" and self._active_combat is None:
            monsters = [m.get("id") for m in state.get("monsters", [])]
            self._active_combat = {
                "act": _global(state).get("act"),
                "floor": _global(state).get("floor", state.get("floor")),
                "encounter": monsters,
                "pre_hp": _global(state).get("hp", state.get("player", {}).get("hp")),
                "pre_state": state,
            }
            if self.combat_snapshot_output is not None:
                started = time.perf_counter()
                env = self.env.unwrapped
                artifact = {
                    'schema_version': 1, 'kind': 'evaluation-combat-start',
                    'episode': self.episode, 'seed': self.seed,
                    'combat_index': len(self.combats),
                    'position': _position(state),
                    'environment_options': {
                        'combat_action_contract': env.combat_action_contract,
                        'ascension': int(env.thick._ascension),
                        'character': str(env.thick._character),
                        **{name: bool(getattr(env, '_' + name)) for name in (
                            'global_map_view', 'card_language', 'combat_language',
                            'deck_size_observation', 'neow_observation',
                            'event_option_observation', 'interactive_reasoning')},
                    },
                    'environment_state': env.export_training_state(),
                    'canonical_state': state,
                    # Keep actual order here. The separate public decision
                    # trace deliberately sorts hidden draw cards.
                    'native_state': env.thick.bc.to_dict(),
                }
                data = (json.dumps(_jsonable(artifact), sort_keys=True, ensure_ascii=False,
                                   allow_nan=False) + '\n').encode('utf-8')
                self.combat_snapshot_output.mkdir(parents=True, exist_ok=True)
                path = self.combat_snapshot_output / (
                    f'seed-{self.seed}-episode-{self.episode:06d}-combat-{len(self.combats):03d}.json')
                # An interrupted write has no completed trajectory reference;
                # refuse overwrite so a rerun cannot replace existing evidence.
                with path.open('xb') as stream:
                    stream.write(data)
                self._active_combat['snapshot'] = {
                    'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
                    'capture_seconds': time.perf_counter() - started,
                }

    def after_step(self, action, reward, terminated, truncated, info,
                   native_action=None):
        state = _snapshot(self.env)
        self._last_after = state
        before = self._last_before or {}
        phase = _phase(before)
        after_phase = _phase(state)
        diff = state_diff(before, state)
        if self._last_entry is not None and phase != "combat":
            self._last_entry["diff"] = diff
        if phase == "event":
            event = self._last_entry if self._last_entry and self._last_entry.get("phase") == "event" else {}
            self.events.append({**_position(before),
                "event": (before.get("options") or {}).get("event"),
                "action": event.get("action"),
                "action_id": event.get("action_id"),
                "option_index": event.get("option_index"),
                "choices": event.get("choices", []),
                "diff": diff,
            })
        if phase == "shop":
            entry = self._last_entry or {}
            item = entry.get("item")
            if item is not None or "buy" in str(entry.get("action", "")):
                transaction = {**_position(before), "action": entry.get("action"),
                               "item": item, "diff": diff}
                self.shops.append(transaction)
                if (item or {}).get("kind") == "remove":
                    self._pending_shop_remove = transaction
        elif phase == "card_select" and self._pending_shop_remove is not None:
            self._pending_shop_remove["selection"] = (self._last_entry or {}).get("action")
            self._pending_shop_remove["selection_diff"] = diff
            self._pending_shop_remove = None
        if phase == "combat_reward":
            entry = self._last_entry or {}
            item = entry.get("reward")
            self.rewards.append({**_position(before), "action": entry.get("action"),
                                 "item": item,
                                 "offers": copy.deepcopy(before.get("options") or {}),
                                 "diff": diff})
        if phase == "chest":
            entry = self._last_entry or {}
            self.chests.append({**_position(before), "action": entry.get("action"), "diff": diff})
        if phase == "boss_relic":
            entry = self._last_entry or {}
            self.boss_relics.append({**_position(before), "action": entry.get("action"),
                                    "candidates": entry.get("candidates", []),
                                    "selected": entry.get("selected"),
                                    "diff": diff})

        if self._active_combat is not None and after_phase != "combat":
            record = self._active_combat
            record.update({
                "post_hp": _global(state).get("hp", state.get("player", {}).get("hp")),
                "hp_loss": (record.get("pre_hp") or 0) - (_global(state).get("hp", 0) or 0),
                "reward_candidates": (copy.deepcopy(state.get("options") or {})
                                      if after_phase == "combat_reward" else {}),
                "post_state": state,
                "diff": state_diff(record["pre_state"], state),
            })
            record.pop("pre_state", None)
            record.pop("post_state", None)
            self.combats.append(record)
            self.timeline.append({"kind": "combat", **record})
            self._active_combat = None

        # Every non-combat decision is a compact point on the chronological
        # timeline.  Combat itself is intentionally represented only by the
        # boundary summary above, never by its individual card actions.
        if phase != "combat" and self._last_entry is not None:
            self.timeline.append({"kind": "action", **copy.deepcopy(self._last_entry)})

    def finish(self, result=None):
        final = self._last_after or _snapshot(self.env)
        if self._active_combat is not None:
            # A truncated evaluator may stop while a battle is live; preserve a
            # clear incomplete marker instead of inventing a post-combat HP.
            self._active_combat["incomplete"] = True
            self.combats.append(self._active_combat)
            self._active_combat = None
        return {
            "seed": self.seed,
            "result": result,
            "map_path": self.map_path,
            "question_mark_events": [event for event in self.events if event.get("event") != "NEOW"],
            "events": self.events,
            "combats": self.combats,
            "combat_snapshot_seconds": sum(
                row.get('snapshot', {}).get('capture_seconds', 0.0) for row in self.combats),
            "shops": self.shops,
            "rewards": self.rewards,
            "chests": self.chests,
            "boss_relics": self.boss_relics,
            "actions": self.actions,
            "timeline": self.timeline,
            "final_state": _persistent_state(final),
        }


def _map_node(state, label):
    match = re.search(r"x=(-?\d+)", label)
    if not match:
        return None
    x = int(match.group(1))
    floor = _global(state).get("floor", state.get("floor"))
    for node in (state.get("options") or {}).get("nodes", []):
        if isinstance(node, (list, tuple)) and int(node[0]) == x:
            return {"x": x, "room": node[1], "from_floor": floor,
                    "floor": (floor or 0) + 1}
    return {"x": x, "from_floor": floor, "floor": (floor or 0) + 1}


def _event_entry(state, label):
    options = state.get("options") or {}
    choices = options.get("event_choices") or options.get("neow", [])
    return {"event": options.get("event"), "option_index": _event_index(label),
            "choices": choices,
            "event_options": options.get("event_options", [])}


def _shop_item(state, label):
    if label == "pay to remove a card":
        return {"kind": "remove", "cost": (state.get("options") or {}).get("shop_remove_cost")}
    match = re.search(r"buy (card|relic|potion) #(\d+)", label)
    if not match:
        return None
    kind, raw_index = match.group(1), int(match.group(2))
    key = {"card": "shop_cards", "relic": "shop_relics", "potion": "shop_potions"}[kind]
    values = (state.get("options") or {}).get(key, [])
    return {"kind": kind, "index": raw_index,
            "offer": values[raw_index] if raw_index < len(values) else None}


def _reward_item(state, label):
    options = state.get("options") or {}
    match = re.search(r"claim (gold|relic|potion) #(\d+)", label)
    if match:
        kind, index = match.group(1), int(match.group(2))
        values = options.get({"gold": "gold", "relic": "relics", "potion": "potions"}[kind], [])
        return {"kind": kind, "index": index,
                "offer": values[index] if index < len(values) else None}
    match = re.search(r"take card: (.+)$", label)
    if match:
        return {"kind": "card", "offer": match.group(1)}
    return None


def _boss_index(label):
    match = re.search(r"boss relic #(\d+)", label)
    return int(match.group(1)) if match else None


def _item_text(item, kind=None):
    if item is None:
        return "未知"
    if isinstance(item, list):
        if kind in {"relic", "potion"} and len(item) >= 2 and isinstance(item[-1], (int, float)):
            return display_name(item[0], kind)
        if len(item) >= 2 and isinstance(item[1], int):
            name = display_name(item[0], kind)
            return f"{name}{'+' * item[1]}"
        return "/".join(_item_text(x, kind) for x in item)
    return display_name(item, kind)


def _diff_text(diff):
    parts = []
    for key in ("hp", "max_hp", "gold"):
        change = diff.get(key)
        if change:
            label = {"hp": "生命", "max_hp": "最大生命", "gold": "金币"}[key]
            color = COLOR["gold"] if key == "gold" else COLOR["hp"]
            parts.append(_colored(
                f"{label} {change['before']}→{change['after']}（{change['delta']:+}）", color))
    for key, singular in (("deck", "card"), ("relics", "relic"), ("potions", "potion")):
        change = diff.get(key, {})
        if change.get("gained"):
            parts.append(_colored("获得：" + ", ".join(_item_text(x, singular) for x in change["gained"]),
                                 COLOR[singular]))
        if change.get("lost"):
            parts.append(_colored("失去：" + ", ".join(_item_text(x, singular) for x in change["lost"]),
                                 COLOR["loss"]))
    return "；".join(parts) or "无持久状态变化"


def _action_text(entry):
    action = str(entry.get("action") or "")
    if action.startswith("event option "):
        return "选择事件选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("Neow option "):
        match = re.search(r"Neow option (\d+)", action)
        return "选择初始选项 " + (match.group(1) if match else "")
    if action.startswith("buy "):
        return "购买" + {"card": "卡牌", "relic": "遗物", "potion": "药水"}.get(
            action.split(" ")[1], "物品") + " " + action.rsplit("#", 1)[-1]
    if action == "pay to remove a card":
        return "删除一张牌"
    if action.startswith("claim "):
        return "领取" + {"gold": "金币", "relic": "遗物", "potion": "药水"}.get(
            action.split(" ")[1], "奖励")
    if action.startswith("take card:"):
        return "选择卡牌"
    if action.startswith("take boss relic"):
        return "选择首领遗物 " + action.rsplit("#", 1)[-1]
    if action == "open chest":
        return "打开宝箱"
    if action.startswith("choose map node"):
        return "选择地图节点"
    if action.startswith("choose "):
        return "选择"
    return action.replace("native combat action", "战斗").replace("action ", "动作 ")


def _event_choice_text(event):
    index = event.get("option_index")
    choices = event.get("choices") or []
    if index is None:
        return ""
    for detail in choices:
        if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
            label = detail.get("label_zh") or detail.get("label") or detail.get("description_zh")
            if label:
                return f"：{escape(str(label))}"
    return f"（选项 {index}）"


def _format_offer(item):
    if item is None:
        return ""
    if isinstance(item, dict):
        kind, offer = item.get("kind"), item.get("offer")
        if kind == "gold":
            return _colored(f"{offer} 金币", COLOR["gold"])
        if kind == "remove":
            cost = item.get("cost")
            return "删除一张牌" + (f"（费用 {_colored(cost, COLOR['gold'])}）" if cost is not None else "")
        if offer is not None:
            text = _item_text(offer, kind)
            # Shop offers conventionally carry the price as their final field.
            if isinstance(offer, list) and offer and isinstance(offer[-1], (int, float)):
                text += f"（{_colored(offer[-1], COLOR['gold'])} 金币）"
            return _colored(text, COLOR.get(kind, COLOR["card"]))
        return _colored(_item_text(offer, kind), COLOR.get(kind, COLOR["card"]))
    return escape(_item_text(item))


def _timeline_from_legacy(trajectory):
    """Rebuild chronological entries for traces written before ``timeline``."""
    actions = list(trajectory.get("actions") or [])
    combats = list(trajectory.get("combats") or [])
    by_floor = {}
    for combat in combats:
        by_floor.setdefault((combat.get("act"), combat.get("floor")), []).append(combat)
    emitted = set()
    timeline = []
    for action in actions:
        key = (action.get("act"), action.get("floor"))
        if action.get("phase") == "combat_reward" and key in by_floor:
            for index, combat in enumerate(by_floor[key]):
                token = id(combat)
                if token not in emitted:
                    timeline.append({"kind": "combat", **combat})
                    emitted.add(token)
        timeline.append({"kind": "action", **action})
    for combat in combats:
        if id(combat) not in emitted:
            timeline.append({"kind": "combat", **combat})
    return timeline


def _timeline(trajectory):
    return trajectory.get("timeline") or _timeline_from_legacy(trajectory)


def render_episode_markdown_legacy(trajectory):
    """Render one compact, outside-combat history suitable for seed comparison."""
    final = trajectory.get("final_state", {})
    lines = [
        f"## Seed {trajectory.get('seed')} — {trajectory.get('result')}", "",
        f"Final: Act {final.get('act')} floor {final.get('floor')}, "
        f"HP {final.get('hp')}/{final.get('max_hp')}, gold {final.get('gold')}", "",
    ]
    route = trajectory.get("map_path", [])
    if route:
        lines.extend(["Route: " + " -> ".join(
            f"F{n.get('floor')} {n.get('room', '?')}" for n in route), ""])
    lines.append("Events:")
    for event in trajectory.get("events", []):
        choice = ""
        choices, index = event.get("choices") or [], event.get("option_index")
        if index is not None:
            for detail in choices:
                if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
                    choice = f"; typed choice={json.dumps(detail, ensure_ascii=False)}"
                    break
        lines.append(f"- A{event.get('act')}F{event.get('floor')} {event.get('event')}: "
                     f"{event.get('action')}{choice}; {_diff_text(event.get('diff', {}))}")
    if not trajectory.get("events"):
        lines.append("- none")
    lines.extend(["", "Combats:"])
    rewards_by_floor = {}
    for reward in trajectory.get("rewards", []):
        rewards_by_floor.setdefault((reward.get("act"), reward.get("floor")), []).append(reward)
    for combat in trajectory.get("combats", []):
        suffix = " incomplete" if combat.get("incomplete") else ""
        lines.append(f"- A{combat.get('act')}F{combat.get('floor')} "
                     f"{', '.join(combat.get('encounter') or ['unknown'])}: "
                     f"HP {combat.get('pre_hp')}->{combat.get('post_hp', '?')} "
                     f"(loss {combat.get('hp_loss', '?')}){suffix}; "
                     f"combat delta: {_diff_text(combat.get('diff', {}))}")
        candidates = combat.get("reward_candidates") or {}
        if candidates:
            lines.append("  - offers: " + json.dumps(candidates, ensure_ascii=False))
        for reward in rewards_by_floor.get((combat.get("act"), combat.get("floor")), []):
            if reward.get("item") is not None or reward.get("diff"):
                lines.append(f"  - reward {reward.get('action')}: "
                             f"{json.dumps(reward.get('item'), ensure_ascii=False)}; "
                             f"{_diff_text(reward.get('diff', {}))}")
    lines.extend(["", "Shops / chests / boss relics:"])
    for shop in trajectory.get("shops", []):
        detail = _diff_text(shop.get("diff", {}))
        if shop.get("selection_diff"):
            detail += "; " + _diff_text(shop["selection_diff"])
        lines.append(f"- shop A{shop.get('act')}F{shop.get('floor')}: {shop.get('action')}; "
                     f"{json.dumps(shop.get('item'), ensure_ascii=False)}; {detail}")
    for chest in trajectory.get("chests", []):
        lines.append(f"- chest A{chest.get('act')}F{chest.get('floor')}: "
                     f"{chest.get('action')}; {_diff_text(chest.get('diff', {}))}")
    for boss in trajectory.get("boss_relics", []):
        lines.append(f"- boss relic A{boss.get('act')}F{boss.get('floor')}: "
                     f"offers={boss.get('candidates')}; selected={boss.get('selected')}; "
                     f"{_diff_text(boss.get('diff', {}))}")
    if not any(trajectory.get(key) for key in ("shops", "chests", "boss_relics")):
        lines.append("- none")
    return "\n".join(lines) + "\n"


# Final helper bindings for the public renderer (defined after legacy helpers
# so old trace compatibility code cannot shadow the Chinese implementation).
def _action_text(entry):
    action = str(entry.get("action") or "")
    if action.startswith("event option "): return "选择事件选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("Neow option "):
        match = re.search(r"Neow option (\d+)", action)
        return "选择初始选项 " + (match.group(1) if match else "")
    if action.startswith("buy "): return "购买" + {"card":"卡牌","relic":"遗物","potion":"药水"}.get(action.split(" ")[1], "物品")
    if action == "pay to remove a card": return "删除一张牌"
    if action.startswith("claim "): return "领取" + {"gold":"金币","relic":"遗物","potion":"药水"}.get(action.split(" ")[1], "奖励")
    if action.startswith("take card:"): return "选择卡牌"
    if action.startswith("take boss relic"): return "选择首领遗物 " + action.rsplit("#", 1)[-1]
    if action == "open chest": return "打开宝箱"
    return action.replace("native combat action", "战斗").replace("action ", "动作 ")


def _event_choice_text(event):
    index = event.get("option_index")
    for detail in event.get("choices") or []:
        if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
            label = detail.get("label_zh") or detail.get("label") or detail.get("description_zh")
            if label: return f"：{escape(str(label))}"
    return f"（选项 {index}）" if index is not None else ""


def _format_offer(item):
    if item is None: return ""
    if isinstance(item, dict):
        kind, offer = item.get("kind"), item.get("offer")
        if kind == "gold": return _colored(f"{offer} 金币", COLOR["gold"])
        if kind == "remove":
            cost = item.get("cost")
            return "删除一张牌" + (f"（费用 {_colored(cost, COLOR['gold'])}）" if cost is not None else "")
        text = _item_text(offer, kind)
        if isinstance(offer, list) and offer and isinstance(offer[-1], (int, float)):
            text += f"（{_colored(offer[-1], COLOR['gold'])} 金币）"
        return _colored(text, COLOR.get(kind, COLOR["card"]))
    return escape(_item_text(item))


def _timeline_line(entry):
    act, floor = entry.get("act", "?"), entry.get("floor", "?")
    prefix = f"A{act} · 第 {floor} 层"
    if entry.get("kind") == "combat":
        encounter = "、".join(ENEMY_NAME_ZH.get(str(x), str(x)) for x in (entry.get("encounter") or ["未知敌人"]))
        pre, post, loss = entry.get("pre_hp"), entry.get("post_hp", "?"), entry.get("hp_loss", "?")
        hp = f"生命 {pre}→{post}" + (f"（{loss:+}）" if isinstance(loss, (int, float)) else "")
        text = f"{prefix}　⚔️ 战斗：{escape(encounter)}　{_colored(hp, COLOR['hp'])}"
        if entry.get("incomplete"): text += "（战斗未完成）"
        delta = _diff_text(entry.get("diff", {}))
        if delta != "无持久状态变化": text += "；" + delta
        offers = entry.get("reward_candidates") or {}
        offered = [_colored(_item_text(x, "card"), COLOR["card"]) for x in offers.get("cards", [])]
        offered += [_colored(_item_text(x, "relic"), COLOR["relic"]) for x in offers.get("relics", [])]
        offered += [_colored(_item_text(x, "potion"), COLOR["potion"]) for x in offers.get("potions", [])]
        if offers.get("gold"): offered.append(_colored(f"{offers['gold'][0]} 金币", COLOR["gold"]))
        if offered: text += "；奖励候选：" + "、".join(offered)
        return "- " + text
    phase, action = entry.get("phase"), _action_text(entry)
    if phase == "map" and entry.get("node"):
        node = entry["node"]; room = ROOM_NAME_ZH.get(node.get("room"), node.get("room", "未知节点"))
        text = f"{prefix}　🗺️ 前往 {room}（路线坐标 {node.get('x', '?')}）"
    elif phase == "event": text = f"{prefix}　❓ {EVENT_NAME_ZH.get(entry.get('event'), entry.get('event', '未知事件'))}：{action}{_event_choice_text(entry)}"
    elif phase == "shop": text = f"{prefix}　🛒 {action}" + (("：" + _format_offer(entry["item"])) if entry.get("item") is not None else "")
    elif phase == "combat_reward": text = f"{prefix}　🎁 {action}" + (("：" + _format_offer(entry["reward"])) if entry.get("reward") is not None else "")
    elif phase == "chest": text = f"{prefix}　🧰 打开宝箱"
    elif phase == "boss_relic":
        candidates = "、".join(_item_text(x, "relic") for x in entry.get("candidates", [])); selected = _item_text(entry.get("selected"), "relic") if entry.get("selected") is not None else "未选择"
        text = f"{prefix}　👑 首领遗物：候选 {escape(candidates) or '未知'}；选择 {_colored(selected, COLOR['relic'])}"
    else: text = f"{prefix}　{action}"
    diff = _diff_text(entry.get("diff", {}))
    if diff != "无持久状态变化": text += "；" + diff
    return "- " + text


# Keep the public renderer's helpers at the end of the module.  This is also
# deliberate compatibility for traces produced while the renderer was being
# migrated: all calls resolve these helpers at render time.
def _action_text(entry):
    action = str(entry.get("action") or "")
    if action.startswith("event option "):
        return "选择事件选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("Neow option "):
        match = re.search(r"Neow option (\d+)", action)
        return "选择初始选项 " + (match.group(1) if match else "")
    if action.startswith("buy "):
        return "购买" + {"card": "卡牌", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "物品")
    if action == "pay to remove a card":
        return "删除一张牌"
    if action.startswith("claim "):
        return "领取" + {"gold": "金币", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "奖励")
    if action.startswith("take card:"):
        return "选择卡牌"
    if action.startswith("take boss relic"):
        return "选择首领遗物 " + action.rsplit("#", 1)[-1]
    if action == "open chest":
        return "打开宝箱"
    return action.replace("native combat action", "战斗").replace("action ", "动作 ")


def _event_choice_text(event):
    index = event.get("option_index")
    for detail in event.get("choices") or []:
        if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
            label = detail.get("label_zh") or detail.get("label") or detail.get("description_zh")
            if label:
                return f"：{escape(str(label))}"
    return f"（选项 {index}）" if index is not None else ""


def _format_offer(item):
    if item is None:
        return ""
    if isinstance(item, dict):
        kind, offer = item.get("kind"), item.get("offer")
        if kind == "gold":
            return _colored(f"{offer} 金币", COLOR["gold"])
        if kind == "remove":
            cost = item.get("cost")
            return "删除一张牌" + (f"（费用 {_colored(cost, COLOR['gold'])}）" if cost is not None else "")
        text = _item_text(offer, kind)
        if isinstance(offer, list) and offer and isinstance(offer[-1], (int, float)):
            text += f"（{_colored(offer[-1], COLOR['gold'])} 金币）"
        return _colored(text, COLOR.get(kind, COLOR["card"]))
    return escape(_item_text(item))


def _timeline_line(entry):
    act, floor = entry.get("act", "?"), entry.get("floor", "?")
    prefix = f"A{act} · 第 {floor} 层"
    if entry.get("kind") == "combat":
        encounter = "、".join(ENEMY_NAME_ZH.get(str(x), str(x)) for x in (entry.get("encounter") or ["未知敌人"]))
        pre, post, loss = entry.get("pre_hp"), entry.get("post_hp", "?"), entry.get("hp_loss", "?")
        hp_text = f"生命 {pre}→{post}" + (f"（{loss:+}）" if isinstance(loss, (int, float)) else "")
        text = f"{prefix}　⚔️ 战斗：{escape(encounter)}　{_colored(hp_text, COLOR['hp'])}"
        if entry.get("incomplete"):
            text += "（战斗未完成）"
        delta = _diff_text(entry.get("diff", {}))
        if delta != "无持久状态变化":
            text += "；" + delta
        offers = entry.get("reward_candidates") or {}
        offered = [_colored(_item_text(x, "card"), COLOR["card"]) for x in offers.get("cards", [])]
        offered += [_colored(_item_text(x, "relic"), COLOR["relic"]) for x in offers.get("relics", [])]
        offered += [_colored(_item_text(x, "potion"), COLOR["potion"]) for x in offers.get("potions", [])]
        if offers.get("gold"):
            offered.append(_colored(f"{offers['gold'][0]} 金币", COLOR["gold"]))
        if offered:
            text += "；奖励候选：" + "、".join(offered)
        return "- " + text
    phase, action = entry.get("phase"), _action_text(entry)
    if phase == "map" and entry.get("node"):
        node = entry["node"]
        room = ROOM_NAME_ZH.get(node.get("room"), node.get("room", "未知节点"))
        text = f"{prefix}　🗺️ 前往 {room}（路线坐标 {node.get('x', '?')}）"
    elif phase == "event":
        text = f"{prefix}　❓ {EVENT_NAME_ZH.get(entry.get('event'), entry.get('event', '未知事件'))}：{action}{_event_choice_text(entry)}"
    elif phase == "shop":
        text = f"{prefix}　🛒 {action}" + (("：" + _format_offer(entry["item"])) if entry.get("item") is not None else "")
    elif phase == "combat_reward":
        text = f"{prefix}　🎁 {action}" + (("：" + _format_offer(entry["reward"])) if entry.get("reward") is not None else "")
    elif phase == "chest":
        text = f"{prefix}　🧰 打开宝箱"
    elif phase == "boss_relic":
        candidates = "、".join(_item_text(x, "relic") for x in entry.get("candidates", []))
        selected = _item_text(entry.get("selected"), "relic") if entry.get("selected") is not None else "未选择"
        text = f"{prefix}　👑 首领遗物：候选 {escape(candidates) or '未知'}；选择 {_colored(selected, COLOR['relic'])}"
    else:
        text = f"{prefix}　{action}"
    diff = _diff_text(entry.get("diff", {}))
    if diff != "无持久状态变化":
        text += "；" + diff
    return "- " + text


def _timeline_from_legacy(trajectory):
    actions = list(trajectory.get("actions") or [])
    combats = list(trajectory.get("combats") or [])
    by_floor = {}
    for combat in combats:
        by_floor.setdefault((combat.get("act"), combat.get("floor")), []).append(combat)
    emitted = set()
    timeline = []
    for action in actions:
        key = (action.get("act"), action.get("floor"))
        if action.get("phase") == "combat_reward" and key in by_floor:
            for combat in by_floor[key]:
                token = id(combat)
                if token not in emitted:
                    timeline.append({"kind": "combat", **combat})
                    emitted.add(token)
        timeline.append({"kind": "action", **action})
    for combat in combats:
        if id(combat) not in emitted:
            timeline.append({"kind": "combat", **combat})
    return timeline


def _action_text(entry):
    action = str(entry.get("action") or "")
    if action.startswith("event option "):
        return "选择事件选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("Neow option "):
        return "选择初始选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("buy "):
        return "购买" + {"card": "卡牌", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "物品")
    if action == "pay to remove a card":
        return "删除一张牌"
    if action.startswith("claim "):
        return "领取" + {"gold": "金币", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "奖励")
    if action.startswith("take card:"):
        return "选择卡牌"
    if action.startswith("take boss relic"):
        return "选择首领遗物 " + action.rsplit("#", 1)[-1]
    if action == "open chest":
        return "打开宝箱"
    return action.replace("native combat action", "战斗").replace("action ", "动作 ")


def _event_choice_text(event):
    index = event.get("option_index")
    for detail in event.get("choices") or []:
        if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
            label = detail.get("label_zh") or detail.get("label") or detail.get("description_zh")
            if label:
                return f"：{escape(str(label))}"
    return f"（选项 {index}）" if index is not None else ""


def _format_offer(item):
    if item is None:
        return ""
    if isinstance(item, dict):
        kind, offer = item.get("kind"), item.get("offer")
        if kind == "remove":
            cost = item.get("cost")
            return "删除一张牌" + (f"（费用 {_colored(cost, COLOR['gold'])}）" if cost is not None else "")
        text = _item_text(offer, kind)
        if isinstance(offer, list) and offer and isinstance(offer[-1], (int, float)):
            text += f"（{_colored(offer[-1], COLOR['gold'])} 金币）"
        return _colored(text, COLOR.get(kind, COLOR["card"]))
    return escape(_item_text(item))


def _timeline_line(entry):
    act, floor = entry.get("act", "?"), entry.get("floor", "?")
    prefix = f"A{act} · 第 {floor} 层"
    if entry.get("kind") == "combat":
        encounter = "、".join(ENEMY_NAME_ZH.get(str(x), str(x))
                              for x in (entry.get("encounter") or ["未知敌人"]))
        pre, post, loss = entry.get("pre_hp"), entry.get("post_hp", "?"), entry.get("hp_loss", "?")
        hp_text = f"生命 {pre}→{post}" + (f"（{loss:+}）" if isinstance(loss, (int, float)) else "")
        text = f"{prefix}　⚔️ 战斗：{escape(encounter)}　{_colored(hp_text, COLOR['hp'])}"
        if entry.get("incomplete"):
            text += "（战斗未完成）"
        delta = _diff_text(entry.get("diff", {}))
        if delta != "无持久状态变化":
            text += "；" + delta
        offers = entry.get("reward_candidates") or {}
        offered = [_colored(_item_text(x, "card"), COLOR["card"]) for x in offers.get("cards", [])]
        offered += [_colored(_item_text(x, "relic"), COLOR["relic"]) for x in offers.get("relics", [])]
        offered += [_colored(_item_text(x, "potion"), COLOR["potion"]) for x in offers.get("potions", [])]
        if offers.get("gold"):
            offered.append(_colored(f"{offers['gold'][0]} 金币", COLOR["gold"]))
        if offered:
            text += "；奖励候选：" + "、".join(offered)
        return "- " + text
    phase, action = entry.get("phase"), _action_text(entry)
    if phase == "map" and entry.get("node"):
        node = entry["node"]
        room = ROOM_NAME_ZH.get(node.get("room"), node.get("room", "未知节点"))
        text = f"{prefix}　🗺️ 前往 {room}（路线坐标 {node.get('x', '?')}）"
    elif phase == "event":
        text = f"{prefix}　❓ {EVENT_NAME_ZH.get(entry.get('event'), entry.get('event', '未知事件'))}：{action}{_event_choice_text(entry)}"
    elif phase == "shop":
        text = f"{prefix}　🛒 {action}" + (("：" + _format_offer(entry["item"])) if entry.get("item") is not None else "")
    elif phase == "combat_reward":
        text = f"{prefix}　🎁 {action}" + (("：" + _format_offer(entry["reward"])) if entry.get("reward") is not None else "")
    elif phase == "chest":
        text = f"{prefix}　🧰 打开宝箱"
    elif phase == "boss_relic":
        candidates = "、".join(_item_text(x, "relic") for x in entry.get("candidates", []))
        selected = _item_text(entry.get("selected"), "relic") if entry.get("selected") is not None else "未选择"
        text = f"{prefix}　👑 首领遗物：候选 {escape(candidates) or '未知'}；选择 {_colored(selected, COLOR['relic'])}"
    else:
        text = f"{prefix}　{action}"
    diff = _diff_text(entry.get("diff", {}))
    if diff != "无持久状态变化":
        text += "；" + diff
    return "- " + text


def _action_text(entry):
    action = str(entry.get("action") or "")
    if action.startswith("event option "):
        return "选择事件选项 " + action.rsplit(" ", 1)[-1]
    if action.startswith("Neow option "):
        match = re.search(r"Neow option (\d+)", action)
        return "选择初始选项 " + (match.group(1) if match else "")
    if action.startswith("buy "):
        return "购买" + {"card": "卡牌", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "物品")
    if action == "pay to remove a card":
        return "删除一张牌"
    if action.startswith("claim "):
        return "领取" + {"gold": "金币", "relic": "遗物", "potion": "药水"}.get(action.split(" ")[1], "奖励")
    if action.startswith("take card:"):
        return "选择卡牌"
    if action.startswith("take boss relic"):
        return "选择首领遗物 " + action.rsplit("#", 1)[-1]
    if action == "open chest":
        return "打开宝箱"
    return action.replace("native combat action", "战斗").replace("action ", "动作 ")


def _event_choice_text(event):
    index = event.get("option_index")
    for detail in event.get("choices") or []:
        if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
            label = detail.get("label_zh") or detail.get("label") or detail.get("description_zh")
            if label:
                return f"：{escape(str(label))}"
    return f"（选项 {index}）" if index is not None else ""


def _format_offer(item):
    if item is None:
        return ""
    if isinstance(item, dict):
        kind, offer = item.get("kind"), item.get("offer")
        if kind == "gold":
            return _colored(f"{offer} 金币", COLOR["gold"])
        if kind == "remove":
            cost = item.get("cost")
            return "删除一张牌" + (f"（费用 {_colored(cost, COLOR['gold'])}）" if cost is not None else "")
        text = _item_text(offer, kind)
        if isinstance(offer, list) and offer and isinstance(offer[-1], (int, float)):
            text += f"（{_colored(offer[-1], COLOR['gold'])} 金币）"
        return _colored(text, COLOR.get(kind, COLOR["card"]))
    return escape(_item_text(item))


def _timeline_line(entry):
    act, floor = entry.get("act", "?"), entry.get("floor", "?")
    prefix = f"A{act} · 第 {floor} 层"
    if entry.get("kind") == "combat":
        encounter = "、".join(ENEMY_NAME_ZH.get(str(x), str(x)) for x in (entry.get("encounter") or ["未知敌人"]))
        pre, post, loss = entry.get("pre_hp"), entry.get("post_hp", "?"), entry.get("hp_loss", "?")
        hp_text = f"生命 {pre}→{post}" + (f"（{loss:+}）" if isinstance(loss, (int, float)) else "")
        text = f"{prefix}　⚔️ 战斗：{escape(encounter)}　{_colored(hp_text, COLOR['hp'])}"
        if entry.get("incomplete"):
            text += "（战斗未完成）"
        delta = _diff_text(entry.get("diff", {}))
        if delta != "无持久状态变化":
            text += "；" + delta
        offers = entry.get("reward_candidates") or {}
        offered = [_colored(_item_text(x, "card"), COLOR["card"]) for x in offers.get("cards", [])]
        offered += [_colored(_item_text(x, "relic"), COLOR["relic"]) for x in offers.get("relics", [])]
        offered += [_colored(_item_text(x, "potion"), COLOR["potion"]) for x in offers.get("potions", [])]
        if offers.get("gold"):
            offered.append(_colored(f"{offers['gold'][0]} 金币", COLOR["gold"]))
        if offered:
            text += "；奖励候选：" + "、".join(offered)
        return "- " + text
    phase, action = entry.get("phase"), _action_text(entry)
    if phase == "map" and entry.get("node"):
        node = entry["node"]
        room = ROOM_NAME_ZH.get(node.get("room"), node.get("room", "未知节点"))
        text = f"{prefix}　🗺️ 前往 {room}（路线坐标 {node.get('x', '?')}）"
    elif phase == "event":
        text = f"{prefix}　❓ {EVENT_NAME_ZH.get(entry.get('event'), entry.get('event', '未知事件'))}：{action}{_event_choice_text(entry)}"
    elif phase == "shop":
        text = f"{prefix}　🛒 {action}" + (("：" + _format_offer(entry["item"])) if entry.get("item") is not None else "")
    elif phase == "combat_reward":
        text = f"{prefix}　🎁 {action}" + (("：" + _format_offer(entry["reward"])) if entry.get("reward") is not None else "")
    elif phase == "chest":
        text = f"{prefix}　🧰 打开宝箱"
    elif phase == "boss_relic":
        candidates = "、".join(_item_text(x, "relic") for x in entry.get("candidates", []))
        selected = _item_text(entry.get("selected"), "relic") if entry.get("selected") is not None else "未选择"
        text = f"{prefix}　👑 首领遗物：候选 {escape(candidates) or '未知'}；选择 {_colored(selected, COLOR['relic'])}"
    else:
        text = f"{prefix}　{action}"
    diff = _diff_text(entry.get("diff", {}))
    if diff != "无持久状态变化":
        text += "；" + diff
    return "- " + text


def render_episode_markdown(trajectory):
    """Render a Chinese, chronological, outside-combat history."""
    final = trajectory.get("final_state", {})
    result = {"win": "胜利", "death": "死亡", "dead_screen": "死亡界面", "step_limit": "达到步数上限"}.get(
        trajectory.get("result"), trajectory.get("result"))
    hp_summary = _colored(f"{final.get('hp')}/{final.get('max_hp')}", COLOR['hp'])
    gold_summary = _colored(final.get('gold'), COLOR['gold'])
    lines = [f"## 第 {trajectory.get('seed')} 局　—　{result}", "",
             f"**最终状态**　Act {final.get('act')} · 第 {final.get('floor')} 层　生命 {hp_summary}　金币 {gold_summary}",
             "", "**时间线**", ""]
    timeline = trajectory.get("timeline") or _timeline_from_legacy(trajectory)
    if timeline:
        lines.extend(_timeline_line(entry) for entry in timeline)
    else:
        lines.append("- （没有记录到行动）")
    lines.extend(["", "<small>说明：战斗仅记录战斗前后生命与资源变化，不展开逐动作过程；生命为红色，金币为金色，获得/失去的牌、遗物和药水使用不同颜色。</small>"])
    return "\n".join(lines) + "\n"
