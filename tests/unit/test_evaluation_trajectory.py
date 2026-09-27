import numpy as np

from rl.evaluation.trajectory import EpisodeTraceCollector, render_episode_markdown, state_diff


def _state(phase, *, hp=70, gold=99, deck=None, relics=None, potions=None,
           options=None, act=1, floor=1):
    return {"phase": phase, "options": options or {}, "global": {
        "hp": hp, "max_hp": 80, "gold": gold, "deck": deck or [["Strike_R", 0]],
        "relics": relics or [], "potions": potions or [], "act": act,
        "floor": floor, "boss_relics": [],
    }}


class FakeEnv:
    def __init__(self, states):
        self.states = iter(states)
        self.state = next(self.states)
        self.unwrapped = self
        self.thick = self

    def obs_canonical(self):
        return self.state

    def describe_action(self, action):
        return {1: "event option 0", 2: "buy card #1", 3: "open chest",
                4: "take boss relic #0"}.get(action, "combat action")


def test_state_diff_tracks_typed_resources_and_hp():
    before = _state("event", hp=50, gold=100, deck=[["Strike_R", 0]], potions=["Fire Potion"])
    after = _state("map", hp=42, gold=75, deck=[["Strike_R", 0], ["Bash", 0]],
                   relics=["Burning Blood"], potions=[])
    diff = state_diff(before, after)
    assert diff["hp"]["delta"] == -8
    assert diff["gold"]["delta"] == -25
    assert diff["deck"]["gained"] == [["Bash", 0]]
    assert diff["relics"]["gained"] == ["Burning Blood"]
    assert diff["potions"]["lost"] == ["Fire Potion"]


def test_event_trace_keeps_choices_and_multiple_steps():
    first = _state("event", options={"event": "Golden Idol",
                                      "event_choices": [{"index": 0, "label": "take"}],
                                      "event_options": [0]})
    second = _state("event", hp=60, options={"event": "Golden Idol",
                                               "event_choices": [{"index": 1, "label": "leave"}],
                                               "event_options": [1]})
    third = _state("map", hp=60)
    env = FakeEnv([first])
    collector = EpisodeTraceCollector(env, 0, 10)
    collector.before_step(1)
    env.state = second
    collector.after_step(1, 0, False, False, {})
    collector.before_step(1)
    env.state = third
    collector.after_step(1, 0, False, False, {})
    result = collector.finish("death")
    assert len(result["question_mark_events"]) == 2
    assert result["question_mark_events"][0]["event"] == "Golden Idol"
    assert result["question_mark_events"][0]["option_index"] == 0
    assert result["question_mark_events"][1]["option_index"] == 0


def test_combat_trace_records_only_boundary_hp():
    before = _state("combat", hp=70, floor=4)
    before["monsters"] = [{"id": "Jaw Worm"}]
    after = _state("combat_reward", hp=61, floor=4)
    env = FakeEnv([before])
    collector = EpisodeTraceCollector(env, 0, 11)
    collector.before_step(9)
    env.state = after
    collector.after_step(9, 0, False, False, {})
    row = collector.finish("death")
    assert row["combats"][0]["pre_hp"] == 70
    assert row["combats"][0]["post_hp"] == 61
    assert row["combats"][0]["hp_loss"] == 9
    assert "actions" in row


def test_markdown_uses_chinese_names_and_colored_resources():
    trajectory = {
        "seed": 1,
        "result": "death",
        "timeline": [
            {"kind": "action", "phase": "shop", "act": 1, "floor": 2,
             "action": "buy card #0", "item": {"kind": "card", "offer": ["Combust", 0, 50]},
             "diff": {"gold": {"before": 99, "after": 49, "delta": -50}}},
            {"kind": "combat", "act": 1, "floor": 3, "encounter": ["Jaw Worm"],
             "pre_hp": 70, "post_hp": 60, "hp_loss": 10, "diff": {}},
        ],
        "final_state": {"act": 1, "floor": 3, "hp": 60, "max_hp": 80, "gold": 49},
    }
    rendered = render_episode_markdown(trajectory)
    assert "自燃" in rendered
    assert "燃烧" not in rendered.replace("自燃", "")
    assert "#c58a00" in rendered  # gold
    assert "#d64545" in rendered  # HP
    assert "**时间线**" in rendered
    assert "Events:" not in rendered
    assert "Combats:" not in rendered


def test_legacy_trace_is_rendered_in_chronological_order():
    trajectory = {
        "seed": 2, "result": "win", "actions": [
            {"phase": "map", "act": 1, "floor": 0, "action": "go to map node x=1",
             "node": {"x": 1, "room": "MONSTER"}, "diff": {}},
            {"phase": "combat_reward", "act": 1, "floor": 1, "action": "claim gold #0",
             "reward": {"kind": "gold", "offer": 12}, "diff": {"gold": {"before": 99, "after": 111, "delta": 12}}},
        ],
        "combats": [{"act": 1, "floor": 1, "encounter": ["Jaw Worm"], "pre_hp": 70,
                      "post_hp": 65, "hp_loss": 5, "diff": {}}],
        "final_state": {"act": 1, "floor": 1, "hp": 65, "max_hp": 80, "gold": 111},
    }
    rendered = render_episode_markdown(trajectory)
    assert rendered.index("前往 普通战斗") < rendered.index("战斗：大颚虫") < rendered.index("领取金币")
