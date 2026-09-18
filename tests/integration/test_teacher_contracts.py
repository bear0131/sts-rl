import copy
import random
import pytest

from rl.rules.data import make_vector_rule_training_data as teacher
from rl.rules.rule_vector_model import StateVectorConfig


@pytest.fixture
def teacher_effect(monkeypatch):
    original = teacher._make_event
    states = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result)
        return result

    monkeypatch.setattr(teacher, "_make_event", capture)

    def run(action, *, hand=None, draw=(), discard=(), powers=None):
        rng = random.Random(0)
        state = teacher._canonical_state(rng, 1, "Jaw Worm", "normal", [], [],
                                        action, hand if hand is not None else [action])
        state["combat"]["player"].update(powers=powers or {}, energy=10)
        state["combat"].update(draw=list(draw), discard=list(discard), exhaust=[])
        before = copy.deepcopy(state)
        teacher._effect_record(state, action, StateVectorConfig(), rng)
        assert state == before
        return states[-1]["combat"]

    return run


@pytest.mark.native
@pytest.mark.parametrize("up", [0, 1])
@pytest.mark.parametrize("card,destination", [
    ("Carnage", "discard"), ("Impervious", "exhaust"), ("Inflame", None),
    ("True Grit", "discard"), ("Burning Pact", "discard"), ("Sentinel", "discard"),
    ("Sever Soul", "discard"), ("Second Wind", "discard"), ("Seeing Red", "exhaust"),
    ("Limit Break", "upgrade-dependent"),
])
def test_played_card_moves_to_correct_zone(teacher_effect, combat_factory, card, destination, up):
    action = {"id": card, "up": up}
    if destination == "upgrade-dependent":
        destination = "discard" if up else "exhaust"
    final = teacher_effect(action)
    assert final["hand"] == []
    for zone in ("discard", "exhaust"):
        assert final[zone] == ([action] if zone == destination else [])

    # Separate oracle: execute the actual card, without copying teacher rules.
    import slaythespire as sts
    native_id = card.upper().replace(" ", "_")
    env = combat_factory([native_id] * 5)
    env.thick.bc.apply_custom_state({"player": {"energy": 10},
                                    "hand": [(native_id, up)], "draw": [],
                                    "discard": [], "exhaust": []})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    native = env.thick.bc.to_dict()
    assert native["hand"] == []
    for zone in ("discard", "exhaust"):
        assert [(c[0], c[1]) for c in native[zone]] == ([(native_id, up)] if zone == destination else [])


@pytest.mark.parametrize("card,zone,created", [
    ("Anger", "discard", {"id": "Anger", "up": 1}),
    ("Immolate", "discard", {"id": "Burn", "up": 0}),
    ("Wild Strike", "draw", {"id": "Wound", "up": 0}),
    ("Reckless Charge", "draw", {"id": "Dazed", "up": 0}),
])
def test_created_cards_have_the_required_identity(teacher_effect, card, zone, created):
    action = {"id": card, "up": 1}
    final = teacher_effect(action)
    assert final["hand"] == []
    assert final["discard"] == [action] + ([created] if zone == "discard" else [])
    assert final["draw"] == ([created] if zone == "draw" else [])
    assert final["exhaust"] == []


def test_draw_moves_exact_cards_instead_of_deleting_them(teacher_effect):
    action = {"id": "Pommel Strike", "up": 1}
    duplicate = dict(action)
    drawn = [{"id": "Defend_R", "up": 1}, {"id": "Bash", "up": 0}]
    final = teacher_effect(action, hand=[action, duplicate], draw=drawn)
    assert final["hand"] == [duplicate, drawn[1], drawn[0]]
    assert final["draw"] == [] and final["discard"] == [action]
    assert final["exhaust"] == []


def test_draw_after_shuffle_conserves_existing_discard(teacher_effect):
    action = {"id": "Pommel Strike", "up": 1}
    drawn = {"id": "Defend_R", "up": 1}
    shuffled = {"id": "Bash", "up": 0}
    final = teacher_effect(action, draw=[drawn], discard=[shuffled])
    assert final["hand"] == [drawn, shuffled]
    assert final["draw"] == [] and final["discard"] == [action]


def test_draw_respects_hand_capacity_without_losing_cards(teacher_effect):
    action = {"id": "Pommel Strike", "up": 1}
    held = [{"id": "Defend_R", "up": 0}] * 9
    drawn = [{"id": "Bash", "up": 0}, {"id": "Defend_R", "up": 1}]
    final = teacher_effect(action, hand=[action] + held, draw=drawn)
    assert final["hand"] == held + [drawn[1]]
    assert final["draw"] == drawn[:1] and final["discard"] == [action]


def test_double_tap_commits_only_one_played_card(teacher_effect):
    action = {"id": "Carnage", "up": 0}
    final = teacher_effect(action, powers={"Double Tap": 1})
    assert final["hand"] == [] and final["discard"] == [action]
    assert final["exhaust"] == []
