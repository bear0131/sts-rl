import random
import pytest

from rl.rules.data import make_vector_rule_training_data as teacher
from rl.rules.rule_vector_model import StateVectorConfig


@pytest.mark.parametrize("card,destination", [
    ("Carnage", "discard"), ("Impervious", "exhaust"), ("Inflame", None),
])
@pytest.mark.xfail(reason="F04: approximate teacher does not conserve card identities",
                   strict=True, raises=AssertionError)
def test_played_card_moves_to_correct_zone(monkeypatch, card, destination):
    action = {"id": card, "up": 0}
    rng = random.Random(0)
    state = teacher._canonical_state(rng, 1, "Jaw Worm", "normal", [], [],
                                    action, [action])
    state["combat"]["player"].update(powers={}, energy=10)
    for zone in ("draw", "discard", "exhaust"):
        state["combat"][zone] = []
    original = teacher._make_event
    states = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result)
        return result

    monkeypatch.setattr(teacher, "_make_event", capture)
    teacher._effect_record(state, action, StateVectorConfig(), rng)
    final = states[-1]["combat"]
    assert final["hand"] == []
    for zone in ("discard", "exhaust"):
        assert final[zone] == ([action] if zone == destination else [])
