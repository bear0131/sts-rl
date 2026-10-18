"""BRIDGE08: known real IDs must not become padding in checkpoint inputs."""
import pytest

from bridge.real_game_env import _potion_id
from bridge.schemas.combat_state import _card
from env.core.cards_vocab import CARD_INDEX
from env.core.vocab import POTION_VOCAB_INDEX


@pytest.mark.xfail(strict=True, reason="BRIDGE08: real potion IDs silently encode as padding")
@pytest.mark.parametrize("real_id,canonical", [
    ("SteroidPotion", "Flex Potion"),
    ("ColorlessPotion", "Colorless Potion"),
    ("SpeedPotion", "Speed Potion"),
    ("GamblersBrew", "Gambler's Brew"),
])
def test_real_potion_identity_matches_checkpoint_token(real_id, canonical):
    expected = POTION_VOCAB_INDEX[canonical]
    assert expected > 0
    assert POTION_VOCAB_INDEX.get(_potion_id(real_id), 0) == expected


@pytest.mark.xfail(strict=True, reason="BRIDGE08: observed colorless card IDs silently encode as padding")
@pytest.mark.parametrize("real_id,canonical", [
    ("HandOfGreed", "Hand Of Greed"),
    ("PanicButton", "Panic Button"),
    ("Flash of Steel", "Flash Of Steel"),
    ("Master of Strategy", "Master Of Strategy"),
])
def test_real_colorless_identity_matches_checkpoint_token(real_id, canonical):
    expected = CARD_INDEX[canonical]
    assert expected > 0
    assert CARD_INDEX.get(_card(real_id, 0)["id"], 0) == expected
