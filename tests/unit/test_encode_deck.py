"""Deck encoding preserves ordered slots, normalization and output ownership."""
import numpy as np
import pytest

from env.core import encode


def _observation(deck):
    return encode.encode({'global': {'deck': deck}}, deck_size_observation=True)


def test_deck_slots_upgrade_precision_and_padding_survive_input_changes():
    deck = [('Strike_R', 0), {'id': 'Bash', 'upgrades': 7}, ['Strike_R', 16777217]]
    first = _observation(deck)
    np.testing.assert_array_equal(
        first['deck_ids'][:3],
        [encode.CARD_INDEX['Strike_R'], encode.CARD_INDEX['Bash'],
         encode.CARD_INDEX['Strike_R']],
    )
    expected = np.asarray([0., 7 / encode.UPGRADE_SCALE,
                           16777217 / encode.UPGRADE_SCALE], dtype=np.float32)
    assert first['deck_upgrades'][:3].tobytes() == expected.tobytes()
    assert not first['deck_ids'][3:].any()
    assert not first['deck_upgrades'][3:].any()

    deck[1]['upgrades'] = 2
    deck[2][0] = 'Bash'
    deck.reverse()
    changed = _observation(deck)
    assert changed['deck_ids'][0] == encode.CARD_INDEX['Bash']
    assert changed['deck_upgrades'][1] == np.float32(2 / encode.UPGRADE_SCALE)
    assert first['deck_upgrades'][:3].tobytes() == expected.tobytes()
    short = _observation(deck[:1])
    assert not short['deck_ids'][1:].any()
    assert not short['deck_upgrades'][1:].any()


def test_repeated_deck_outputs_own_storage_and_keep_live_fields():
    canon = {'global': {'deck': [('Bash', 1)], 'gold': 31, 'hp': 40, 'max_hp': 80}}
    first = encode.encode(canon)
    first['deck_ids'].fill(0)
    first['deck_upgrades'].fill(99)
    canon['global'].update(gold=73, hp=19)
    second = encode.encode(canon)
    assert second['deck_ids'][0] == encode.CARD_INDEX['Bash']
    assert second['deck_upgrades'][0] == np.float32(1 / encode.UPGRADE_SCALE)
    assert second['gold'][0] == np.float32(73 / encode.GOLD_SCALE)
    assert second['run_hp'][0] == np.float32(19 / encode.HP_SCALE)
    assert not np.shares_memory(first['deck_ids'], second['deck_ids'])
    empty = _observation([])
    assert not empty['deck_ids'].any() and not empty['deck_upgrades'].any()


@pytest.mark.parametrize('deck,error', [
    ([('not-a-card', 0)], ValueError),
    ([(encode.EMPTY, 0)], ValueError),
    ([('Strike_R',)], ValueError),
    ([{'id': 'Strike_R'}], ValueError),
    ([('Strike_R', None)], TypeError),
    ([('Strike_R', 0)] * (encode.DECK_SLOTS + 1), ValueError),
])
def test_deck_encoding_does_not_accept_invalid_cards(deck, error):
    _observation([('Strike_R', 0)])
    with pytest.raises(error):
        _observation(deck)
