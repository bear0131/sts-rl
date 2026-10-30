"""Violence transfers physical cards even after generated attacks exceed 64."""
from collections import Counter

import pytest
import slaythespire as sts

from tests.bridge_fixtures import native_card, native_card_state

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('attacks', [0, 2, 10, 63, 64, 65, 128])
@pytest.mark.parametrize('upgraded', [0, 1])
@pytest.mark.parametrize('filler', [0, 8, 9])
def test_violence_long_pile_conservation_and_order(combat_factory, attacks, upgraded, filler):
    bc = combat_factory(['VIOLENCE'] * 5, 'Cultist', seed=42).thick.bc
    hand = [native_card('VIOLENCE', 0, upgrades=upgraded, base_cost=0, cost=0)]
    hand += [native_card('DEFEND_RED', i + 1) for i in range(filler)]
    draw = [native_card('ANGER', 100 + i) for i in range(attacks)]
    draw += [native_card('DEFEND_RED', 1000 + i) for i in range(7)]
    before = native_card_state(hand=hand, draw=draw, next_uid=2000)
    bc.set_card_state(before)
    sibling = bc.clone()
    action = sts.Action(sts.ActionType.CARD, 0)
    bc.execute(action)
    after = bc.card_state()

    moved = after['hand'][filler:] + after['discard']
    assert len(moved) == min(attacks, 3 + upgraded)
    assert all(c['id'] == 'ANGER' for c in moved)
    assert len(after['hand']) == min(10, filler + len(moved))
    assert after['hand'][:filler] == hand[1:]
    assert [c['uid'] for c in after['exhaust']] == [0]
    selected = {c['uid'] for c in moved}
    assert after['draw'] == [c for c in draw if c['uid'] not in selected]
    assert Counter(c['uid'] for zone in ('hand', 'draw', 'discard', 'exhaust')
                   for c in after[zone]) == Counter(c['uid'] for c in hand + draw)
    # Fixed pre-fix seed-42 choices: retain insertion/shuffle RNG consumption.
    if attacks == 10:
        assert [c['uid'] for c in moved] == [103, 109, 107, 100][:3 + upgraded]
    assert sibling.card_state() == before
    sibling.execute(action)
    assert sibling.card_state() == after
    assert sibling.nosl_future_key() == bc.nosl_future_key()
