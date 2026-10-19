"""A real target index belongs to the client list, including gone monsters."""
import numpy as np
import pytest

from bridge.real_game_env import RealGameAdapter
from env.core import action_space as A, vocab as V
from schemas.combat_state import normalize_real, diff


@pytest.mark.parametrize('target', [1, 3, 4])
def test_real_observation_mask_and_command_share_client_slot(target):
    monsters = [dict(id='GremlinWarrior', current_hp=0, max_hp=20, block=0,
        is_gone=True, powers=[]) for _ in range(target)]
    monsters.append(dict(id='GremlinLeader', current_hp=140, max_hp=140, block=0,
        is_gone=False, powers=[]))
    gs = dict(room_phase='COMBAT', screen_type='NONE',
        combat_state=dict(turn=1, monsters=monsters,
            player=dict(current_hp=80, max_hp=80, block=0, energy=3, powers=[]),
            hand=[dict(id='Bash', upgrades=0, cost=2, has_target=True, is_playable=True)],
            draw_pile=[], discard_pile=[], exhaust_pile=[]))
    adapter = RealGameAdapter()
    adapter.update(dict(in_game=True, available_commands=['PLAY', 'END'], game_state=gs))
    assert len(normalize_real(gs)['monsters']) == target + 1
    obs = adapter.observation()
    assert obs['monster_id'][target] == V.MONSTER_VOCAB_INDEX['GremlinLeader']
    assert not np.any(obs['monster_scalars'][:target, 0])
    legal = adapter.legal_semantic_actions()
    semantic, pending = A.resolve(A._PLAY, legal, None, 'combat')
    if pending is not None:
        semantic, pending = A.resolve(A._TARGET + target, legal, pending, 'combat')
    assert pending is None
    assert adapter.command_for_semantic(semantic) == f'PLAY 1 {target}'
    # Differential comparison can discard gone positions, observation cannot.
    full = normalize_real(gs)
    live = dict(full, monsters=[full['monsters'][target]])
    assert diff(full, live) == []
