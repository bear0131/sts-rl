"""Native roots at the custom anti-stall rule's exact thresholds."""
from tests.bridge_fixtures import native_card, native_card_state


def punishment_battle(env, scenario='cards'):
    import slaythespire as sts
    relics = ['TungstenRod'] if scenario == 'mitigation' else []
    if scenario == 'revive': relics = ['Lizard Tail']
    if scenario == 'draw_hooks': relics = ['Centennial Puzzle', 'Runic Cube']
    env.thick.gc.set_relics(relics)
    env.thick.gc.set_potion_state(['Fairy Potion'] if scenario == 'fairy' else [])
    env.thick.bc = bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    bc.set_card_state(native_card_state(
        hand=[native_card('DEFEND_RED', i, cost=0) for i in range(3)],
        draw=[native_card('STRIKE_RED', i) for i in range(3, 9)]))
    bc.apply_custom_state({'turn': 19 if scenario == 'turns' else 0,
        'player': {'hp': 10 if scenario in ('fatal', 'revive', 'fairy') else 80,
                   'max_hp': 80, 'block': 999, 'energy': 10,
                   'counters': {'cards_played': 99}},
        'monsters': [{'hp': 1000, 'max_hp': 1000}]})
    bc.set_monster_status(0, 'STRENGTH', -99)
    if scenario == 'mitigation':
        bc.set_player_status('BUFFER', 3)
        bc.set_player_status('INTANGIBLE', 3)
    if scenario == 'autoplay': bc.set_player_status('BURST', 1)
    bc.heavenly_punishment = True
    env._invalidate_backend_cache()
    return bc
