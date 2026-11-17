"""Decision boundaries from the native card/potion and trigger call-site audit."""
from tests.bridge_fixtures import native_card, native_card_state

SELECTION_CARDS = (
    'ARMAMENTS', 'BURNING_PACT', 'DISCOVERY', 'DUAL_WIELD', 'EXHUME',
    'FORETHOUGHT', 'HEADBUTT', 'PURITY', 'SECRET_TECHNIQUE', 'SECRET_WEAPON',
    'THINKING_AHEAD', 'TRUE_GRIT', 'WARCRY',
)
SELECTION_POTIONS = (
    'Attack Potion', 'Colorless Potion', 'Skill Potion', 'Power Potion',
    'Elixir Potion', 'Gamblers Brew', 'Liquid Memories',
)
TRIGGER_CASES = (
    'attack_relics', 'skill_relics', 'attack_powers', 'skill_powers',
    'juggernaut', 'void', 'evolve_fire', 'shuffle', 'sharp_hide',
    'flight', 'flight_last', 'hand_drill', 'envenom_sadistic', 'death_relics', 'stasis',
    'stasis_full', 'stasis_discard', 'darkling', 'awakened', 'leader',
    'double_tap', 'burst', 'duplication', 'echo_form', 'necronomicon',
    'havoc', 'distilled_chaos', 'mayhem', 'duplication_potion', 'exhaust_hooks', 'pain_draw',
)
AUDITED_CASES = (tuple(f'card:{card}:{up}' for card in SELECTION_CARDS for up in (0, 1))
    + tuple(f'potion:{potion}:{bark}' for potion in SELECTION_POTIONS for bark in (0, 1))
    + TRIGGER_CASES)


def large_card_pile(env, zone, count):
    """Keep every generated card while crossing the GPU vector's inline size."""
    import slaythespire as sts
    env.thick.gc.set_deck([('TRUE_GRIT', 1)] + [('STRIKE_RED', 0)] * 9)
    env.thick.gc.set_relics([])
    env.thick.gc.set_potion_state([])
    env.thick.bc = bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    piles = dict(hand=[native_card('TRUE_GRIT', 0, upgrades=1),
                       native_card('STRIKE_RED', 1), native_card('DEFEND_RED', 2)],
                 draw=[native_card('STRIKE_RED', 3), native_card('DEFEND_RED', 4)],
                 discard=[native_card('STRIKE_RED', 5)], exhaust=[])
    piles[zone] = [native_card('SLIMED', 32 + i) for i in range(count)]
    bc.set_card_state(native_card_state(**piles, next_uid=32 + count))
    bc.set_player_energy(5)
    bc.apply_custom_state({'monsters': [{'hp': 1000, 'max_hp': 1000}]})
    env._invalidate_backend_cache()
    return bc


def large_selection(env, zone, count):
    """Real selection handles at and beyond the 96-slot policy boundary."""
    import slaythespire as sts
    card = {'draw': 'SECRET_TECHNIQUE', 'discard': 'HEADBUTT', 'exhaust': 'EXHUME'}[zone]
    env.thick.gc.set_deck([(card, 0)] + [('DEFEND_RED', 0)] * 9)
    env.thick.gc.set_relics([])
    env.thick.gc.set_potion_state([])
    env.thick.bc = bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    env.thick.combat_exit_anchor_pending = False
    cards = ['DEFEND_RED'] * count if count == 96 else ['DEFEND_RED'] * 100 + [('DEFEND_RED', 1)]
    assert count in (96, 101)
    bc.apply_custom_state({'hand': [card], zone: cards})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    handles = [c['native_index'] for c in bc.select_info()['candidates']]
    assert handles == (list(range(96)) if count == 96 else [0, 100])
    return bc


def pending_continuation(env, case):
    """Use the original native action to reach a choice; never inject a queue."""
    import slaythespire as sts
    card, upgrade, potion, bark = 'HEADBUTT', 0, None, False
    if case.startswith('card:'):
        _, card, up = case.split(':'); upgrade = int(up)
    elif case.startswith('potion:'):
        _, potion, value = case.split(':'); bark = bool(int(value))
    elif case in ('skill_relics', 'skill_powers', 'void', 'evolve_fire', 'shuffle',
                  'burst', 'duplication', 'echo_form', 'pain_draw', 'duplication_potion'):
        card = 'WARCRY'
    elif case == 'juggernaut':
        card = 'ARMAMENTS'
    elif case == 'exhaust_hooks':
        card = 'BURNING_PACT'
    elif case == 'havoc':
        card = 'HAVOC'
    elif case == 'distilled_chaos':
        potion = 'Distilled Chaos'
    elif case == 'mayhem':
        card = 'WARCRY'
    if case == 'duplication_potion': potion = 'Duplication Potion'
    relics = []
    if potion:
        relics = ['Toy Ornithopter'] + (['SacredBark'] if bark else [])
    if case == 'attack_relics':
        relics = ['InkBottle', 'Kunai', 'Ornamental Fan', 'Shuriken', 'Pen Nib',
                  'Yang', 'Nunchaku', 'OrangePellets']
    if case == 'skill_relics':
        relics = ['InkBottle', 'Letter Opener', 'OrangePellets']
    if case == 'shuffle': relics = ['TheAbacus', 'Sundial']
    if case == 'hand_drill': relics = ['HandDrill', 'Champion Belt']
    if case == 'death_relics': relics = ['Gremlin Horn', 'The Specimen']
    if case == 'necronomicon': relics = ['Necronomicon']
    if case == 'exhaust_hooks': relics = ['Charon\'s Ashes', 'Dead Branch']
    if case == 'pain_draw': relics = ['Centennial Puzzle', 'Runic Cube']
    if case == 'stasis_full': relics = ['InkBottle']
    if case == 'stasis_discard': relics = ['Runic Cube']
    encounter = {
        'sharp_hide': 'The Guardian', 'flight': 'Three Byrds', 'flight_last': 'Three Byrds',
        'death_relics': 'Two Louse', 'stasis': 'Automaton', 'stasis_full': 'Automaton',
        'stasis_discard': 'Automaton',
        'darkling': 'Three Darklings', 'awakened': 'Awakened One', 'leader': 'Gremlin Leader',
    }.get(case, 'Jaw Worm')
    env.thick.gc.set_deck([(card, upgrade)] + [('STRIKE_RED', 0)] * 19)
    env.thick.gc.set_relics(relics)
    env.thick.gc.set_potion_state([potion] if potion else [])
    env.thick.bc = bc = sts.create_battle(env.thick.gc, encounter)
    counters = dict(attacks_played=2, skills_played=2, ink_bottle=9, pen_nib=8,
                    nunchaku=9, orange_attack=1, orange_skill=1, orange_power=1,
                    sundial=2, panache_counter=1)
    bc.apply_custom_state({'player': {'hp': 40, 'counters': counters}})
    hand = [native_card(card, 0, upgrades=upgrade, cost=2 if case == 'necronomicon' else 1)]
    hand += [native_card(name, i) for i, name in enumerate(
        ['STRIKE_RED', 'DEFEND_RED', 'BASH', 'DEFEND_RED'], 1)]
    draw = [native_card('STRIKE_RED' if i % 2 else 'DEFEND_RED', i) for i in range(5, 15)]
    if case in ('void', 'evolve_fire', 'pain_draw'):
        draw[-1] = native_card('VOID' if case == 'void' else 'DAZED', 14)
    if case == 'pain_draw': hand[-1] = native_card('PAIN', 4)
    if case == 'shuffle': draw = []
    if case in ('havoc', 'distilled_chaos'):
        draw[-3:] = [native_card('WARCRY', 12), native_card('BURNING_PACT', 13),
                     native_card('HEADBUTT', 14)]
    if case == 'mayhem': draw = [native_card('WARCRY', i) for i in range(5, 15)]
    stasis = (None, None)
    target = 0
    if case.startswith('stasis'):
        target = 0
        bc.apply_custom_state({'monsters': [{'id': 'BRONZE_ORB', 'hp': 1,
            'max_hp': 50, 'block': 0}]})
        bc.set_monster_status(target, 'STASIS', 1)
        stasis = (native_card('RITUAL_DAGGER', 30, upgrades=1, special=23, cost=0), None)
        if case in ('stasis_full', 'stasis_discard'):
            hand += [native_card('DEFEND_RED', i) for i in range(20, 25)]
        if case == 'stasis_discard': hand[-1] = native_card('PAIN', 24)
    discard = [native_card('STRIKE_RED', 15), native_card('DEFEND_RED', 16)]
    if potion == 'Liquid Memories': discard.append(native_card('BASH', 19))
    bc.set_card_state(native_card_state(hand=hand, draw=draw, discard=discard,
        exhaust=[native_card('DEFEND_RED', 17), native_card('BASH', 18)], stasis=stasis))
    bc.set_player_energy(10)
    powers = {}
    if case in ('attack_relics', 'skill_relics'): powers.update(WEAK=2, FRAIL=2)
    if case in ('attack_powers', 'skill_powers'):
        powers = dict(AFTER_IMAGE=3, PANACHE=5, THOUSAND_CUTS=2)
        if case == 'attack_powers': powers.update(RAGE=3, VIGOR=2, PEN_NIB=1)
        else: powers['HEX'] = 1
    if case == 'juggernaut': powers['JUGGERNAUT'] = 5
    if case in ('evolve_fire', 'pain_draw'): powers.update(EVOLVE=1, FIRE_BREATHING=3)
    if case == 'hand_drill': powers['SADISTIC'] = 3
    if case == 'envenom_sadistic': powers.update(ENVENOM=2, SADISTIC=3)
    if case in ('double_tap', 'burst', 'duplication', 'echo_form'):
        powers[case.upper()] = 1
    if case == 'exhaust_hooks': powers.update(DARK_EMBRACE=1, FEEL_NO_PAIN=3)
    if case == 'mayhem': powers['MAYHEM'] = 2
    for name, amount in powers.items(): bc.set_player_status(name, amount)
    if 'PANACHE' in powers:
        # Restoring the power initializes its countdown, so restore the
        # actual fifth-card counter afterwards and require its damage below.
        bc.apply_custom_state({'player': {'counters': {'panache_counter': 1}}})
    if case == 'sharp_hide': bc.set_monster_status(0, 'SHARP_HIDE', 3)
    if case in ('flight', 'flight_last'): bc.set_monster_status(0, 'FLIGHT', 1 if case.endswith('last') else 3)
    if case == 'hand_drill': bc.set_monster_block(0, 1)
    if case in ('death_relics', 'darkling', 'awakened', 'leader'):
        if case == 'awakened': target = 2
        if case == 'leader': target = 3
        bc.apply_custom_state({'monsters': [{}] * target + [{'hp': 1, 'block': 0}]})
        if case == 'death_relics':
            bc.apply_custom_state({'monsters': [{}, {'hp': 100, 'max_hp': 100}]})
            bc.set_monster_status(0, 'POISON', 4)
            bc.set_monster_status(0, 'CORPSE_EXPLOSION', 1)
    env._invalidate_backend_cache()
    if case == 'mayhem':
        env.step(0)
        return bc
    actions = env._joint_policy_actions()
    action = next(action for action, bits in actions
        if sts.Action(bits).action_type == (sts.ActionType.POTION if potion else sts.ActionType.CARD)
        and sts.Action(bits).source_idx == 0 and sts.Action(bits).target_idx == target)
    env.step(action)
    if case == 'duplication_potion':
        action = next(action for action, bits in env._joint_policy_actions()
            if sts.Action(bits).action_type == sts.ActionType.CARD and sts.Action(bits).source_idx == 0)
        env.step(action)
    return bc
