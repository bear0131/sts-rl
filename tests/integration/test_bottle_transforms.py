"""BOTTLE01: transform eligibility depends on the originating event/relic."""
import subprocess

import pytest


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(17), ids=['jax', 'jax-egg-fish', 'mutagens',
    'duplicate-mutagens', 'duplicate-circlet', 'curse-omamori0', 'curse-omamori1',
    'curse-omamori2', 'bloom-omamori0', 'bloom-omamori1', 'bloom-omamori2',
    'parasite-min-hp', 'bulk-curse', 'wounded-curse', 'wounded-bloom', 'non-curse', 'parasite-at-one'])
def test_augmenter_rewards_and_curse_side_effects(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'augmenter-reward', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(6), ids=['augmenter-candidates', 'augmenter-result',
    'ordinary-transform', 'ordinary-remove', 'astrolabe', 'augmenter-count'])
def test_bottle_transform_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'bottle-transform', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(18), ids=[
    *[f'cage-{n}' for n in range(5)], *[f'astrolabe-{n}' for n in range(5)],
    'boss-cage-auto', 'boss-cage-select', 'boss-astrolabe-auto', 'boss-astrolabe-select',
    'reward-cage-auto', 'reward-cage-select', 'reward-astrolabe-auto', 'reward-astrolabe-select'])
def test_relic_selection_threshold_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'relic-selection', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(3), ids=['colored', 'colorless', 'curse'])
def test_transform_upgrade_color_and_rng(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'transform-upgrade', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('count', [1, 4], ids=['automatic', 'selection'])
@pytest.mark.parametrize('eggs', [False, True])
def test_astrolabe_exports_upgraded_colorless_cards(ascension, count, eggs):
    import slaythespire as sts

    # Controlled boss entry, not a natural run: seed 25 offers Astrolabe in slot 2.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 25, ascension)
    gc.skip_battles = True
    gc.cur_map_node_y = 14
    gc.transition_to_map_node(0)
    gc.reward_proceed()
    assert gc.boss_relic_ids()[2] == 'Astrolabe'
    gc.set_deck([('FLASH_OF_STEEL', 0)] * count + [('ASCENDERS_BANE', 0)])
    if eggs:
        gc.set_relics(['Molten Egg 2', 'Toxic Egg 2', 'Frozen Egg 2'])
    gc.choose_boss_relic(2)
    transformed = min(count, 3)
    if count > 3:
        assert gc.select_screen_info()['context'] == 'TRANSFORM_UPGRADE'
        assert gc.get_select_cards() == [('FLASH_OF_STEEL', 0)] * count
        for _ in range(transformed):
            gc.choose_select_card_option(0)
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN and gc.act == 2
    deck = gc.deck_ids()
    assert deck[:-transformed] == [('FLASH_OF_STEEL', 0)] * (count - transformed) + [('ASCENDERS_BANE', 0)]
    assert all(card != 'FLASH_OF_STEEL' and upgrades == 1 for card, upgrades in deck[-transformed:])
    assert len(deck) == count + 1


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('charges', [0, 1, 2])
@pytest.mark.parametrize('bloom', [False, True])
def test_astrolabe_curse_obtain_effects_reach_public_state(ascension, charges, bloom):
    import slaythespire as sts
    from env.core import encode
    from schemas.combat_state import normalize_sim_global

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 25, ascension)
    gc.skip_battles = True
    gc.cur_map_node_y = 14
    gc.transition_to_map_node(0)
    gc.reward_proceed()
    assert gc.boss_relic_ids()[2] == 'Astrolabe'
    gc.set_deck([('PARASITE', 0), ('PAIN', 0), ('ASCENDERS_BANE', 0)])
    relics = [dict(id='Darkstone Periapt', counter=0), dict(id='CeramicFish', counter=0),
              dict(id='Omamori', counter=charges)]
    if bloom:
        relics.append(dict(id='Mark of the Bloom', counter=0))
    gc.set_relic_state(relics)
    gc.cur_hp, gc.max_hp, gc.gold = 40, 80, 12
    gc.choose_boss_relic(2)
    obtained = 2 - charges
    assert gc.max_hp == 77 + 6 * obtained
    assert gc.gold == 12 + 9 * obtained
    assert len(gc.deck_ids()) == 1 + obtained
    assert gc.deck_ids()[0] == ('ASCENDERS_BANE', 0)
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN and gc.act == 2
    if bloom:
        assert gc.cur_hp == 40  # Blocks both obtain healing and the act transition heal.
    public = normalize_sim_global(gc)
    assert public['max_hp'] == 77 + 6 * obtained and public['hp'] == gc.cur_hp
    obs = encode.encode({'phase': 'map', 'global': public})
    assert obs['run_hp'][1] == pytest.approx(gc.max_hp / encode.HP_SCALE)
