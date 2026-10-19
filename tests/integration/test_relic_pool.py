"""RELICPOOL01: native empty-pool faults must fail a bounded child, not pytest."""

import subprocess
import sys

import pytest

_CASES = (
    [(mode, tier, front) for mode in ('nonempty', 'exhausted', 'filtered')
     for tier in range(5) for front in (0, 1)]
    + [(mode, tier, front) for mode in ('fallback', 'shop_fallback')
       for tier in (0, 1, 4) for front in (0, 1)]
    + [('retry', tier, 1) for tier in range(5)]
    + [('invalid', tier, 1) for tier in (5, 6, 7, 99)]
)


@pytest.mark.timeout(240)  # The first case also builds the isolated C++ executable.
@pytest.mark.parametrize('mode,tier,front', _CASES)
def test_relic_pool_boundaries(game_context_executable, mode, tier, front):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), mode, str(tier), str(front)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


def test_courier_can_restock_until_relic_pools_are_exhausted():
    # Real Python extension + native shop/courier path. No internal pool edits.
    code = r'''
from env.core import _BUILD
import slaythespire as sts
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
gc.set_relics(['The Courier'])
gc.cur_map_node_y = 0
gc.gold = 100000
gc.transition_to_map_node(2)
assert gc.screen_state_name.endswith('SHOP_ROOM')
for purchase in range(200):
    item = gc.shop_contents()['relics'][0]
    assert ('relic', 0) in gc.legal_shop_options()
    if item['id'] == 'Circlet':
        print('PASS: exhausted via Courier', purchase)
        break
    gc.shop_buy_relic(0)
    for _ in range(10):
        if gc.screen_state_name.endswith('SHOP_ROOM'): break
        if gc.screen_state_name.endswith('CARD_SELECT'): gc.choose_select_card_option(0)
        elif gc.screen_state_name.endswith('REWARDS'): gc.reward_proceed()
        else: raise AssertionError(gc.screen_state_name)
    assert gc.screen_state_name.endswith('SHOP_ROOM')
else:
    raise AssertionError('bounded Courier purchases did not exhaust a pool')
'''
    # resource is POSIX-only; omit that diagnostic safeguard on Windows.
    if sys.platform == 'win32':
        code = code.replace('import resource\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n', '')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: exhausted via Courier' in result.stdout
