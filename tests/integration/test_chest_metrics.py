import pytest

from env.core import action_space as A, game_actions as ga

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('take', [False, True])
def test_chest_choice_ignores_gold_clicks_and_survives_restore(chest_env_type, take):
    env, restored = chest_env_type(), chest_env_type()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(A._CHEST)
        assert info['chest_opened'] is True
        assert 'chest_relic_taken' not in info
        assert env.thick.gc.get_reward_info()['gold']
        # Use the thin codec for all transitions and keep its checkpoint journal.
        gold = next(i for i in range(env.action_space.n)
                    if env.action_masks()[i] and env.describe_action(i).startswith('claim gold'))
        info = env.step(gold)[-1]
        assert 'chest_opened' not in info and 'chest_relic_taken' not in info
        state = env.export_training_state()
        restored.restore_training_state(state)
        for current in (env, restored):
            action = (next(i for i in range(current.action_space.n)
                           if current.action_masks()[i] and current.describe_action(i).startswith('claim relic'))
                      if take else A._PROCEED)
            info = current.step(action)[-1]
            assert info['chest_relic_taken'] is take
            assert 'chest_opened' not in info
            if take:
                assert 'chest_relic_taken' not in current.step(A._PROCEED)[-1]
    finally:
        env.close()
        restored.close()


def test_skip_chest_and_reset_do_not_count_as_relic_refusal(chest_env_type):
    env = chest_env_type()
    try:
        env.reset(seed=0)
        info = env.step(A._CHEST + 1)[-1]
        assert info['chest_opened'] is False
        assert 'chest_relic_taken' not in info
        env.reset(seed=0)
        env.step(A._CHEST)
        assert env.thick._chest_relic_pending
        env.reset(seed=0)
        assert not env.thick._chest_relic_pending
    finally:
        env.close()


@pytest.mark.parametrize('relic,offers', [('Matryoshka', 2), ('NlothsMask', 0)])
def test_multiple_or_empty_chest_counts_at_most_once(chest_env_type, relic, offers):
    env = chest_env_type()
    try:
        env.reset(seed=0)
        env.thick.gc.set_relic_state([{'id': relic, 'counter': 2}])
        info = env.thick.step(ga.chest(True))[-1]
        assert info['chest_opened'] is True
        assert len(env.thick.gc.get_reward_info()['relics']) == offers
        for index in range(offers):
            info = env.thick.step(ga.claim(ga.CLAIM_RELIC))[-1]
            assert info.get('chest_relic_taken') is (True if index == 0 else None)
        if ga.proceed() in env.thick.legal_semantic_actions():
            assert 'chest_relic_taken' not in env.thick.step(ga.proceed())[-1]
    finally:
        env.close()
