import numpy as np
import pytest

from env.sts2.game import EnvConfig, Sts2Env

pytestmark = pytest.mark.sts2


def test_native_clone_replay_and_invalid_action():
    env = Sts2Env()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    original = env.fingerprint()
    with pytest.raises(ValueError):
        env.step(len(env.candidates))
    assert env.fingerprint() == original
    for _ in range(7):
        env.step(0)
    twin = env.clone()
    state = env.snapshot()
    assert twin.fingerprint() == env.fingerprint()
    env.step(0)
    assert twin.fingerprint() != env.fingerprint()
    twin.step(0)
    assert twin.fingerprint() == env.fingerprint()
    restored = Sts2Env()
    restored.restore(state)
    restored.step(0)
    assert restored.fingerprint() == env.fingerprint()
    for item in (env, twin, restored):
        item.close()


def test_truncation_is_not_death_and_keeps_bootstrap_observation():
    env = Sts2Env(EnvConfig(max_steps=1))
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated, info['outcome']) == (0, False, True, 'truncated')
    assert np.any(obs['mask'])
    with pytest.raises(RuntimeError, match='reset required'):
        env.step(0)
    env.restore(env.snapshot())
    env.close()


def test_real_combat_has_natural_outcome():
    env = Sts2Env(EnvConfig(max_steps=300))
    env.reset(seed=42)
    for _ in range(300):
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['outcome'] in ('win', 'death')
    assert reward == (1 if info['outcome'] == 'win' else -1)
    assert not env.action_masks().any()
    env.close()
