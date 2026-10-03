"""Offline combat artifacts replay real native transitions without strategy input changes."""
import hashlib
import json

import numpy as np
import pytest

from env.core.native_actions import complete_actions
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.trajectory import EpisodeTraceCollector, restore_combat_snapshot


pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.fixture
def natural_combat_pair():
    # Do not use combat_factory's injected native state: only a real seeded run
    # and its committed actions can be reconstructed by the recovery journal.
    options = dict(combat_action_contract='native-v1', card_language=True,
                   global_map_view=True)
    source, restored = SlayTheSpireEnv(**options), SlayTheSpireEnv(**options)
    try:
        source.reset(seed=200464)
        for _ in range(100):
            if source.thick.bc is not None:
                break
            source.step(int(np.flatnonzero(source.action_masks())[0]))
        assert source.thick.bc is not None
        yield source, restored
    finally:
        source.close()
        restored.close()


def _equal_arrays(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_combat_snapshot_restores_native_state_and_next_actions(tmp_path, natural_combat_pair):
    source, restored = natural_combat_pair
    before = source.export_training_state()
    observations = {key: value.copy() for key, value in source._obs().items()}
    mask = source.action_masks().copy()
    raw = source.thick.bc.to_dict()
    collector = EpisodeTraceCollector(source, 0, 200464, combat_snapshot_output=tmp_path)
    first = complete_actions(source.thick.bc)[0]
    collector.before_step(None, first)
    # Capturing evidence must not mutate observations, native state, or journal.
    assert source.export_training_state() == before
    assert source.thick.bc.to_dict() == raw
    _equal_arrays(source._obs(), observations)
    np.testing.assert_array_equal(source.action_masks(), mask)

    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    artifact = json.loads(files[0].read_text(encoding='utf-8'))
    assert artifact['environment_state']['actions']
    assert artifact['native_state']['draw'] == json.loads(json.dumps(raw['draw']))
    sha = hashlib.sha256(files[0].read_bytes()).hexdigest()
    obs, _ = restore_combat_snapshot(restored, files[0], expected_sha256=sha)
    _equal_arrays(obs, observations)
    np.testing.assert_array_equal(restored.action_masks(), mask)
    assert restored.thick.bc.to_dict() == raw

    # Include several enemy/next-player boundaries so draw/RNG continuation is
    # checked, not just an initial observation matching a lossy visible state.
    for step in range(64):
        actions = complete_actions(source.thick.bc)
        assert actions == complete_actions(restored.thick.bc)
        # First advance a draw boundary, then play available cards to complete
        # this seeded encounter and check the finalized combat reference too.
        action = next((a for a in actions if int(a) >> 29 == 4), actions[0]) if step == 0 else actions[0]
        collector.before_step(None, action)
        left = source.step_native_combat(action)
        right = restored.step_native_combat(action)
        _equal_arrays(left[0], right[0])
        assert left[1:4] == right[1:4]
        np.testing.assert_array_equal(source.action_masks(), restored.action_masks())
        assert source.thick.obs_canonical() == restored.thick.obs_canonical()
        collector.after_step(None, *left[1:4], left[4], native_action=action)
        if left[2] or left[3] or source.thick.bc is None:
            break
    trajectory = collector.finish('fixture_stop')
    assert 'incomplete' not in trajectory['combats'][0]
    assert len(list(tmp_path.glob('*.json'))) == 1
    reference = trajectory['combats'][0]['snapshot']
    assert reference['sha256'] == sha
    assert reference['path'] == str(files[0].resolve())
    assert trajectory['combat_snapshot_seconds'] == reference['capture_seconds'] > 0
    # Later gameplay cannot silently overwrite the captured starting point.
    assert hashlib.sha256(files[0].read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match='checksum'):
        restore_combat_snapshot(restored, files[0], expected_sha256='0' * 64)


def test_combat_snapshot_rejects_overwriting_evidence(tmp_path, natural_combat_pair):
    source, _ = natural_combat_pair
    action = complete_actions(source.thick.bc)[0]
    first = EpisodeTraceCollector(source, 0, 200464, combat_snapshot_output=tmp_path)
    first.before_step(None, action)
    path = next(tmp_path.glob('*.json'))
    original = path.read_bytes()
    second = EpisodeTraceCollector(source, 0, 200464, combat_snapshot_output=tmp_path)
    with pytest.raises(FileExistsError):
        second.before_step(None, action)
    assert path.read_bytes() == original
