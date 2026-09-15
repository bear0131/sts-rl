"""Check probes against the native encoder, not a duplicate scalar formula."""
import numpy as np
import pytest

from rl.evaluation.nosl.hp_sensitivity import (
    intervene, maximum_hp, summarize, save_corpus, load_corpus, corpus_digest,
)


def test_enemy_intervention_matches_native_reencoding(combat_factory):
    env = combat_factory(['STRIKE_RED']*5, encounter='Cultist')
    base = env._obs()
    raw = env.thick.bc.to_dict()
    assert maximum_hp(base, 'enemy') == raw['monsters'][0]['max_hp']
    original = {k: v.copy() for k, v in base.items()}
    changed = intervene(base, 'enemy', 13)
    env.thick.bc.apply_custom_state({'monsters': [{'hp': 13}]})
    env._invalidate_backend_cache()
    encoded = env._obs()
    for key in changed:
        np.testing.assert_array_equal(changed[key], encoded[key], err_msg=key)
        np.testing.assert_array_equal(base[key], original[key], err_msg=key)


def test_player_intervention_matches_native_reencoding(combat_factory):
    env = combat_factory(['STRIKE_RED']*5)
    base = env._obs()
    changed = intervene(base, 'player', 17)
    env.thick.gc.cur_hp = 17
    env.thick.bc.apply_custom_state({'player': {'hp': 17}})
    env._invalidate_backend_cache()
    for key, value in env._obs().items():
        np.testing.assert_array_equal(changed[key], value, err_msg=key)


def test_corpus_roundtrip_preserves_observation_bytes(tmp_path, combat_factory):
    corpus = [({'encounter': 'Cultist'}, combat_factory(['STRIKE_RED']*5)._obs())]
    path = tmp_path/'corpus.npz'
    save_corpus(corpus, path)
    assert corpus_digest(load_corpus(path)) == corpus_digest(corpus)
    with pytest.raises(FileExistsError):
        save_corpus(corpus, path)


def test_inversions_are_local_not_just_endpoint_slope():
    # Net enemy response is correct, but the middle segment is inverted.
    result = summarize([1, 2, 3, 4], [.4, .2, .3, .1], 'enemy')
    assert result['adverse_steps'] == 1
    assert result['largest_adverse_step'] == pytest.approx(.1)
    assert summarize([1, 2, 3], [.1, .2, .3], 'player')['adverse_steps'] == 0
    assert not summarize([1, 2], [float('nan'), .1], 'enemy')['finite']


def test_hp_intervention_rejects_inconsistent_language_aliases(combat_factory):
    base = combat_factory(['STRIKE_RED']*5)._obs()
    base['player_status_language'] = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match='tokenized'):
        intervene(base, 'player', 20)
