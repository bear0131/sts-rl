"""Internal row reuse must preserve the public schema and output isolation."""
import itertools

import numpy as np
import pytest

from env.core import encode


_FLAGS = ('global_map_view', 'card_language', 'combat_language',
          'deck_size_observation', 'neow_observation', 'event_option_observation',
          'interactive_reasoning', 'upper_public_context')


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=len(_FLAGS))))
def test_reused_observation_row_clears_absent_groups_and_preserves_neighbors(flags):
    options = dict(zip(_FLAGS, flags))
    populated = {'phase': 'none', 'global': dict(deck=[('Bash', 2)], gold=79, hp=54, max_hp=80)}
    reference = encode.encode(populated, **options)
    backing = {key: np.full((3, *value.shape), 17, dtype=value.dtype)
               for key, value in reference.items()}
    destination = {key: value[1] for key, value in backing.items()}
    identities = {key: id(value) for key, value in destination.items()}
    for canon in (populated, dict(phase='none'), populated):
        expected = encode.encode(canon, **options)
        actual = encode.encode(canon, **options, _out=destination)
        assert actual is destination
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            assert id(value) == identities[key]
            assert value.shape == expected[key].shape and value.dtype == expected[key].dtype
            assert value.tobytes() == expected[key].tobytes(), key
            assert not np.shares_memory(value, expected[key])
            assert np.all(backing[key][0] == 17) and np.all(backing[key][2] == 17)
            value.fill(31)  # The next encoding must reset every absent/PAD field.
        assert reference['gold'][0] == np.float32(79 / encode.GOLD_SCALE)
