"""Returned step metadata must retain deepcopy isolation and alias contracts."""
from copy import deepcopy

import numpy as np
import pytest

from rl.training.native_vec_env import _copy_step_infos


def test_step_info_copy_isolates_nested_observations_and_preserves_aliases():
    observation = {'hp': np.array([20., 30.], dtype=np.float32)}
    stats = {'floor': 17, 'won': False, 'reward': 0.5, 'name': 'boss',
             'selected': [1, 2, None]}
    infos = [{'run_stats': stats, 'terminal_observation': observation},
             {'run_stats': stats, 'terminal_observation': observation}]
    infos[0]['self'] = infos
    copied = _copy_step_infos(infos)
    assert copied[0]['self'] is copied
    assert copied[0]['run_stats'] is copied[1]['run_stats']
    assert copied[0]['terminal_observation'] is copied[1]['terminal_observation']
    assert copied[0]['run_stats'] == stats
    assert copied[0]['terminal_observation']['hp'].tobytes() == observation['hp'].tobytes()
    copied[0]['terminal_observation']['hp'][0] = -1
    copied[0]['run_stats']['selected'].append(3)
    assert observation['hp'][0] == 20
    assert stats['selected'] == [1, 2, None]


def test_step_info_copy_honors_subclasses_and_custom_copy_protocol():
    class Number(int):
        def __deepcopy__(self, memo):
            return Number(int(self) + 10)

    class Info(dict):
        pass

    shared = [Number(3)]
    object_array = np.empty(1, dtype=object)
    object_array[0] = shared
    source = [Info(value=Number(2)), {'tuple': (shared,), 'array': object_array,
                                    ('custom', 1): Number(4)}]
    expected = deepcopy(source)
    copied = _copy_step_infos(source)
    assert type(copied[0]) is Info
    assert copied[0] == expected[0] == {'value': 12}
    assert copied[1][('custom', 1)] == 14
    assert copied[1]['tuple'][0] is copied[1]['array'][0]
    assert copied[1]['tuple'][0] == expected[1]['tuple'][0] == [13]
    assert shared == [3]


@pytest.mark.parametrize('layout', ['C', 'F', 'slice', 'scalar', 'structured'])
def test_numeric_info_arrays_match_deepcopy_layout_and_aliases(layout):
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    if layout == 'F':
        array = np.asfortranarray(array)
    elif layout == 'slice':
        array = array[::-2, ::2]
    elif layout == 'scalar':
        array = np.array(7, dtype=np.int64)
    elif layout == 'structured':
        array = np.array([(1, 2.)], dtype=[('id', 'i4'), ('value', 'f4')])
    array.flags.writeable = False
    source = [{'array': array, 'nested': (array,)}]
    expected, copied = deepcopy(source), _copy_step_infos(source)
    result = copied[0]['array']
    assert result is copied[0]['nested'][0]
    assert result.dtype == expected[0]['array'].dtype
    assert result.strides == expected[0]['array'].strides
    assert result.flags.writeable == expected[0]['array'].flags.writeable
    assert result.tobytes() == expected[0]['array'].tobytes()
    assert not np.shares_memory(result, array)


def test_info_array_subclass_preserves_custom_deepcopy():
    class Tagged(np.ndarray):
        def __deepcopy__(self, memo):
            return np.asarray(self).copy() + 100

    original = np.array([1, 2]).view(Tagged)
    result = _copy_step_infos([{'value': original}])
    np.testing.assert_array_equal(result[0]['value'], [101, 102])
