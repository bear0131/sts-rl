"""Grouped leaf transport preserves row ownership and decoder ordering."""

from collections.abc import Mapping
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from rl.search import native
from rl.training.cached_mask_vec_env import (
    _pack_leaf_responses,
    _unpack_leaf_responses,
)


def _response(index, count=1):
    rows = np.arange(count, dtype=np.int32) + 10 * index
    return {
        'done': False,
        'leaf_meta': [{'request_id': int(row)} for row in rows],
        'observation_batch': {
            'ids': np.stack((rows, rows + 1), axis=1),
            'hp': (rows.astype(np.float32) / 8)[:, None],
        },
    }


def _roundtrip(responses):
    return _unpack_leaf_responses(pickle.loads(pickle.dumps(
        _pack_leaf_responses(responses))))


class _CountingFields(dict):
    reads = None

    def __init__(self, fields):
        super().__init__(fields)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


class GroupedNativeTransportTest(unittest.TestCase):
    def test_unpack_is_lazy_and_adjacent_spans_read_each_group_field_once(self):
        packet = _pack_leaf_responses({index: _response(index) for index in range(16)})
        source = _CountingFields(packet.observations)
        packet.observations = source
        responses = _unpack_leaf_responses(packet)
        self.assertEqual(source.reads, [])
        self.assertIsInstance(responses[0]['observation_batch'], Mapping)
        records, observations = native.decode_leaves(responses)
        self.assertEqual(source.reads, ['ids', 'hp'])
        self.assertEqual([(i, leaf['request_id']) for i, leaf in records],
                         [(i, i * 10) for i in range(16)])
        np.testing.assert_array_equal(observations['ids'][:, 0], np.arange(16) * 10)
        self.assertEqual(observations['ids'].dtype, np.int32)
        self.assertEqual(observations['hp'].dtype, np.float32)
        for key, result in observations.items():
            self.assertTrue(result.flags.c_contiguous)
            self.assertFalse(np.shares_memory(result, dict.__getitem__(source, key)))
        observations['hp'].fill(-99)
        self.assertEqual(responses[0]['observation_batch']['hp'][0, 0], 0)

    def test_mixed_transports_keep_sorted_workers_and_original_request_order(self):
        grouped = _roundtrip({4: _response(4, 2), 1: _response(1, 3)})
        legacy = _response(2, 2)
        chunks, spec, offset = [], [], 0
        for key, value in legacy['observation_batch'].items():
            payload = value.tobytes()
            spec.append((key, value.dtype.str, value.shape, offset, len(payload)))
            chunks.append(payload)
            offset += len(payload)
        legacy.update(observation_batch=None, observation_buffer=b''.join(chunks),
                      observation_spec=spec)
        plain = _response(3)
        raw_observation = _response(0)['observation_batch']
        raw = {'leaves': [{'request_id': 0, 'observation': {
            key: value[0] for key, value in raw_observation.items()}}]}
        responses = {4: grouped[4], 3: plain, 2: legacy, 1: grouped[1], 0: raw}
        records, observations = native.decode_leaves(responses)
        expected_rows = [0, 10, 11, 12, 20, 21, 30, 40, 41]
        self.assertEqual([leaf['request_id'] for _, leaf in records], expected_rows)
        self.assertEqual([i for i, _ in records], [0, 1, 1, 1, 2, 2, 3, 4, 4])
        np.testing.assert_array_equal(observations['ids'][:, 0], expected_rows)
        np.testing.assert_array_equal(observations['hp'][:, 0],
                                      np.asarray(expected_rows, dtype=np.float32) / 8)

    def test_nonadjacent_subset_does_not_import_omitted_tree_rows(self):
        packet = _pack_leaf_responses({index: _response(index) for index in range(3)})
        source = _CountingFields(packet.observations)
        packet.observations = source
        responses = _unpack_leaf_responses(packet)
        records, observations = native.decode_leaves({2: responses[2], 0: responses[0]})
        self.assertEqual([i for i, _ in records], [0, 2])
        np.testing.assert_array_equal(observations['ids'][:, 0], [0, 20])
        self.assertEqual(source.reads, ['ids', 'ids', 'hp', 'hp'])

    def test_retained_response_does_not_alias_later_packets(self):
        inputs = {2: _response(2), 0: {'done': True, 'result': {'action': 7}},
                  1: _response(1)}
        first, later = _roundtrip(inputs), _roundtrip(inputs)
        self.assertEqual(list(first), [2, 0, 1])
        self.assertEqual(first[0], inputs[0])
        later[1]['observation_batch']['hp'].fill(-1)
        np.testing.assert_array_equal(first[1]['observation_batch']['hp'], [[1.25]])

    def test_rejects_bad_metadata_after_an_earlier_span_from_same_source(self):
        responses = _roundtrip({0: _response(0), 1: _response(1)})
        responses[1]['leaf_meta'].append({'request_id': 999})
        with self.assertRaisesRegex(ValueError, 'batch size'):
            native.decode_leaves(responses)

    def test_rejects_mismatched_source_fields_and_cross_source_keys(self):
        packet = _pack_leaf_responses({0: _response(0), 1: _response(1)})
        packet.observations['hp'] = packet.observations['hp'][:1]
        with self.assertRaisesRegex(ValueError, 'batch size'):
            native.decode_leaves(_unpack_leaf_responses(packet))
        responses = _roundtrip({0: _response(0), 1: _response(1)})
        responses[2] = {'leaf_meta': [{'request_id': 2}],
                        'observation_batch': {'other': np.zeros((1, 2))}}
        with self.assertRaisesRegex(ValueError, 'schemas'):
            native.decode_leaves(responses)

    def test_cross_source_dtype_promotion_matches_numpy_concatenation(self):
        responses = _roundtrip({0: _response(0), 1: _response(1)})
        responses[2] = _response(2)
        responses[2]['observation_batch']['ids'] = np.asarray([[20.5, 21]], dtype=np.float64)
        _, observations = native.decode_leaves(responses)
        self.assertEqual(observations['ids'].dtype, np.float64)
        np.testing.assert_array_equal(observations['ids'][:, 0], [0, 10, 20.5])

    def test_count_uses_metadata_and_decode_still_rejects_malformed_buffer(self):
        response = {'leaf_meta': [{'request_id': 1}, {'request_id': 2}],
                    'observation_batch': None, 'observation_buffer': b'',
                    'observation_spec': [('x', '<i8', (2, 1), 0, 16)]}
        with patch.object(native, 'unpack_nosl_observation_buffer',
                          side_effect=AssertionError('count must not decode fields')):
            self.assertEqual(native._leaf_count(response), 2)
            self.assertEqual(native._leaf_count({'leaves': [{}, {}, {}]}), 3)
            self.assertEqual(native._leaf_count([{}, {}]), 2)
        with self.assertRaises(ValueError):
            native.decode_leaves({0: response})


if __name__ == '__main__':
    unittest.main()
