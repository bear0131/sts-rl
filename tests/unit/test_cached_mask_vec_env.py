from __future__ import annotations

import os
import pickle
import multiprocessing as mp

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecMonitor

from rl.training.cached_mask_vec_env import (
    CachedMaskSubprocVecEnv,
    GroupedCachedMaskSubprocVecEnv,
    _pack_leaf_responses,
    _unpack_leaf_responses,
    _SharedArrayBuffer,
    _SharedArrayMessage,
)
from rl.training.distributed_sb3 import build_parser as build_distributed_parser


class _MaskInfoEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 2, shape=(1,), dtype=np.int64)

    def __init__(self) -> None:
        self.position = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return np.array([0]), {"action_mask": np.array([True, False])}

    def step(self, action):
        self.position += 1
        done = self.position == 2
        mask = np.array([False, not done])
        return np.array([self.position]), 0.0, done, False, {
            "action_mask": mask
        }

    def action_masks(self):
        raise AssertionError("cached info mask should avoid worker method calls")

    def nosl_is_az_cpp_begin(self, value):
        return {'done':False,'leaf_meta':[{'request_id':value}],
                'observation_batch':{'x':np.array([[value]],dtype=np.int64)}}

    def export_training_state(self):
        return {"position": self.position}

    def restore_training_state(self, state):
        self.position = int(state["position"])
        mask = np.array([self.position % 2 == 0, self.position % 2 == 1])
        return np.array([self.position]), {"action_mask": mask}

    def hard_exit(self):
        os._exit(23)


def _make_mask_info_env():
    return _MaskInfoEnv()


def test_shared_search_arrays_keep_bits_and_own_received_storage():
    buffer = _SharedArrayBuffer(mp.get_context('spawn'), 65536)
    payload = {'ids': np.array([2**63 + 7], dtype=np.uint64),
               'policy': np.arange(724, dtype=np.float64).reshape(4, 181),
               'values': np.array([0x80000000, 0x00000001, 0x3DCCCCCD,
                                   0x7F800000, 0xFF800000, 0x7FC01234],
                                  dtype=np.uint32).view(np.float32),
               'mask': np.array([True, False, True]),
               'result': {'done': True, 'action': 5}}
    wire = buffer.pack(payload)
    assert isinstance(wire, _SharedArrayMessage)
    assert len(pickle.dumps(wire)) < len(pickle.dumps(payload)) / 4
    received = buffer.unpack(pickle.loads(pickle.dumps(wire)))
    for key in ('ids', 'policy', 'values', 'mask'):
        assert received[key].dtype == payload[key].dtype
        np.testing.assert_array_equal(received[key], payload[key])
        # Numeric equality alone would miss signed zero or a changed NaN payload.
        assert received[key].tobytes() == payload[key].tobytes()
    assert received['result'] == payload['result']
    # Both caller mutation and slot reuse are independent of the old reply.
    received['policy'][0, 0] = -9
    assert payload['policy'][0, 0] == 0
    buffer.pack({'next': np.zeros(1024)})
    assert received['policy'][0, 0] == -9
    assert received['ids'][0] == 2**63 + 7


def test_shared_search_arrays_fall_back_without_truncation():
    buffer = _SharedArrayBuffer(mp.get_context('spawn'), 128)
    for payload in ({'data': np.arange(10000)}, {'data': np.arange(12)[::2]},
                    {'data': np.array(['x', {'k': 1}], dtype=object)}, {'done': True}):
        received = buffer.unpack(pickle.loads(pickle.dumps(buffer.pack(payload))))
        if 'data' in payload:
            np.testing.assert_array_equal(received['data'], payload['data'])
        else:
            assert received == payload


def test_grouped_leaf_wire_roundtrip_preserves_rows_dtypes_and_terminals():
    inputs = {
        4:{'leaf_meta':[{'request_id':7}], 'observation_batch':{
            'ids':np.array([[1,2]],dtype=np.int32),'hp':np.array([[.125]],dtype=np.float32)}},
        1:{'done':True,'result':{'action':3}},
        8:{'leaf_meta':[{'request_id':8},{'request_id':9}], 'observation_batch':{
            'ids':np.arange(8,dtype=np.int32).reshape(2,4)[:,::2],
            'hp':np.array([[.25],[.75]],dtype=np.float32)}},
    }
    restored = _unpack_leaf_responses(pickle.loads(pickle.dumps(_pack_leaf_responses(inputs))))
    assert list(restored)==list(inputs)
    assert restored[1]==inputs[1]
    for index in (4,8):
        assert restored[index]['leaf_meta']==inputs[index]['leaf_meta']
        for key,value in inputs[index]['observation_batch'].items():
            actual=restored[index]['observation_batch'][key]
            assert actual.dtype==value.dtype
            np.testing.assert_array_equal(actual,value)
    # Unpacking a later message cannot overwrite results retained by search.
    later=pickle.loads(pickle.dumps(_pack_leaf_responses(inputs)))
    _unpack_leaf_responses(later)[4]['observation_batch']['hp'].fill(9)
    assert restored[4]['observation_batch']['hp'][0,0]==.125


def test_grouped_leaf_wire_works_for_synchronous_and_streamed_rpc():
    env=GroupedCachedMaskSubprocVecEnv([_make_mask_info_env]*3,
                                    envs_per_worker=2,start_method='spawn')
    try:
        calls={index:((index+10,),{}) for index in range(3)}
        expected=env.env_method_batch('nosl_is_az_cpp_begin',calls)
        stream=env.env_method_stream()
        stream.submit('nosl_is_az_cpp_begin',calls)
        actual={}
        while stream.pending:
            for packet in stream.receive():actual.update(packet.responses)
        for index in range(3):
            assert actual[index]['leaf_meta']==expected[index]['leaf_meta']
            np.testing.assert_array_equal(actual[index]['observation_batch']['x'],[[index+10]])
    finally:
        env.close()


def test_cached_masks_follow_step_and_autoreset_info():
    base = CachedMaskSubprocVecEnv(
        [_make_mask_info_env], start_method="spawn"
    )
    env = VecMonitor(base)
    try:
        env.reset()
        np.testing.assert_array_equal(get_action_masks(env), [[True, False]])

        env.step(np.array([0]))
        np.testing.assert_array_equal(get_action_masks(env), [[False, True]])

        _obs, _reward, done, _info = env.step(np.array([1]))
        assert done.tolist() == [True]
        np.testing.assert_array_equal(get_action_masks(env), [[True, False]])
    finally:
        env.close()


def test_grouped_cached_masks_preserve_logical_order():
    base = GroupedCachedMaskSubprocVecEnv(
        [_make_mask_info_env, _make_mask_info_env, _make_mask_info_env],
        envs_per_worker=2,
        start_method="spawn",
    )
    env = VecMonitor(base)
    try:
        assert len(base.processes) == 2
        assert callable(getattr(env, "env_method_stream"))
        env.reset()
        np.testing.assert_array_equal(
            get_action_masks(env),
            [[True, False], [True, False], [True, False]],
        )
        env.step(np.array([0, 0, 0]))
        np.testing.assert_array_equal(
            get_action_masks(env),
            [[False, True], [False, True], [False, True]],
        )
        # Distinct calls in one group must be routed back to their logical
        # indices without exposing the group's local numbering.
        values = base.env_method_batch(
            "get_wrapper_attr",
            {
                0: (("position",), {}),
                1: (("position",), {}),
                2: (("position",), {}),
            },
        )
        assert values == {0: 1, 1: 1, 2: 1}
    finally:
        env.close()


def test_grouped_environment_restore_preserves_order_and_refreshes_masks():
    base = GroupedCachedMaskSubprocVecEnv(
        [_make_mask_info_env, _make_mask_info_env, _make_mask_info_env],
        envs_per_worker=2,
        start_method="spawn",
    )
    try:
        base.reset()
        observations = base.restore_training_states([
            {"position": 2}, {"position": 5}, {"position": 8},
        ])
        np.testing.assert_array_equal(observations[:, 0], [2, 5, 8])
        np.testing.assert_array_equal(
            get_action_masks(base),
            [[True, False], [False, True], [True, False]],
        )
        assert base.env_method("get_wrapper_attr", "position") == [2, 5, 8]
    finally:
        base.close()


def test_grouped_method_stream_reuses_groups_without_changing_env_order():
    base = GroupedCachedMaskSubprocVecEnv(
        [_make_mask_info_env, _make_mask_info_env, _make_mask_info_env],
        envs_per_worker=2,
        start_method="spawn",
    )
    try:
        base.reset()
        base.step(np.array([0, 0, 0]))
        stream = base.env_method_stream()
        calls = {
            index: (("position",), {})
            for index in range(3)
        }
        stream.submit("get_wrapper_attr", calls)
        packets = []
        while stream.pending:
            packets.extend(stream.receive())

        assert {packet.group_index for packet in packets} == {0, 1}
        assert {
            index: value
            for packet in packets
            for index, value in packet.responses.items()
        } == {0: 1, 1: 1, 2: 1}

        stream.submit("get_wrapper_attr", calls)
        repeated = {}
        while stream.pending:
            for packet in stream.receive():
                repeated.update(packet.responses)
        assert repeated == {0: 1, 1: 1, 2: 1}
    finally:
        base.close()


def test_grouped_worker_hard_exit_fails_fast_and_is_reaped():
    base = GroupedCachedMaskSubprocVecEnv(
        [_make_mask_info_env, _make_mask_info_env],
        envs_per_worker=2,
        start_method="spawn",
    )
    try:
        base.reset()
        with pytest.raises(
            RuntimeError,
            match=r"group=0 logical_envs=\[0, 1\].*exitcode=23",
        ):
            base.env_method("hard_exit", indices=[0])
        assert not base.processes[0].is_alive()
    finally:
        base.close()


def test_distributed_parser_accepts_grouped_worker_geometry():
    args = build_distributed_parser().parse_args([
        "--rank", "0",
        "--world-size", "1",
        "--checkpoint", "model.zip",
        "--n-envs", "288",
        "--envs-per-worker", "12",
        "--nosl-is-az-replay-max-old-fraction", "0.9",
        "--nosl-is-az-inference-batch-size", "48",
        "--nosl-is-az-inference-max-wait-ms", "0.25",
        "--nosl-is-az-inference-schedule", "adaptive",
    ])
    assert args.n_envs == 288
    assert args.envs_per_worker == 12
    assert args.nosl_is_az_replay_max_old_fraction == 0.9
    assert args.nosl_is_az_inference_batch_size == 48
    assert args.nosl_is_az_inference_max_wait_ms == 0.25
    assert args.nosl_is_az_inference_schedule == "adaptive"
