import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from bridge.schemas.combat_state import normalize_sim
from env.core import vocab as V
from rl.accumulating_ppo import (
    AccumulatingMaskablePPO,
    _leaf_observation_as_tensor,
    _fill_nosl_combat_value_targets,
    ppo_loss_components,
)
from rl.nosl_is_az import NOSLISAZConfig, NOSLISAZSearch, NOSLISAZSession
from rl.nosl_replay import NOSLISAZReplayBuffer
from rl.nosl_sampler import NOSLSampler


class _Battle:
    redeterminizations = 0

    def __init__(self):
        self.is_battle_over = False
        self.steps = 0

    def clone(self):
        return copy.copy(self)

    def nosl_redeterminize(self, nonce):
        type(self).redeterminizations += 1
        self.nonce = int(nonce)

    def projected_exit_hp(self, _gc):
        return 100.0 - self.steps


class _Branch:
    def __init__(self):
        self.thick = SimpleNamespace(bc=_Battle(), gc=object())

    def fork_combat(self):
        branch = copy.copy(self)
        branch.thick = SimpleNamespace(
            bc=self.thick.bc.clone(), gc=self.thick.gc
        )
        return branch

    def action_masks(self):
        return np.asarray([True, True, False], dtype=bool)

    def _obs(self):
        return {"phase": np.asarray([1], dtype=np.int64)}

    def search_step(self, action, *, validate=False):
        del validate
        self.thick.bc.steps += 1 + int(action)
        if self.thick.bc.steps >= 4:
            self.thick.bc.is_battle_over = True
        return self.thick.bc.is_battle_over, self.action_masks()


class _DeepBranch(_Branch):
    def action_masks(self):
        return np.asarray([True, False, False], dtype=bool)

    def search_step(self, action, *, validate=False):
        del validate
        if int(action) != 0:
            raise AssertionError(f"unexpected action: {action}")
        self.thick.bc.steps += 1
        return False, self.action_masks()


class _ToySplitPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.shared = torch.nn.Linear(2, 2)
        self.core.combat = torch.nn.Linear(2, 2)
        self.core.overworld = torch.nn.Linear(2, 2)
        self.outside_core = torch.nn.Parameter(torch.ones(1))


class _ReplayCore(torch.nn.Module):
    combat_observation_keys = frozenset({"state"})

    def __init__(self):
        super().__init__()
        self.policy_head = torch.nn.Linear(1, 2)
        self.value_head = torch.nn.Linear(1, 1)

    def actor_critic(self, obs):
        state = obs["state"].float()
        return self.policy_head(state), self.value_head(state)

    combat_actor_critic = actor_critic


class _ReplayPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _ReplayCore()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05)


class NOSLISAZTest(unittest.TestCase):
    def test_native_boundary_marker_survives_canonical_normalization(self):
        raw = {
            "turn": 0,
            "post_enemy_turn_boundary": True,
            "player": {
                "hp": 10,
                "max_hp": 20,
                "block": 0,
                "energy": 3,
                "powers": [],
                "combat_counters": {},
            },
            "monsters": [],
            "hand": [],
            "draw": [],
            "discard": [],
            "exhaust": [],
        }
        self.assertTrue(normalize_sim(raw)["post_enemy_turn_boundary"])
        raw.pop("post_enemy_turn_boundary")
        self.assertFalse(normalize_sim(raw)["post_enemy_turn_boundary"])

    def test_leaf_dtype_packing_preserves_fields(self):
        observation = {
            "category": np.arange(6, dtype=np.int64).reshape(2, 3),
            "scalar": np.arange(8, dtype=np.float32).reshape(2, 4),
        }
        packed = _leaf_observation_as_tensor(observation, "cpu")
        self.assertTrue(torch.equal(
            packed["category"], torch.as_tensor(observation["category"])
        ))
        self.assertTrue(torch.equal(
            packed["scalar"], torch.as_tensor(observation["scalar"])
        ))

    @staticmethod
    def _replay_sample(value: int) -> dict:
        return {
            "observation": {
                "category": np.asarray([value, value + 1], dtype=np.int64),
                "scalar": np.asarray([value / 10.0], dtype=np.float32),
            },
            "action_mask": np.asarray([True, True, False]),
            "policy_target": np.asarray([0.25, 0.75, 0.0], dtype=np.float32),
            "value_target": value / 10.0,
        }

    def test_replay_is_fifo_and_samples_without_replacement(self):
        observation_space = spaces.Dict({
            "category": spaces.Box(0, 1000, shape=(2,), dtype=np.int64),
            "scalar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        replay = NOSLISAZReplayBuffer(
            capacity=3,
            observation_space=observation_space,
            observation_keys=observation_space.spaces,
            action_dim=3,
        )
        replay.add_samples([self._replay_sample(i) for i in range(5)])

        self.assertEqual(replay.size, 3)
        self.assertEqual(replay.total_added, 5)
        all_samples = replay.get_batch(np.arange(replay.size))
        self.assertEqual(set(all_samples["sample_ids"].tolist()), {2, 3, 4})
        self.assertTrue(np.all(all_samples["policy_weights"] == 1.0))
        indices = replay.sample_indices(3, np.random.default_rng(7))
        self.assertEqual(len(set(indices.tolist())), 3)

    def test_replay_sampling_caps_old_fraction_and_covers_small_fresh_set(self):
        observation_space = spaces.Dict({
            "category": spaces.Box(0, 1000, shape=(2,), dtype=np.int64),
            "scalar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        replay = NOSLISAZReplayBuffer(
            capacity=10,
            observation_space=observation_space,
            observation_keys=observation_space.spaces,
            action_dim=3,
        )
        replay.add_samples([self._replay_sample(i) for i in range(8)])
        fresh_start_id = replay.total_added
        replay.add_samples([self._replay_sample(i) for i in range(8, 10)])

        indices = replay.sample_indices(
            10,
            np.random.default_rng(7),
            fresh_start_id=fresh_start_id,
            max_old_fraction=0.5,
        )
        batch = replay.get_batch(indices)
        old = batch["sample_ids"] < fresh_start_id

        self.assertEqual(indices.shape, (10,))
        self.assertEqual(int(old.sum()), 5)
        self.assertEqual(set(batch["sample_ids"][~old].tolist()), {8, 9})

    def test_replay_freshness_protection_rejects_an_all_old_batch(self):
        observation_space = spaces.Dict({
            "category": spaces.Box(0, 1000, shape=(2,), dtype=np.int64),
            "scalar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        replay = NOSLISAZReplayBuffer(
            capacity=4,
            observation_space=observation_space,
            observation_keys=observation_space.spaces,
            action_dim=3,
        )
        replay.add_samples([self._replay_sample(i) for i in range(4)])

        with self.assertRaisesRegex(ValueError, "without fresh roots"):
            replay.sample_indices(
                4,
                np.random.default_rng(7),
                fresh_start_id=replay.total_added,
                max_old_fraction=0.9,
            )

    def test_replay_uses_compact_lazy_storage(self):
        observation_space = spaces.Dict({
            "category": spaces.Box(0, 1000, shape=(2,), dtype=np.int64),
            "scalar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        replay = NOSLISAZReplayBuffer(
            capacity=4,
            observation_space=observation_space,
            observation_keys=observation_space.spaces,
            action_dim=3,
        )
        self.assertEqual(replay.allocated_bytes, 0)
        replay.add_samples([self._replay_sample(1)])
        self.assertEqual(replay.observations["category"].dtype, np.uint16)
        self.assertEqual(replay.observations["scalar"].dtype, np.float16)
        self.assertGreater(replay.allocated_bytes, 0)
        self.assertEqual(
            replay.allocated_bytes, replay.estimated_capacity_bytes()
        )

    def test_replay_preserves_value_only_boundary_weight(self):
        observation_space = spaces.Dict({
            "category": spaces.Box(0, 1000, shape=(2,), dtype=np.int64),
            "scalar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        replay = NOSLISAZReplayBuffer(
            capacity=2,
            observation_space=observation_space,
            observation_keys=observation_space.spaces,
            action_dim=3,
        )
        root = self._replay_sample(1)
        boundary = self._replay_sample(2)
        boundary["policy_weight"] = 0.0
        replay.add_samples([root, boundary])

        batch = replay.get_batch(np.arange(2))
        self.assertEqual(batch["policy_weights"].tolist(), [1.0, 0.0])

    def test_training_samples_old_and_new_roots_from_replay(self):
        model = object.__new__(AccumulatingMaskablePPO)
        # Simulate an older checkpoint: the replay toggle did not exist yet.
        # The training path must retain the historical replay-enabled default.
        self.assertFalse(hasattr(model, "nosl_is_az_replay_enabled"))
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        model.policy = _ReplayPolicy()
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 1.0
        model.nosl_is_az_batch_size = 4
        model.nosl_is_az_micro_batch_size = 2
        model.nosl_is_az_replay_capacity = 8
        model.nosl_is_az_replay_min_size = 1
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 1.0
        model.nosl_is_az_value_coef = 0.5
        model._nosl_is_az_replay = None
        model._nosl_is_az_replay_prepare_stats = {}
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)

        def sample(value):
            return {
                "observation": {
                    "state": np.asarray([value], dtype=np.float32),
                },
                "action_mask": np.asarray([True, True]),
                "policy_target": np.asarray([0.75, 0.25], dtype=np.float32),
                "value_target": value / 10.0,
            }

        model._nosl_is_az_samples = [sample(value) for value in range(4)]
        first = model._train_nosl_is_az()
        self.assertEqual(first["fresh_roots"], 4.0)
        self.assertEqual(first["replay_size"], 4.0)
        self.assertEqual(first["gradient_batches"], 2.0)
        self.assertEqual(first["updates"], 1.0)

        np.random.seed(9)
        model._nosl_is_az_samples = [sample(4), sample(5)]
        second = model._train_nosl_is_az()
        self.assertEqual(second["fresh_roots"], 2.0)
        self.assertEqual(second["replay_size"], 6.0)
        self.assertEqual(second["replay_sampled_roots"], 4.0)
        self.assertGreaterEqual(second["replay_old_fraction"], 0.5)

        model._nosl_is_az_samples = []
        third = model._train_nosl_is_az()
        self.assertEqual(third["fresh_roots"], 0.0)
        self.assertEqual(third["replay_sampled_roots"], 4.0)
        self.assertEqual(third["replay_old_fraction"], 1.0)

    def test_replay_disabled_trains_all_fresh_samples_without_allocating_buffer(self):
        model = object.__new__(AccumulatingMaskablePPO)
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        model.policy = _ReplayPolicy()
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 1.0
        # Deliberately make batch_size smaller than the fresh sample count.
        model.nosl_is_az_batch_size = 2
        model.nosl_is_az_micro_batch_size = 3
        model.nosl_is_az_replay_capacity = 8
        model.nosl_is_az_replay_min_size = 1
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 1.0
        model.nosl_is_az_value_coef = 0.5
        model.nosl_is_az_replay_enabled = False
        model._nosl_is_az_replay = None
        model._nosl_is_az_replay_prepare_stats = {}
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)

        def sample(value):
            return {
                "observation": {
                    "state": np.asarray([value], dtype=np.float32),
                },
                "action_mask": np.asarray([True, True]),
                "policy_target": np.asarray([0.75, 0.25], dtype=np.float32),
                "value_target": value / 10.0,
            }

        model._nosl_is_az_samples = [sample(value) for value in range(7)]
        result = model._train_nosl_is_az()

        # Every valid fresh root is used; nosl_is_az_batch_size is a replay
        # sampling limit and must not truncate this no-replay path.
        self.assertEqual(result["fresh_roots"], 7.0)
        self.assertEqual(result["roots"], 7.0)
        self.assertEqual(result["replay_sampled_roots"], 0.0)
        self.assertEqual(result["replay_old_fraction"], 0.0)
        self.assertEqual(result["replay_size"], 0.0)
        self.assertEqual(result["replay_enabled"], 0.0)
        self.assertEqual(result["gradient_batches"], 3.0)
        self.assertEqual(result["updates"], 1.0)
        self.assertEqual(model._nosl_is_az_samples, [])
        self.assertIsNone(model._nosl_is_az_replay)

    def test_replay_disabled_can_defer_consuming_fresh_samples(self):
        model = object.__new__(AccumulatingMaskablePPO)
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        model.policy = _ReplayPolicy()
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 1.0
        model.nosl_is_az_batch_size = 4
        model.nosl_is_az_micro_batch_size = 2
        model.nosl_is_az_replay_capacity = 8
        model.nosl_is_az_replay_min_size = 1
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 1.0
        model.nosl_is_az_value_coef = 0.5
        model.nosl_is_az_replay_enabled = False
        model._nosl_is_az_replay = None
        model._nosl_is_az_replay_prepare_stats = {}
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)

        def sample(value):
            return {
                "observation": {
                    "state": np.asarray([value], dtype=np.float32),
                },
                "action_mask": np.asarray([True, True]),
                "policy_target": np.asarray([0.75, 0.25], dtype=np.float32),
                "value_target": value / 10.0,
            }

        model._nosl_is_az_samples = [sample(value) for value in range(4)]
        first = model._train_nosl_is_az(consume_fresh_samples=False)
        second = model._train_nosl_is_az(
            prepare_replay=False,
            consume_fresh_samples=True,
        )

        self.assertEqual(first["roots"], 4.0)
        self.assertEqual(second["roots"], 4.0)
        self.assertEqual(first["value_target_mean"], second["value_target_mean"])
        self.assertEqual(first["value_target_std"], second["value_target_std"])
        self.assertEqual(model._nosl_is_az_fresh_samples, [])

    def test_replay_disabled_with_no_fresh_samples_keeps_buffer_unallocated(self):
        model = object.__new__(AccumulatingMaskablePPO)
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        model.policy = _ReplayPolicy()
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 1.0
        model.nosl_is_az_batch_size = 2
        model.nosl_is_az_micro_batch_size = 3
        model.nosl_is_az_replay_capacity = 8
        model.nosl_is_az_replay_min_size = 1
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 1.0
        model.nosl_is_az_value_coef = 0.5
        model.nosl_is_az_replay_enabled = False
        model._nosl_is_az_replay = None
        model._nosl_is_az_replay_prepare_stats = {}
        model._nosl_is_az_samples = []
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)

        result = model._train_nosl_is_az()

        self.assertEqual(result["fresh_roots"], 0.0)
        self.assertEqual(result["roots"], 0.0)
        self.assertEqual(result["updates"], 0.0)
        self.assertEqual(result["replay_size"], 0.0)
        self.assertEqual(result["replay_sampled_roots"], 0.0)
        self.assertEqual(result["replay_old_fraction"], 0.0)
        self.assertEqual(result["replay_enabled"], 0.0)
        self.assertIsNone(model._nosl_is_az_replay)
        self.assertEqual(model._nosl_is_az_samples, [])

    def test_value_only_boundary_trains_with_an_empty_action_mask(self):
        model = object.__new__(AccumulatingMaskablePPO)
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        model.policy = _ReplayPolicy()
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 1.0
        model.nosl_is_az_batch_size = 2
        model.nosl_is_az_micro_batch_size = 2
        model.nosl_is_az_replay_capacity = 8
        model.nosl_is_az_replay_min_size = 1
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 1.0
        model.nosl_is_az_value_coef = 1.0
        model.nosl_is_az_replay_enabled = False
        model._nosl_is_az_replay = None
        model._nosl_is_az_replay_prepare_stats = {}
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)
        model._nosl_is_az_samples = [
            {
                "observation": {"state": np.asarray([1], dtype=np.float32)},
                "action_mask": np.asarray([True, True]),
                "policy_target": np.asarray([0.75, 0.25], dtype=np.float32),
                "policy_weight": 1.0,
                "value_target": 0.2,
            },
            {
                "observation": {"state": np.asarray([2], dtype=np.float32)},
                "action_mask": np.asarray([False, False]),
                "policy_target": np.asarray([0.0, 0.0], dtype=np.float32),
                "policy_weight": 0.0,
                "value_target": 0.7,
            },
        ]
        value_before = model.policy.core.value_head.weight.detach().clone()

        result = model._train_nosl_is_az()

        self.assertEqual(result["roots"], 2.0)
        self.assertEqual(result["metric_count"], 2.0)
        self.assertEqual(model._nosl_is_az_policy_weight_sum, 1.0)
        self.assertTrue(np.isfinite(result["policy_loss"]))
        self.assertTrue(np.isfinite(result["value_loss"]))
        self.assertTrue(np.isfinite(result["finite_fraction"]))
        self.assertFalse(torch.equal(
            value_before, model.policy.core.value_head.weight.detach()
        ))

    def test_root_redeterminizes_once_per_simulation_and_returns_legal_target(self):
        _Battle.redeterminizations = 0
        search = NOSLISAZSearch(
            NOSLISAZConfig(simulations=6, depth=2)
        )
        source = _Branch()
        result = search.choose(
            source,
            np.asarray([True, True, False]),
            prior_logits=np.asarray([3.0, 0.0, -100.0]),
            sampler=NOSLSampler(123),
            state_nonce=456,
            decision_index=7,
        )

        self.assertIn(result.action, (0, 1))
        self.assertEqual(int(result.root_visits.sum()), 6)
        self.assertAlmostEqual(float(result.visit_policy.sum()), 1.0)
        self.assertEqual(float(result.visit_policy[2]), 0.0)
        self.assertGreaterEqual(result.root_value, 0.0)
        self.assertEqual(_Battle.redeterminizations, 6)
        self.assertEqual(source.thick.bc.steps, 0)

    def test_nonterminal_leaf_uses_value_bootstrap_without_rollout(self):
        search = NOSLISAZSearch(NOSLISAZConfig(simulations=4, depth=1))
        source = _Branch()
        result = search.choose(
            source,
            np.asarray([True, True, False]),
            prior_logits=np.asarray([0.0, 0.0, -100.0]),
            sampler=NOSLSampler(123),
            state_nonce=456,
            decision_index=0,
            leaf_value=0.375,
        )
        self.assertAlmostEqual(result.root_value, 0.375)

    def test_combat_samples_are_excluded_from_ppo_objective(self):
        phase = torch.tensor(
            [
                [V.PHASE_VOCAB_INDEX["combat"]],
                [V.PHASE_VOCAB_INDEX["map"]],
            ]
        )
        result = ppo_loss_components(
            hierarchical=True,
            normalize_advantage=False,
            observations={"phase": phase},
            advantages=torch.ones(2),
            returns=torch.ones(2),
            old_values=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            values=torch.zeros(2, requires_grad=True),
            log_prob=torch.zeros(2),
            entropy=torch.zeros(2),
            clip_range=0.2,
            clip_range_vf=None,
            skip_combat=True,
        )
        self.assertEqual(result.combat_samples, 0)
        self.assertEqual(result.overworld_samples, 1)
        self.assertEqual(result.valid_samples, 1)

    def test_all_roots_in_a_completed_fight_share_final_hp_target(self):
        samples = [
            {"value_target": np.nan},
            {"value_target": np.nan},
            {"value_target": np.nan},
        ]
        count = _fill_nosl_combat_value_targets(samples, 0.73)
        self.assertEqual(count, 3)
        self.assertEqual(
            [sample["value_target"] for sample in samples],
            [0.73, 0.73, 0.73],
        )

    def test_value_function_can_inspect_the_leaf_branch(self):
        branch = _Branch()
        search = NOSLISAZSearch(NOSLISAZConfig(simulations=1, depth=1))
        result = search.choose(
            branch,
            np.asarray([True, True, False]),
            prior_logits=np.asarray([1.0, 0.0, -100.0]),
            sampler=NOSLSampler(123),
            state_nonce=456,
            decision_index=0,
            value_fn=lambda leaf: float(leaf.thick.bc.steps) / 10.0,
        )
        self.assertGreaterEqual(result.root_value, 0.0)

    def test_staged_session_requests_and_backprops_each_leaf(self):
        session = NOSLISAZSession(
            _Branch(),
            np.asarray([True, True, False]),
            prior_logits=np.asarray([1.0, 0.0, -100.0]),
            sampler=NOSLSampler(123),
            state_nonce=456,
            decision_index=0,
            config=NOSLISAZConfig(simulations=3, depth=1),
        )
        request = session.next_request()
        requests = 0
        while not request["done"]:
            self.assertIn("observation", request)
            requests += 1
            request = session.backprop(
                0.25,
                policy_logits=np.asarray([1.0, 0.0, -100.0]),
            )
        self.assertEqual(requests, 3)
        self.assertEqual(int(request["result"]["root_visits"].sum()), 3)

    def test_staged_session_expands_first_unseen_leaf_progressively(self):
        session = NOSLISAZSession(
            _DeepBranch(),
            np.asarray([True, False, False]),
            prior_logits=np.asarray([1.0, -100.0, -100.0]),
            sampler=NOSLSampler(123),
            state_nonce=456,
            decision_index=0,
            config=NOSLISAZConfig(simulations=3, depth=3),
        )

        leaf_depths = []
        request = session.next_request()
        while not request["done"]:
            leaf_depths.append(session.pending[0].thick.bc.steps)
            request = session.backprop(
                0.25,
                policy_logits=np.asarray([1.0, -100.0, -100.0]),
            )

        self.assertEqual(leaf_depths, [1, 2, 3])
        self.assertEqual(int(request["result"]["root_visits"].sum()), 3)

    def test_inner_only_freezes_everything_except_combat_specialist(self):
        model = object.__new__(AccumulatingMaskablePPO)
        model.nosl_is_az = True
        model.nosl_is_az_inner_only = True
        model.policy = _ToySplitPolicy()

        frozen, trainable = model.configure_nosl_is_az_inner_only()

        self.assertGreater(frozen, 0)
        self.assertGreater(trainable, 0)
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.policy.core.combat.parameters()
        ))
        self.assertTrue(all(
            not parameter.requires_grad
            for name, parameter in model.policy.named_parameters()
            if not name.startswith("core.combat.")
        ))


if __name__ == "__main__":
    unittest.main()
