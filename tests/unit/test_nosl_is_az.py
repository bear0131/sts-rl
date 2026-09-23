import copy
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from bridge.schemas.combat_state import normalize_sim
from env.core import vocab as V
from rl.algorithms.accumulating_ppo import (
    AccumulatingMaskablePPO,
    _append_nosl_value_only_sample,
    _leaf_observation_as_tensor,
    _fill_nosl_combat_value_targets,
    _normalized_player_hp,
    ppo_loss_components,
)
from env.search.nosl_is_az import NOSLISAZConfig, NOSLISAZSearch, NOSLISAZSession
from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer
from env.search.nosl_sampler import NOSLSampler


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


class NOSLLearnerDiagnosticsTest(unittest.TestCase):
    """Diagnostics must not change the bounded, sample-weighted objective."""

    def setUp(self):
        self.model = object.__new__(AccumulatingMaskablePPO)
        model = self.model
        model.observation_space = spaces.Dict({
            "state": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
        })
        model.action_space = spaces.Discrete(2)
        with torch.random.fork_rng():
            torch.manual_seed(5311)
            model.policy = _ReplayPolicy()
        model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.01)
        model.device = torch.device("cpu")
        model.amp_mode = "off"
        model.max_grad_norm = 0.5
        model.nosl_is_az_batch_size = 5
        model.nosl_is_az_micro_batch_size = 2
        model.nosl_is_az_replay_enabled = False
        model.nosl_is_az_updates = 1
        model.nosl_is_az_policy_coef = 0.7
        model.nosl_is_az_value_coef = 1.3
        model._logger = SimpleNamespace(record=lambda *_args, **_kwargs: None)
        # A complete value-only microbatch and a short, fractionally weighted
        # final microbatch distinguish policy population from policy weights.
        weights = [1.0, 2.0, 0.0, 0.0, 0.25]
        self.samples = [{
            "observation": {"state": np.asarray([index - 1], dtype=np.float32)},
            "action_mask": np.asarray([weight > 0, weight > 0]),
            "policy_target": np.asarray(
                [0.75, 0.25] if weight else [0.0, 0.0], dtype=np.float32),
            "policy_weight": weight,
            "value_target": (index + 1) / 8.0,
        } for index, weight in enumerate(weights)]
        model._nosl_is_az_samples = self.samples

    def _reference_gradients(self, reference, samples, *, gradient_scale=1.0, totals=None):
        if totals is None:
            totals = ({}, {}, {}, np.zeros(3, dtype=np.float64))
        expected_sum, expected_sumsq, expected_count, expected_losses = totals
        reference.optimizer.zero_grad()
        for start in range(0, len(samples), 2):
            rows = samples[start:start + 2]
            obs = {"state": torch.from_numpy(np.stack([
                row["observation"]["state"] for row in rows]))}
            logits, values = reference.core.actor_critic(obs)
            masks = torch.from_numpy(np.stack([row["action_mask"] for row in rows]))
            targets = torch.from_numpy(np.stack([row["policy_target"] for row in rows]))
            weights = torch.tensor([row["policy_weight"] for row in rows])
            value_targets = torch.tensor([row["value_target"] for row in rows])
            log_probs = torch.log_softmax(torch.where(
                masks, logits, torch.full_like(logits, -1e8)), dim=-1)
            total = targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            targets = targets / total
            cross_entropy = -(targets * log_probs).sum(dim=-1)
            squared_error = torch.nn.functional.mse_loss(
                values.float().flatten(), value_targets, reduction="none")
            # The reference objective specifies the same microbatch and
            # distributed scale; no diagnostics participate in backward.
            objective = 0.7 * cross_entropy * weights + 1.3 * squared_error
            (objective.sum() / len(samples) * gradient_scale).backward()
            expected_losses += [
                float((cross_entropy * weights).sum().detach()),
                float(weights.sum()), float(squared_error.sum().detach()),
            ]
            target_log = torch.where(targets > 0, targets.clamp_min(1e-8).log(),
                                     torch.zeros_like(targets))
            target_entropy = -(targets * target_log).sum(dim=-1)
            probabilities = torch.where(masks, log_probs.exp(), torch.zeros_like(log_probs))
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            policy_log = torch.where(probabilities > 0, probabilities.clamp_min(1e-8).log(),
                                     torch.zeros_like(probabilities))
            predicted = values.float().flatten()
            policy_metrics = {
                "target_entropy": target_entropy,
                "policy_entropy": -(probabilities * policy_log).sum(dim=-1),
                "target_policy_kl": cross_entropy - target_entropy,
                "policy_cross_entropy": cross_entropy,
                "target_effective_actions": target_entropy.exp(),
                "target_top1_agreement": (targets.argmax(-1) == probabilities.argmax(-1)).float(),
                "legal_actions": masks.sum(dim=-1).float(),
                "illegal_target_mass": (targets * (~masks).to(targets.dtype)).sum(dim=-1),
                "target_sum_error": (total.squeeze(-1) - 1.0).abs(),
            }
            metrics = {name: value[weights > 0] for name, value in policy_metrics.items()}
            metrics.update({
                "value_target": value_targets,
                "pred_value": predicted,
                "value_mae": (predicted - value_targets).abs(),
                "value_bias": predicted - value_targets,
                "finite_fraction": (torch.isfinite(logits).all(-1) & torch.isfinite(predicted)).float(),
            })
            for name, tensor in metrics.items():
                array = tensor.detach().float().numpy()
                array = array[np.isfinite(array)].astype(np.float64)
                expected_sum[name] = expected_sum.get(name, 0.0) + float(array.sum())
                expected_sumsq[name] = expected_sumsq.get(name, 0.0) + float(np.square(array).sum())
                expected_count[name] = expected_count.get(name, 0) + array.size
        return totals

    def test_diagnostics_gradients_and_adam_match_fixed_microbatch_objective(self):
        model = self.model
        reference = copy.deepcopy(model.policy)
        expected_sum, expected_sumsq, expected_count, expected_losses = self._reference_gradients(
            reference, self.samples, gradient_scale=1.5)
        result = model._train_nosl_is_az(step_optimizer=False, gradient_scale=1.5)
        self.assertEqual(result["gradient_batches"], 3)
        self.assertEqual(model._nosl_is_az_metric_sums, expected_sum)
        self.assertEqual(model._nosl_is_az_metric_sumsq, expected_sumsq)
        self.assertEqual(model._nosl_is_az_metric_counts, expected_count)
        self.assertEqual(model._nosl_is_az_policy_loss_sum, expected_losses[0])
        self.assertEqual(model._nosl_is_az_policy_weight_sum, expected_losses[1])
        self.assertEqual(model._nosl_is_az_value_loss_sum, expected_losses[2])
        for actual, expected in zip(model.policy.parameters(), reference.parameters(), strict=True):
            self.assertTrue(torch.equal(actual.grad, expected.grad))
        for policy in (model.policy, reference):
            torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
        for actual, expected in zip(model.policy.parameters(), reference.parameters(), strict=True):
            self.assertTrue(torch.equal(actual, expected))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(model.policy.optimizer.state[actual][key],
                                            reference.optimizer.state[expected][key]))

    def _assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self._assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected, strict=True):
                self._assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def _policy_snapshot(policy):
        return {
            "weights": copy.deepcopy(policy.state_dict()),
            "adam": copy.deepcopy(policy.optimizer.state_dict()),
            "gradients": [None if p.grad is None else p.grad.detach().clone()
                          for p in policy.parameters()],
        }

    def test_two_update_diagnostics_reuse_matches_sequential_objective_and_rng(self):
        saved_rng = np.random.get_state()
        try:
            for replay_enabled in (False, True):
                with self.subTest(replay=replay_enabled):
                    self.setUp()
                    model = self.model
                    model.nosl_is_az_replay_enabled = replay_enabled
                    model.nosl_is_az_replay_capacity = 16
                    model.nosl_is_az_updates = 2
                    reference = copy.deepcopy(model.policy)
                    np.random.seed(5311)
                    for _round in range(2):
                        model._nosl_is_az_samples = copy.deepcopy(self.samples)
                        prepare = model._prepare_nosl_is_az_replay()
                        replay = copy.deepcopy(model._nosl_is_az_replay) if replay_enabled else None
                        initial_rng = np.random.get_state()
                        initial_torch_rng = torch.get_rng_state()
                        totals = None
                        expected_states, expected_norms = [], []
                        for _ in range(2):
                            rows = self.samples
                            if replay is not None:
                                indices = replay.sample_indices(
                                    5, fresh_start_id=int(prepare["fresh_start_id"]),
                                    max_old_fraction=1.0)
                                batch = replay.get_batch(indices)
                                rows = [{
                                    "observation": {"state": np.asarray(
                                        batch["observations"]["state"][i], dtype=np.float32)},
                                    "action_mask": batch["action_masks"][i],
                                    "policy_target": np.asarray(batch["policy_targets"][i], dtype=np.float32),
                                    "policy_weight": float(batch["policy_weights"][i]),
                                    "value_target": float(batch["value_targets"][i]),
                                } for i in range(5)]
                            totals = self._reference_gradients(reference, rows, totals=totals)
                            norm = torch.nn.utils.clip_grad_norm_(
                                reference.parameters(), model.max_grad_norm)
                            expected_norms.append(float(norm))
                            reference.optimizer.step()
                            expected_states.append(self._policy_snapshot(reference))
                            reference.optimizer.zero_grad()
                        expected_rng = np.random.get_state()
                        np.random.set_state(initial_rng)
                        actual_states = []
                        original_step = model.policy.optimizer.step

                        def record_step(*args, **kwargs):
                            result = original_step(*args, **kwargs)
                            actual_states.append(self._policy_snapshot(model.policy))
                            return result

                        with patch.object(model.policy.optimizer, "step", record_step):
                            result = model._train_nosl_is_az(prepare_replay=False)
                        expected_sum, expected_sumsq, expected_count, expected_losses = totals
                        self._assert_nested_equal(actual_states, expected_states)
                        self._assert_nested_equal(np.random.get_state(), expected_rng)
                        self._assert_nested_equal(torch.get_rng_state(), initial_torch_rng)
                        self.assertEqual(model._nosl_is_az_metric_sums, expected_sum)
                        self.assertEqual(model._nosl_is_az_metric_sumsq, expected_sumsq)
                        self.assertEqual(model._nosl_is_az_metric_counts, expected_count)
                        self.assertEqual(model._nosl_is_az_policy_loss_sum, expected_losses[0])
                        self.assertEqual(model._nosl_is_az_policy_weight_sum, expected_losses[1])
                        self.assertEqual(model._nosl_is_az_value_loss_sum, expected_losses[2])
                        self.assertEqual(result["policy_loss"], expected_losses[0] / max(1.0, expected_losses[1]))
                        self.assertEqual(result["value_loss"], expected_losses[2] / 10)
                        self.assertEqual(result["gradient_norm_preclip"], np.mean(expected_norms))
                        self.assertEqual(result["gradient_batches"], 6)
                        self.assertEqual(result["metric_count"], 10)
                        self.assertEqual(result["updates"], 2)
                        timing_fields = (
                            "learner_replay_sample_sec",
                            "learner_materialize_sec",
                            "learner_cpu_metrics_sec",
                            "learner_h2d_sec",
                            "learner_forward_sec",
                            "learner_backward_sec",
                            "learner_optimizer_sec",
                        )
                        for name in timing_fields:
                            self.assertTrue(np.isfinite(result[name]), name)
                            self.assertGreaterEqual(result[name], 0.0, name)
                        self.assertFalse(
                            any(name in model.__dict__ for name in timing_fields)
                        )
                        self.assertTrue(all(p.grad is None for p in model.policy.parameters()))
                        if not replay_enabled:
                            self.assertEqual(model._nosl_is_az_fresh_samples, [])
        finally:
            np.random.set_state(saved_rng)

    def test_two_update_timing_resets_between_calls(self):
        model = self.model
        model.nosl_is_az_updates = 2
        timing_fields = (
            "learner_replay_sample_sec",
            "learner_materialize_sec",
            "learner_cpu_metrics_sec",
            "learner_h2d_sec",
            "learner_forward_sec",
            "learner_backward_sec",
            "learner_optimizer_sec",
        )

        def timed_call():
            model._nosl_is_az_samples = copy.deepcopy(self.samples)
            ticks = iter(float(value) for value in range(1000))
            with patch(
                "rl.algorithms.accumulating_ppo.time.perf_counter",
                side_effect=lambda: next(ticks),
            ):
                return model._train_nosl_is_az()

        first = timed_call()
        second = timed_call()

        self.assertEqual(first["updates"], 2)
        self.assertEqual(second["updates"], 2)
        for name in timing_fields:
            self.assertEqual(first[name], second[name], name)
            self.assertTrue(np.isfinite(first[name]), name)
            self.assertGreaterEqual(first[name], 0.0, name)
        self.assertFalse(any(name in model.__dict__ for name in timing_fields))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device; CPU objective test always runs")
    def test_cuda_temporary_host_sources_match_blocking_transfers_and_are_released(self):
        # Two fields per dtype exercise temporary packed host allocations.
        self.model.observation_space = spaces.Dict({
            **self.model.observation_space.spaces,
            "extra": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
            "category": spaces.Box(0, 10, shape=(1,), dtype=np.int64),
            "category_extra": spaces.Box(0, 10, shape=(1,), dtype=np.int64),
        })
        self.model.policy.core.combat_observation_keys = frozenset(self.model.observation_space.spaces)
        for index, sample in enumerate(self.samples):
            sample["observation"].update({
                "extra": np.asarray([index / 4], dtype=np.float32),
                "category": np.asarray([index], dtype=np.int64),
                "category_extra": np.asarray([5 - index], dtype=np.int64),
            })
        original_to, original_from_numpy = torch.Tensor.to, torch.from_numpy

        def run(blocking):
            model = copy.deepcopy(self.model)
            model.device = torch.device("cuda")
            model.policy.to(model.device)
            core = model.policy.core

            def combat(obs):
                # Consume every field; allocation churn occurs while preceding
                # nonblocking copies may still be queued on this stream.
                scratch = [np.full((2, 2), -100, dtype=dtype)
                           for dtype in (np.float32, np.int64) for _ in range(8)]
                state = obs["state"] + obs["extra"] + (
                    obs["category"] + obs["category_extra"]).float() / 8
                del scratch
                return core.policy_head(state), core.value_head(state)

            core.combat_actor_critic = combat
            sources = []

            def track_source(array):
                source = original_from_numpy(array)
                sources.append(weakref.ref(source))
                return source

            def transfer(source, *args, **kwargs):
                if blocking and source.device.type == "cpu":
                    kwargs["non_blocking"] = False
                return original_to(source, *args, **kwargs)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            results = []
            with torch.cuda.stream(stream), patch.object(torch, "from_numpy", track_source), \
                    patch.object(torch.Tensor, "to", transfer):
                for count in (5, 3):
                    model._nosl_is_az_samples = copy.deepcopy(self.samples[:count])
                    result = model._train_nosl_is_az(update_count=2)
                    self.assertTrue(sources)
                    self.assertTrue(all(source() is None for source in sources))
                    results.append((result, copy.deepcopy(model._nosl_is_az_metric_sums),
                                    copy.deepcopy(model._nosl_is_az_metric_sumsq),
                                    copy.deepcopy(model._nosl_is_az_metric_counts),
                                    self._policy_snapshot(model.policy)))
            stream.synchronize()
            return results

        self._assert_nested_equal(run(blocking=False), run(blocking=True))

    def test_relic_width_preserves_sparse_slots_and_microbatch_boundaries(self):
        model = self.model
        model.observation_space = spaces.Dict({
            **model.observation_space.spaces,
            "relics": spaces.Box(0, 10, shape=(4,), dtype=np.int64),
        })
        model.policy.core.combat_observation_keys = frozenset({"state", "relics"})
        relics = [[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 3]]
        for sample, ids in zip(self.samples, relics, strict=True):
            sample["observation"]["relics"] = np.asarray(ids, dtype=np.int64)
        widths = []
        def combat(obs, *, active_relic_width):
            widths.append(active_relic_width)
            return model.policy.core.actor_critic(obs)
        model.policy.core.combat_actor_critic = combat
        model._train_nosl_is_az(step_optimizer=False)
        self.assertEqual(widths, [3, 1, 4])

    def test_nonfinite_predictions_are_excluded_per_metric(self):
        model = self.model
        def combat(obs):
            logits, values = model.policy.core.actor_critic(obs)
            state = obs["state"].flatten()
            offsets = torch.where(state == 0, float("nan"),
                                  torch.where(state == 1, float("inf"), 0.0))
            return logits, values + offsets[:, None]
        model.policy.core.combat_actor_critic = combat
        result = model._train_nosl_is_az(step_optimizer=False)
        self.assertEqual(result["finite_fraction"], 3 / 5)
        counts = model._nosl_is_az_metric_counts
        self.assertEqual(counts["value_target"], 5)
        self.assertEqual(counts["pred_value"], 3)
        self.assertEqual(counts["value_mae"], 3)
        self.assertEqual(counts["value_bias"], 3)
        self.assertEqual(counts["policy_entropy"], 3)
        self.assertTrue(all(np.isfinite(value) for value in model._nosl_is_az_metric_sums.values()))


class NOSLISAZTest(unittest.TestCase):
    def test_normalized_player_hp_prefers_public_run_hp(self):
        observation = {
            "run_hp": np.asarray([0.32, 0.4, 0.8], dtype=np.float32),
            "player_scalars": np.asarray([0.31, 0.4, 0, 0, 0, 0.8]),
        }
        self.assertAlmostEqual(_normalized_player_hp(observation), 0.32)
        self.assertAlmostEqual(
            _normalized_player_hp({"player_scalars": observation["player_scalars"]}),
            0.31,
        )
        self.assertTrue(np.isnan(_normalized_player_hp({})))

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

    def test_observed_value_only_states_never_create_policy_targets(self):
        samples = []
        source = {"state": np.asarray([3.0], dtype=np.float32)}

        for kind in ("end_turn_boundary", "combat_terminal"):
            self.assertTrue(_append_nosl_value_only_sample(
                samples, source, 3, sample_kind=kind
            ))
        self.assertFalse(_append_nosl_value_only_sample(
            samples, None, 3, sample_kind="combat_terminal"
        ))
        source["state"][0] = 99.0

        self.assertEqual(
            [sample["sample_kind"] for sample in samples],
            ["end_turn_boundary", "combat_terminal"],
        )
        for sample in samples:
            np.testing.assert_array_equal(sample["observation"]["state"], [3.0])
            self.assertFalse(sample["action_mask"].any())
            self.assertFalse(sample["policy_target"].any())
            self.assertEqual(sample["policy_weight"], 0.0)
            self.assertTrue(np.isnan(sample["search_value"]))
            self.assertTrue(np.isnan(sample["value_target"]))

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
