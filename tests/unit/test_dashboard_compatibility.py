from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest
import torch

from env.core.cards_vocab import CARD_VOCAB
from apps import dashboard
from rl.rules.visualize_card_vectors import _semantic_card_vectors
from rl.policies.checkpoint import custom_objects
from rl.search.native import NativeSearchConfig, search, decode_leaves, forward_leaves
from stable_baselines3.common.save_util import data_to_json
from zipfile import ZipFile


@pytest.mark.parametrize('extension', ['', '.zip'])
def test_remote_card_vector_path_is_rebased_to_project_root(tmp_path: Path, extension):
    local = tmp_path / "card-vector-run" / "last.pt"
    local.parent.mkdir()
    local.touch()

    checkpoint = tmp_path / 'model.zip'
    with ZipFile(checkpoint, 'w') as archive:
        archive.writestr('data', data_to_json({'policy_kwargs': {
            'card_vector_checkpoint': '/root/autodl-tmp/sts-rl/card-vector-run/last.pt'}}))
    resolved = Path(custom_objects(tmp_path / ('model' + extension))['policy_kwargs']['card_vector_checkpoint'])

    assert resolved == local


def test_card_vector_path_can_be_overridden_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    local = tmp_path / "encoder.pt"
    local.touch()
    monkeypatch.setenv("STS_CARD_VECTOR_CHECKPOINT", str(local))

    checkpoint = tmp_path / 'model.zip'
    with ZipFile(checkpoint, 'w') as archive:
        archive.writestr('data', data_to_json({'policy_kwargs': {
            'card_vector_checkpoint': '/remote/missing/card-vector-run/last.pt'}}))
    resolved = Path(custom_objects(checkpoint)['policy_kwargs']['card_vector_checkpoint'])

    assert resolved == local


def test_semantic_vectors_support_encoder_without_weight_attribute():
    class RuleEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 3)

        def versioned_forward(self, ids, upgrades):
            values = ids.to(torch.float32) + upgrades
            return self.projection(values.unsqueeze(-1))

    encoder = RuleEncoder()
    vectors, upgrades = _semantic_card_vectors(encoder, upgrades=1)

    assert vectors.shape == (len(CARD_VOCAB), 3)
    assert upgrades.device == encoder.projection.weight.device


def test_fullrun_search_config_uses_checkpoint_defaults_and_request_overrides():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        nosl_is_az_simulations = 512
        nosl_is_az_depth = 0
        nosl_is_az_c_puct = 1.75
        nosl_is_az_leaf_batch_size = 16
        nosl_is_az_master_nonce = 1234

    saved = dashboard._evaluation_search_config(Model())
    overridden = dashboard._evaluation_search_config(
        Model(), simulations=128, depth=3
    )

    assert saved == {
        "method": "nosl_is_az_cpp",
        "supported": True,
        "search_mode": "puct",
        "search_mode_label": "PUCT",
        "root_search_mode": 0,
        "simulations": 512,
        "depth": 0,
        "prior_temperature": 1.0,
        "c_puct": 1.75,
        "leaf_batch_size": 16,
        "master_nonce": 1234,
        "gumbel_candidates": 0,
        "gumbel_scale": 1.0,
        "gumbel_value_scale": 0.1,
        "gumbel_max_visit_init": 50.0,
        "gumbel_rescale_q": True,
        "gumbel_use_mixed_value": True,
        "gumbel_interior_uniformity": 0.0,
        "gumbel_interior_coverage_fraction": 0.0,
        "gumbel_interior_value_mode": 2,
        "paper_full_gumbel": False,
        "end_turn_as_leaf": False,
        "stochastic": False,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
    }
    assert overridden["simulations"] == 128
    assert overridden["depth"] == 3


def test_paper_full_gumbel_config_uses_mctx_defaults():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        nosl_is_az_simulations = 512
        nosl_is_az_search_mode = "full_gumbel"
        nosl_is_az_paper_full_gumbel = True
        nosl_is_az_gumbel_candidates = 0
        nosl_is_az_gumbel_value_scale = 3.0
        nosl_is_az_gumbel_interior_coverage_fraction = 0.4
        nosl_is_az_gumbel_interior_value_mode = 2
        nosl_is_az_dirichlet_alpha = 0.3
        nosl_is_az_dirichlet_epsilon = 0.25

    config = dashboard._evaluation_search_config(Model())

    assert config["search_mode"] == "full_gumbel"
    assert config["search_mode_label"] == "Strict paper Full Gumbel"
    assert config["paper_full_gumbel"] is True
    assert config["leaf_batch_size"] == 1
    assert config["gumbel_candidates"] == 16
    assert config["gumbel_scale"] == 1.0
    assert config["gumbel_value_scale"] == 0.1
    assert config["gumbel_max_visit_init"] == 50.0
    assert config["gumbel_rescale_q"] is True
    assert config["gumbel_use_mixed_value"] is True
    assert config["gumbel_interior_uniformity"] == 0.0
    assert config["gumbel_interior_coverage_fraction"] == 0.0
    assert config["gumbel_interior_value_mode"] == 0


def test_paper_full_gumbel_dashboard_override_uses_requested_value():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        nosl_is_az_simulations = 512
        nosl_is_az_search_mode = "full_gumbel"
        nosl_is_az_paper_full_gumbel = True

    config = dashboard._evaluation_search_config(Model(), gumbel_scale=0.25)

    assert config["paper_full_gumbel"] is False
    assert config["search_mode"] == "full_gumbel"
    assert config["gumbel_scale"] == pytest.approx(0.25)


def test_explicit_paper_full_gumbel_ignores_noncanonical_dashboard_values():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        nosl_is_az_paper_full_gumbel = False

    config = dashboard._evaluation_search_config(
        Model(),
        simulations=32,
        leaf_batch_size=64,
        search_mode="puct",
        gumbel_scale=0.25,
        paper_full_gumbel=True,
    )

    assert config["paper_full_gumbel"] is True
    assert config["search_mode"] == "full_gumbel"
    assert config["leaf_batch_size"] == 1
    assert config["gumbel_scale"] == pytest.approx(1.0)


def test_explicit_paper_full_gumbel_rejects_direct_policy():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()

    with pytest.raises(ValueError, match="requires simulations >= 1"):
        dashboard._evaluation_search_config(
            Model(), simulations=0, paper_full_gumbel=True
        )


def test_native_strict_config_forces_sequential_leaf_evaluation():
    strict = NativeSearchConfig(leaf_batch_size=64, paper_full_gumbel=True)
    experimental = NativeSearchConfig(
        leaf_batch_size=64, paper_full_gumbel=False
    )

    assert strict.leaf_batch_size == 1
    assert strict.gumbel_interior_min_visits == 0
    assert experimental.leaf_batch_size == 64


def test_watch_strict_mode_disables_post_search_visit_sampling():
    session = dashboard.WatchSession.__new__(dashboard.WatchSession)
    session.mcts_paper_full_gumbel = True
    session.mcts_simulations = 32
    session.mcts_search_mode = "puct"
    session.mcts_root_search_mode = 0
    session.mcts_leaf_batch_size = 64
    session.mcts_prior_temperature = 2.0
    session.mcts_gumbel_candidates = 0
    session.mcts_gumbel_scale = 0.25
    session.mcts_gumbel_value_scale = 3.0
    session.mcts_gumbel_max_visit_init = 5.0
    session.mcts_gumbel_rescale_q = False
    session.mcts_gumbel_use_mixed_value = False
    session.mcts_gumbel_interior_uniformity = 0.5
    session.mcts_gumbel_interior_coverage_fraction = 0.4
    session.mcts_gumbel_interior_value_mode = 2
    session.stochastic = True

    session._apply_paper_full_gumbel_defaults()

    assert session.mcts_search_mode == "full_gumbel"
    assert session.mcts_leaf_batch_size == 1
    assert session.stochastic is False


def test_dashboard_assets_expose_and_send_strict_paper_mode():
    static = Path(dashboard.__file__).with_name("dashboard_static")
    html = (static / "index.html").read_text(encoding="utf-8")
    javascript = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="watchPaperFullGumbel"' in html
    assert 'id="evalPaperFullGumbel"' in html
    assert "paper_full_gumbel: $(`${prefix}PaperFullGumbel`).checked" in javascript
    assert '"Strict paper Full Gumbel"' in javascript


def test_paper_full_gumbel_dashboard_can_select_direct_policy():
    class Core:
        actor_critic = object()

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        nosl_is_az_simulations = 512
        nosl_is_az_search_mode = "full_gumbel"
        nosl_is_az_paper_full_gumbel = True

    config = dashboard._evaluation_search_config(Model(), simulations=0)

    assert config["paper_full_gumbel"] is False
    assert config["search_mode"] == "direct"
    assert config["simulations"] == 0


def test_fullrun_search_calls_cpp_neural_nosl_without_root_noise(monkeypatch):
    class Core:
        def actor_critic(self, _observation):
            raise AssertionError("root inference is replaced by this unit test")

    class Policy:
        core = Core()

    class Model:
        policy = Policy()

    class BaseEnv:
        def __init__(self):
            self.call = None

        def nosl_is_az_cpp_begin(self, logits, **kwargs):
            self.call = (np.asarray(logits), kwargs)
            return {
                "done": True,
                "result": {"action": 17, "simulations": kwargs["simulations"]},
            }

    class Env:
        def __init__(self):
            self.unwrapped = BaseEnv()

    config = dashboard._evaluation_search_config(
        Model(), simulations=32, depth=0
    )
    env = Env()

    result = search(Model(), env, {}, NativeSearchConfig.from_mapping(config),
                    nonce=7, decision_index=11,
                    root_outputs=(np.asarray([1.0, 2.0], dtype=np.float32), 0.37))

    logits, call = env.unwrapped.call
    assert logits.tolist() == [1.0, 2.0]
    assert call["simulations"] == 32
    assert call["depth"] == 0
    assert call["stochastic"] is False
    assert call["dirichlet_alpha"] == 0.0
    assert call["dirichlet_epsilon"] == 0.0
    assert call["root_value"] == pytest.approx(0.37)
    assert call["prior_temperature"] == pytest.approx(1.0)
    assert result["action"] == 17
    assert result["leaf_evaluations"] == 0


def test_watch_leaf_inference_uses_explicit_combat_head():
    class Core:
        combat_observation_keys = ("phase", "relics")

        def __init__(self):
            self.called = None

        def combat_actor_critic(self, observations, *, active_relic_width):
            self.called = (observations, active_relic_width)
            batch = observations["phase"].shape[0]
            return torch.zeros((batch, 200)), torch.full((batch, 1), 0.42)

        def actor_critic(self, _observations):
            raise AssertionError("Watch search leaves must use combat_actor_critic")

    class Policy:
        core = Core()

    class Model:
        policy = Policy()
        device = torch.device("cpu")

    runner = dashboard.WatchSession.__new__(dashboard.WatchSession)
    runner.model = Model()
    runner.model_lock = threading.Lock()
    runner.compact_observations = True
    leaves = [{
        "observation": {
            "phase": np.asarray([3.0], dtype=np.float32),
            "relics": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        },
        "needs_policy": True,
    }]

    _, observations = decode_leaves({0: leaves})
    logits, values = forward_leaves(runner.model, observations, compact=True,
                                   lock=runner.model_lock)

    assert logits.shape == (1, 200)
    assert values.tolist() == pytest.approx([0.42])
    assert Model.policy.core.called[1] == 1
