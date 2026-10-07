from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

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


def test_tree_snapshot_keeps_direct_model_value_distinct_from_backed_up_value():
    session = object.__new__(dashboard.WatchSession)
    session.env = SimpleNamespace(
        unwrapped=SimpleNamespace(_native_action_to_thin=lambda _bits: None),
        describe_action=lambda _action: "unused",
    )
    formatted = session._format_native_tree_snapshot({
        "valid": True,
        "path": [],
        "visits": 9,
        "value_visits": 8,
        "value": 0.83,
        "sample_value": 0.62,
        "network_value_valid": True,
        "network_value": 0.41,
        "children": [{
            "edge_index": 0,
            "action_bits": 4 << 29,
            "successor_model_value_valid": True,
            "successor_model_value": 0.37,
            "successor_model_value_count": 3,
        }],
    })

    assert formatted["network_value_valid"] is True
    assert formatted["network_value"] == pytest.approx(0.41)
    assert formatted["value"] == pytest.approx(0.83)
    assert formatted["sample_value"] == pytest.approx(0.62)
    assert formatted["children"][0]["successor_model_value"] == pytest.approx(0.37)
    assert formatted["children"][0]["successor_model_value_count"] == 3


def test_search_tree_ui_labels_direct_model_value_without_fallback_to_backup():
    source = (dashboard.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "模型直出 V" in source
    assert "network_value_valid" in source
    assert "后继模型直出 V" in source
    assert "snapshot?.network_value ?? snapshot?.value" not in source


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


def test_entity_checkpoint_dashboard_projects_cards_and_language(combat_factory):
    from rl.policies.entity_combat import EntityCombatCore, EntityCombatConfig
    from rl.rules.visualize_card_vectors import resolve_encoder, extract_vectors

    env = combat_factory(['STRIKE_RED', 'DEFEND_RED'],
                         combat_action_contract='autoregressive-v2', card_language=True)
    core = EntityCombatCore(env.observation_space,
                            EntityCombatConfig(width=32, layers=1, heads=4))
    model = SimpleNamespace(policy=SimpleNamespace(core=core))
    assert resolve_encoder(model) is core
    base, _ = extract_vectors(core, 'hand', upgrades=0)
    upgraded, _ = extract_vectors(core, 'hand', upgrades=1)
    assert base.shape == upgraded.shape == (len(CARD_VOCAB), 32)
    assert np.isfinite(base).all() and np.isfinite(upgraded).all()
    app = dashboard.DashboardState()
    projected = app._embedding(model)
    assert len(projected['points']) == 2 * projected['card_count'] > 0
    language = app._token_embeddings(model)
    assert len(language['card']['points']) == len(dashboard.CARD_LANGUAGE_TOKENS)


def test_dashboard_delegates_checkpoint_algorithm_selection(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'joint.zip'
    checkpoint.write_bytes(b'PK\x05\x06' + b'\0' * 18)
    calls = []

    class Model:
        policy = object()

        def set_random_seed(self, seed):
            calls.append(('seed', seed))

    model = Model()
    watch = object()
    monkeypatch.setattr(
        dashboard,
        '_load_checkpoint',
        lambda path, device: calls.append(('load', path, device)) or model,
    )
    monkeypatch.setattr(
        dashboard.DashboardState, '_embedding', lambda self, loaded: {}
    )
    monkeypatch.setattr(
        dashboard.DashboardState, '_token_embeddings', lambda self, loaded: {}
    )
    monkeypatch.setattr(
        dashboard, 'WatchSession',
        lambda loaded, lock, seed, boss_opening_pool: watch,
    )

    app = dashboard.DashboardState()
    app.load_checkpoint(checkpoint, episodes=0)

    assert calls == [('load', checkpoint, 'auto'), ('seed', 0)]
    assert app.model is model
    assert app.watch is watch


def test_watch_displays_boolean_powers_without_changing_policy_inputs(combat_factory):
    from rl.policies.entity_combat import EntityCombatPolicy
    from sb3_contrib import MaskablePPO

    env = combat_factory(['CORRUPTION', 'DEFEND_RED', 'STRIKE_RED'],
                         combat_action_contract='autoregressive-v2', card_language=True)
    model = MaskablePPO(EntityCombatPolicy, env, n_steps=2, batch_size=2, device='cpu',
                       policy_kwargs={'entity_config': {'width': 32, 'layers': 1, 'heads': 4}})
    watch = dashboard.WatchSession(model, threading.Lock(), seed=42)
    watch.env.close()
    watch.env = env
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'block': 0},
        'hand': ['CORRUPTION', 'DEFEND_RED', 'STRIKE_RED'],
        'draw': ['DEFEND_RED'] * 5, 'discard': [], 'exhaust': [],
    })
    env._invalidate_backend_cache()
    assert 'Corruption' not in battle.player_power_flags()
    env.step(6)  # Play the real power, rather than fabricating its status map.
    assert 'Corruption' in battle.player_power_flags()
    assert env.thick.obs_canonical()['hand'][0]['cost'] == 0
    observation = {k: v.copy() for k, v in env._obs().items()}
    mask = env.action_masks().copy()
    watch.obs, watch.info = env._obs(), env._info()
    shown = watch.snapshot()['canon']['player']
    assert 'Corruption' in shown['power_flags']
    assert all(np.array_equal(v, env._obs()[k]) for k, v in observation.items())
    assert np.array_equal(mask, env.action_masks())
    env.step(6)  # The Defend is free and goes into exhaust.
    raw = battle.to_dict()
    assert raw['player']['energy'] == 0
    assert any(card[0] == 'DEFEND_RED' for card in raw['exhaust'])
    battle.set_player_status('CORRUPTION', 0)
    env._invalidate_backend_cache()
    assert 'Corruption' not in watch.snapshot()['canon']['player']['power_flags']


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
        "fixed_worldline": False,
        "stochastic": False,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
    }
    assert overridden["simulations"] == 128
    assert overridden["depth"] == 3


def test_paper_full_gumbel_config_preserves_explicit_q_scale():
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
    assert config["search_mode_label"] == "Paper Full Gumbel (custom Q)"
    assert config["paper_full_gumbel"] is True
    assert config["leaf_batch_size"] == 1
    assert config["gumbel_candidates"] == 16
    assert config["gumbel_scale"] == 1.0
    assert config["gumbel_value_scale"] == 3.0
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
    assert config["gumbel_scale"] == pytest.approx(0.25)


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
    assert 'id="watchFixedWorldline"' in html
    assert 'id="evalFixedWorldline"' in html
    assert "paper_full_gumbel: $(`${prefix}PaperFullGumbel`).checked" in javascript
    assert "fixed_worldline: $(`${prefix}FixedWorldline`).checked" in javascript
    assert '"Strict paper Full Gumbel"' in javascript


def test_dashboard_assets_expose_route_only_boss_fast_forward():
    static = Path(dashboard.__file__).with_name("dashboard_static")
    html = (static / "index.html").read_text(encoding="utf-8")
    javascript = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="watchBossAct"' in html
    assert 'id="watchBossButton"' in html
    assert 'id="watchUpperStrategy"' in html
    assert 'data-upper-strategy="rules" class="active"' in html
    assert 'id="watchCombatStrategy"' in html
    assert 'data-combat-strategy="model" class="active"' in html
    assert 'data-combat-strategy="planner"' in html
    assert 'ironclad-turn-v8-awakened-phase' in html
    assert 'id="decisionHeading"' in html
    assert "target_boss_act: targetAct" in javascript
    assert "upper_strategy: watchUpperStrategyValue()" in javascript
    assert "combat_strategy: watchCombatStrategyValue()" in javascript
    assert "忽略 HP/药水" in javascript


def test_dashboard_assets_expose_independent_evaluation_strategies():
    static = Path(dashboard.__file__).with_name("dashboard_static")
    html = (static / "index.html").read_text(encoding="utf-8")
    javascript = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="evalUpperStrategy"' in html
    assert 'data-eval-upper-strategy="model" class="active"' in html
    assert 'id="evalCombatStrategy"' in html
    assert 'data-eval-combat-strategy="model" class="active"' in html
    assert 'data-eval-combat-strategy="planner"' in html
    assert "upper_strategy: evalUpperStrategyValue()" in javascript
    assert "combat_strategy: evalCombatStrategyValue()" in javascript
    assert "function drawDistribution()" in javascript
    assert '$("distributionCanvas").addEventListener("mousemove"' in javascript


def test_dashboard_composes_evaluation_upper_and_combat_strategies():
    model = object()
    lock = threading.Lock()
    strategy = dashboard._evaluation_strategy(
        model,
        lock,
        {"method": "direct_policy"},
        upper_strategy="rules",
        combat_strategy="planner",
    )

    assert isinstance(strategy, dashboard.PhaseStrategy)
    assert isinstance(strategy.default, dashboard.OverworldRuleStrategy)
    assert isinstance(strategy.default.fallback, dashboard.PolicyStrategy)
    combat = strategy.routes["combat"]
    assert isinstance(combat, dashboard.TurnPlannerStrategy)
    assert combat.config == dashboard._TURN_PLANNER_CONFIG

    strategy = dashboard._evaluation_strategy(
        model,
        lock,
        {
            "method": "nosl_is_az_cpp",
            "simulations": 1,
            "paper_full_gumbel": False,
            "root_search_mode": 0,
        },
        upper_strategy="model",
        combat_strategy="model",
    )
    assert isinstance(strategy, dashboard.PhaseStrategy)
    assert isinstance(strategy.default, dashboard.PolicyStrategy)
    combat = strategy.routes["combat"]
    assert isinstance(combat, dashboard.PhaseStrategy)
    assert isinstance(combat.default, dashboard.PolicyStrategy)
    assert isinstance(combat.routes["combat"], dashboard.NativeStrategy)


def test_dashboard_evaluation_status_records_selected_strategies(monkeypatch):
    state = dashboard.DashboardState()
    state.model = object()
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)

    state.start_evaluation(
        episodes=1,
        seed=0,
        upper_strategy="rules",
        combat_strategy="planner",
    )

    assert state.eval_status["upper_strategy"] == "rules"
    assert state.eval_status["upper_strategy_version"] == "ironclad-overworld-v3"
    assert state.eval_status["upper_fallback"] == "masked_argmax"
    assert state.eval_status["combat_strategy"] == "planner"
    assert state.eval_status["combat_strategy_version"] == "ironclad-turn-v9-time-clock"
    assert state.eval_status["search_method"] == "turn_planner"

    with pytest.raises(ValueError, match="upper strategy"):
        state.start_evaluation(1, 0, upper_strategy="unknown")
    with pytest.raises(ValueError, match="combat strategy"):
        state.start_evaluation(1, 0, combat_strategy="unknown")


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
        Model(), simulations=32, depth=0, fixed_worldline=True
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
    assert call["fixed_worldline"] is True
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
