"""Local browser workbench for inspecting and evaluating sts-rl checkpoints."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from sb3_contrib import MaskablePPO

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
import slaythespire as sts
from rl.evaluation.runner import evaluate, summarize, Scenario, EvaluationConfig, EvaluationCancelled
from rl.evaluation.runner import is_win
from rl.evaluation.strategies import (
    DecisionRequest,
    NativeStrategy,
    PhaseStrategy,
    PolicyStrategy,
)
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy
from rl.policies.checkpoint import custom_objects as _sb3_load_custom_objects
from dataclasses import asdict, fields
from rl.search.native import (
    STRICT_SEARCH_DEFAULTS,
    NativeSearchConfig,
    search as native_search,
)
from rl.rules.visualize_card_vectors import (
    extract_state_dict_vectors,
    extract_vectors,
    pca_project,
    resolve_encoder,
)
from apps.watch import _action_dist
from env.search.nosl_sampler import NOSLSampler
from env.search.overworld_rules import OVERWORLD_RULE_VERSION
from env.search.turn_planner import PLANNER_VERSION, TurnPlannerConfig
from env.core.cards_vocab import CARD_INDEX
from env.core.card_language import (
    TOKEN_DOC as CARD_LANGUAGE_DOC,
    TOKEN_VOCAB as CARD_LANGUAGE_TOKENS,
)
from env.core.combat_language import TOKEN_VOCAB as COMBAT_LANGUAGE_TOKENS
from env.core.event_descriptions import describe_event
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers
from env.core.card_language import STATUS_CARDS
from env.core.relics import RELIC_NAME_ZH

# Register custom policy classes before checkpoint loading.
from rl.policies import model as _model  # noqa: F401
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


STATIC_DIR = Path(__file__).with_name("dashboard_static")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path("runs/dashboard/uploads")
def _default_dashboard_checkpoint() -> Path | None:
    """Pick the newest local training checkpoint for a no-argument launch.

    An explicit ``STS_RL_CHECKPOINT`` always wins.  Otherwise prefer the
    newest ``nosl_az_*_steps.zip`` under ``runs``; this keeps the normal
    ``uv run python -m apps.dashboard`` command on the latest downloaded model
    without hard-coding a step number.
    """
    configured = os.environ.get("STS_RL_CHECKPOINT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = PROJECT_ROOT / "runs"
    candidates = list(root.glob("**/nosl_az_*_steps.zip")) if root.is_dir() else []
    if not candidates:
        return None
    def checkpoint_rank(path: Path) -> tuple[float, int]:
        match = re.search(r"(\d+)_steps\.zip$", path.name)
        step = int(match.group(1)) if match else -1
        return path.stat().st_mtime, step
    return max(candidates, key=checkpoint_rank)


def _observation_for_policy(model: Any, observation: dict) -> dict:
    """Compatibility alias for the shared inference boundary."""
    return _model.observation_for_policy(model, observation)


MAX_UPLOAD_BYTES = 512 * 1024 * 1024

# Interactive combat search follows the C++ neural NOSL-IS-AZ configuration
# saved in the checkpoint.  The browser can raise the simulation count for a
# slower, wider test, or set it to zero to use the policy directly.  Route
# preprocessing remains opt-in because it executes the whole policy-controlled
# run before the first frame is shown.
DEFAULT_MCTS_SIMULATIONS = 512
MAX_MCTS_SIMULATIONS = 4096
DEFAULT_MCTS_DEPTH = 0
MAX_MCTS_DEPTH = 256
DEFAULT_EVAL_WORKERS = 4
MAX_EVAL_WORKERS = 64
DEFAULT_MCTS_PRIOR_TEMPERATURE = 1.0
MIN_MCTS_PRIOR_TEMPERATURE = 0.05
MAX_MCTS_PRIOR_TEMPERATURE = 20.0
MAX_GUMBEL_CANDIDATES = 4096
MAX_GUMBEL_SCALE = 100.0
MAX_GUMBEL_VALUE_SCALE = 100.0
MAX_GUMBEL_MAX_VISIT_INIT = 1_000_000.0
MCTS_MASTER_NONCE = 0x4E4F534C_64617368

# Search modes exposed by the dashboard.  ``direct`` is deliberately a
# separate mode from PUCT with zero simulations: it keeps the user's choice
# stable when simulation counts are edited and makes the effective algorithm
# unambiguous in the API/status panel.
SEARCH_MODE_IDS = {
    "puct": 0,
    "gumbel": 1,
    "full_gumbel": 2,
}
SEARCH_MODE_LABELS = {
    "direct": "Direct policy",
    "puct": "PUCT",
    "gumbel": "Simplified Gumbel",
    "full_gumbel": "Full Gumbel AZ",
}
STRICT_PAPER_SEARCH_LABEL = "Strict paper Full Gumbel"


def _strict_paper_search_settings_match(
    *,
    search_mode: str,
    leaf_batch_size: int,
    prior_temperature: float,
    gumbel_candidates: int,
    gumbel_scale: float,
    gumbel_value_scale: float,
    gumbel_max_visit_init: float,
    gumbel_rescale_q: bool,
    gumbel_use_mixed_value: bool,
    gumbel_interior_uniformity: float,
    gumbel_interior_coverage_fraction: float,
    gumbel_interior_value_mode: int,
) -> bool:
    """Return whether dashboard values still describe strict paper search."""
    return (
        search_mode == "full_gumbel"
        and int(leaf_batch_size) == STRICT_SEARCH_DEFAULTS["leaf_batch_size"]
        and np.isclose(
            float(prior_temperature), STRICT_SEARCH_DEFAULTS["prior_temperature"]
        )
        and int(gumbel_candidates) == STRICT_SEARCH_DEFAULTS["gumbel_candidates"]
        and np.isclose(float(gumbel_scale), STRICT_SEARCH_DEFAULTS["gumbel_scale"])
        and np.isclose(
            float(gumbel_value_scale), STRICT_SEARCH_DEFAULTS["gumbel_value_scale"]
        )
        and np.isclose(
            float(gumbel_max_visit_init),
            STRICT_SEARCH_DEFAULTS["gumbel_max_visit_init"],
        )
        and bool(gumbel_rescale_q) is STRICT_SEARCH_DEFAULTS["gumbel_rescale_q"]
        and bool(gumbel_use_mixed_value)
        is STRICT_SEARCH_DEFAULTS["gumbel_use_mixed_value"]
        and np.isclose(
            float(gumbel_interior_uniformity),
            STRICT_SEARCH_DEFAULTS["gumbel_interior_uniformity"],
        )
        and np.isclose(
            float(gumbel_interior_coverage_fraction),
            STRICT_SEARCH_DEFAULTS["gumbel_interior_coverage_fraction"],
        )
        and int(gumbel_interior_value_mode)
        == STRICT_SEARCH_DEFAULTS["gumbel_interior_value_mode"]
    )


def _normalize_search_mode(value: Any, default: str = "puct") -> str:
    """Normalize a persisted/API search mode without accepting ambiguity."""
    if isinstance(value, (int, np.integer)):
        return {0: "puct", 1: "gumbel", 2: "full_gumbel"}.get(
            int(value), default
        )
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_")
        aliases = {
            "direct_policy": "direct",
            "policy": "direct",
            "mcts": "puct",
            "gumbel_sequential_halving": "gumbel",
            "full": "full_gumbel",
            "full_gumbel_az": "full_gumbel",
        }
        candidate = aliases.get(candidate, candidate)
        if candidate in SEARCH_MODE_LABELS:
            return candidate
    return default if default in SEARCH_MODE_LABELS else "puct"


def _model_search_bool(model: Any, name: str, default: bool) -> bool:
    value = _model_search_param(model, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)

IRONCLAD_VECTOR_CARDS = list(IRONCLAD_CARDS)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not np.isfinite(parsed):
        parsed = float(default)
    return max(float(minimum), min(parsed, float(maximum)))












def _model_search_param(model: Any, name: str, default: Any) -> Any:
    """Read a saved search setting from an algorithm or its policy."""
    for owner in (model, getattr(model, "policy", None)):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    return default


def _model_search_float(model: Any, name: str, default: float) -> float:
    try:
        value = float(_model_search_param(model, name, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if np.isfinite(value) and value >= 0.0 else float(default)


def _supports_neural_nosl(model: Any) -> bool:
    core = getattr(getattr(model, "policy", None), "core", None)
    return core is not None and hasattr(core, "actor_critic")


def _evaluation_search_config(
    model: Any,
    *,
    simulations: int | None = None,
    depth: int | None = None,
    leaf_batch_size: int | None = None,
    prior_temperature: float | None = None,
    search_mode: str | None = None,
    gumbel_candidates: int | None = None,
    gumbel_scale: float | None = None,
    gumbel_value_scale: float | None = None,
    gumbel_max_visit_init: float | None = None,
    gumbel_rescale_q: bool | None = None,
    gumbel_use_mixed_value: bool | None = None,
    gumbel_interior_uniformity: float | None = None,
    gumbel_interior_coverage_fraction: float | None = None,
    gumbel_interior_value_mode: int | None = None,
    paper_full_gumbel: bool | None = None,
    end_turn_as_leaf: bool | None = None,
    fixed_worldline: bool | None = None,
) -> dict[str, Any]:
    supported = _supports_neural_nosl(model)
    default_simulations = _clamp_int(
        _model_search_param(
            model,
            "nosl_is_az_simulations",
            DEFAULT_MCTS_SIMULATIONS if supported else 0,
        ),
        DEFAULT_MCTS_SIMULATIONS if supported else 0,
        0,
        MAX_MCTS_SIMULATIONS,
    )
    selected_simulations = _clamp_int(
        default_simulations if simulations is None else simulations,
        default_simulations,
        0,
        MAX_MCTS_SIMULATIONS,
    )
    default_depth = _clamp_int(
        _model_search_param(model, "nosl_is_az_depth", DEFAULT_MCTS_DEPTH),
        DEFAULT_MCTS_DEPTH,
        0,
        MAX_MCTS_DEPTH,
    )
    selected_depth = _clamp_int(
        default_depth if depth is None else depth,
        default_depth,
        0,
        MAX_MCTS_DEPTH,
    )
    selected_leaf_batch_size = _clamp_int(
        _model_search_param(model, "nosl_is_az_leaf_batch_size", 64)
        if leaf_batch_size is None else leaf_batch_size,
        64,
        1,
        256,
    )
    default_prior_temperature = _clamp_float(
        _model_search_param(
            model,
            "nosl_is_az_prior_temperature",
            DEFAULT_MCTS_PRIOR_TEMPERATURE,
        ),
        DEFAULT_MCTS_PRIOR_TEMPERATURE,
        MIN_MCTS_PRIOR_TEMPERATURE,
        MAX_MCTS_PRIOR_TEMPERATURE,
    )
    selected_prior_temperature = _clamp_float(
        default_prior_temperature
        if prior_temperature is None else prior_temperature,
        default_prior_temperature,
        MIN_MCTS_PRIOR_TEMPERATURE,
        MAX_MCTS_PRIOR_TEMPERATURE,
    )
    saved_mode = _normalize_search_mode(_model_search_param(
        model,
        "nosl_is_az_search_mode",
        _model_search_param(model, "nosl_is_az_root_search_mode", "puct"),
    ))
    selected_mode = _normalize_search_mode(
        saved_mode if search_mode is None else search_mode,
        saved_mode,
    )
    saved_paper_full_gumbel = _model_search_bool(
        model, "nosl_is_az_paper_full_gumbel", False
    )
    selected_paper_full_gumbel = (
        saved_paper_full_gumbel
        if paper_full_gumbel is None
        else bool(paper_full_gumbel)
    )
    if selected_simulations <= 0:
        selected_mode = "direct"
    selected_candidates = _clamp_int(
        _model_search_param(model, "nosl_is_az_gumbel_candidates", 0)
        if gumbel_candidates is None else gumbel_candidates,
        0,
        0,
        MAX_GUMBEL_CANDIDATES,
    )
    selected_gumbel_scale = _clamp_float(
        _model_search_param(model, "nosl_is_az_gumbel_scale", 1.0)
        if gumbel_scale is None else gumbel_scale,
        1.0,
        0.0,
        MAX_GUMBEL_SCALE,
    )
    selected_gumbel_value_scale = _clamp_float(
        _model_search_param(model, "nosl_is_az_gumbel_value_scale", 0.1)
        if gumbel_value_scale is None else gumbel_value_scale,
        0.1,
        0.0,
        MAX_GUMBEL_VALUE_SCALE,
    )
    selected_gumbel_max_visit_init = _clamp_float(
        _model_search_param(model, "nosl_is_az_gumbel_max_visit_init", 50.0)
        if gumbel_max_visit_init is None else gumbel_max_visit_init,
        50.0,
        0.0,
        MAX_GUMBEL_MAX_VISIT_INIT,
    )
    selected_rescale_q = _model_search_bool(
        model, "nosl_is_az_gumbel_rescale_q", True
    ) if gumbel_rescale_q is None else bool(gumbel_rescale_q)
    selected_mixed_value = _model_search_bool(
        model, "nosl_is_az_gumbel_use_mixed_value", True
    ) if gumbel_use_mixed_value is None else bool(gumbel_use_mixed_value)
    selected_interior_uniformity = _clamp_float(
        _model_search_param(model, "nosl_is_az_gumbel_interior_uniformity", 0.0)
        if gumbel_interior_uniformity is None else gumbel_interior_uniformity,
        0.0, 0.0, 1.0,
    )
    selected_coverage_fraction = _clamp_float(
        _model_search_param(
            model, "nosl_is_az_gumbel_interior_coverage_fraction", 0.0
        ) if gumbel_interior_coverage_fraction is None
        else gumbel_interior_coverage_fraction,
        0.0, 0.0, 1.0,
    )
    selected_value_mode = int(
        _model_search_param(model, "nosl_is_az_gumbel_interior_value_mode", 2)
        if gumbel_interior_value_mode is None
        else gumbel_interior_value_mode
    )
    selected_end_turn_as_leaf = _model_search_bool(
        model, "nosl_is_az_end_turn_as_leaf", False
    ) if end_turn_as_leaf is None else bool(end_turn_as_leaf)
    selected_fixed_worldline = _model_search_bool(
        model, "nosl_is_az_fixed_worldline", False
    ) if fixed_worldline is None else bool(fixed_worldline)
    explicit_search_override = any(
        value is not None
        for value in (
            simulations,
            leaf_batch_size,
            prior_temperature,
            search_mode,
            gumbel_candidates,
            gumbel_scale,
            gumbel_value_scale,
            gumbel_max_visit_init,
            gumbel_rescale_q,
            gumbel_use_mixed_value,
            gumbel_interior_uniformity,
            gumbel_interior_coverage_fraction,
            gumbel_interior_value_mode,
        )
    )
    # Old dashboard clients do not send paper_full_gumbel. Preserve strict
    # checkpoint behavior only while their explicit values remain canonical.
    if (
        paper_full_gumbel is None
        and selected_paper_full_gumbel
        and explicit_search_override
        and not _strict_paper_search_settings_match(
            search_mode=selected_mode,
            leaf_batch_size=selected_leaf_batch_size,
            prior_temperature=selected_prior_temperature,
            gumbel_candidates=selected_candidates,
            gumbel_scale=selected_gumbel_scale,
            gumbel_value_scale=selected_gumbel_value_scale,
            gumbel_max_visit_init=selected_gumbel_max_visit_init,
            gumbel_rescale_q=selected_rescale_q,
            gumbel_use_mixed_value=selected_mixed_value,
            gumbel_interior_uniformity=selected_interior_uniformity,
            gumbel_interior_coverage_fraction=selected_coverage_fraction,
            gumbel_interior_value_mode=selected_value_mode,
        )
    ):
        selected_paper_full_gumbel = False
    if selected_paper_full_gumbel and selected_simulations <= 0:
        if paper_full_gumbel is True:
            raise ValueError("strict paper Full Gumbel requires simulations >= 1")
        selected_paper_full_gumbel = False
    if selected_paper_full_gumbel:
        # Keep the dashboard's reported/requested settings identical to the
        # native strict-paper canonicalization.  In particular, mctx's
        # default max_num_considered_actions is 16; zero means "all legal"
        # only in the legacy project mode.
        selected_mode = "full_gumbel"
        selected_leaf_batch_size = STRICT_SEARCH_DEFAULTS["leaf_batch_size"]
        selected_candidates = STRICT_SEARCH_DEFAULTS["gumbel_candidates"]
        selected_prior_temperature = STRICT_SEARCH_DEFAULTS["prior_temperature"]
        selected_gumbel_scale = STRICT_SEARCH_DEFAULTS["gumbel_scale"]
        selected_gumbel_value_scale = STRICT_SEARCH_DEFAULTS[
            "gumbel_value_scale"
        ]
        selected_gumbel_max_visit_init = STRICT_SEARCH_DEFAULTS[
            "gumbel_max_visit_init"
        ]
        selected_rescale_q = STRICT_SEARCH_DEFAULTS["gumbel_rescale_q"]
        selected_mixed_value = STRICT_SEARCH_DEFAULTS["gumbel_use_mixed_value"]
        selected_interior_uniformity = STRICT_SEARCH_DEFAULTS[
            "gumbel_interior_uniformity"
        ]
        selected_coverage_fraction = STRICT_SEARCH_DEFAULTS[
            "gumbel_interior_coverage_fraction"
        ]
        selected_value_mode = STRICT_SEARCH_DEFAULTS["gumbel_interior_value_mode"]
    if selected_simulations > 0 and not supported and selected_mode != "direct":
        raise ValueError(
            "this checkpoint does not expose the combat policy/value heads "
            "required by neural NOSL MCTS; set simulations to 0 for direct policy"
        )
    return {
        "method": (
            "nosl_is_az_cpp"
            if selected_simulations > 0 and selected_mode != "direct"
            else "direct_policy"
        ),
        "supported": supported,
        "search_mode": selected_mode,
        "search_mode_label": (
            STRICT_PAPER_SEARCH_LABEL
            if selected_paper_full_gumbel
            else SEARCH_MODE_LABELS[selected_mode]
        ),
        "root_search_mode": SEARCH_MODE_IDS.get(selected_mode, 0),
        "paper_full_gumbel": selected_paper_full_gumbel,
        "simulations": selected_simulations,
        "depth": selected_depth,
        "c_puct": _model_search_float(model, "nosl_is_az_c_puct", 1.5),
        "prior_temperature": selected_prior_temperature,
        "leaf_batch_size": _clamp_int(
            selected_leaf_batch_size,
            64,
            1,
            256,
        ),
        "master_nonce": int(
            _model_search_param(model, "nosl_is_az_master_nonce", MCTS_MASTER_NONCE)
        ),
        "gumbel_candidates": selected_candidates,
        "gumbel_scale": selected_gumbel_scale,
        "gumbel_value_scale": selected_gumbel_value_scale,
        "gumbel_max_visit_init": selected_gumbel_max_visit_init,
        "gumbel_rescale_q": selected_rescale_q,
        "gumbel_use_mixed_value": selected_mixed_value,
        "gumbel_interior_uniformity": selected_interior_uniformity,
        "gumbel_interior_coverage_fraction": selected_coverage_fraction,
        "gumbel_interior_value_mode": selected_value_mode,
        "end_turn_as_leaf": selected_end_turn_as_leaf,
        "fixed_worldline": selected_fixed_worldline,
        # Evaluation selects the root visit argmax and never injects training noise.
        "stochastic": False,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
    }




_ROUTE_STEP_NORMAL = "normal"
_ROUTE_STEP_SKIP_COMBAT = "skip_combat"
_ROUTE_STEP_ENTER_BOSS = "enter_boss"
_UPPER_STRATEGY_MODEL = "model"
_UPPER_STRATEGY_RULES = "rules"
_UPPER_STRATEGIES = {_UPPER_STRATEGY_MODEL, _UPPER_STRATEGY_RULES}
_COMBAT_STRATEGY_MODEL = "model"
_COMBAT_STRATEGY_PLANNER = "planner"
_COMBAT_STRATEGIES = {_COMBAT_STRATEGY_MODEL, _COMBAT_STRATEGY_PLANNER}
_TURN_PLANNER_CONFIG = TurnPlannerConfig(
    worlds=2,
    node_budget=8192,
    beam_width=8,
    max_depth=16,
    lookahead_turns=1,
    master_nonce=20260918,
)


def _evaluation_strategy(
    model,
    model_lock,
    search_config,
    *,
    upper_strategy: str,
    combat_strategy: str,
):
    """Compose independent upper/combat choices on the shared eval contract."""
    policy = PolicyStrategy(model, lock=model_lock)
    upper = (
        OverworldRuleStrategy(policy)
        if upper_strategy == _UPPER_STRATEGY_RULES
        else policy
    )
    if combat_strategy == _COMBAT_STRATEGY_PLANNER:
        combat = TurnPlannerStrategy(_TURN_PLANNER_CONFIG)
    elif search_config.get("method") == "nosl_is_az_cpp":
        combat = PhaseStrategy(
            policy,
            {
                "combat": NativeStrategy(
                    model,
                    NativeSearchConfig.from_mapping(search_config),
                    master_nonce=search_config.get("master_nonce", 0),
                    lock=model_lock,
                )
            },
            predicates={
                "combat": lambda request: getattr(
                    request.env.unwrapped, "_pending", None
                )
                is None
            },
        )
    else:
        combat = policy
    if upper is policy and combat is policy:
        return policy
    return PhaseStrategy(upper, {"combat": combat})


class WatchSession:
    def __init__(self, model: Any, model_lock: threading.Lock, seed: int = 0):
        self.model = model
        self.model_lock = model_lock
        set_training_mode = getattr(getattr(model, "policy", None), "set_training_mode", None)
        if set_training_mode is not None:
            set_training_mode(False)
        core = getattr(getattr(model, "policy", None), "core", None)
        # Match training/full-run evaluation for search leaves.  The root
        # still uses the routed whole-run actor_critic because it is a full
        # observation; all native search leaves are combat states.
        self.compact_observations = bool(
            core is not None
            and hasattr(core, "combat_actor_critic")
            and hasattr(core, "combat_observation_keys")
        )
        self.env = SlayTheSpireEnv(
            combat_action_contract="native-v1",
            global_map_view=_model.uses_global_map_observation(model),
            card_language=_model.uses_card_language(model),
            combat_language=_model.uses_combat_language(model),
            deck_size_observation=_model.uses_deck_size_observation(model),
            neow_observation=_model.uses_neow_observation(model),
            event_option_observation=_model.uses_event_option_observation(model),
        )
        self.seed = seed
        self.history: list[int | None] = []
        self.native_history: list[int | None] = []
        self.native_direct_history: list[bool] = []
        self.position = 0
        self.timeline: list[dict] = []
        self.timeline_exhausted = False
        self.obs = None
        self.info = None
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.decision = None
        self.map_state: dict | None = None
        self.map_states: dict[int, dict] = {}
        self.card_reward_choices: list[dict] = []
        self._search_generation = 0
        self.mcts_simulations = _clamp_int(
            self._model_search_param("nosl_is_az_simulations", DEFAULT_MCTS_SIMULATIONS),
            DEFAULT_MCTS_SIMULATIONS, 0, MAX_MCTS_SIMULATIONS,
        )
        self.mcts_depth = _clamp_int(
            self._model_search_param("nosl_is_az_depth", DEFAULT_MCTS_DEPTH),
            DEFAULT_MCTS_DEPTH, 0, MAX_MCTS_DEPTH,
        )
        self.mcts_end_turn_as_leaf = _model_search_bool(
            model, "nosl_is_az_end_turn_as_leaf", True
        )
        self.mcts_fixed_worldline = False
        self.mcts_prior_temperature = _clamp_float(
            self._model_search_param(
                "nosl_is_az_prior_temperature",
                DEFAULT_MCTS_PRIOR_TEMPERATURE,
            ),
            DEFAULT_MCTS_PRIOR_TEMPERATURE,
            MIN_MCTS_PRIOR_TEMPERATURE,
            MAX_MCTS_PRIOR_TEMPERATURE,
        )
        self.mcts_c_puct = self._model_search_float("nosl_is_az_c_puct", 1.5)
        self.mcts_search_mode = "full_gumbel"
        self.mcts_paper_full_gumbel = _model_search_bool(
            model, "nosl_is_az_paper_full_gumbel", False
        )
        self.mcts_root_search_mode = SEARCH_MODE_IDS.get(
            self.mcts_search_mode, 0
        )
        self.mcts_gumbel_candidates = _clamp_int(
            self._model_search_param("nosl_is_az_gumbel_candidates", 0),
            0, 0, MAX_GUMBEL_CANDIDATES,
        )
        self.mcts_gumbel_scale = _clamp_float(
            self._model_search_param("nosl_is_az_gumbel_scale", 1.0),
            1.0, 0.0, MAX_GUMBEL_SCALE,
        )
        self.mcts_gumbel_value_scale = _clamp_float(
            self._model_search_param("nosl_is_az_gumbel_value_scale", 3.0),
            3.0, 0.0, MAX_GUMBEL_VALUE_SCALE,
        )
        self.mcts_gumbel_max_visit_init = _clamp_float(
            self._model_search_param("nosl_is_az_gumbel_max_visit_init", 50.0),
            50.0, 0.0, MAX_GUMBEL_MAX_VISIT_INIT,
        )
        self.mcts_gumbel_rescale_q = _model_search_bool(
            model, "nosl_is_az_gumbel_rescale_q", True
        )
        self.mcts_gumbel_use_mixed_value = _model_search_bool(
            model, "nosl_is_az_gumbel_use_mixed_value", True
        )
        self.mcts_gumbel_interior_uniformity = _clamp_float(
            self._model_search_param(
                "nosl_is_az_gumbel_interior_uniformity", 0.0
            ), 0.0, 0.0, 1.0
        )
        self.mcts_gumbel_interior_coverage_fraction = _clamp_float(
            self._model_search_param(
                "nosl_is_az_gumbel_interior_coverage_fraction", 0.4
            ), 0.4, 0.0, 1.0
        )
        self.mcts_gumbel_interior_value_mode = int(
            self._model_search_param("nosl_is_az_gumbel_interior_value_mode", 2)
        )
        self.mcts_leaf_batch_size = _clamp_int(
            self._model_search_param("nosl_is_az_leaf_batch_size", 1),
            1, 1, 256,
        )
        self.mcts_master_nonce = int(
            self._model_search_param("nosl_is_az_master_nonce", MCTS_MASTER_NONCE)
        )
        self._apply_paper_full_gumbel_defaults()
        self.stochastic = False
        self.upper_strategy = _UPPER_STRATEGY_RULES
        self.combat_strategy = _COMBAT_STRATEGY_MODEL
        self._turn_planner = TurnPlannerStrategy(_TURN_PLANNER_CONFIG)
        self.precompute_enabled = False
        self.route_step_modes: list[str] = []
        self.route_preprocess_target_act: int | None = None
        self.route_preprocess_hp: int | None = None
        self.route_preprocess_boss_step: int | None = None
        self._mcts_sampler = NOSLSampler(MCTS_MASTER_NONCE)
        self.reset(seed)

    def _model_search_param(self, name: str, default: Any) -> Any:
        """Read a saved search setting from the algorithm or policy object."""
        return _model_search_param(self.model, name, default)

    def _model_search_float(self, name: str, default: float) -> float:
        return _model_search_float(self.model, name, default)

    def _set_combat_strategy(self, strategy: str | None) -> None:
        if strategy is None:
            return
        candidate = str(strategy).strip().lower()
        if candidate not in _COMBAT_STRATEGIES:
            raise ValueError(
                "combat strategy must be one of "
                f"{sorted(_COMBAT_STRATEGIES)}, got {strategy!r}"
            )
        self.combat_strategy = candidate

    def _apply_paper_full_gumbel_defaults(self, *, preserve_overrides: bool = False):
        """Keep Watch's displayed/native settings on the mctx paper path."""
        if not self.mcts_paper_full_gumbel:
            return
        if self.mcts_simulations <= 0:
            self.mcts_paper_full_gumbel = False
            return
        if preserve_overrides and not _strict_paper_search_settings_match(
            search_mode=self.mcts_search_mode,
            leaf_batch_size=self.mcts_leaf_batch_size,
            prior_temperature=self.mcts_prior_temperature,
            gumbel_candidates=self.mcts_gumbel_candidates,
            gumbel_scale=self.mcts_gumbel_scale,
            gumbel_value_scale=self.mcts_gumbel_value_scale,
            gumbel_max_visit_init=self.mcts_gumbel_max_visit_init,
            gumbel_rescale_q=self.mcts_gumbel_rescale_q,
            gumbel_use_mixed_value=self.mcts_gumbel_use_mixed_value,
            gumbel_interior_uniformity=self.mcts_gumbel_interior_uniformity,
            gumbel_interior_coverage_fraction=self.mcts_gumbel_interior_coverage_fraction,
            gumbel_interior_value_mode=self.mcts_gumbel_interior_value_mode,
        ):
            self.mcts_paper_full_gumbel = False
            return
        self.mcts_search_mode = "full_gumbel"
        self.mcts_root_search_mode = SEARCH_MODE_IDS["full_gumbel"]
        self.mcts_leaf_batch_size = STRICT_SEARCH_DEFAULTS["leaf_batch_size"]
        self.mcts_prior_temperature = STRICT_SEARCH_DEFAULTS["prior_temperature"]
        self.mcts_gumbel_candidates = STRICT_SEARCH_DEFAULTS["gumbel_candidates"]
        self.mcts_gumbel_scale = STRICT_SEARCH_DEFAULTS["gumbel_scale"]
        self.mcts_gumbel_value_scale = STRICT_SEARCH_DEFAULTS["gumbel_value_scale"]
        self.mcts_gumbel_max_visit_init = STRICT_SEARCH_DEFAULTS[
            "gumbel_max_visit_init"
        ]
        self.mcts_gumbel_rescale_q = STRICT_SEARCH_DEFAULTS["gumbel_rescale_q"]
        self.mcts_gumbel_use_mixed_value = STRICT_SEARCH_DEFAULTS[
            "gumbel_use_mixed_value"
        ]
        self.mcts_gumbel_interior_uniformity = STRICT_SEARCH_DEFAULTS[
            "gumbel_interior_uniformity"
        ]
        self.mcts_gumbel_interior_coverage_fraction = STRICT_SEARCH_DEFAULTS[
            "gumbel_interior_coverage_fraction"
        ]
        self.mcts_gumbel_interior_value_mode = STRICT_SEARCH_DEFAULTS[
            "gumbel_interior_value_mode"
        ]
        self.stochastic = False

    def close(self):
        self.env.close()

    def reset(
        self,
        seed: int,
        precompute: bool = False,
        simulations: int | None = None,
        depth: int | None = None,
        leaf_batch_size: int | None = None,
        prior_temperature: float | None = None,
        search_mode: str | None = None,
        gumbel_candidates: int | None = None,
        gumbel_scale: float | None = None,
        gumbel_value_scale: float | None = None,
        gumbel_max_visit_init: float | None = None,
        gumbel_rescale_q: bool | None = None,
        gumbel_use_mixed_value: bool | None = None,
        gumbel_interior_uniformity: float | None = None,
        gumbel_interior_coverage_fraction: float | None = None,
        gumbel_interior_value_mode: int | None = None,
        paper_full_gumbel: bool | None = None,
        end_turn_as_leaf: bool | None = None,
        fixed_worldline: bool | None = None,
        stochastic: bool | None = None,
        upper_strategy: str | None = None,
        combat_strategy: str | None = None,
        target_boss_act: int | None = None,
    ):
        requested_simulations = (
            self.mcts_simulations if simulations is None else int(simulations)
        )
        if paper_full_gumbel is True and requested_simulations <= 0:
            raise ValueError("strict paper Full Gumbel requires simulations >= 1")
        self.seed = int(seed)
        if paper_full_gumbel is not None:
            self.mcts_paper_full_gumbel = bool(paper_full_gumbel)
        if simulations is not None:
            self.mcts_simulations = _clamp_int(
                simulations,
                DEFAULT_MCTS_SIMULATIONS,
                0,
                MAX_MCTS_SIMULATIONS,
            )
        if depth is not None:
            self.mcts_depth = _clamp_int(
                depth,
                DEFAULT_MCTS_DEPTH,
                0,
                MAX_MCTS_DEPTH,
            )
        if leaf_batch_size is not None:
            self.mcts_leaf_batch_size = _clamp_int(
                leaf_batch_size,
                self.mcts_leaf_batch_size,
                1,
                256,
            )
        if prior_temperature is not None:
            self.mcts_prior_temperature = _clamp_float(
                prior_temperature,
                self.mcts_prior_temperature,
                MIN_MCTS_PRIOR_TEMPERATURE,
                MAX_MCTS_PRIOR_TEMPERATURE,
            )
        if search_mode is not None:
            self.mcts_search_mode = _normalize_search_mode(
                search_mode, self.mcts_search_mode
            )
            self.mcts_root_search_mode = SEARCH_MODE_IDS.get(
                self.mcts_search_mode, 0
            )
        if gumbel_candidates is not None:
            self.mcts_gumbel_candidates = _clamp_int(
                gumbel_candidates, self.mcts_gumbel_candidates,
                0, MAX_GUMBEL_CANDIDATES,
            )
        if gumbel_scale is not None:
            self.mcts_gumbel_scale = _clamp_float(
                gumbel_scale, self.mcts_gumbel_scale,
                0.0, MAX_GUMBEL_SCALE,
            )
        if gumbel_value_scale is not None:
            self.mcts_gumbel_value_scale = _clamp_float(
                gumbel_value_scale, self.mcts_gumbel_value_scale,
                0.0, MAX_GUMBEL_VALUE_SCALE,
            )
        if gumbel_max_visit_init is not None:
            self.mcts_gumbel_max_visit_init = _clamp_float(
                gumbel_max_visit_init, self.mcts_gumbel_max_visit_init,
                0.0, MAX_GUMBEL_MAX_VISIT_INIT,
            )
        if gumbel_rescale_q is not None:
            self.mcts_gumbel_rescale_q = bool(gumbel_rescale_q)
        if gumbel_use_mixed_value is not None:
            self.mcts_gumbel_use_mixed_value = bool(gumbel_use_mixed_value)
        if gumbel_interior_uniformity is not None:
            self.mcts_gumbel_interior_uniformity = _clamp_float(
                gumbel_interior_uniformity, self.mcts_gumbel_interior_uniformity,
                0.0, 1.0
            )
        if gumbel_interior_coverage_fraction is not None:
            self.mcts_gumbel_interior_coverage_fraction = _clamp_float(
                gumbel_interior_coverage_fraction,
                self.mcts_gumbel_interior_coverage_fraction, 0.0, 1.0
            )
        if gumbel_interior_value_mode is not None:
            self.mcts_gumbel_interior_value_mode = max(0, min(2, int(gumbel_interior_value_mode)))
        if end_turn_as_leaf is not None:
            self.mcts_end_turn_as_leaf = bool(end_turn_as_leaf)
        if fixed_worldline is not None:
            self.mcts_fixed_worldline = bool(fixed_worldline)
        if stochastic is not None:
            self.stochastic = bool(stochastic)
        if upper_strategy is not None:
            candidate = str(upper_strategy).strip().lower()
            if candidate not in _UPPER_STRATEGIES:
                raise ValueError(
                    "upper strategy must be one of "
                    f"{sorted(_UPPER_STRATEGIES)}, got {upper_strategy!r}"
                )
            self.upper_strategy = candidate
        self._set_combat_strategy(combat_strategy)
        self._apply_paper_full_gumbel_defaults(
            preserve_overrides=paper_full_gumbel is not True
        )
        if target_boss_act is not None:
            target_boss_act = int(target_boss_act)
            if target_boss_act < 1 or target_boss_act > 3:
                raise ValueError(
                    f"target boss act must be between 1 and 3, got {target_boss_act}"
                )
            if precompute:
                raise ValueError(
                    "whole-run precompute and route-only boss fast-forward are exclusive"
                )
        self.precompute_enabled = bool(precompute)
        self.history = []
        self.native_history = []
        self.native_direct_history = []
        self.route_step_modes = []
        self.position = 0
        self.timeline = []
        self.timeline_exhausted = False
        self.route_preprocess_target_act = target_boss_act
        self.route_preprocess_hp = None
        self.route_preprocess_boss_step = None
        self.obs, self.info = self.env.reset(seed=self.seed)
        if target_boss_act is not None:
            self.route_preprocess_hp = int(self.env.unwrapped.thick.gc.max_hp)
            self._apply_route_step_mode(self._route_mode_for_current_state())
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.map_state = None
        self.map_states = {}
        self.card_reward_choices = []
        self._search_generation = 0
        self._prepare_decision()
        if target_boss_act is not None:
            return self.fast_forward_to_boss(target_boss_act)
        if precompute:
            return self.precompute_run()
        return self.snapshot()

    def apply_custom_state(self, spec: dict):
        """Construct a combat from an editor-supplied state and enter Watch.

        The native simulator remains authoritative: encounter construction,
        card legality, intent execution and NOSL search all operate on the
        resulting BattleContext.  ``move`` is intentionally a one-turn
        override; ending the turn lets the simulator roll the next intent.
        """
        if not isinstance(spec, dict):
            raise ValueError("自定义状态必须是 JSON 对象")
        thick = self.env.unwrapped.thick
        if thick is None:
            self.reset(self.seed)
            thick = self.env.unwrapped.thick
        if hasattr(thick.gc, "skip_battles"):
            thick.gc.skip_battles = False
        encounter = str(spec.get("encounter") or "Cultist")
        names = set(sts.encounter_names())
        if encounter not in names:
            raise ValueError(f"未知敌人/战斗：{encounter}")
        player = spec.get("player") or {}
        relics = spec.get("relics")
        if relics is not None:
            if isinstance(relics, dict):
                relics = [
                    {"id": str(k), "counter": int(v)}
                    for k, v in relics.items()
                ]
            relic_ids = [
                (x.get("id") if isinstance(x, dict) else str(x))
                for x in relics
            ]
            # set_relics accepts stable simulator ids and preserves counters
            # only when supplied through relic_state; use the richer API when
            # available, otherwise fall back to ids.
            try:
                thick.gc.set_relic_state(relics)
            except Exception:
                thick.gc.set_relics(relic_ids)
        if "hp" in player:
            thick.gc.cur_hp = int(player["hp"])
        if "max_hp" in player:
            thick.gc.max_hp = int(player["max_hp"])
        thick.bc = sts.create_battle(thick.gc, encounter)
        thick.bc.apply_custom_state(spec)
        # The thin wrapper caches phase/legal masks and the model observation.
        self.env._pending = None
        self.env._multi = None
        self.env._nosl_cpp_az_session = None
        self.env._nosl_cpp_last_session = None
        self.env._nosl_cpp_pending = None
        self.env._invalidate_backend_cache()
        self.history = []
        self.native_history = []
        self.native_direct_history = []
        self.route_step_modes = []
        self.position = 0
        self.timeline = []
        self.timeline_exhausted = False
        self.route_preprocess_target_act = None
        self.route_preprocess_hp = None
        self.route_preprocess_boss_step = None
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.map_state = None
        self.map_states = {}
        self.card_reward_choices = []
        self._search_generation += 1
        self.obs, self.info = self.env._obs(), self.env._info()
        self._prepare_decision()
        return self.snapshot()

    def configure_search(
        self,
        simulations: int | None = None,
        depth: int | None = None,
        leaf_batch_size: int | None = None,
        prior_temperature: float | None = None,
        search_mode: str | None = None,
        gumbel_candidates: int | None = None,
        gumbel_scale: float | None = None,
        gumbel_value_scale: float | None = None,
        gumbel_max_visit_init: float | None = None,
        gumbel_rescale_q: bool | None = None,
        gumbel_use_mixed_value: bool | None = None,
        gumbel_interior_uniformity: float | None = None,
        gumbel_interior_coverage_fraction: float | None = None,
        gumbel_interior_value_mode: int | None = None,
        paper_full_gumbel: bool | None = None,
        end_turn_as_leaf: bool | None = None,
        fixed_worldline: bool | None = None,
        stochastic: bool | None = None,
        combat_strategy: str | None = None,
    ):
        """Apply search settings without resetting the current run."""
        requested_simulations = (
            self.mcts_simulations if simulations is None else int(simulations)
        )
        if paper_full_gumbel is True and requested_simulations <= 0:
            raise ValueError("strict paper Full Gumbel requires simulations >= 1")
        if paper_full_gumbel is not None:
            self.mcts_paper_full_gumbel = bool(paper_full_gumbel)
        if simulations is not None:
            self.mcts_simulations = _clamp_int(
                simulations,
                self.mcts_simulations,
                0,
                MAX_MCTS_SIMULATIONS,
            )
        if depth is not None:
            self.mcts_depth = _clamp_int(
                depth,
                self.mcts_depth,
                0,
                MAX_MCTS_DEPTH,
            )
        if leaf_batch_size is not None:
            self.mcts_leaf_batch_size = _clamp_int(
                leaf_batch_size,
                self.mcts_leaf_batch_size,
                1,
                256,
            )
        if prior_temperature is not None:
            self.mcts_prior_temperature = _clamp_float(
                prior_temperature,
                self.mcts_prior_temperature,
                MIN_MCTS_PRIOR_TEMPERATURE,
                MAX_MCTS_PRIOR_TEMPERATURE,
            )
        if search_mode is not None:
            self.mcts_search_mode = _normalize_search_mode(
                search_mode, self.mcts_search_mode
            )
            self.mcts_root_search_mode = SEARCH_MODE_IDS.get(
                self.mcts_search_mode, 0
            )
        if gumbel_candidates is not None:
            self.mcts_gumbel_candidates = _clamp_int(
                gumbel_candidates, self.mcts_gumbel_candidates,
                0, MAX_GUMBEL_CANDIDATES,
            )
        if gumbel_scale is not None:
            self.mcts_gumbel_scale = _clamp_float(
                gumbel_scale, self.mcts_gumbel_scale,
                0.0, MAX_GUMBEL_SCALE,
            )
        if gumbel_value_scale is not None:
            self.mcts_gumbel_value_scale = _clamp_float(
                gumbel_value_scale, self.mcts_gumbel_value_scale,
                0.0, MAX_GUMBEL_VALUE_SCALE,
            )
        if gumbel_max_visit_init is not None:
            self.mcts_gumbel_max_visit_init = _clamp_float(
                gumbel_max_visit_init, self.mcts_gumbel_max_visit_init,
                0.0, MAX_GUMBEL_MAX_VISIT_INIT,
            )
        if gumbel_rescale_q is not None:
            self.mcts_gumbel_rescale_q = bool(gumbel_rescale_q)
        if gumbel_use_mixed_value is not None:
            self.mcts_gumbel_use_mixed_value = bool(gumbel_use_mixed_value)
        if gumbel_interior_uniformity is not None:
            self.mcts_gumbel_interior_uniformity = _clamp_float(
                gumbel_interior_uniformity,
                self.mcts_gumbel_interior_uniformity,
                0.0,
                1.0,
            )
        if gumbel_interior_coverage_fraction is not None:
            self.mcts_gumbel_interior_coverage_fraction = _clamp_float(
                gumbel_interior_coverage_fraction,
                self.mcts_gumbel_interior_coverage_fraction,
                0.0,
                1.0,
            )
        if gumbel_interior_value_mode is not None:
            self.mcts_gumbel_interior_value_mode = max(
                0, min(2, int(gumbel_interior_value_mode))
            )
        if end_turn_as_leaf is not None:
            self.mcts_end_turn_as_leaf = bool(end_turn_as_leaf)
        if fixed_worldline is not None:
            self.mcts_fixed_worldline = bool(fixed_worldline)
        if stochastic is not None:
            self.stochastic = bool(stochastic)
        self._set_combat_strategy(combat_strategy)
        self._apply_paper_full_gumbel_defaults(
            preserve_overrides=paper_full_gumbel is not True
        )
        # A previous neural decision may have retained a complete native
        # action until its thin root id is submitted.  Changing settings means
        # that pending action is no longer the displayed decision and must not
        # leak into the next direct-policy/search step.
        self.env.unwrapped._nosl_cpp_pending = None
        self._prepare_decision()
        return self.snapshot()

    def _at_route_target_boss(self) -> bool:
        target_act = self.route_preprocess_target_act
        thick = self.env.unwrapped.thick
        if target_act is None or thick is None or thick.bc is None:
            return False
        room = str(thick.gc.cur_room).split(".")[-1]
        return (
            self.info is not None
            and self.info.get("phase") == "combat"
            and int(thick.gc.act) == int(target_act)
            and room == "BOSS"
        )

    def _route_mode_for_current_state(self) -> str:
        target_act = self.route_preprocess_target_act
        if target_act is None:
            return _ROUTE_STEP_NORMAL
        if (
            self.route_preprocess_boss_step is not None
            and self.position >= self.route_preprocess_boss_step
        ):
            return _ROUTE_STEP_NORMAL
        phase = self.info.get("phase") if self.info else None
        if phase == "combat":
            return _ROUTE_STEP_NORMAL
        gc = self.env.unwrapped.thick.gc
        if (
            phase == "map"
            and int(gc.act) == int(target_act)
            and int(gc.cur_map_node_y) >= 14
        ):
            return _ROUTE_STEP_ENTER_BOSS
        return _ROUTE_STEP_SKIP_COMBAT

    def _apply_route_step_mode(self, mode: str) -> None:
        if mode not in {
            _ROUTE_STEP_NORMAL,
            _ROUTE_STEP_SKIP_COMBAT,
            _ROUTE_STEP_ENTER_BOSS,
        }:
            raise ValueError(f"unknown route preprocessing mode: {mode!r}")
        thick = self.env.unwrapped.thick
        gc = thick.gc
        if mode == _ROUTE_STEP_NORMAL:
            if hasattr(gc, "skip_battles"):
                gc.skip_battles = False
            return
        if thick.bc is not None:
            raise RuntimeError("route preprocessing cannot mutate a live combat")
        if not hasattr(gc, "skip_battles"):
            raise RuntimeError(
                "Boss fast-forward requires a simulator build with "
                "GameContext.skip_battles"
            )
        gc.skip_battles = mode == _ROUTE_STEP_SKIP_COMBAT
        hp = self.route_preprocess_hp
        if hp is None:
            raise RuntimeError("route preprocessing HP baseline is missing")
        gc.max_hp = max(1, int(hp))
        gc.cur_hp = max(1, int(hp))
        thick._prev_hp = int(gc.cur_hp)
        for slot in range(len(gc.overworld_potions)):
            gc.discard_potion(slot)
        self.env._invalidate_backend_cache()
        self.obs, self.info = self.env._obs(), self.env._info()

    def fast_forward_to_boss(self, target_act: int):
        target_act = int(target_act)
        if target_act < 1 or target_act > 3:
            raise ValueError(
                f"target boss act must be between 1 and 3, got {target_act}"
            )
        if self.route_preprocess_target_act != target_act:
            raise RuntimeError("Boss fast-forward must start from reset")

        steps = 0
        while not self._at_route_target_boss():
            if self.terminated or self.truncated:
                stats = self.info.get("run_stats", {}) if self.info else {}
                raise RuntimeError(
                    "route preprocessing ended before the requested boss "
                    f"(target Act {target_act}, reached Act {stats.get('act', '?')} "
                    f"floor {stats.get('floor', '?')})"
                )
            if self.decision is None:
                phase = self.info.get("phase") if self.info else "unknown"
                raise RuntimeError(
                    "route preprocessing reached a screen with no model action "
                    f"(phase={phase})"
                )
            mode = self._route_mode_for_current_state()
            if mode == _ROUTE_STEP_NORMAL:
                raise RuntimeError(
                    "route preprocessing reached combat before the requested boss"
                )
            previous_position = self.position
            self._advance_one(route_mode=mode)
            steps += 1
            if self.position == previous_position:
                raise RuntimeError("route preprocessing made no progress")
            if mode == _ROUTE_STEP_ENTER_BOSS and not self._at_route_target_boss():
                raise RuntimeError(
                    f"Act {target_act} boss entrance did not create a live combat"
                )
            if steps >= 5000:
                raise RuntimeError("route preprocessing exceeded 5000 actions")
        return self.snapshot()

    def precompute_run(self):
        original_position = self.position
        steps = 0
        while not (self.terminated or self.truncated):
            previous_position = self.position
            self._advance_one()
            steps += 1
            if self.position == previous_position:
                break
            if steps >= 5000:
                raise RuntimeError("route preprocessing exceeded 5000 actions")
        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        return self.seek(original_position)

    def advance_to_timeline(self, target_index: int):
        target_index = int(target_index)
        if target_index < 0 or target_index >= 48:
            raise ValueError(f"timeline index must be between 0 and 47, got {target_index}")
        if target_index < len(self.timeline):
            return self.seek(self.timeline[target_index]["step"])

        steps = 0
        while len(self.timeline) <= target_index and not (
            self.terminated or self.truncated
        ):
            previous_position = self.position
            self._advance_one()
            steps += 1
            if self.position == previous_position:
                break
            if steps >= 5000:
                raise RuntimeError("fast-forward exceeded 5000 actions")

        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        if target_index < len(self.timeline):
            return self.snapshot()
        return self.snapshot()

    def preview_map(self, act: int):
        act = int(act)
        if act < 1 or act > 3:
            raise ValueError(f"act must be between 1 and 3, got {act}")
        original_position = self.position
        # Each act contributes 15 visible map rooms plus a y=15 boss entry.
        # Preview through the last visible room; entering a later act naturally
        # records the previous act's boss checkpoint as well.
        target_index = (act - 1) * 16 + 14
        if target_index >= len(self.timeline) and not self.timeline_exhausted:
            self.advance_to_timeline(target_index)
        if self.position != original_position:
            self.seek(original_position)
        return self.snapshot()

    def _policy_outputs(self, observation, mask):
        """Return masked root probabilities, value, and raw policy logits.

        The C++ search consumes the same unmasked logits that the training
        collector passes to ``nosl_is_az_cpp_begin``.  Keeping this forward in
        one helper prevents the dashboard from accidentally using a different
        policy head or a second, stale value path.
        """
        import torch

        policy = getattr(self.model, "policy", None)
        core = getattr(policy, "core", None)
        if policy is None or core is None or not hasattr(core, "actor_critic"):
            with self.model_lock:
                probs, value = _action_dist(self.model, observation, mask)
            return probs, value, None
        obs_tensor, _ = policy.obs_to_tensor(
            _observation_for_policy(self.model, observation)
        )
        with self.model_lock, torch.inference_mode():
            logits, values = core.actor_critic(obs_tensor)
        logits = logits.detach().float().reshape(-1)
        raw_logits = logits.cpu().numpy()
        legal = np.flatnonzero(np.asarray(mask, dtype=bool))
        probs = np.zeros(raw_logits.size, dtype=np.float32)
        if legal.size:
            selected = raw_logits[legal].astype(np.float64, copy=False)
            selected = selected - np.max(selected)
            weights = np.exp(np.clip(selected, -80.0, 80.0))
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0.0:
                weights = np.ones(legal.size, dtype=np.float64)
                total = float(legal.size)
            probs[legal] = (weights / total).astype(np.float32)
        value = float(values.detach().float().reshape(-1)[0].cpu().item())
        return probs, value, raw_logits


    def _run_native_search(self, root_logits, root_value):
        config = NativeSearchConfig(**{
            field.name: getattr(self, "mcts_" + field.name, field.default)
            for field in fields(NativeSearchConfig)
        })
        nonce = int(self.mcts_master_nonce) ^ int(self.seed) ^ (int(self.position) * 0x9E3779B97F4A7C15)
        return native_search(self.model, self.env, None, config, nonce=nonce,
                             decision_index=self.position, stochastic=self.stochastic,
                             lock=self.model_lock, root_outputs=(root_logits, root_value))

    def _abort_native_search(self):
        self.env.unwrapped.nosl_is_az_cpp_abort()

    def _select_upper_action(self, model_action: int, legal, phase: str):
        if phase == "combat" or self.upper_strategy == _UPPER_STRATEGY_MODEL:
            return int(model_action), _UPPER_STRATEGY_MODEL
        rule_action = self.env.overworld_rule_decision()
        if rule_action is None:
            return int(model_action), "model_fallback"
        rule_action = int(rule_action)
        if not np.any(np.asarray(legal, dtype=np.int64) == rule_action):
            raise RuntimeError(f"illegal overworld rule action: {rule_action}")
        return rule_action, "overworld_rules"

    def _select_combat_planner_action(self, legal):
        request = DecisionRequest(
            self.env,
            self.obs,
            self.info,
            self.seed,
            self.position,
        )
        decision = self._turn_planner.decide([request])[0]
        if decision.native_action is not None:
            native = int(decision.native_action)
            if native not in self.env.native_combat_actions():
                raise RuntimeError(f"illegal turn-planner native action: {native}")
            action = self.env.unwrapped._native_action_to_thin(native)
            return action, native, dict(decision.diagnostics), decision.source
        action = int(decision.action)
        if not np.any(np.asarray(legal, dtype=np.int64) == action):
            raise RuntimeError(f"illegal turn-planner fallback action: {action}")
        return action, None, dict(decision.diagnostics), decision.source

    def _describe_native_action(self, action_bits: int, thin_action: int | None):
        action = sts.Action(int(action_bits))
        if thin_action is not None:
            label = self.env.describe_action(int(thin_action))
        elif action.action_type == sts.ActionType.SINGLE_CARD_SELECT:
            candidate = next(
                (
                    item
                    for item in self.env.combat_selection_candidates()
                    if item.action_bits == int(action_bits)
                ),
                None,
            )
            label = (
                f"select {candidate.card[0]}"
                if candidate is not None and candidate.card is not None
                else "skip card selection"
                if candidate is not None
                else f"select card #{action.source_idx + 1}"
            )
        elif action.action_type == sts.ActionType.MULTI_CARD_SELECT:
            selected = [
                str(index + 1)
                for index in range(16)
                if int(action.source_idx) & (1 << index)
            ]
            label = "confirm cards " + (", ".join(selected) if selected else "none")
        elif action.action_type == sts.ActionType.END_TURN:
            label = "end turn"
        else:
            label = f"native action 0x{int(action_bits):08x}"
        if action.action_type in (sts.ActionType.CARD, sts.ActionType.POTION):
            if self.env.unwrapped.thick.bc.is_targeted_action(int(action_bits)):
                label = f"{label} -> target monster #{int(action.target_idx) + 1}"
        return label

    def _prepare_decision(self):
        if self.terminated or self.truncated:
            self.decision = None
            return
        # The native combat bridge can advance the thick simulator while the
        # Gym wrapper returns the pre-transition observation (notably when an
        # event action enters combat).  Refresh from the authoritative env
        # state before constructing the next decision; otherwise the UI may
        # retain a stale Neow/event action after the screen already shows
        # combat.
        try:
            self.obs, self.info = self.env._obs(), self.env._info()
        except Exception:
            pass
        mask = self.env.action_masks()
        base = self.env.unwrapped
        probs, value, root_logits = self._policy_outputs(self.obs, mask)
        legal = np.flatnonzero(mask)
        is_combat = self.info.get("phase") == "combat"
        planner_active = (
            is_combat and self.combat_strategy == _COMBAT_STRATEGY_PLANNER
        )
        if legal.size == 0 and not planner_active:
            self.decision = None
            return
        if legal.size == 0:
            model_action = None
            ranked = []
        elif probs is None:
            model_action = int(legal[0])
            ranked = [(model_action, None)]
        else:
            model_action = int(max(legal, key=lambda i: probs[i]))
            ranked = sorted(legal, key=lambda i: probs[i], reverse=True)[:6]
            ranked = [(int(i), float(probs[i])) for i in ranked]
        chosen = None
        decision_source = self._turn_planner.name if planner_active else None
        if not planner_active:
            chosen, decision_source = self._select_upper_action(
                model_action,
                legal,
                self.info.get("phase") if self.info else "unknown",
            )
        search = None
        combat_planner = None
        native_action_bits = None
        native_direct = False
        if planner_active:
            chosen, native_action_bits, combat_planner, decision_source = (
                self._select_combat_planner_action(legal)
            )
            native_direct = native_action_bits is not None
            if chosen is not None and all(
                action != chosen for action, _probability in ranked
            ):
                ranked.append(
                    (chosen, None if probs is None else float(probs[chosen]))
                )
        native_root = getattr(self.env.unwrapped, "_pending", None) is None
        if (
            is_combat
            and self.combat_strategy == _COMBAT_STRATEGY_MODEL
            and native_root
            and self.mcts_search_mode != "direct"
            and self.mcts_simulations > 0
            and root_logits is not None
        ):
            try:
                result = self._run_native_search(root_logits, value)
            except (
                RuntimeError,
                ValueError,
                AttributeError,
                IndexError,
                KeyError,
            ) as exc:
                # Keep the run usable with an old simulator build, but never
                # silently fall back to the historical random-rollout MCTS.
                # pybind11 translates std::out_of_range from the native
                # simulator (notably ``map::at``) into IndexError.  A failed
                # leaf must not strand the watch session after the backend
                # has already advanced to the next state.
                self._abort_native_search()
                result = None
                search = {
                    "method": "direct_policy",
                    "requested_method": "nosl_is_az_cpp",
                    "error": str(exc),
                }
            if result is not None:
                chosen = int(result["action"])
                decision_source = "combat_search"
                selected_target = int(result.get("selected_target", -1))
                selected_targeted = bool(result.get("selected_action_targeted", False))
                native_action_bits = int(result.get("native_action_bits", 0xFFFFFFFF))
                root_visits = np.asarray(result.get("root_visits", ()), dtype=np.int32)
                total_visits = int(root_visits.sum())
                root_visit_policy = np.asarray(
                    result.get("root_visit_policy", ()), dtype=np.float64
                ).reshape(-1)
                improved_policy = np.asarray(
                    result.get("improved_policy", result.get("visit_policy", ())),
                    dtype=np.float64,
                ).reshape(-1)
                used_completed_policy = bool(
                    result.get("used_completed_policy", False)
                )
                search_actions = []
                for aid in legal:
                    visits = max(0, int(root_visits[int(aid)])) if int(aid) < root_visits.size else 0
                    visit_prob = (
                        float(root_visit_policy[int(aid)])
                        if int(aid) < root_visit_policy.size
                        else (float(visits) / total_visits if total_visits > 0 else 0.0)
                    )
                    policy_prob = (
                        float(improved_policy[int(aid)])
                        if used_completed_policy and int(aid) < improved_policy.size
                        else visit_prob
                    )
                    search_actions.append({
                        "action": int(aid),
                        "label": self.env.describe_action(int(aid)),
                        "visits": visits,
                        "value": float(result.get("root_value", 0.0)),
                        # ``prob`` is the policy used by Full Gumbel's target;
                        # retain visit_prob for diagnosing sparse visits.
                        "prob": policy_prob,
                        "visit_prob": visit_prob,
                        "improved_prob": policy_prob if used_completed_policy else None,
                        "target_slot": (
                            selected_target if selected_targeted and int(aid) == chosen else None
                        ),
                        "selected": int(aid) == chosen,
                    })
                search_actions.sort(
                    key=lambda item: (item["prob"], item["visits"], -item["action"]),
                    reverse=True,
                )
                search = {
                    "method": "nosl_is_az_cpp",
                    "search": (
                        "full_gumbel_alpha_zero"
                        if self.mcts_search_mode == "full_gumbel"
                        else "gumbel_sequential_halving"
                        if self.mcts_search_mode == "gumbel"
                        else "neural_puct"
                    ),
                    "search_mode": self.mcts_search_mode,
                    "search_mode_label": (
                        STRICT_PAPER_SEARCH_LABEL
                        if self.mcts_paper_full_gumbel
                        else SEARCH_MODE_LABELS.get(
                            self.mcts_search_mode, self.mcts_search_mode
                        )
                    ),
                    "paper_full_gumbel": self.mcts_paper_full_gumbel,
                    "fixed_worldline": self.mcts_fixed_worldline,
                    "simulations": int(result.get("simulations", self.mcts_simulations)),
                    "depth": self.mcts_depth,
                    "prior_temperature": self.mcts_prior_temperature,
                    "c_puct": self.mcts_c_puct,
                    "leaf_batch_size": self.mcts_leaf_batch_size,
                    "root_search_mode": self.mcts_root_search_mode,
                    "gumbel_candidates": self.mcts_gumbel_candidates,
                    "gumbel_scale": self.mcts_gumbel_scale,
                    "gumbel_value_scale": self.mcts_gumbel_value_scale,
                    "gumbel_max_visit_init": self.mcts_gumbel_max_visit_init,
                    "gumbel_rescale_q": self.mcts_gumbel_rescale_q,
                    "gumbel_use_mixed_value": self.mcts_gumbel_use_mixed_value,
                    "gumbel_interior_uniformity": self.mcts_gumbel_interior_uniformity,
                    "used_completed_policy": used_completed_policy,
                    "nosl": True,
                    "stochastic": self.stochastic,
                    "model_action": model_action,
                    "action": chosen,
                    "native_action_bits": native_action_bits,
                    "selected_target": selected_target if selected_targeted else None,
                    "selected_action_targeted": selected_targeted,
                    "tree_id": f"{self.position}:{self._search_generation + 1}",
                    "root_value": float(result.get("root_value", 0.0)),
                    "selection": (
                        "root_visits_sample" if self.stochastic
                        else "gumbel_completed_score"
                        if self.mcts_search_mode == "full_gumbel"
                        else "root_visits_argmax"
                    ),
                    "root_actions": search_actions,
                }
                self._search_generation += 1
                search["tree_id"] = f"{self.position}:{self._search_generation}"
                tree = base.nosl_is_az_cpp_tree_snapshot((), max_children=64)
                search["tree"] = self._format_native_tree_snapshot(
                    tree, native_action_bits
                )
        elif (
            probs is not None
            and self.stochastic
            and (
                (is_combat and self.combat_strategy == _COMBAT_STRATEGY_MODEL)
                or (
                    not is_combat
                    and self.upper_strategy == _UPPER_STRATEGY_MODEL
                )
            )
        ):
            weights = np.asarray([max(float(probs[i]), 0.0) for i in legal], dtype=np.float64)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0.0:
                weights = np.ones(len(legal), dtype=np.float64)
                total = float(len(legal))
            chosen = int(np.random.default_rng(
                self._mcts_sampler.nonce(
                    state_nonce=self.seed & ((1 << 64) - 1),
                    simulation_index=self.position * 1_000_000 + 777_777,
                )
            ).choice(legal, p=weights / total))
            decision_source = "model_sample"
        selected_target = (
            int(search.get("selected_target"))
            if search and search.get("selected_target") is not None
            else -1
        )
        selected_targeted = bool(search and search.get("selected_action_targeted"))
        label = (
            self._describe_native_action(native_action_bits, chosen)
            if native_direct
            else self.env.describe_action(chosen)
        )
        if selected_targeted and selected_target >= 0:
            label = f"{label} -> target monster #{selected_target + 1}"
        animation = self._action_animation(chosen) if chosen is not None else None
        if native_direct:
            native_action = sts.Action(int(native_action_bits))
            if native_action.action_type in (sts.ActionType.CARD, sts.ActionType.POTION):
                if base.thick.bc.is_targeted_action(int(native_action_bits)):
                    selected_target = int(native_action.target_idx)
                    selected_targeted = True
        if selected_targeted and selected_target >= 0 and animation is not None:
            animation["target_slot"] = selected_target
        self.decision = {
            "action": chosen,
            "label": label,
            "value": value,
            "model_action": model_action,
            "source": decision_source,
            "combat_planner": combat_planner,
            "native_action_bits": native_action_bits,
            "native_direct": native_direct,
            "search": search,
            "animation": animation,
            "top_actions": [
                {"action": aid, "label": self.env.describe_action(aid), "prob": prob}
                for aid, prob in ranked
            ],
        }

    def _format_native_tree_snapshot(self, snapshot, selected_native_bits=None):
        """Add stable UI labels to the compact C++ lazy-tree response."""
        if not snapshot:
            return {
                "valid": False,
                "path": [],
                "visits": 0,
                "value": 0.0,
                "network_value_valid": False,
                "network_value": None,
                "children": [],
            }
        root_level = not snapshot.get("path")
        children = []
        for item in snapshot.get("children", ()):
            bits = int(item.get("action_bits", 0))
            action_type = int(item.get("action_type", (bits >> 29) & 0xF))
            thin = self.env.unwrapped._native_action_to_thin(bits)
            native_label = str(item.get("label", "")).strip()
            if not root_level and native_label:
                # The C++ edge was expanded against this hypothetical branch
                # state, so its label reflects upgrades, hand reordering, and
                # the branch's monster names. Never decode it from live root
                # slots at deeper levels.
                label = native_label
            elif thin is not None:
                label = self.env.describe_action(thin)
            else:
                label = native_label or f"native action 0x{bits:08x}"
            if root_level and bool(item.get("targeted", False)):
                target = int(item.get("target_idx", -1))
                if target >= 0 and target < A.MAX_MONSTERS:
                    label = f"{label} -> target monster #{target + 1}"
            if action_type == 4:
                label = "end turn"
            elif action_type == 2:
                label = f"select card #{int(item.get('source_idx', 0)) + 1}"
            elif action_type == 3:
                label = "confirm card selection"
            children.append({
                **item,
                "label": label,
                "thin_action": thin,
                "selected": (
                    root_level
                    and
                    selected_native_bits is not None
                    and bits == int(selected_native_bits)
                ),
            })
        return {
            "valid": bool(snapshot.get("valid", False)),
            "path": [int(x) for x in snapshot.get("path", ())],
            "visits": int(snapshot.get("visits", 0)),
            "value_visits": int(snapshot.get("value_visits", 0)),
            "value": float(snapshot.get("value", 0.0)),
            "sample_value": float(snapshot.get(
                "sample_value", snapshot.get("value", 0.0)
            )),
            "network_value_valid": bool(snapshot.get(
                "network_value_valid", False
            )),
            "network_value": (
                float(snapshot["network_value"])
                if bool(snapshot.get("network_value_valid", False))
                else None
            ),
            "children": children,
        }

    def search_tree(self, path=(), max_children: int = 64):
        """Read one node from the currently displayed completed search tree."""
        search = self.decision.get("search") if self.decision else None
        if not search or search.get("method") != "nosl_is_az_cpp":
            return self._format_native_tree_snapshot(None)
        indexes = [int(index) for index in (path or ())]
        if len(indexes) > 128:
            raise ValueError("search tree path is too deep")
        max_children = _clamp_int(max_children, 64, 1, 128)
        raw = self.env.unwrapped.nosl_is_az_cpp_tree_snapshot(
            indexes, max_children=max_children
        )
        return self._format_native_tree_snapshot(
            raw, search.get("native_action_bits")
        )

    def _action_animation(self, action: int):
        play_base = A.seg_base("PLAY_CARD")
        target_base = A.seg_base("PICK_TARGET")
        if play_base <= action < play_base + A.HAND_SLOTS:
            return {"kind": "play_card", "slot": action - play_base}
        if target_base <= action < target_base + A.MAX_MONSTERS:
            return {"kind": "target", "slot": action - target_base}
        if action == A.seg_base("END_TURN"):
            return {"kind": "end_turn"}
        return None

    def step(self):
        if self.decision is None:
            return self.snapshot()
        self._advance_one()
        return self.snapshot()

    def _advance_one(self, *, route_mode: str | None = None):
        if self.decision is None:
            return
        decision_before = self.decision
        displayed_action = decision_before.get("action")
        displayed_action = (
            int(displayed_action) if displayed_action is not None else None
        )
        direct_native = bool(decision_before.get("native_direct"))
        decision_native = (
            int(decision_before["native_action_bits"])
            if direct_native
            else None
        )
        pending = self.env.unwrapped._nosl_cpp_pending
        pending_native = (
            int(pending[1])
            if pending is not None and int(pending[0]) == displayed_action
            else None
        )
        displayed_native = decision_native if direct_native else pending_native

        # ``seek``/``undo`` rebuild the current decision but historically left
        # the future action history intact.  Replaying that suffix blindly can
        # execute an action different from the one currently shown in the UI
        # (especially after changing search settings or a native target).  Only
        # reuse a historical transition when both the thin action and complete
        # native action still match the freshly prepared decision; otherwise
        # branch from the current state and discard the stale suffix.
        replaying = self.position < len(self.history)
        historical_action = (
            self.history[self.position] if replaying else None
        )
        if historical_action is not None:
            historical_action = int(historical_action)
        historical_native = (
            int(self.native_history[self.position])
            if replaying and self.position < len(self.native_history)
            and self.native_history[self.position] is not None
            else None
        )
        historical_direct = bool(
            replaying
            and self.position < len(self.native_direct_history)
            and self.native_direct_history[self.position]
        )
        requested_route_mode = (
            self._route_mode_for_current_state()
            if route_mode is None
            else str(route_mode)
        )
        historical_route_mode = (
            self.route_step_modes[self.position]
            if replaying and self.position < len(self.route_step_modes)
            else _ROUTE_STEP_NORMAL
        )
        same_action = replaying and historical_action == displayed_action
        same_native = historical_native == displayed_native
        same_native_mode = historical_direct == direct_native
        same_route_mode = historical_route_mode == requested_route_mode
        if (
            replaying
            and same_action
            and same_native
            and same_native_mode
            and same_route_mode
        ):
            action = historical_action
            native = historical_native
            execute_native = historical_direct
            active_route_mode = historical_route_mode
            if native is not None and not execute_native:
                self.env.unwrapped._nosl_cpp_pending = (
                    int(action), int(native)
                )
        else:
            if replaying:
                del self.history[self.position:]
                del self.native_history[self.position:]
                del self.native_direct_history[self.position:]
                del self.route_step_modes[self.position:]
                if (
                    self.route_preprocess_boss_step is not None
                    and self.position < self.route_preprocess_boss_step
                ):
                    self.route_preprocess_boss_step = None
            action = displayed_action
            native = displayed_native
            execute_native = direct_native
            active_route_mode = requested_route_mode
            self.history.append(action)
            self.native_history.append(native)
            self.native_direct_history.append(execute_native)
            self.route_step_modes.append(active_route_mode)
        self._apply_route_step_mode(active_route_mode)
        label = (
            decision_before.get("label")
            if displayed_action == action
            and displayed_native == native
            and direct_native == execute_native
            else self._describe_native_action(native, action)
            if execute_native
            else self.env.describe_action(action)
        )
        animation = (
            decision_before.get("animation")
            if displayed_action == action
            and displayed_native == native
            and direct_native == execute_native
            else self._action_animation(action) if action is not None else None
        )
        canon_before = self.env.unwrapped.thick.obs_canonical()
        if action is not None:
            self._record_card_reward_choice(action, canon_before)
        if execute_native:
            self.obs, reward, self.terminated, self.truncated, self.info = (
                self.env.step_native_combat(native)
            )
        else:
            self.obs, reward, self.terminated, self.truncated, self.info = (
                self.env.step(action)
            )
        self.position += 1
        if active_route_mode == _ROUTE_STEP_SKIP_COMBAT:
            self._apply_route_step_mode(active_route_mode)
        elif (
            active_route_mode == _ROUTE_STEP_ENTER_BOSS
            and self._at_route_target_boss()
        ):
            self.route_preprocess_boss_step = self.position
        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        self.last_action = {
            "action": action,
            "label": label,
            "animation": animation,
        }
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self._prepare_decision()
        canon = self.env.unwrapped.thick.obs_canonical()
        map_snapshot = self._update_map(canon)
        self._record_timeline(map_snapshot, canon)

    def undo(self):
        if self.position <= 0:
            return self.snapshot()
        return self.seek(self.position - 1)

    def seek(self, target_step: int):
        target_step = int(target_step)
        if target_step < 0 or target_step > len(self.history):
            raise ValueError(
                f"step must be between 0 and {len(self.history)}, got {target_step}"
            )
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.position = 0
        self.map_state = None
        self.card_reward_choices = []
        for index, action in enumerate(self.history[:target_step]):
            route_mode = (
                self.route_step_modes[index]
                if index < len(self.route_step_modes)
                else _ROUTE_STEP_NORMAL
            )
            self._apply_route_step_mode(route_mode)
            direct_native = bool(
                index < len(self.native_direct_history)
                and self.native_direct_history[index]
            )
            native = None
            if index < len(self.native_history):
                native = self.native_history[index]
                if native is not None and not direct_native:
                    self.env.unwrapped._nosl_cpp_pending = (int(action), int(native))
            label = (
                self._describe_native_action(native, action)
                if direct_native
                else self.env.describe_action(action)
            )
            canon_before = self.env.unwrapped.thick.obs_canonical()
            if action is not None:
                self._record_card_reward_choice(
                    action, canon_before, include_current_prob=False
                )
            if direct_native:
                self.obs, reward, self.terminated, self.truncated, self.info = (
                    self.env.step_native_combat(native)
                )
            else:
                self.obs, reward, self.terminated, self.truncated, self.info = (
                    self.env.step(action)
                )
            self.position += 1
            if route_mode == _ROUTE_STEP_SKIP_COMBAT:
                self._apply_route_step_mode(route_mode)
            self.total_reward += float(reward)
            self.last_reward = float(reward)
            self.last_action = {
                "action": action,
                "label": label,
                "animation": self._action_animation(action) if action is not None else None,
            }
            canon = self.env.unwrapped.thick.obs_canonical()
            map_snapshot = self._update_map(canon)
            self._record_timeline(map_snapshot, canon)
            if self.terminated or self.truncated:
                break
        next_route_mode = (
            self.route_step_modes[self.position]
            if self.position < len(self.route_step_modes)
            else self._route_mode_for_current_state()
        )
        self._apply_route_step_mode(next_route_mode)
        self._prepare_decision()
        return self.snapshot()

    def _record_timeline(self, map_snapshot, canon):
        if not map_snapshot or not map_snapshot.get("current"):
            return
        act = int(map_snapshot["act"])
        x, y = (int(v) for v in map_snapshot["current"])
        if any(
            node["act"] == act and node["x"] == x and node["y"] == y
            for node in self.timeline
        ):
            return
        room = next(
            (
                node.get("room", "")
                for node in map_snapshot.get("nodes", [])
                if int(node.get("x", -1)) == x and int(node.get("y", -1)) == y
            ),
            "",
        )
        global_state = canon.get("global") or {}
        stats = self.info.get("run_stats", {}) if self.info else {}
        self.timeline.append({
            "step": self.position,
            "act": act,
            "floor": int(stats.get("floor", global_state.get("floor", 0)) or 0),
            "x": x,
            "y": y,
            "room": room,
        })

    def _update_map(self, canon):
        global_state = canon.get("global") or {}
        act = int(global_state.get("act", 0) or 0)
        if act <= 0:
            return None
        if act not in self.map_states:
            self.map_states[act] = {
                "act": act,
                "boss": global_state.get("boss", ""),
                "nodes": [],
                "visited": set(),
                "current": None,
            }

        state = self.map_states[act]
        self.map_state = state
        state["boss"] = global_state.get("boss", state["boss"])
        full_map = (canon.get("options") or {}).get("full_map")
        thick = self.env.unwrapped.thick
        gc = getattr(thick, "gc", None)
        current_x = getattr(gc, "cur_map_node_x", -1)
        current_y = getattr(gc, "cur_map_node_y", -1)
        if not full_map:
            run_map = getattr(thick, "rmap", None)
            if run_map is not None:
                full_map = run_map.full_view(current_x, current_y)
        if full_map:
            state["nodes"] = [dict(node) for node in full_map]
        if current_x >= 0 and current_y >= 0:
            current = (int(current_x), int(current_y))
            state["current"] = current
            state["visited"].add(current)
        for node in state["nodes"]:
            if node.get("current"):
                current = (int(node["x"]), int(node["y"]))
                state["current"] = current
                state["visited"].add(current)
        return self._map_snapshot(state)

    @staticmethod
    def _map_snapshot(state):
        return {
            "act": state["act"],
            "boss": state["boss"],
            "nodes": state["nodes"],
            "visited": [list(node) for node in sorted(state["visited"])],
            "current": list(state["current"]) if state["current"] else None,
        }

    def _record_card_reward_choice(
        self, action: int, canon: dict, *, include_current_prob: bool = True
    ) -> None:
        phase = self.info.get("phase") if self.info else canon.get("phase")
        if phase != "combat_reward":
            return
        options = canon.get("options") or {}
        cards = options.get("cards") or []
        if not cards:
            return
        card_base = A.seg_base("REWARD_CARD")
        skip_action = A.seg_base("REWARD_SKIP")
        selected_index = None
        skipped = False
        if card_base <= action < card_base + A.REWARD_CARD:
            selected_index = int(action - card_base)
            if selected_index >= len(cards):
                return
        elif action == skip_action:
            skipped = True
        else:
            return

        stats = self.info.get("run_stats", {}) if self.info else {}
        global_state = canon.get("global") or {}
        prob = None
        if include_current_prob and self.decision:
            for ranked in self.decision.get("top_actions", []):
                if int(ranked.get("action", -1)) == int(action):
                    prob = ranked.get("prob")
                    break
        self.card_reward_choices.append({
            "step": self.position,
            "floor": int(stats.get("floor", global_state.get("floor", 0)) or 0),
            "cards": list(cards),
            "selected_index": selected_index,
            "skipped": skipped,
            "relics": list(global_state.get("relics", [])),
            "action": int(action),
            "action_label": self.env.describe_action(action),
            "prob": prob,
        })

    def snapshot(self):
        canon = self.env.unwrapped.thick.obs_canonical()
        map_snapshot = self._update_map(canon)
        self._record_timeline(map_snapshot, canon)
        stats = self.info.get("run_stats", {}) if self.info else {}
        pending = self.env._pending
        pending_card_slot = (
            int(pending[1])
            if pending is not None and pending[0] == "play"
            else None
        )
        event = describe_event(
            (canon.get("options") or {}).get("event"),
            (canon.get("options") or {}).get("event_options", []),
        )
        if event is not None:
            event_base = A.seg_base("EVENT_OPT")
            for option in event["options"]:
                option["action"] = event_base + option["index"]
        return _jsonable({
            "seed": self.seed,
            "step": self.position,
            "max_step": len(self.history),
            "precomputed_steps": len(self.history) if self.timeline_exhausted else 0,
            "precompute_enabled": self.precompute_enabled,
            "upper_strategy": self.upper_strategy,
            "upper_strategy_version": (
                OVERWORLD_RULE_VERSION
                if self.upper_strategy == _UPPER_STRATEGY_RULES
                else "checkpoint-model"
            ),
            "upper_fallback": (
                "masked_argmax"
                if self.upper_strategy == _UPPER_STRATEGY_RULES
                else None
            ),
            "combat_strategy": self.combat_strategy,
            "combat_strategy_version": (
                PLANNER_VERSION
                if self.combat_strategy == _COMBAT_STRATEGY_PLANNER
                else "checkpoint-model+configured-search"
            ),
            "combat_strategy_config": (
                asdict(_TURN_PLANNER_CONFIG)
                if self.combat_strategy == _COMBAT_STRATEGY_PLANNER
                else None
            ),
            "route_preprocess": {
                "enabled": self.route_preprocess_target_act is not None,
                "target_act": self.route_preprocess_target_act,
                "reached": (
                    self.route_preprocess_boss_step is not None
                    and self.position >= self.route_preprocess_boss_step
                ),
                "boss_step": self.route_preprocess_boss_step,
                "baseline_hp": self.route_preprocess_hp,
                "ignored_resources": ["hp", "max_hp", "potions"],
                "route_steps": sum(
                    mode != _ROUTE_STEP_NORMAL
                    for mode in self.route_step_modes[:self.position]
                ),
            },
            "mcts_simulations": self.mcts_simulations,
            "mcts_depth": self.mcts_depth,
            "mcts_end_turn_as_leaf": self.mcts_end_turn_as_leaf,
            "mcts_fixed_worldline": self.mcts_fixed_worldline,
            "mcts_c_puct": self.mcts_c_puct,
            "mcts_prior_temperature": self.mcts_prior_temperature,
            "mcts_leaf_batch_size": self.mcts_leaf_batch_size,
            "mcts_search_mode": self.mcts_search_mode,
            "mcts_search_mode_label": (
                STRICT_PAPER_SEARCH_LABEL
                if self.mcts_paper_full_gumbel
                else SEARCH_MODE_LABELS.get(
                    self.mcts_search_mode, self.mcts_search_mode
                )
            ),
            "mcts_paper_full_gumbel": self.mcts_paper_full_gumbel,
            "mcts_root_search_mode": self.mcts_root_search_mode,
            "mcts_gumbel_candidates": self.mcts_gumbel_candidates,
            "mcts_gumbel_scale": self.mcts_gumbel_scale,
            "mcts_gumbel_value_scale": self.mcts_gumbel_value_scale,
            "mcts_gumbel_max_visit_init": self.mcts_gumbel_max_visit_init,
            "mcts_gumbel_rescale_q": self.mcts_gumbel_rescale_q,
            "mcts_gumbel_use_mixed_value": self.mcts_gumbel_use_mixed_value,
            "mcts_gumbel_interior_uniformity": self.mcts_gumbel_interior_uniformity,
            "mcts_gumbel_interior_coverage_fraction": self.mcts_gumbel_interior_coverage_fraction,
            "mcts_gumbel_interior_value_mode": self.mcts_gumbel_interior_value_mode,
            "search_backend": (
                "nosl_is_az_cpp"
                if self.mcts_simulations > 0 and self.mcts_search_mode != "direct"
                else "direct_policy"
            ),
            "stochastic": self.stochastic,
            "can_undo": self.position > 0,
            "done": self.terminated or self.truncated,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "won": is_win(self.env) if self.terminated else False,
            "total_reward": self.total_reward,
            "last_reward": self.last_reward,
            "last_action": self.last_action,
            "decision": self.decision,
            "phase": self.info.get("phase") if self.info else None,
            "pending_card_slot": pending_card_slot,
            "stats": stats,
            "canon": canon,
            "event": event,
            "map": map_snapshot,
            "maps": [
                self._map_snapshot(self.map_states[act])
                for act in sorted(self.map_states)
            ],
            "timeline": self.timeline,
            "timeline_exhausted": self.timeline_exhausted,
            "card_reward_choices": self.card_reward_choices,
        })


class DashboardState:
    def __init__(self):
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.checkpoint: Path | None = None
        self.checkpoint_format: str | None = None
        self.model: Any | None = None
        self.embedding: dict | None = None
        self.token_embeddings: dict = {}
        self.watch: WatchSession | None = None
        self.eval_status: dict = {"state": "idle"}
        self.eval_token = ""
        self.error: str | None = None


    def public_state(self):
        with self.lock:
            return {
                "checkpoint": self.checkpoint.name if self.checkpoint else None,
                "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
                "checkpoint_format": self.checkpoint_format,
                "capabilities": {
                    "global_map": _model.uses_global_map_observation(self.model)
                    if self.model else False,
                    "map_dag": _model.uses_map_dag(self.model) if self.model else False,
                    "card_language": _model.uses_card_language(self.model)
                    if self.model else False,
                    "combat_language": _model.uses_combat_language(self.model)
                    if self.model else False,
                    "deck_size": _model.uses_deck_size_observation(self.model)
                    if self.model else False,
                    "neow": _model.uses_neow_observation(self.model)
                    if self.model else False,
                    "event_option": _model.uses_event_option_observation(self.model)
                    if self.model else False,
                },
                "embedding": self.embedding,
                "token_embeddings": self.token_embeddings,
                "card_catalog": {
                    card_id: {
                        "name_zh": info.name_zh,
                        "kind": info.kind,
                        "cost": list(info.cost),
                        "damage": list(info.damage),
                        "block": list(info.block),
                        "effect_en": info.effect_en,
                        "effect_zh": info.effect_zh,
                    }
                    for card_id, info in IRONCLAD_CARDS.items()
                } | {
                    card_id: {
                        "name_zh": info["name_zh"],
                        "kind": "status",
                        "cost": [-1, -1],
                        "damage": [0, 0],
                        "block": [0, 0],
                        "effect_en": info.get("effect_en", ""),
                        "effect_zh": info.get("effect_zh", ""),
                    }
                    for card_id, info in STATUS_CARDS.items()
                },
                "relic_catalog": {
                    relic_id: {
                        "name_zh": name_zh,
                    }
                    for relic_id, name_zh in RELIC_NAME_ZH.items()
                },
                "watch": self.watch.snapshot() if self.watch else None,
                "evaluation": self.eval_status,
                "error": self.error,
            }


    def load_checkpoint(self, path: Path, episodes: int):
        suffix = path.suffix.lower()
        if suffix == ".zip":
            model = MaskablePPO.load(
                path,
                device="auto",
                custom_objects=_sb3_load_custom_objects(path),
            )
            checkpoint_format = "sb3"
        else:
            raise ValueError("checkpoint must be an SB3 .zip")
        model.set_random_seed(0)
        embedding = self._embedding(model)
        token_embeddings = self._token_embeddings(model)
        watch = WatchSession(model, self.model_lock, seed=0)
        with self.lock:
            old_watch = self.watch
            self.checkpoint = path
            self.checkpoint_format = checkpoint_format
            self.model = model
            self.embedding = embedding
            self.token_embeddings = token_embeddings
            self.watch = watch
            self.error = None
        if old_watch:
            old_watch.close()
        self.start_evaluation(episodes=episodes, seed=0)

    def _embedding(self, model: Any):
        try:
            encoder = resolve_encoder(model)
            base, description = extract_vectors(encoder, "hand", upgrades=0)
            upgraded, _ = extract_vectors(encoder, "hand", upgrades=1)
        except RuntimeError:
            # Older flat policies have no pointer core; retain their state-dict
            # compatibility path (they also predate the card language).
            state = model.policy.state_dict()
            base, description = extract_state_dict_vectors(
                state, "hand", upgrades=0
            )
            upgraded, _ = extract_state_dict_vectors(
                state, "hand", upgrades=1
            )
        indices = np.asarray([CARD_INDEX[name] for name in IRONCLAD_VECTOR_CARDS])
        vectors = np.concatenate([base[indices], upgraded[indices]], axis=0)
        coords, ratio, norms = pca_project(vectors, dim=2, normalize=False)
        points = []
        for variant in (0, 1):
            for i, card_id in enumerate(IRONCLAD_VECTOR_CARDS):
                row = variant * len(IRONCLAD_VECTOR_CARDS) + i
                cost, damage, block = card_numbers(card_id, variant)
                info = IRONCLAD_CARDS[card_id]
                points.append({
                    "card_id": card_id,
                    "name": info.name_zh + ("+" if variant else ""),
                    "kind": info.kind,
                    "upgrade": variant,
                    "x": float(coords[row, 0]),
                    "y": float(coords[row, 1]),
                    "norm": float(norms[row]),
                    "cost": cost,
                    "damage": damage,
                    "block": block,
                })
        return {
            "points": points,
            "explained_ratio": ratio.tolist(),
            "description": description,
            "card_count": len(IRONCLAD_VECTOR_CARDS),
        }

    @staticmethod
    def _token_group(token: str) -> str:
        if token == "PAD" or token.endswith(("_BEGIN", "_END")):
            return "structure"
        if token.startswith(("DIGIT_", "NUM_", "VALUE_")):
            return "number"
        if token.startswith(("TYPE_", "RARITY_")) or token == "COST":
            return "metadata"
        if token in {"DO", "TRIGGER", "IF", "TRAIT"}:
            return "operator"
        return "semantic"

    def _token_embeddings(self, model: Any):
        """Project each learned DSL token table in its own parameter space."""
        try:
            encoder = resolve_encoder(model)
        except RuntimeError:
            return {}

        sources = []
        card = getattr(getattr(encoder, "shared", None), "card", None)
        card_tokens = getattr(card, "token_embedding", None)
        if card_tokens is not None:
            sources.append((
                "card", "卡牌语言", CARD_LANGUAGE_TOKENS,
                CARD_LANGUAGE_DOC, card_tokens.weight,
            ))
        combat_encoder = getattr(getattr(encoder, "combat", None), "encoder", None)
        combat = getattr(combat_encoder, "combat_language", None)
        combat_tokens = getattr(combat, "token_embedding", None)
        if combat_tokens is not None:
            sources.append((
                "combat", "战斗语言", COMBAT_LANGUAGE_TOKENS,
                {}, combat_tokens.weight,
            ))

        result = {}
        for source, label, vocabulary, docs, weight in sources:
            vectors = weight.detach().float().cpu().numpy()
            if vectors.shape[0] != len(vocabulary):
                continue
            coords, ratio, norms = pca_project(vectors, dim=2, normalize=False)
            points = []
            for index, token in enumerate(vocabulary):
                doc = docs.get(token)
                points.append({
                    "index": index,
                    "token": token,
                    "group": self._token_group(token),
                    "x": float(coords[index, 0]),
                    "y": float(coords[index, 1]),
                    "norm": float(norms[index]),
                    "description_en": doc[0] if doc else None,
                    "description_zh": doc[1] if doc else None,
                })
            result[source] = {
                "label": label,
                "points": points,
                "explained_ratio": ratio.tolist(),
            }
        return result

    def start_evaluation(
        self,
        episodes: int,
        seed: int,
        simulations: int | None = None,
        depth: int | None = None,
        leaf_batch_size: int | None = None,
        prior_temperature: float | None = None,
        search_mode: str | None = None,
        gumbel_candidates: int | None = None,
        gumbel_scale: float | None = None,
        gumbel_value_scale: float | None = None,
        gumbel_max_visit_init: float | None = None,
        gumbel_rescale_q: bool | None = None,
        gumbel_use_mixed_value: bool | None = None,
        gumbel_interior_uniformity: float | None = None,
        gumbel_interior_coverage_fraction: float | None = None,
        gumbel_interior_value_mode: int | None = None,
        paper_full_gumbel: bool | None = None,
        end_turn_as_leaf: bool | None = None,
        fixed_worldline: bool | None = None,
        workers: int | None = None,
        upper_strategy: str = _UPPER_STRATEGY_MODEL,
        combat_strategy: str = _COMBAT_STRATEGY_MODEL,
    ):
        episodes = max(1, min(int(episodes), 10_000))
        upper_strategy = str(upper_strategy).strip().lower()
        combat_strategy = str(combat_strategy).strip().lower()
        if upper_strategy not in _UPPER_STRATEGIES:
            raise ValueError(
                "upper strategy must be one of "
                f"{sorted(_UPPER_STRATEGIES)}, got {upper_strategy!r}"
            )
        if combat_strategy not in _COMBAT_STRATEGIES:
            raise ValueError(
                "combat strategy must be one of "
                f"{sorted(_COMBAT_STRATEGIES)}, got {combat_strategy!r}"
            )
        worker_count = _clamp_int(
            workers,
            min(DEFAULT_EVAL_WORKERS, episodes),
            1,
            min(MAX_EVAL_WORKERS, episodes),
        )
        token = uuid.uuid4().hex
        with self.lock:
            if self.model is None:
                raise RuntimeError("no checkpoint loaded")
            model = self.model
            if combat_strategy == _COMBAT_STRATEGY_PLANNER:
                search_config = _evaluation_search_config(
                    model,
                    simulations=0,
                    search_mode="direct",
                    paper_full_gumbel=False,
                    fixed_worldline=False,
                )
            else:
                search_config = _evaluation_search_config(
                    model,
                    simulations=simulations,
                    depth=depth,
                    leaf_batch_size=leaf_batch_size,
                    prior_temperature=prior_temperature,
                    search_mode=search_mode,
                    gumbel_candidates=gumbel_candidates,
                    gumbel_scale=gumbel_scale,
                    gumbel_value_scale=gumbel_value_scale,
                    gumbel_max_visit_init=gumbel_max_visit_init,
                    gumbel_rescale_q=gumbel_rescale_q,
                    gumbel_use_mixed_value=gumbel_use_mixed_value,
                    gumbel_interior_uniformity=gumbel_interior_uniformity,
                    gumbel_interior_coverage_fraction=gumbel_interior_coverage_fraction,
                    gumbel_interior_value_mode=gumbel_interior_value_mode,
                    paper_full_gumbel=paper_full_gumbel,
                    end_turn_as_leaf=end_turn_as_leaf,
                    fixed_worldline=fixed_worldline,
                )
            self.eval_token = token
            self.eval_status = {
                "state": "running", "episodes": episodes, "completed": 0,
                "seed": seed, "started_at": time.time(), "death_floors": {},
                "death_seeds": {}, "win_seeds": [], "current_episode": 1,
                "current_steps": 0, "current_floor": 0,
                "current_phase": None, "current_action": None,
                "upper_strategy": upper_strategy,
                "upper_strategy_version": (
                    OVERWORLD_RULE_VERSION
                    if upper_strategy == _UPPER_STRATEGY_RULES
                    else "checkpoint-model"
                ),
                "upper_fallback": (
                    "masked_argmax"
                    if upper_strategy == _UPPER_STRATEGY_RULES
                    else None
                ),
                "combat_strategy": combat_strategy,
                "combat_strategy_version": (
                    PLANNER_VERSION
                    if combat_strategy == _COMBAT_STRATEGY_PLANNER
                    else "checkpoint-model+configured-search"
                ),
                "combat_strategy_config": (
                    asdict(_TURN_PLANNER_CONFIG)
                    if combat_strategy == _COMBAT_STRATEGY_PLANNER
                    else None
                ),
                "search_method": (
                    "turn_planner"
                    if combat_strategy == _COMBAT_STRATEGY_PLANNER
                    else search_config["method"]
                ),
                "configured_search_method": search_config["method"],
                "search_supported": search_config["supported"],
                "search_mode": search_config["search_mode"],
                "search_mode_label": search_config["search_mode_label"],
                "paper_full_gumbel": search_config["paper_full_gumbel"],
                "root_search_mode": search_config["root_search_mode"],
                "simulations": search_config["simulations"],
                "depth": search_config["depth"],
                "prior_temperature": search_config["prior_temperature"],
                "c_puct": search_config["c_puct"],
                "leaf_batch_size": search_config["leaf_batch_size"],
                "gumbel_candidates": search_config["gumbel_candidates"],
                "gumbel_scale": search_config["gumbel_scale"],
                "gumbel_value_scale": search_config["gumbel_value_scale"],
                "gumbel_max_visit_init": search_config["gumbel_max_visit_init"],
                "gumbel_rescale_q": search_config["gumbel_rescale_q"],
                "gumbel_use_mixed_value": search_config["gumbel_use_mixed_value"],
                "gumbel_interior_uniformity": search_config[
                    "gumbel_interior_uniformity"
                ],
                "gumbel_interior_coverage_fraction": search_config[
                    "gumbel_interior_coverage_fraction"
                ],
                "gumbel_interior_value_mode": search_config[
                    "gumbel_interior_value_mode"
                ],
                "end_turn_as_leaf": search_config["end_turn_as_leaf"],
                "fixed_worldline": search_config["fixed_worldline"],
                "stochastic": False,
                "root_noise": False,
                "search_decisions": 0,
                "search_simulations": 0,
                "search_leaf_evaluations": 0,
                "search_seconds": 0.0,
                "workers": worker_count,
                "active_workers": 0,
            }
        threading.Thread(
            target=self._evaluate,
            args=(
                token,
                model,
                episodes,
                int(seed),
                search_config,
                worker_count,
                upper_strategy,
                combat_strategy,
            ),
            daemon=True,
            name="dashboard-eval",
        ).start()

    def _evaluate(
        self,
        token,
        model,
        episodes,
        seed,
        search_config,
        worker_count,
        upper_strategy,
        combat_strategy,
    ):
        rows = []

        def cancelled():
            return token != self.eval_token

        def progress(event):
            with self.lock:
                if not cancelled():
                    self.eval_status.update(event, last_progress_at=time.time())

        try:
            strategy = _evaluation_strategy(
                model,
                self.model_lock,
                search_config,
                upper_strategy=upper_strategy,
                combat_strategy=combat_strategy,
            )
            progress({"strategy": strategy.name})
            scenario = Scenario("wholerun", lambda: SlayTheSpireEnv(
                combat_action_contract=(
                    "native-v1"
                    if combat_strategy == _COMBAT_STRATEGY_PLANNER
                    else "legacy"
                ),
                global_map_view=_model.uses_global_map_observation(model),
                card_language=_model.uses_card_language(model),
                combat_language=_model.uses_combat_language(model),
                deck_size_observation=_model.uses_deck_size_observation(model),
                neow_observation=_model.uses_neow_observation(model),
                event_option_observation=_model.uses_event_option_observation(model)))
            for row in evaluate(strategy, scenario, EvaluationConfig(
                    episodes=episodes, seed=seed, num_envs=worker_count),
                    cancelled=cancelled, progress=progress):
                rows.append(row)
                with self.lock:
                    if not cancelled():
                        elapsed = time.time() - self.eval_status["started_at"]
                        self.eval_status.update(summarize(rows), elapsed_seconds=elapsed,
                            eta_seconds=elapsed / len(rows) * (episodes - len(rows)))
            progress({"state": "complete", "active_workers": 0})
        except EvaluationCancelled:
            return
        except Exception as exc:
            progress({"state": "error", "error": str(exc), "active_workers": 0})








APP = DashboardState()


class Handler(BaseHTTPRequestHandler):
    server_version = "StSRLWorkbench/1.0"

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} {fmt % args}")

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(APP.public_state())
            return
        if path == "/api/custom-combat/encounters":
            self._send_json({"encounters": sts.encounter_names()})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        # Dashboard assets are edited during local experiments.  Prevent a
        # stale browser-cached app.js/index.html from silently dropping new
        # search controls (for example Interior C) on Apply.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/upload":
                self._upload()
            elif path == "/api/watch/reset":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.reset(
                        int(body.get("seed", 0)),
                        precompute=bool(body.get("precompute", False)),
                        simulations=body.get("simulations"),
                        depth=body.get("depth"),
                        leaf_batch_size=body.get("leaf_batch_size"),
                        prior_temperature=body.get("prior_temperature"),
                        search_mode=body.get("search_mode"),
                        gumbel_candidates=body.get("gumbel_candidates"),
                        gumbel_scale=body.get("gumbel_scale"),
                        gumbel_value_scale=body.get("gumbel_value_scale"),
                        gumbel_max_visit_init=body.get("gumbel_max_visit_init"),
                        gumbel_rescale_q=body.get("gumbel_rescale_q"),
                        gumbel_use_mixed_value=body.get("gumbel_use_mixed_value"),
                        gumbel_interior_uniformity=body.get("gumbel_interior_uniformity"),
                        gumbel_interior_coverage_fraction=body.get("gumbel_interior_coverage_fraction"),
                        gumbel_interior_value_mode=body.get("gumbel_interior_value_mode"),
                        paper_full_gumbel=body.get("paper_full_gumbel"),
                        end_turn_as_leaf=body.get("end_turn_as_leaf"),
                        fixed_worldline=body.get("fixed_worldline"),
                        stochastic=body.get("stochastic"),
                        upper_strategy=body.get("upper_strategy"),
                        combat_strategy=body.get("combat_strategy"),
                        target_boss_act=body.get("target_boss_act"),
                    )
                self._send_json(result)
            elif path == "/api/watch/apply":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.configure_search(
                        simulations=body.get("simulations"),
                        depth=body.get("depth"),
                        leaf_batch_size=body.get("leaf_batch_size"),
                        prior_temperature=body.get("prior_temperature"),
                        search_mode=body.get("search_mode"),
                        gumbel_candidates=body.get("gumbel_candidates"),
                        gumbel_scale=body.get("gumbel_scale"),
                        gumbel_value_scale=body.get("gumbel_value_scale"),
                        gumbel_max_visit_init=body.get("gumbel_max_visit_init"),
                        gumbel_rescale_q=body.get("gumbel_rescale_q"),
                        gumbel_use_mixed_value=body.get("gumbel_use_mixed_value"),
                        gumbel_interior_uniformity=body.get("gumbel_interior_uniformity"),
                        gumbel_interior_coverage_fraction=body.get("gumbel_interior_coverage_fraction"),
                        gumbel_interior_value_mode=body.get("gumbel_interior_value_mode"),
                        paper_full_gumbel=body.get("paper_full_gumbel"),
                        end_turn_as_leaf=body.get("end_turn_as_leaf"),
                        fixed_worldline=body.get("fixed_worldline"),
                        stochastic=body.get("stochastic"),
                        combat_strategy=body.get("combat_strategy"),
                    )
                self._send_json(result)
            elif path == "/api/custom-combat/apply":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.apply_custom_state(body)
                self._send_json(result)
            elif path == "/api/watch/search-tree":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.search_tree(
                        body.get("path", ()),
                        max_children=body.get("max_children", 64),
                    )
                self._send_json(result)
            elif path == "/api/watch/step":
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.step()
                self._send_json(result)
            elif path == "/api/watch/undo":
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.undo()
                self._send_json(result)
            elif path == "/api/watch/seek":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.seek(int(body.get("step", 0)))
                self._send_json(result)
            elif path == "/api/watch/advance":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.advance_to_timeline(int(body.get("index", 0)))
                self._send_json(result)
            elif path == "/api/watch/map-preview":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.preview_map(int(body.get("act", 1)))
                self._send_json(result)
            elif path == "/api/eval/start":
                body = self._read_json()
                APP.start_evaluation(
                    body.get("episodes", 50),
                    body.get("seed", 0),
                    simulations=body.get("simulations"),
                    depth=body.get("depth"),
                    leaf_batch_size=body.get("leaf_batch_size"),
                    prior_temperature=body.get("prior_temperature"),
                    search_mode=body.get("search_mode"),
                    gumbel_candidates=body.get("gumbel_candidates"),
                    gumbel_scale=body.get("gumbel_scale"),
                    gumbel_value_scale=body.get("gumbel_value_scale"),
                    gumbel_max_visit_init=body.get("gumbel_max_visit_init"),
                    gumbel_rescale_q=body.get("gumbel_rescale_q"),
                    gumbel_use_mixed_value=body.get("gumbel_use_mixed_value"),
                    gumbel_interior_uniformity=body.get("gumbel_interior_uniformity"),
                    gumbel_interior_coverage_fraction=body.get("gumbel_interior_coverage_fraction"),
                    gumbel_interior_value_mode=body.get("gumbel_interior_value_mode"),
                    paper_full_gumbel=body.get("paper_full_gumbel"),
                    end_turn_as_leaf=body.get("end_turn_as_leaf"),
                    fixed_worldline=body.get("fixed_worldline"),
                    workers=body.get("workers"),
                    upper_strategy=body.get(
                        "upper_strategy", _UPPER_STRATEGY_MODEL
                    ),
                    combat_strategy=body.get(
                        "combat_strategy", _COMBAT_STRATEGY_MODEL
                    ),
                )
                self._send_json(APP.public_state()["evaluation"])
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _upload(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("upload must be a non-empty file smaller than 512 MB")
        original = self.headers.get("X-Filename", "checkpoint.zip")
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original).name)
        suffix = Path(filename).suffix.lower()
        if suffix != ".zip":
            raise ValueError(
                "only SB3 .zip checkpoints are accepted"
            )
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{filename}"
        partial = target.with_suffix(target.suffix + ".part")
        try:
            remaining = length
            with partial.open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            f"upload ended early: received {length - remaining} "
                            f"of {length} bytes"
                        )
                    stream.write(chunk)
                    remaining -= len(chunk)
            with partial.open("rb") as stream:
                signature = stream.read(4)
            if suffix == ".zip":
                if signature not in (
                    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"
                ):
                    raise ValueError(
                        "uploaded file does not have a ZIP signature; re-download "
                        "the checkpoint and drag the .zip file itself"
                    )
            os.replace(partial, target)
            if suffix == ".zip":
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("checkpoint ZIP failed integrity validation")
            episodes = int(self.headers.get("X-Eval-Episodes", "50"))
            APP.load_checkpoint(target, episodes)
        except Exception:
            partial.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        self._send_json(APP.public_state(), HTTPStatus.CREATED)


def main():
    parser = argparse.ArgumentParser(description="Run the local sts-rl test dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    checkpoint = args.checkpoint or _default_dashboard_checkpoint()
    if checkpoint:
        APP.load_checkpoint(checkpoint, args.eval_episodes)
    else:
        print("[dashboard] no checkpoint found; upload one from the browser", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
