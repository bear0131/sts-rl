"""Automatic CommunicationMod orchestrator backed by a MaskablePPO checkpoint.

CommunicationMod launches this script as a subprocess:

    stdin  <- one JSON game-state frame per line
    stdout -> one command line per frame

Never write diagnostics to stdout.  Everything except commands goes to
``agent_orchestrator.log`` next to this file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from env.core import native_actions as C
import numpy as np

from bridge.real_game_env import RealDecision, RealGameAdapter, validate_transport_frame
from bridge.campaign import SeedCampaign
from bridge.event_contract import EventEffectAck
from bridge.schemas.entity_ids import CHEST_REWARD_SCREEN_CONTRACTS, REWARD_ROOM_EXIT_CONTRACTS
from env.core import action_space as A
from env.core import game_actions as ga

# Register custom policy/extractor classes before SB3 unpickles the checkpoint.
from rl.policies import model as _model  # noqa: F401
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file


try:
    sys.stdin.reconfigure(encoding="gbk", errors="replace")
    sys.stdout.reconfigure(encoding="gbk")
except Exception:
    # reconfigure() is unavailable on a few embedded streams; the commands are
    # ASCII, so the default encoding is still usable.
    pass


LOG_PATH = Path(__file__).with_name("agent_orchestrator.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("agent_orchestrator")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive the real Slay the Spire game through CommunicationMod."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=os.environ.get("STS_RL_CHECKPOINT"),
        help="MaskablePPO .zip checkpoint. May also be set with STS_RL_CHECKPOINT.",
    )
    parser.add_argument("--device", default=os.environ.get("STS_RL_DEVICE", "auto"))
    parser.add_argument(
        "--ascension",
        type=int,
        default=int(os.environ.get("STS_RL_ASCENSION", "0")),
    )
    parser.add_argument(
        "--character",
        default=os.environ.get("STS_RL_CHARACTER", "IRONCLAD"),
        choices=("IRONCLAD", "SILENT", "DEFECT", "WATCHER"),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=os.environ.get("STS_RL_STOCHASTIC", "").lower() in {"1", "true", "yes"},
        help="Sample from the masked policy instead of deterministic argmax.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("STS_RL_SEED", "10000")),
        help="Random seed for stochastic policy sampling.",
    )
    parser.add_argument(
        "--start-seed",
        default=os.environ.get("STS_RL_START_SEED"),
        help="If not currently in a run, send START with this game seed.",
    )
    parser.add_argument(
        "--start-seeds",
        default=os.environ.get("STS_RL_START_SEEDS"),
        help="Comma-separated distinct game seeds for a bounded sequential campaign.",
    )
    parser.add_argument(
        "--campaign-results",
        default=os.environ.get("STS_RL_CAMPAIGN_RESULTS"),
        help="Append one JSON object per naturally completed campaign game.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=os.environ.get("STS_RL_AUTO_START", "").lower() in {"1", "true", "yes"},
        help="Start a new Ironclad run when the frame says in_game=false.",
    )
    parser.add_argument(
        "--no-auto-enter-shop",
        action="store_true",
        help="Do not auto-enter SHOP_SCREEN from SHOP_ROOM.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=int(os.environ.get("STS_RL_LOG_TOPK", "5")),
        help="How many legal policy choices to log per decision.",
    )
    parser.add_argument(
        "--max-inner-decisions",
        type=int,
        default=4,
        help="Max policy calls within one frame for autoregressive target picks.",
    )
    parser.add_argument(
        "--card-vector-checkpoint",
        default=os.environ.get("STS_RL_CARD_VECTOR_CHECKPOINT"),
        help=(
            "Local replacement for a frozen card-vector checkpoint saved with a "
            "machine-specific path."
        ),
    )
    parser.add_argument(
        "--allow-direct-combat-policy",
        action="store_true",
        default=os.environ.get("STS_RL_ALLOW_DIRECT_COMBAT_POLICY", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Allow checkpoints trained with equivalent-play grouping to execute "
            "their direct combat policy. This validates the real-game transport "
            "but does not reproduce simulator-backed Full-Gumbel search."
        ),
    )
    parser.add_argument(
        "--combat-search",
        action="store_true",
        default=os.environ.get("STS_RL_COMBAT_SEARCH", "").lower()
        in {"1", "true", "yes"},
        help="Use the checkpoint's simulator-backed Full-Gumbel combat search.",
    )
    parser.add_argument(
        "--native-search-threads",
        type=int,
        default=int(os.environ.get("STS_RL_NATIVE_SEARCH_THREADS", "4")),
        help="Persistent native search worker threads for the single real-game mirror.",
    )
    parser.add_argument(
        "--gumbel-scale", type=float, default=None,
        help="Override combat-search Gumbel noise; 0 disables it, omitted inherits checkpoint.",
    )
    parser.add_argument('--continuous-test', action='store_true',
                        help='Explicit authorization for automated main-menu starts; also requires --auto-start.')
    parser.add_argument('--reward-policy-contract', choices=_model.REWARD_POLICY_CONTRACTS,
                        default='native',
                        help='Explicit checkpoint-v7-act3 preserves old key-free reward decisions; world state stays complete.')
    parser.add_argument('--attach-existing-seed', type=int, default=None,
                        help='Wait for user/UI save resume, never START; verify this internal seed.')
    parser.add_argument('--attach-existing-floor', type=int, default=None,
                        help='Verify the first loaded floor when attaching to an existing save.')
    parser.add_argument('--attach-search-index', type=int, default=None,
                        help='Restore logical search index on first validated seed/floor attachment only.')
    return parser


def send(command: str) -> None:
    log.info("SEND %s", command)
    sys.stdout.write(command + "\n")
    sys.stdout.flush()


def _checkpoint_model_class(checkpoint: Path):
    """Use the algorithm class that owns every optimizer saved in the archive."""
    try:
        with ZipFile(checkpoint) as archive:
            names = set(archive.namelist())
    except BadZipFile:
        # Unit-test doubles and very old callers may provide a placeholder path.
        names = set()
    return (
        AccumulatingMaskablePPO
        if "policy.upper_optimizer.pth" in names
        else MaskablePPO
    )


def _load_model(
    path: str,
    device: str,
    *,
    card_vector_checkpoint: str | None = None,
    allow_direct_combat_policy: bool = False,
    simulator_combat_search: bool = False,
) -> MaskablePPO:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    custom_objects = None
    if card_vector_checkpoint is not None:
        local_card_vectors = Path(card_vector_checkpoint)
        if not local_card_vectors.is_file():
            raise FileNotFoundError(
                f"card-vector checkpoint not found: {local_card_vectors}"
            )
        data, _, _ = load_from_zip_file(checkpoint, device="cpu")
        policy_kwargs = deepcopy(data.get("policy_kwargs") or {})
        entity_config = dict(policy_kwargs.get("entity_config") or {})
        saved_path = entity_config.get("card_vector_checkpoint")
        entity_config["card_vector_checkpoint"] = str(local_card_vectors.resolve())
        policy_kwargs["entity_config"] = entity_config
        custom_objects = {"policy_kwargs": policy_kwargs}
        log.info(
            "relocating frozen card vectors saved=%s local=%s",
            saved_path,
            local_card_vectors.resolve(),
        )
    log.info("loading checkpoint=%s device=%s", checkpoint, device)
    model_class = _checkpoint_model_class(checkpoint)
    log.info("checkpoint algorithm class=%s", model_class.__name__)
    model = model_class.load(
        checkpoint,
        device=device,
        custom_objects=custom_objects,
    )
    grouped_combat = (
        getattr(model, "nosl_action_group_contract", None) == "equivalent-plays-v1"
    )
    if grouped_combat and not (allow_direct_combat_policy or simulator_combat_search):
        raise ValueError(
            "equivalent-plays-v1 currently requires simulator-backed PolicyStrategy; "
            "CommunicationMod lacks the per-instance fields needed for identical grouping. "
            "Use --allow-direct-combat-policy only for an explicitly ungrouped "
            "real-game integration diagnostic."
        )
    if grouped_combat and allow_direct_combat_policy and not simulator_combat_search:
        log.warning(
            "direct combat policy explicitly enabled: equivalent-play probability "
            "mass and simulator-backed search are unavailable in CommunicationMod"
        )
    log.info(
        "loaded checkpoint; obs flags global_map=%s card_language=%s combat_language=%s "
        "deck_size=%s neow=%s event_option=%s",
        _model.uses_global_map_observation(model),
        _model.uses_card_language(model),
        _model.uses_combat_language(model),
        _model.uses_deck_size_observation(model),
        _model.uses_neow_observation(model),
        _model.uses_event_option_observation(model),
    )
    return model


def _adapter_for_model(model: MaskablePPO, args: argparse.Namespace) -> RealGameAdapter:
    return RealGameAdapter(
        character=args.character,
        ascension=args.ascension,
        global_map_view=_model.uses_global_map_observation(model),
        card_language=_model.uses_card_language(model),
        combat_language=_model.uses_combat_language(model),
        deck_size_observation=_model.uses_deck_size_observation(model),
        neow_observation=_model.uses_neow_observation(model),
        event_option_observation=_model.uses_event_option_observation(model),
        combat_action_contract=(C.policy_contract(model.observation_space)),
    )


class RealPolicyAgent:
    def __init__(
        self,
        model: MaskablePPO,
        adapter: RealGameAdapter,
        *,
        stochastic: bool,
        topk: int,
        max_inner_decisions: int,
        auto_enter_shop: bool,
        auto_start: bool,
        start_seed: str | None,
        ascension: int,
        character: str,
        campaign: SeedCampaign | None = None,
        combat_search=None,
        reward_policy_contract: str = "native",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.stochastic = bool(stochastic)
        self.topk = max(0, int(topk))
        self.max_inner_decisions = max(1, int(max_inner_decisions))
        self.auto_enter_shop = bool(auto_enter_shop)
        self.auto_start = bool(auto_start)
        self.start_seed = start_seed
        self.ascension = int(ascension)
        self.character = character
        self.campaign = campaign
        self.combat_search = combat_search
        _model.validate_reward_policy_contract(model, reward_policy_contract)
        self.reward_policy_contract = reward_policy_contract
        log.info("reward_policy_contract=%s; no automatic key claiming", reward_policy_contract)

    def decide(self, frame: dict[str, Any]) -> RealDecision:
        self.adapter.update(frame)
        if self.adapter.in_game():
            _model.validate_reward_policy_scope(
                self.reward_policy_contract, self.adapter.gs["act"])
        if frame.get("in_game") is False and not self.auto_start:
            return RealDecision("STATE", "await_manual_continue")
        if self.combat_search is not None:
            self.combat_search.observe_frame(frame)
        if (self.campaign is not None and self.combat_search is not None
                and self.adapter.in_game() and self.adapter.screen_type() == "GAME_OVER"
                and self.combat_search.run_errors):
            raise RuntimeError(
                "campaign game is invalid because bridge/search errors occurred: "
                f"{self.combat_search.run_errors}")
        campaign_result = (
            self.campaign.observe(
                frame,
                metadata={
                    **(self.combat_search.run_summary() if self.combat_search is not None else {}),
                    "reward_policy_contract": self.reward_policy_contract,
                },
            )
            if self.campaign is not None else None
        )
        if campaign_result is not None:
            log.info("campaign game complete %s", campaign_result)
        if (self.campaign is not None and self.adapter.in_game()
                and self.adapter.screen_type() == "GAME_OVER"):
            return RealDecision("PROCEED", "campaign_game_over")

        # CommunicationMod error frames omit ``in_game`` and ``game_state``.
        # Refresh the state instead of mistaking a rejected command for a run
        # ending and repeatedly issuing START while the combat is still live.
        deferred = self.adapter._deferred_combat_reward_action
        auto = self.adapter.auto_command(auto_enter_shop=self.auto_enter_shop)
        if auto is not None:
            command, reason = auto
            return RealDecision(command, reason, semantic=(deferred if reason == "complete_deferred_reward_action" else None))

        if not self.adapter.in_game():
            if not self.auto_start:
                return RealDecision("STATE", "await_manual_continue")
            if self.combat_search is not None:
                self.combat_search.reset_run()
            if self.campaign is not None:
                command = self.campaign.start_command(self.character, self.ascension)
                return RealDecision(command or "STATE", "campaign_start" if command else "campaign_complete")
            if self.auto_start or self.start_seed is not None:
                seed_suffix = "" if self.start_seed is None else f" {self.start_seed}"
                return RealDecision(
                    f"START {self.character} {self.ascension}{seed_suffix}",
                    "auto_start",
                )
            return RealDecision("STATE", "not_in_game")

        pending: tuple | None = None
        if (getattr(getattr(getattr(self.model,'policy',None),'core',None),'combat_only',False)
                and self.adapter.phase() != "combat"):
            return RealDecision("STATE", "combat_only_requires_manual_overworld")
        if self.adapter.phase() == "combat" and self.combat_search is not None:
            return self.combat_search.decide()
        trace: list[tuple[int, tuple | None, str]] = []
        for inner in range(max(self.max_inner_decisions, 11 if self.adapter.combat_action_contract == C.AUTOREGRESSIVE_CONTRACT else 1)):
            obs = self.adapter.observation(pending=pending)
            world_mask = self.adapter.action_mask(pending=pending)
            mask = _model.action_mask_for_policy(
                self.model, world_mask, phase=self.adapter.phase(), act=self.adapter.gs["act"],
                reward_policy_contract=self.reward_policy_contract)
            excluded = np.flatnonzero(world_mask & ~mask)
            if len(excluded):
                log.info("reward_policy_contract=%s policy_excluded_ids=%s actual_reward_keys=%s run_keys=%s",
                         self.reward_policy_contract, excluded.tolist(),
                         obs["reward_keys"].tolist(), obs["run_keys"].tolist())
            legal_ids = tuple(int(i) for i in np.flatnonzero(mask))
            if not legal_ids:
                raise RuntimeError(
                    "no legal policy actions; refusing fallback: "
                    f"screen={self.adapter.screen_type()} phase={self.adapter.phase()} "
                    f"choices={self.adapter.choices()!r} pending={pending!r}"
                )

            if (self.adapter.combat_action_contract in C.NEURAL_CONTRACTS
                    and legal_ids == (C.policy_offset(self.adapter.combat_action_contract),)):
                return RealDecision("END", "forced_end_turn", action_id=C.policy_offset(self.adapter.combat_action_contract), legal_ids=legal_ids)
            action_id, top = self._predict(obs, mask, pending=pending)
            if self.adapter.combat_action_contract in C.NEURAL_CONTRACTS and self.adapter.phase() == "combat":
                command = self.adapter.command_for_joint_action(action_id)
                if command is None:
                    continue
                return RealDecision(command, "policy_joint",
                                    action_id=action_id, legal_ids=legal_ids, top=top)
            desc = self.adapter.select_descriptor()
            if A.is_multi_select(desc):
                command = self.adapter.command_for_multi_action(action_id)
                return RealDecision(
                    command,
                    "policy_multi",
                    action_id=action_id,
                    semantic=None,
                    legal_ids=legal_ids,
                    top=top,
                )

            legal_semantic = self.adapter.legal_semantic_actions()
            semantic, next_pending = A.resolve(
                action_id, legal_semantic, pending, self.adapter.phase()
            )
            trace.append((action_id, pending, self.adapter.describe_action_id(action_id, pending)))
            if semantic is None:
                pending = next_pending
                continue

            command = self.adapter.command_for_semantic(semantic)
            return RealDecision(
                command,
                "policy",
                action_id=action_id,
                semantic=semantic,
                legal_ids=legal_ids,
                top=top,
            )

        raise RuntimeError(f"inner decision limit reached; refusing fallback: trace={trace!r}")

    def _predict(
        self, obs: dict[str, np.ndarray], mask: np.ndarray, *, pending: tuple | None
    ) -> tuple[int, tuple[tuple[int, float, str], ...]]:
        action, _ = self.model.predict(
            _model.observation_for_policy(self.model, obs),
            action_masks=mask,
            deterministic=not self.stochastic,
        )
        action_id = int(np.asarray(action).item())
        top = self._top_actions(obs, mask, action_id, pending=pending)
        log.debug(
            "policy action=%s label=%s top=%s",
            action_id,
            self.adapter.describe_action_id(action_id, pending),
            top,
        )
        return action_id, top

    def _top_actions(
        self,
        obs: dict[str, np.ndarray],
        mask: np.ndarray,
        chosen: int,
        *,
        pending: tuple | None,
    ) -> tuple[tuple[int, float, str], ...]:
        if self.topk <= 0:
            return ()
        try:
            import torch

            obs_tensor, _ = self.model.policy.obs_to_tensor(
                _model.observation_for_policy(self.model, obs)
            )
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor, action_masks=mask)
                probs = dist.distribution.probs[0].detach().cpu().numpy()
            legal = np.flatnonzero(mask)
            ranked = sorted(legal, key=lambda i: probs[i], reverse=True)[: self.topk]
            out = []
            for i in ranked:
                label = self.adapter.describe_action_id(int(i), pending)
                if int(i) == chosen:
                    label = "> " + label
                out.append((int(i), float(probs[i]), label))
            return tuple(out)
        except Exception:
            log.debug("could not compute action distribution", exc_info=True)
            return ()


def _log_decision(frame: dict[str, Any], decision: RealDecision, adapter: RealGameAdapter) -> None:
    gs = frame.get("game_state") or {}
    log.info(
        "decision reason=%s command=%s action_id=%s semantic=%s screen=%s phase=%s "
        "floor=%s act=%s hp=%s/%s choices=%s legal_ids=%s top=%s",
        decision.reason,
        decision.command,
        decision.action_id,
        decision.semantic,
        gs.get("screen_type"),
        adapter.phase(),
        gs.get("floor"),
        gs.get("act"),
        gs.get("current_hp"),
        gs.get("max_hp"),
        gs.get("choice_list"),
        list(decision.legal_ids),
        decision.top,
    )


def send_decision(frame: dict[str, Any], decision: RealDecision) -> None:
    gs = frame.get("game_state") or {}
    screen = str(gs.get("screen_type") or "").upper()
    verb = decision.command.split()[0].upper()
    outside_combat = str(gs.get("room_phase") or "").upper() != "COMBAT"
    if verb not in {"STATE", "WAIT"} and (
        screen in {"EVENT", "SHOP_SCREEN", "SHOP_ROOM"}
        or (outside_combat and screen in {"GRID", "CARD_REWARD", "COMBAT_REWARD"})
    ):
        time.sleep(0.5)
    send(decision.command)


def searched_command_still_current(frame, original, command):
    return ("error" not in frame and frame.get("ready_for_command", False)
            and frame.get("game_state") == original.get("game_state")
            and command.split()[0].lower() in frame.get("available_commands", []))


def stale_combat_play_frame(frame: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    """A mutating command must change its real frame before another decision."""
    return pending is not None and frame.get("game_state") == pending


def combat_command_needs_ack(frame: dict[str, Any], command: str) -> bool:
    gs = frame.get("game_state") or {}
    # Keep the existing helper name, but protect mutations outside combat too:
    # shop purchases, dialogue choices, and overworld potions can also receive
    # an identical old frame. Dedicated effect receipts add stronger checks;
    # they do not replace this common prohibition on acting on the old frame.
    return bool(gs) and command.split()[0].upper() in {
        "PLAY", "POTION", "CHOOSE", "CONFIRM", "RETURN", "LEAVE", "SKIP", "PROCEED",
    }


def stale_combat_end_frame(frame: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    """END advances to another turn/room or a required in-combat selection."""
    if pending is None or "error" in frame:
        return False
    gs = frame.get("game_state") or {}
    combat = gs.get("combat_state")
    if gs.get("screen_type") == "GAME_OVER" or not combat or gs.get("room_phase") != "COMBAT":
        return False
    if any(gs.get(k) != pending.get(k) for k in ("seed", "floor")):
        return False
    if "turn" not in combat or type(combat["turn"]) is not int:
        raise ValueError("combat END acknowledgement is missing an integer turn")
    if (frame.get("ready_for_command") is True
            and gs.get("screen_type") in {"CARD_REWARD", "HAND_SELECT", "GRID"}):
        # Codex/retain and other queued selections must resolve before the
        # turn can advance. Release only this wait barrier; the normal adapter
        # and search still validate source, context, candidates and legal picks.
        # No command is chosen here and ordinary changed flags remain stale.
        return False
    return combat["turn"] <= pending["combat_state"]["turn"]


class MapCommandAck:
    """Do not reissue a map click or ignore its Wing Boots charge effect."""

    def __init__(self):
        self.pending = None

    def arm(self, frame, decision, now, adapter):
        if not decision.semantic or decision.semantic[0] != ga.MAP:
            return
        if self.pending is not None:
            raise RuntimeError('map move already awaiting acknowledgement')
        if adapter.screen_type() != 'MAP':
            raise ValueError('map move receipt armed outside MAP')
        expected_command = adapter.command_for_semantic(decision.semantic)
        if decision.command != expected_command:
            raise ValueError('map move command disagrees with its semantic destination')
        self.pending = (adapter.map_move_expectation(decision.semantic[1]), now)

    def waiting(self, frame, now):
        if self.pending is None:
            return False
        expected, since = self.pending
        gs = frame.get('game_state') or {}
        if (gs.get('seed') != expected['seed'] or gs.get('act') != expected['act']
                or type(gs.get('floor')) is not int):
            raise ValueError('map move receipt changed run or lacks floor')
        if frame.get('ready_for_command') is not True:
            if now - since > 10.0:
                raise RuntimeError('map move did not settle for 10 seconds')
            return True
        counter = RealGameAdapter.map_flight_counter(gs)
        if gs['floor'] == expected['floor']:
            position = gs.get('simulator_current_node')
            if (not isinstance(position, dict)
                    or any(type(position.get(k)) is not int for k in ('x', 'y'))
                    or (position['x'], position['y']) != expected['position']):
                raise ValueError('map move receipt lacks or differs from the actual destination')
            if counter != expected['flight_counter']:
                raise ValueError('map move Wing Boots counter differs from its connection cost')
            self.pending = None
            return False
        if gs['floor'] != expected['floor'] - 1:
            raise ValueError('map move advanced to an unexpected floor')
        if now - since > 10.0:
            raise RuntimeError('map move did not advance for 10 seconds')
        return True


class RewardCommandAck:
    """One in-flight reward mutation; UI retries must never become card picks."""

    def __init__(self):
        self.pending = None

    @staticmethod
    def card_identity(card):
        if (not isinstance(card, dict) or not isinstance(card.get("id"), str)
                or type(card.get("upgrades")) is not int or card["upgrades"] < 0):
            raise ValueError("reward card receipt requires explicit id and upgrades")
        return card["id"], card["upgrades"]

    @staticmethod
    def snapshot(gs):
        return {k: deepcopy(gs.get(k)) for k in (
            "seed", "act", "floor", "room_type", "room_phase", "simulator_current_node",
            "screen_type", "screen_state", "choice_list",
            "deck", "gold", "relics", "potions", "max_hp",
            "has_ruby_key", "has_emerald_key", "has_sapphire_key",
        )}

    @staticmethod
    def room_exit_expectation(gs, command):
        if command.upper() != "PROCEED":
            return None
        contract = REWARD_ROOM_EXIT_CONTRACTS.get(gs.get("room_type"))
        if (contract is None or gs.get("act") not in contract["source_acts"]
                or gs.get("screen_type") not in contract["source_screens"]):
            return None
        node = gs.get("simulator_current_node")
        if (any(type(gs.get(k)) is not int for k in ("seed", "act", "floor"))
                or gs.get("room_phase") != "COMPLETE"
                or not isinstance(node, dict)
                or any(type(node.get(k)) is not int for k in ("x", "y"))
                or (node["x"], node["y"]) != contract["source_node"]
                or node.get("room_class") != gs["room_type"]):
            raise ValueError("reward room exit lacks its explicit completed source identity")
        return {
            "act": gs["act"] + contract["act_delta"],
            "floor": gs["floor"] + contract["floor_delta"],
            "room_type": contract["target_room"], "room_phase": "COMPLETE",
            "screen_type": contract["target_screen"], "node": contract["target_node"],
        }

    def arm(self, frame, decision, now):
        gs = frame.get("game_state") or {}
        if gs.get("room_phase") == "COMBAT":
            # Discovery/Codex reuse CARD_REWARD, but choose a temporary combat
            # card, not a permanent deck reward. Their source/candidates and
            # result are owned by the combat adapter/search transaction.
            return
        room_exit = self.room_exit_expectation(gs, decision.command)
        if (gs.get("screen_type") not in {"COMBAT_REWARD", "CARD_REWARD", "BOSS_REWARD", "CHEST"}
                and room_exit is None):
            return
        if decision.command.split()[0].upper() not in {"CHOOSE", "SKIP", "PROCEED", "RETURN"}:
            return
        expected = None
        if decision.semantic and decision.semantic[0] == "reward_card":
            index = int(decision.semantic[1])
            cards = (gs.get("screen_state") or {}).get("cards", [])
            if gs.get("screen_type") == "CARD_REWARD":
                if not 0 <= index < len(cards):
                    raise ValueError("reward card index is outside the offered cards")
                expected = self.card_identity(cards[index])
        claim = None
        if gs.get("screen_type") == "CHEST" and decision.command.upper().startswith("CHOOSE "):
            ss = gs.get("screen_state") or {}
            chest_type = ss.get("chest_type")
            if (decision.semantic != (ga.CHEST, True) or decision.command.upper() != "CHOOSE 0"
                    or ss.get("chest_open") is not False
                    or chest_type not in CHEST_REWARD_SCREEN_CONTRACTS):
                raise ValueError("chest opening receipt requires the explicit unopened chest and OPEN action")
            claim = ("chest_open", chest_type, None)
        if (gs.get("screen_type") == "CARD_REWARD" and decision.semantic
                and decision.semantic[0] == "skip_cards"
                and decision.command.upper().startswith("CHOOSE ")):
            index = int(decision.command.split()[1])
            if gs.get("choice_list", [])[index] != "bowl":
                raise ValueError("skip_cards CHOOSE must target Singing Bowl")
            claim = ("bowl", None, None)
        if (gs.get("screen_type") == "COMBAT_REWARD" and decision.command.upper().startswith("CHOOSE ")
                and decision.semantic and decision.semantic[0] == "claim"):
            kind, index = decision.semantic[1:3]
            reward_types = {
                "gold": {"gold", "stolen_gold"},
                "relic": {"relic"},
                "potion": {"potion"},
                "key": {"emerald_key", "sapphire_key"},
            }.get(kind)
            if reward_types is None:
                raise ValueError(f"unsupported reward claim kind: {kind!r}")
            rewards = [r for r in (gs.get("screen_state") or {}).get("rewards", [])
                       if r.get("reward_type", "").lower() in reward_types]
            if type(index) is not int or not 0 <= index < len(rewards):
                raise ValueError(f"reward claim identity missing: {kind}/{index}")
            reward = rewards[index]
            claim = (
                kind,
                reward.get(kind, reward.get("reward_type")),
                reward.get("gold"),
            )
            if kind == "gold" and (type(claim[2]) is not int or claim[2] < 0):
                raise ValueError("gold reward receipt requires the explicit offered amount")
        verb = decision.command.split()[0].upper()
        if (verb == "CHOOSE" and decision.reason != "open_card_reward"
                and expected is None and claim is None):
            if decision.semantic and decision.semantic[0] == ga.BOSS_RELIC:
                index = decision.semantic[1]
                relics = gs.get("screen_state", {}).get("relics", [])
                if type(index) is not int or not 0 <= index < len(relics):
                    raise ValueError("boss relic receipt is missing the chosen identity")
                claim = ("relic", relics[index], None)
            else:
                raise ValueError("reward CHOOSE has no explicit effect receipt")
        if self.pending is not None:
            raise RuntimeError("reward command already awaiting acknowledgement")
        self.pending = (self.snapshot(gs), decision.reason, expected, now, claim, room_exit)

    def waiting(self, frame, now):
        if self.pending is None:
            return False
        if "error" in frame:
            raise RuntimeError(f"reward command rejected before acknowledgement: {frame['error']}")
        before, reason, expected, since, claim, room_exit = self.pending
        gs = frame.get("game_state") or {}
        after = self.snapshot(gs)
        if gs and gs.get("seed") != before["seed"]:
            raise RuntimeError("reward receipt crossed a different run")
        # A fade can expose a partially switched room. Do not release the
        # transaction or send another action until its destination is ready.
        if not gs or frame.get("ready_for_command") is not True:
            if now - since > 10.0:
                raise RuntimeError("reward command did not settle for 10 seconds; not resending")
            return True
        # A gold animation or changing choice label does not acknowledge a
        # SKIP/RETURN/PROCEED transaction.  Its addressed screen must close.
        acknowledged = bool(gs) and gs.get("screen_type") != before["screen_type"]
        if room_exit is not None:
            node = gs.get("simulator_current_node")
            acknowledged = (
                all(gs.get(k) == room_exit[k] for k in
                    ("act", "floor", "room_type", "room_phase", "screen_type"))
                and isinstance(node, dict)
                and all(type(node.get(k)) is int for k in ("x", "y"))
                and (node["x"], node["y"]) == room_exit["node"]
                and node.get("room_class") == room_exit["room_type"]
            )
            if not acknowledged and any(gs.get(k) != before[k] for k in ("act", "floor", "room_type")):
                raise RuntimeError(f"reward room exit reached an unexpected destination; expected {room_exit!r}")
        elif any(gs.get(key) != before[key] for key in ("act", "floor", "room_type")
                 if before[key] is not None):
            raise RuntimeError("reward receipt crossed a different act/floor/room")
        if reason == "open_card_reward":
            acknowledged = gs.get("screen_type") == "CARD_REWARD"
        elif expected is not None:
            def card_count(deck):
                if not isinstance(deck, list):
                    raise ValueError("reward card receipt is missing the deck")
                return Counter(self.card_identity(card) for card in deck)
            acknowledged = (card_count(gs["deck"])[expected]
                            == card_count(before["deck"])[expected] + 1)
        elif claim is not None:
            kind, entity, amount = claim
            if kind == "chest_open":
                # An old CHEST response can arrive after CHOOSE was sent. Never
                # resend it: a second CHOOSE 0 would select the first boss relic.
                acknowledged = gs.get("screen_type") == CHEST_REWARD_SCREEN_CONTRACTS[entity]
            elif kind == "bowl":
                acknowledged = (gs.get("max_hp", 0) == (before.get("max_hp") or 0) + 2
                                and gs.get("screen_type") != "CARD_REWARD")
            elif kind == "gold":
                if type(gs.get("gold")) is not int or type(before["gold"]) is not int:
                    raise ValueError("gold reward receipt is missing the gold balance")
                blocked = any(r.get("id") == 'Ectoplasm' for r in before["relics"] or [])
                expected_gold = before["gold"] + (0 if blocked else amount)
                acknowledged = gs["gold"] == expected_gold
                if blocked:
                    acknowledged = acknowledged and after["screen_state"] != before["screen_state"]
            elif kind in {"potion", "relic"}:
                entity_id = entity.get("id") if isinstance(entity, dict) else entity
                if not entity_id:
                    raise ValueError(f"reward {kind} missing stable identity")
                key = "potions" if kind == "potion" else "relics"
                count = lambda items: sum(x.get("id") == entity_id for x in (items or []))
                acknowledged = count(gs.get(key)) == count(before.get(key)) + 1
                if kind == "relic" and entity_id == "Circlet":
                    def circlet_counter(items):
                        records = [item for item in (items or []) if item.get("id") == "Circlet"]
                        if len(records) != 1 or type(records[0].get("counter")) is not int:
                            raise ValueError("Circlet receipt requires exactly one relic and its explicit counter")
                        return records[0]["counter"]
                    if count(before.get(key)) == 1:
                        acknowledged = (count(gs.get(key)) == 1 and
                            circlet_counter(gs[key]) == circlet_counter(before[key]) + 1)
                    elif acknowledged:
                        acknowledged = circlet_counter(gs[key]) == 1
            elif kind == "key":
                key_field = {
                    "EMERALD_KEY": "has_emerald_key",
                    "SAPPHIRE_KEY": "has_sapphire_key",
                }.get(str(entity).upper())
                if key_field is None:
                    raise ValueError(f"unknown key reward acknowledgement: {entity!r}")
                acknowledged = (
                    before.get(key_field) is False and gs.get(key_field) is True
                )
            else:
                raise ValueError(f"unsupported reward claim acknowledgement: {kind}")
        if acknowledged:
            if room_exit is not None:
                log.info("reward room exit acknowledged source=%s/%s/%s target=%s/%s/%s",
                         before["act"], before["floor"], before["room_type"],
                         gs["act"], gs["floor"], gs["room_type"])
            self.pending = None
            return False
        if now - since > 10.0:
            raise RuntimeError("reward command not acknowledged for 10 seconds; not resending")
        return True


class UiPauseClock:
    """Suspend acknowledgement time only for an explicitly exported settings UI."""
    def __init__(self):
        self.last = None
        self.paused = False
        self.excluded = 0.0

    def update(self, frame, now):
        gs = frame.get('game_state') or {}
        paused = bool(gs.get('is_screen_up')) and gs.get('screen_name') == 'SETTINGS'
        if self.last is not None and self.paused:
            self.excluded += now - self.last
        self.last, self.paused = now, paused
        return now - self.excluded, paused


class CombatFrameStability:
    """Require a fresh, unchanged settled frame before ordinary combat actions."""

    def __init__(self, seconds: float = 0.5):
        self.seconds = seconds
        self.state = None
        self.since = 0.0

    def waiting(self, frame: dict[str, Any], now: float) -> bool:
        gs = frame.get("game_state") or {}
        if (not gs.get("combat_state") or gs.get("room_phase") != "COMBAT"
                or gs.get("screen_type") != "NONE" or "error" in frame):
            self.state = None
            return False  # Selection screens have their own readiness contract.
        if (not frame.get("ready_for_command", False) or gs.get("current_action")
                or gs.get("action_phase") not in (None, "WAITING_ON_USER")
                or gs.get("is_screen_up")):
            self.state = None
            return True
        if self.state != gs:
            self.state = deepcopy(gs)
            self.since = now
            return True
        return now - self.since < self.seconds


def restore_attached_search_index(args, search, gs) -> None:
    index = args.attach_search_index
    if index is None:
        return
    if (index < 0 or args.attach_existing_seed is None or args.attach_existing_floor is None
            or search is None or gs.get('seed') != args.attach_existing_seed
            or gs.get('floor') != args.attach_existing_floor):
        raise RuntimeError('search index restore requires matching save seed/floor and combat search')
    search.decision_index = index
    log.info('restored save search index=%s seed=%s floor=%s', index, gs['seed'], gs['floor'])


def rebuild_save_attachment(agent, model, args):
    """Discard all prior battle/multi-selection state when reloading a fixed save."""
    from bridge.search_bridge import RealCombatSearch
    old_search = agent.combat_search
    if old_search is not None:
        old_search.close()
    model.set_random_seed(args.seed)
    adapter = _adapter_for_model(model, args)
    search = (RealCombatSearch(model, adapter, native_search_threads=args.native_search_threads,
                              master_nonce=args.seed, gumbel_scale=args.gumbel_scale)
              if args.combat_search else None)
    agent.adapter = adapter
    agent.combat_search = search
    return adapter, search


def main() -> None:
    args = build_parser().parse_args()
    if args.attach_search_index is not None and (
            args.attach_search_index < 0 or args.attach_existing_seed is None
            or args.attach_existing_floor is None or not args.combat_search):
        raise SystemExit('--attach-search-index requires nonnegative index, seed, floor and combat search')
    log.info("=== agent session start argv=%s cwd=%s ===", sys.argv, Path.cwd())
    if not args.checkpoint:
        raise SystemExit("checkpoint path required or STS_RL_CHECKPOINT must be set")

    model = _load_model(
        args.checkpoint,
        args.device,
        card_vector_checkpoint=args.card_vector_checkpoint,
        allow_direct_combat_policy=args.allow_direct_combat_policy,
        simulator_combat_search=args.combat_search,
    )
    model.set_random_seed(args.seed)
    adapter = _adapter_for_model(model, args)
    campaign = None
    if args.start_seeds:
        seeds = [part.strip() for part in args.start_seeds.split(",") if part.strip()]
        campaign = SeedCampaign(seeds, args.campaign_results)
    combat_search = None
    if args.combat_search:
        from bridge.search_bridge import RealCombatSearch

        combat_search = RealCombatSearch(
            model,
            adapter,
            native_search_threads=args.native_search_threads,
            master_nonce=args.seed,
            gumbel_scale=args.gumbel_scale,
        )
    agent = RealPolicyAgent(
        model,
        adapter,
        stochastic=args.stochastic,
        topk=args.topk,
        max_inner_decisions=args.max_inner_decisions,
        auto_enter_shop=not args.no_auto_enter_shop,
        auto_start=args.auto_start and args.continuous_test,
        start_seed=args.start_seed,
        ascension=args.ascension,
        character=args.character,
        campaign=campaign,
        combat_search=combat_search,
        reward_policy_contract=args.reward_policy_contract,
    )

    try:
        send("ready")
        pending_play_frame = None
        pending_play_since = 0.0
        pending_end_frame = None
        pending_end_since = 0.0
        combat_stability = CombatFrameStability()
        reward_ack = RewardCommandAck()
        event_ack = EventEffectAck()
        map_ack = MapCommandAck()
        unsent_search = None
        pause_clock = UiPauseClock()
        attach_pending = args.attach_existing_seed is not None
        attached_live = False
        rebuild_pending = False
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            log.debug("RECV %s", line)
            try:
                frame = json.loads(line)
                validate_transport_frame(frame)
                if "error" in frame:
                    raise RuntimeError(f"CommunicationMod rejected command: {frame['error']}")
                if frame.get('in_game') is False and args.attach_existing_seed is not None:
                    if attached_live:
                        # Menu round-trip is a save-load boundary, not an overlay.
                        # Never carry a prepared command or ACK across that boundary.
                        rebuild_pending = True
                        attach_pending = True
                        attached_live = False
                        pending_play_frame = pending_end_frame = unsent_search = None
                        pending_play_since = pending_end_since = 0.0
                        combat_stability = CombatFrameStability()
                        reward_ack = RewardCommandAck()
                        event_ack = EventEffectAck()
                        map_ack = MapCommandAck()
                        pause_clock = UiPauseClock()
                        log.info('save reload boundary: discarded pending commands and acknowledgements')
                    time.sleep(0.1)
                    send('STATE')
                    continue
                if frame.get('in_game') is False and not (args.auto_start and args.continuous_test):
                    time.sleep(0.1)
                    send('STATE')
                    continue
                if args.attach_existing_seed is not None:
                    gs = frame.get('game_state') or {}
                    if not frame.get('in_game'):
                        time.sleep(0.1)
                        send('STATE')
                        continue
                    if (int(gs.get('seed', -1)) != args.attach_existing_seed
                            or (attach_pending and args.attach_existing_floor is not None
                                and gs.get('floor') != args.attach_existing_floor)):
                        raise RuntimeError('loaded save seed/floor mismatch')
                    if attach_pending:
                        if rebuild_pending:
                            adapter, combat_search = rebuild_save_attachment(agent, model, args)
                            rebuild_pending = False
                        restore_attached_search_index(args, combat_search, gs)
                        attach_pending = False
                        attached_live = True
                        log.info('attached existing save seed=%s floor=%s; diagnostic SL resume', gs.get('seed'), gs.get('floor'))
                now, paused = pause_clock.update(frame, time.monotonic())
                if paused:
                    time.sleep(0.1)
                    send('STATE')
                    continue
                if event_ack.waiting(frame, now):
                    time.sleep(0.1)
                    send("STATE")
                    continue
                if map_ack.waiting(frame, now):
                    time.sleep(0.1)
                    send("STATE")
                    continue
                if reward_ack.waiting(frame, now):
                    time.sleep(0.1)
                    send("STATE")
                    continue
                if unsent_search is not None:
                    original, prepared = unsent_search
                    unsent_search = None
                    if searched_command_still_current(frame, original, prepared.command):
                        reward_ack.arm(frame, prepared, now)
                        send_decision(frame, prepared)
                        if combat_command_needs_ack(frame, prepared.command):
                            pending_play_frame = deepcopy(frame.get("game_state"))
                            pending_play_since = now
                        if prepared.command.upper() == "END":
                            pending_end_frame = deepcopy(frame.get("game_state"))
                            pending_end_since = now
                        continue
                    combat_search.cancel_unsent_command()
                    log.info("discarded unsent search command after fresh state changed")
                if stale_combat_end_frame(frame, pending_end_frame):
                    if now - pending_end_since > 10.0:
                        raise RuntimeError("game turn did not advance for 10 seconds after END")
                    time.sleep(0.05)
                    send("STATE")
                    continue
                pending_end_frame = None
                if stale_combat_play_frame(frame, pending_play_frame):
                    if now - pending_play_since > 10.0:
                        raise RuntimeError("game state unchanged for 10 seconds after mutating command")
                    time.sleep(0.05)
                    send("STATE")
                    continue
                pending_play_frame = None
                if combat_stability.waiting(frame, now):
                    time.sleep(0.1)
                    send("STATE")
                    continue
                decision = agent.decide(frame)
                _log_decision(frame, decision, adapter)
                if decision.reason == "native_full_gumbel":
                    unsent_search = (deepcopy(frame), decision)
                    send("STATE")
                    continue
                event_ack.arm(frame, decision, now)
                map_ack.arm(frame, decision, now, adapter)
                reward_ack.arm(frame, decision, now)
                send_decision(frame, decision)
                if combat_command_needs_ack(frame, decision.command):
                    pending_play_frame = deepcopy(frame.get("game_state"))
                    pending_play_since = now
                if decision.command.upper() == "END":
                    pending_end_frame = deepcopy(frame.get("game_state"))
                    pending_end_since = now
            except Exception:
                if combat_search is not None:
                    combat_search.note_error()
                log.error("failed to handle frame:\n%s\n%s", line, traceback.format_exc())
                send("STATE")
    finally:
        if combat_search is not None:
            combat_search.close()
        log.info("stdin EOF; session end")


if __name__ == "__main__":
    main()
