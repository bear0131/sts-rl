"""Automatic CommunicationMod orchestrator backed by a MaskablePPO checkpoint.

CommunicationMod launches this script as a subprocess:

    stdin  <- one JSON game-state frame per line
    stdout -> one command line per frame

Never write diagnostics to stdout.  Everything except commands goes to
``agent_orchestrator.log`` next to this file.
"""

from __future__ import annotations

import argparse
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

from bridge.real_game_env import RealDecision, RealGameAdapter
from bridge.campaign import SeedCampaign
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

    def decide(self, frame: dict[str, Any]) -> RealDecision:
        self.adapter.update(frame)
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
                metadata=(self.combat_search.run_summary()
                          if self.combat_search is not None else None),
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
        auto = self.adapter.auto_command(auto_enter_shop=self.auto_enter_shop)
        if auto is not None:
            command, reason = auto
            return RealDecision(command, reason)

        if not self.adapter.in_game():
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
            mask = self.adapter.action_mask(pending=pending)
            legal_ids = tuple(int(i) for i in np.flatnonzero(mask))
            if not legal_ids:
                fallback = self.adapter.fallback_command()
                log.warning(
                    "no legal policy actions; fallback=%s screen=%s phase=%s choices=%s",
                    fallback,
                    self.adapter.screen_type(),
                    self.adapter.phase(),
                    self.adapter.choices(),
                )
                return RealDecision(fallback, "fallback_no_legal")

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

        fallback = self.adapter.fallback_command()
        log.warning("inner decision limit reached; trace=%s fallback=%s", trace, fallback)
        return RealDecision(fallback, "fallback_inner_limit")

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


def stale_combat_play_frame(frame: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    """A combat command must change its real frame before another decision."""
    return pending is not None and frame.get("game_state") == pending


def combat_command_needs_ack(frame: dict[str, Any], command: str) -> bool:
    gs = frame.get("game_state") or {}
    return bool(gs.get("combat_state")) and command.split()[0].upper() in {
        "PLAY", "POTION", "CHOOSE", "CONFIRM", "RETURN",
    }


def stale_combat_end_frame(frame: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    """END is acknowledged by a new turn/room, not changed transport flags."""
    if pending is None or "error" in frame:
        return False
    gs = frame.get("game_state") or {}
    combat = gs.get("combat_state")
    if gs.get("screen_type") == "GAME_OVER" or not combat or gs.get("room_phase") != "COMBAT":
        return False
    if any(gs.get(k) != pending.get(k) for k in ("seed", "floor")):
        return False
    return int(combat.get("turn", 0)) <= int(pending["combat_state"]["turn"])


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
        if (not frame.get("ready_for_command", True) or gs.get("current_action")
                or gs.get("action_phase") not in (None, "WAITING_ON_USER")
                or gs.get("is_screen_up")):
            self.state = None
            return True
        if self.state != gs:
            self.state = deepcopy(gs)
            self.since = now
            return True
        return now - self.since < self.seconds


def main() -> None:
    args = build_parser().parse_args()
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
        auto_start=args.auto_start,
        start_seed=args.start_seed,
        ascension=args.ascension,
        character=args.character,
        campaign=campaign,
        combat_search=combat_search,
    )

    try:
        send("ready")
        pending_play_frame = None
        pending_play_since = 0.0
        pending_end_frame = None
        pending_end_since = 0.0
        combat_stability = CombatFrameStability()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            log.debug("RECV %s", line)
            try:
                frame = json.loads(line)
                if stale_combat_end_frame(frame, pending_end_frame):
                    if time.monotonic() - pending_end_since > 10.0:
                        raise RuntimeError("game turn did not advance for 10 seconds after END")
                    time.sleep(0.05)
                    send("STATE")
                    continue
                pending_end_frame = None
                if stale_combat_play_frame(frame, pending_play_frame):
                    if time.monotonic() - pending_play_since > 10.0:
                        raise RuntimeError("game state unchanged for 10 seconds after combat command")
                    time.sleep(0.05)
                    send("STATE")
                    continue
                pending_play_frame = None
                if combat_stability.waiting(frame, time.monotonic()):
                    time.sleep(0.1)
                    send("STATE")
                    continue
                decision = agent.decide(frame)
                _log_decision(frame, decision, adapter)
                send_decision(frame, decision)
                if combat_command_needs_ack(frame, decision.command):
                    pending_play_frame = deepcopy(frame.get("game_state"))
                    pending_play_since = time.monotonic()
                if decision.command.upper() == "END":
                    pending_end_frame = deepcopy(frame.get("game_state"))
                    pending_end_since = time.monotonic()
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
