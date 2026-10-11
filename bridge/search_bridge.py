"""Simulator-backed Full-Gumbel combat decisions for CommunicationMod.

The real game remains authoritative.  A persistent native battle mirrors every
accepted action, is reconciled against each settled CommunicationMod frame, and
provides the hidden simulator state required by the checkpoint's NOSL search.
Any state or legal-action mismatch is an error; this module never falls back to
the checkpoint's direct combat policy.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import logging
from typing import Any

import numpy as np

from bridge.real_game_env import RealDecision, RealGameAdapter
from env.core import native_actions as C
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.strategies import DecisionRequest, NativeStrategy
from rl.policies import model as model_utils
from rl.policies.precision import bf16_autocast
from rl.search.native import NativeSearchConfig
from schemas.combat_state import (
    _MONSTER_SIM2REAL,
    diff,
    normalize_real,
    normalize_sim,
    real_to_sim_card_enum,
    real_to_sim_monster_move,
)

import slaythespire as sts


LOG = logging.getLogger(__name__)

_REAL_TO_SIM_MONSTER = {real: sim for sim, real in _MONSTER_SIM2REAL.items()}
_SUPPORTED_PLAYER_POWERS = frozenset({
    "STRENGTH", "DEXTERITY", "WEAK", "FRAIL", "VULNERABLE",
    "ARTIFACT", "DOUBLE_DAMAGE", "INTANGIBLE", "BARRICADE",
    "CORRUPTION", "CONFUSED", "HEX", "DRAW_REDUCTION", "BLASPHEMER",
    "ELECTRO", "SURROUNDED", "MASTER_REALITY", "PEN_NIB",
    "WRATH_NEXT_TURN", "DOUBLE_TAP", "BRUTALITY", "COMBUST",
    "DARK_EMBRACE", "DEMON_FORM", "EVOLVE", "FEEL_NO_PAIN",
    "FIRE_BREATHING", "FLAME_BARRIER", "JUGGERNAUT", "METALLICIZE",
    "RAGE", "RUPTURE", "THORNS", "VIGOR",
})
_SUPPORTED_MONSTER_POWERS = frozenset({
    "ARTIFACT", "BLOCK_RETURN", "CHOKED", "CORPSE_EXPLOSION",
    "LOCK_ON", "MARK", "METALLICIZE", "PLATED_ARMOR", "POISON",
    "REGEN", "SHACKLED", "STRENGTH", "VULNERABLE", "WEAK",
    "ANGRY", "BEAT_OF_DEATH", "CURIOSITY", "CURL_UP", "ENRAGE",
    "FADING", "FLIGHT", "GENERIC_STRENGTH_UP", "INTANGIBLE",
    "MALLEABLE", "MODE_SHIFT", "RITUAL", "SLOW", "SPORE_CLOUD",
    "THIEVERY", "THORNS", "TIME_WARP", "INVINCIBLE", "REACTIVE",
    "SHARP_HIDE", "ASLEEP", "BARRICADE", "MINION", "MINION_LEADER",
    "PAINFUL_STABS", "REGROW", "SHIFTING", "STASIS",
})
_POWER_ALIASES = {
    "WEAKENED": "WEAK",
    "ANGER": "ENRAGE",
    "DOUBLE_DAMAGE": "DOUBLE_DAMAGE",
    "DRAW_REDUCTION": "DRAW_REDUCTION",
    "MASTER_REALITY": "MASTER_REALITY",
    "PEN_NIB": "PEN_NIB",
    "WRATH_NEXT_TURN": "WRATH_NEXT_TURN",
    "FEEL_NO_PAIN": "FEEL_NO_PAIN",
    "FIRE_BREATHING": "FIRE_BREATHING",
    "FLAME_BARRIER": "FLAME_BARRIER",
    "PLATED_ARMOR": "PLATED_ARMOR",
}


def _enumish(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    return _POWER_ALIASES.get(text, text)


def _card_rows(cards: list[dict[str, Any]], *, costs: bool = False) -> list[tuple]:
    result = []
    for card in cards:
        row = (real_to_sim_card_enum(card["id"]), int(card["up"]))
        if costs:
            row += (int(card.get("cost", 0)),)
        result.append(row)
    return result


def _power_rows(powers: dict[str, int], supported: frozenset[str]) -> list[tuple[str, int]]:
    result = []
    for name, amount in powers.items():
        enum_name = _enumish(name)
        if enum_name in supported:
            result.append((enum_name, int(amount)))
    return result


def _empty_potion_name(name: str) -> str:
    return "EMPTY_POTION_SLOT" if str(name).upper() in {"PAD", "EMPTY_POTION_SLOT"} else str(name)


def _monster_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(
        monster["id"] for monster in normalize_sim(raw)["monsters"]
        if not monster["gone"] and monster["id"] != "PAD"
    ))


class RealCombatSearch:
    """One persistent native combat mirror bound to a Full-Gumbel strategy."""

    def __init__(self, model, adapter: RealGameAdapter, *, native_search_threads: int = 4,
                 master_nonce: int = 0) -> None:
        self.model = model
        self.adapter = adapter
        self.contract = C.policy_contract(model.observation_space)
        if self.contract not in C.NEURAL_CONTRACTS:
            raise ValueError("real-game native search requires a complete neural combat contract")
        self.config = NativeSearchConfig.from_model(model)
        if self.config.root_search_mode != 2 or not self.config.paper_full_gumbel:
            raise ValueError("real-game combat search requires the checkpoint's strict Full-Gumbel mode")
        autocast = (
            (lambda: bf16_autocast(enabled=True))
            if str(model.device).startswith("cuda") else nullcontext
        )
        self.strategy = NativeStrategy(
            model,
            self.config,
            master_nonce=int(master_nonce),
            native_search_threads=int(native_search_threads),
            root_autocast=autocast,
        )
        self.env = SlayTheSpireEnv(
            character=sts.CharacterClass.IRONCLAD,
            ascension=adapter.ascension,
            global_map_view=model_utils.uses_global_map_observation(model),
            card_language=model_utils.uses_card_language(model),
            combat_language=model_utils.uses_combat_language(model),
            deck_size_observation=model_utils.uses_deck_size_observation(model),
            neow_observation=model_utils.uses_neow_observation(model),
            event_option_observation=model_utils.uses_event_option_observation(model),
            combat_action_contract=self.contract,
        )
        self.env.reset(seed=0)
        self.strategy.bind_environments([self.env])
        self._combat_branch_bound = False
        self.identity: tuple[int, int] | None = None
        self.decision_index = 0
        self.sim_to_real_targets: dict[int, int] = {}
        self._rollback_bc = None
        self._rollback_thin = None
        self._awaiting_command = False
        self.run_decisions = 0
        self.run_commands = 0
        self.run_alignment_checks = 0
        self.run_errors = 0
        self.run_encounters = 0
        LOG.info("native search configured %s threads=%s", asdict(self.config), native_search_threads)

    def close(self) -> None:
        self.strategy.unbind_environments()
        self.env.close()

    def observe_frame(self, frame: dict[str, Any]) -> None:
        """Rollback an optimistically mirrored action when the mod rejects it."""
        if "error" in frame and self._awaiting_command:
            if self._rollback_bc is None:
                raise RuntimeError("rejected real-game command has no native rollback state")
            self.env.thick.bc = self._rollback_bc
            thin = self._rollback_thin or {}
            self.env._pending = deepcopy(thin.get("pending"))
            self.env._multi = deepcopy(thin.get("multi"))
            self.env._nosl_cpp_pending = deepcopy(thin.get("nosl_cpp_pending"))
            self.adapter.multi_selected = set(thin.get("real_multi_selected", ()))
            self.adapter._joint_multi_plan = deepcopy(thin.get("real_joint_multi_plan"))
            self.env._invalidate_backend_cache()
            LOG.error("CommunicationMod rejected the mirrored search action; native state rolled back: %s",
                      frame.get("error"))
        if "error" not in frame:
            self._rollback_bc = None
            self._rollback_thin = None
            self._awaiting_command = False

    def reset_run(self) -> None:
        self.identity = None
        self.decision_index = 0
        self.sim_to_real_targets = {}
        self._rollback_bc = None
        self._rollback_thin = None
        self._awaiting_command = False
        self.run_decisions = 0
        self.run_commands = 0
        self.run_alignment_checks = 0
        self.run_errors = 0
        self.run_encounters = 0

    def note_error(self) -> None:
        self.run_errors += 1

    def run_summary(self) -> dict[str, int]:
        return {
            "search_decisions": self.run_decisions,
            "search_commands": self.run_commands,
            "search_simulations": self.run_decisions * int(self.config.simulations),
            "search_alignment_checks": self.run_alignment_checks,
            "search_errors": self.run_errors,
            "search_encounters": self.run_encounters,
        }

    def decide(self) -> RealDecision:
        real = normalize_real(self.adapter.gs)
        if real is None:
            raise RuntimeError("native combat search called outside combat")
        identity = (int(self.adapter.gs.get("seed", 0)), int(self.adapter.gs.get("floor", 0)))
        if self.identity != identity or self.env.thick.bc is None or self.env.thick.bc.is_battle_over:
            self._attach_battle(real, identity)

        self._reconcile_and_sync(real)
        last_diagnostics: dict[str, Any] = {}
        for _ in range(12):
            self._assert_mask_alignment()
            request = DecisionRequest(
                self.env,
                self.env._obs(),
                self.env._info(),
                identity[0],
                self.decision_index,
            )
            result = self.strategy.decide([request], check=lambda: None)[0]
            if result.action is None:
                raise RuntimeError("native search returned no thin policy action")
            action = int(result.action)
            self.run_decisions += 1
            last_diagnostics = dict(result.diagnostics)
            rollback = self.env.thick.bc.clone()
            rollback_thin = {
                "pending": deepcopy(self.env._pending),
                "multi": deepcopy(self.env._multi),
                "nosl_cpp_pending": deepcopy(self.env._nosl_cpp_pending),
                "real_multi_selected": set(self.adapter.multi_selected),
                "real_joint_multi_plan": deepcopy(self.adapter._joint_multi_plan),
            }
            real_action = self._map_action_to_real(action)
            command = self.adapter.command_for_joint_action(real_action)

            _obs, _reward, terminated, truncated, _info = self.env.step(action)
            if truncated:
                raise RuntimeError("native shadow truncated while applying a searched action")
            self.decision_index += 1
            if command is None:
                # Autoregressive MULTI prefix; both adapters now hold the same
                # selection prefix and the next search decision chooses another
                # card or STOP without issuing a game command yet.
                continue

            self._rollback_bc = rollback
            self._rollback_thin = rollback_thin
            self._awaiting_command = True
            self.run_commands += 1
            LOG.info(
                "full-gumbel decision index=%s sim_action=%s real_action=%s command=%s "
                "terminated=%s diagnostics=%s",
                self.decision_index - 1,
                action,
                real_action,
                command,
                terminated,
                last_diagnostics,
            )
            return RealDecision(
                command,
                "native_full_gumbel",
                action_id=real_action,
                legal_ids=tuple(int(i) for i in np.flatnonzero(self.adapter.action_mask())),
            )
        raise RuntimeError("native search exceeded autoregressive MULTI prefix limit")

    def _attach_battle(self, real: dict[str, Any], identity: tuple[int, int]) -> None:
        seed, floor = identity
        global_state = self.adapter.obs_canonical()["global"]
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, self.adapter.ascension)
        gc.act = max(1, int(global_state.get("act", 1) or 1))
        gc.floor_num = floor
        gc.max_hp = max(1, int(global_state["max_hp"]))
        gc.cur_hp = min(gc.max_hp, max(0, int(global_state["hp"])))
        gc.gold = max(0, int(global_state.get("gold", 0)))
        gc.set_deck(_card_rows([
            {"id": cid, "up": up} for cid, up in global_state.get("deck", [])
        ]))
        gc.set_relic_state([
            {"id": rid, "counter": int(counter)}
            for rid, counter in zip(
                global_state.get("relics", []), global_state.get("relic_counters", [])
            )
        ])
        gc.set_potion_state([_empty_potion_name(p) for p in global_state.get("potions", [])])
        room = str(self.adapter.gs.get("room_type", "")).upper()
        gc.cur_room = (
            sts.Room.BOSS if "BOSS" in room else
            sts.Room.ELITE if "ELITE" in room else
            sts.Room.MONSTER
        )
        boss = str(global_state.get("boss") or "")
        if boss and hasattr(sts.MonsterEncounter, boss):
            gc.boss = getattr(sts.MonsterEncounter, boss)

        wanted = tuple(sorted(m["id"] for m in real["monsters"] if not m["gone"]))
        candidates = []
        for encounter in sts.encounter_names():
            try:
                battle = sts.create_battle(gc, encounter)
                if _monster_ids(battle.to_dict()) == wanted:
                    candidates.append((encounter, battle))
            except Exception:
                continue
        if not candidates:
            raise RuntimeError(f"no native encounter matches real monsters {wanted!r}")
        encounter, battle = candidates[0]
        if len(candidates) > 1:
            LOG.warning("ambiguous encounter %s candidates=%s; using=%s", wanted,
                        [name for name, _ in candidates], encounter)
        self.env.thick.gc = gc
        self.env.thick.bc = battle
        self.env.thick._rebuild_map()
        if not self._combat_branch_bound:
            # A real-game shadow must stop at combat terminal states.  The
            # whole-run backend would call BattleContext.exit_battle() and try
            # to open simulator rewards through a callback that does not exist
            # for this manually attached battle.
            self.env.thick = self.env.thick.fork_combat()
            self._combat_branch_bound = True
        self.env._pending = None
        self.env._multi = None
        self.env._nosl_cpp_pending = None
        self.env._invalidate_backend_cache()
        self.identity = identity
        self.sim_to_real_targets = self._target_mapping(real)
        self.run_encounters += 1
        LOG.info("attached native battle seed=%s floor=%s encounter=%s monsters=%s",
                 seed, floor, encounter, wanted)

    def _target_mapping(self, real: dict[str, Any]) -> dict[int, int]:
        sim = normalize_sim(self.env.thick.bc.to_dict())
        unused = set(range(len(real["monsters"])))
        mapping: dict[int, int] = {}
        for sim_index, monster in enumerate(sim["monsters"]):
            if monster["gone"] or monster["id"] == "PAD":
                continue
            matches = [i for i in sorted(unused)
                       if not real["monsters"][i]["gone"]
                       and real["monsters"][i]["id"] == monster["id"]]
            if not matches:
                raise RuntimeError(
                    f"cannot map native monster slot {sim_index} {monster['id']} to real slots"
                )
            mapping[sim_index] = matches[0]
            unused.remove(matches[0])
        return mapping

    def _state_spec(self, real: dict[str, Any]) -> dict[str, Any]:
        self.sim_to_real_targets = self._target_mapping(real)
        native_count = len(self.env.thick.bc.to_dict()["monsters"])
        monsters: list[dict[str, Any]] = [{} for _ in range(native_count)]
        for sim_index, real_index in self.sim_to_real_targets.items():
            monster = real["monsters"][real_index]
            move = real_to_sim_monster_move(monster["id"], monster["real_move_id"])
            last_move = real_to_sim_monster_move(
                monster["id"], monster["real_last_move_id"]
            )
            monsters[sim_index] = {
                "hp": int(monster["hp"]),
                "max_hp": int(monster["max_hp"]),
                "block": int(monster["block"]),
                "move_history": [move, last_move],
                "powers": _power_rows(monster["powers"], _SUPPORTED_MONSTER_POWERS),
            }
        player = real["player"]
        return {
            "turn": max(0, int(real["turn"]) - 1),
            "player": {
                "hp": int(player["hp"]),
                "max_hp": int(player["max_hp"]),
                "block": int(player["block"]),
                "energy": int(player["energy"]),
                "gold": int(player.get("gold", 0)),
                "counters": dict(player.get("combat_counters", {})),
                "powers": _power_rows(player["powers"], _SUPPORTED_PLAYER_POWERS),
            },
            "monsters": monsters,
            "hand": _card_rows(real["hand"], costs=True),
            "draw": _card_rows(real["draw"]),
            "discard": _card_rows(real["discard"]),
            "exhaust": _card_rows(real["exhaust"]),
            "potions": [_empty_potion_name(p) for p in self.adapter._potions()],
        }

    def _reconcile_and_sync(self, real: dict[str, Any]) -> None:
        selection = self.adapter.select_descriptor()
        if selection is not None:
            self._reconcile_selection(selection)
            return
        predicted = normalize_sim(self.env.thick.bc.to_dict())
        before = diff(real, predicted)
        if before:
            LOG.warning("native shadow pre-sync differs count=%s sample=%s", len(before), before[:8])
        self.env.thick.bc.apply_custom_state(self._state_spec(real))
        self.env.thick.gc.max_hp = max(1, int(real["player"]["max_hp"]))
        self.env.thick.gc.cur_hp = min(
            self.env.thick.gc.max_hp, max(0, int(real["player"]["hp"]))
        )
        self.env.thick.gc.gold = max(0, int(real["player"].get("gold", 0)))
        self.env._pending = None
        self.env._multi = None
        self.env._nosl_cpp_pending = None
        self.env._invalidate_backend_cache()
        after = diff(real, normalize_sim(self.env.thick.bc.to_dict()))
        if after:
            raise RuntimeError(f"native real-game state alignment failed: {after[:12]!r}")
        self.run_alignment_checks += 1

    @staticmethod
    def _selection_candidates(desc: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        rows = []
        for candidate in desc.get("candidates", ()):
            card = candidate.get("card")
            rows.append((
                int(candidate.get("native_index", -1)),
                None if card is None else real_to_sim_card_enum(card["id"]),
                None if card is None else int(card.get("up", 0)),
            ))
        return tuple(rows)

    def _reconcile_selection(self, real_desc: dict[str, Any]) -> None:
        sim_desc = self.env.thick.select_descriptor()
        if sim_desc is None:
            raise RuntimeError("real game entered combat selection but native shadow did not")
        real_context = str(real_desc.get("context") or "PAD")
        sim_context = str(sim_desc.get("context") or "PAD")
        if real_context != "PAD" and real_context != sim_context:
            raise RuntimeError(
                f"combat selection context mismatch real={real_context} sim={sim_context}")
        real_candidates = self._selection_candidates(real_desc)
        if sim_context in {"CODEX", "DISCOVERY"}:
            generated = [card for _native, card, _up in real_candidates if card is not None]
            if len(generated) != 3:
                raise RuntimeError("generated combat selection did not expose three real cards")
            self.env.thick.bc.set_generated_selection_cards(generated)
            sim_desc = self.env.thick.select_descriptor()
        sim_candidates = self._selection_candidates(sim_desc)
        if real_candidates != sim_candidates:
            raise RuntimeError(
                "combat selection candidates differ: "
                f"real={real_candidates!r} sim={sim_candidates!r}")
        for key in ("multi", "pick_max", "cards_in_hand"):
            if int(real_desc.get(key, 0)) != int(sim_desc.get(key, 0)):
                raise RuntimeError(
                    f"combat selection {key} mismatch real={real_desc.get(key)!r} "
                    f"sim={sim_desc.get(key)!r}")
        self.env._invalidate_backend_cache()
        self.run_alignment_checks += 1

    def _map_action_to_real(self, action: int) -> int:
        aid = int(action)
        bases = (C.AUTOREGRESSIVE_PLAY, C.AUTOREGRESSIVE_POTION)
        limits = (C.AUTOREGRESSIVE_POTION, C.AUTOREGRESSIVE_PICK)
        for base, limit in zip(bases, limits):
            if base <= aid < limit:
                source, target = divmod(aid - base, 6)
                if target < 5:
                    if target not in self.sim_to_real_targets:
                        raise RuntimeError(f"searched action targets unmapped native slot {target}")
                    target = self.sim_to_real_targets[target]
                return base + source * 6 + target
        return aid

    def _remap_real_mask(self, real_mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(real_mask, dtype=bool)
        reverse = {real: sim for sim, real in self.sim_to_real_targets.items()}
        for aid in np.flatnonzero(real_mask):
            mapped = int(aid)
            for base, limit in ((C.AUTOREGRESSIVE_PLAY, C.AUTOREGRESSIVE_POTION),
                                (C.AUTOREGRESSIVE_POTION, C.AUTOREGRESSIVE_PICK)):
                if base <= aid < limit:
                    source, target = divmod(int(aid) - base, 6)
                    if target < 5:
                        if target not in reverse:
                            continue
                        target = reverse[target]
                    mapped = base + source * 6 + target
                    break
            result[mapped] = True
        return result

    def _assert_mask_alignment(self) -> None:
        real = self._remap_real_mask(self.adapter.action_mask())
        sim = np.asarray(self.env.action_masks(), dtype=bool)
        if real.shape != sim.shape or not np.array_equal(real, sim):
            only_real = np.flatnonzero(real & ~sim).tolist()
            only_sim = np.flatnonzero(sim & ~real).tolist()
            raise RuntimeError(
                "native real-game legal-action alignment failed: "
                f"only_real={only_real} only_sim={only_sim}"
            )
        self.run_alignment_checks += 1
