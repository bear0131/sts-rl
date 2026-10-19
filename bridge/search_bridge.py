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
from bridge.native_state import make_native_combat_root, native_combat_room
from bridge.schemas.card_state import (
    CombatCardIdentityMap, native_combat_card_state, real_master_deck_state,
)
from env.core import native_actions as C
from env.core.encounters import ACT_POOLS
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.strategies import DecisionRequest, NativeStrategy
from rl.policies import model as model_utils
from rl.policies.precision import bf16_autocast
from rl.search.native import NativeSearchConfig
from schemas.combat_state import (
    _MONSTER_SIM2REAL,
    _canonical_real_monster_id,
    combat_select_options,
    diff,
    normalize_real,
    normalize_sim,
    real_to_sim_card_enum,
    real_to_sim_monster_move,
)
from schemas.entity_ids import (
    NATIVE_MONSTER_STATUS_IDS,
    NATIVE_PLAYER_STATUS_IDS,
    native_power_enum,
)

import slaythespire as sts


LOG = logging.getLogger(__name__)

_REAL_TO_SIM_MONSTER = {
    _canonical_real_monster_id(real): sim
    for sim, real in _MONSTER_SIM2REAL.items()
}
_GREMLIN_LEADER_MINIONS = frozenset({
    "GremlinFat", "GremlinWarrior", "GremlinTsundere", "GremlinThief",
    "GremlinWizard",
})
_SUPPORTED_PLAYER_POWERS = frozenset(NATIVE_PLAYER_STATUS_IDS) - {"THE_BOMB"}
_SUPPORTED_MONSTER_POWERS = frozenset(NATIVE_MONSTER_STATUS_IDS)


def _enumish(value: Any) -> str:
    return native_power_enum(value or "")


def _power_rows(powers: dict[str, int], supported: frozenset[str]) -> list[tuple[str, int]]:
    result = []
    for name, amount in powers.items():
        enum_name = _enumish(name)
        if enum_name not in supported:
            raise ValueError(
                f"canonical power {name!r} maps to unsupported native status "
                f"{enum_name!r}")
        result.append((enum_name, int(amount)))
    return result


def _empty_potion_name(name: str) -> str:
    if name == "PAD":
        return "EMPTY_POTION_SLOT"
    if str(name).upper() == "EMPTY_POTION_SLOT":
        raise ValueError(
            "canonical potion state used a native empty sentinel instead of PAD"
        )
    return str(name)


def _monster_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(
        monster["id"] for monster in normalize_sim(raw)["monsters"]
        if not monster["gone"] and monster["id"] != "PAD"
    ))


def _find_shadow_battle(make_gc, wanted: tuple[str, ...], encounters,
                        *, max_seed_nonces: int = 64,
                        allow_gremlin_leader_dynamic: bool = False):
    """Find a deterministic native encounter instance with the public roster.

    A few encounter templates choose their concrete monsters from combat RNG.
    CommunicationMod does not expose that RNG state, so a bounded shadow seed
    nonce supplies a matching template instance before public state injection.
    """
    for seed_nonce in range(max_seed_nonces):
        for encounter in encounters:
            gc = make_gc(seed_nonce)
            try:
                battle = sts.create_battle(gc, encounter)
            except Exception:
                continue
            found = _monster_ids(battle.to_dict())
            if found == wanted or (
                allow_gremlin_leader_dynamic
                and found.count("GremlinLeader") == 1
                and all(monster == "GremlinLeader"
                        or monster in _GREMLIN_LEADER_MINIONS
                        for monster in found)
            ):
                return gc, battle, encounter, seed_nonce
    raise RuntimeError(f"no native encounter matches real monsters {wanted!r}")


class RealCombatSearch:
    """One persistent native combat mirror bound to a Full-Gumbel strategy."""

    def __init__(self, model, adapter: RealGameAdapter, *, native_search_threads: int = 4,
                 master_nonce: int = 0, gumbel_scale: float | None = None) -> None:
        self.model = model
        self.adapter = adapter
        self.contract = C.policy_contract(model.observation_space)
        if self.contract not in C.NEURAL_CONTRACTS:
            raise ValueError("real-game native search requires a complete neural combat contract")
        if gumbel_scale is not None and (not np.isfinite(gumbel_scale) or gumbel_scale < 0):
            raise ValueError("gumbel_scale must be finite and nonnegative")
        self.config = NativeSearchConfig.from_model(model, gumbel_scale=gumbel_scale)
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
        self.card_identities = CombatCardIdentityMap()
        self.decision_index = 0
        self.sim_to_real_targets: dict[int, int] = {}
        self.sim_to_real_selection_handles: dict[int, int] = {}
        self.sim_to_real_selection_actions: dict[int, int] = {}
        self._rollback_bc = None
        self._rollback_thin = None
        self._awaiting_command = False
        self.run_decisions = 0
        self.run_commands = 0
        self.run_alignment_checks = 0
        self.run_errors = 0
        self.run_rejected_commands = 0
        self._consecutive_rejected_commands = 0
        self.run_encounters = 0
        LOG.info("native search configured %s threads=%s", asdict(self.config), native_search_threads)

    def close(self) -> None:
        self.strategy.unbind_environments()
        self.env.close()

    def cancel_unsent_command(self) -> None:
        """Undo a speculative command when a fresh pre-send frame invalidates it."""
        if self._awaiting_command:
            if self._rollback_bc is None:
                raise RuntimeError("rejected real-game command has no native rollback state")
            self.env.thick.bc = self._rollback_bc
            thin = self._rollback_thin or {}
            self.decision_index = thin["decision_index"]
            self.env._pending = deepcopy(thin.get("pending"))
            self.env._multi = deepcopy(thin.get("multi"))
            self.env._nosl_cpp_pending = deepcopy(thin.get("nosl_cpp_pending"))
            self.adapter.multi_selected = set(thin.get("real_multi_selected", ()))
            self.adapter._joint_multi_plan = deepcopy(thin.get("real_joint_multi_plan"))
            self.env._invalidate_backend_cache()
        self._rollback_bc = None
        self._rollback_thin = None
        self._awaiting_command = False

    def observe_frame(self, frame: dict[str, Any]) -> None:
        """Rollback an optimistically mirrored action when the mod rejects it."""
        if "error" in frame and self._awaiting_command:
            # CommandExecutor checks availability before dispatch. This exact
            # rejection guarantees the command never ran; refresh and search again,
            # rather than replaying the stale command across an overlay.
            verb = getattr(self, "_awaiting_command_text", "").split(" ", 1)[0].upper()
            rejected_overlay_command = (
                verb in {"END", "PLAY", "POTION", "CHOOSE", "CONFIRM", "RETURN"}
                and frame["error"] in {
                    f"Invalid command: {verb.lower()}. Possible commands: [key, click, wait, state]",
                    f"Invalid command: {verb.lower()}. Possible commands: [choose, return, key, click, wait, state]",
                }
            )
            self.cancel_unsent_command()
            if rejected_overlay_command:
                self.run_rejected_commands += 1
                self._consecutive_rejected_commands += 1
                if self._consecutive_rejected_commands > 3:
                    raise RuntimeError("command availability race exceeded 3 consecutive rejections")
                LOG.warning("combat command rejected before dispatch; native rollback, fresh-state recovery; "
                            "transport_rejected_commands=%s error=%s",
                            self.run_rejected_commands, frame["error"])
                return
            LOG.error("CommunicationMod rejected the mirrored search action; native state rolled back: %s",
                      frame.get("error"))
        if "error" not in frame:
            if self._awaiting_command:
                self._consecutive_rejected_commands = 0
            self._rollback_bc = None
            self._rollback_thin = None
            self._awaiting_command = False

    def reset_run(self) -> None:
        self.identity = None
        self.card_identities = CombatCardIdentityMap()
        self.decision_index = 0
        self.sim_to_real_targets = {}
        self.sim_to_real_selection_handles = {}
        self.sim_to_real_selection_actions = {}
        self._rollback_bc = None
        self._rollback_thin = None
        self._awaiting_command = False
        self.run_decisions = 0
        self.run_commands = 0
        self.run_alignment_checks = 0
        self.run_errors = 0
        self.run_rejected_commands = 0
        self._consecutive_rejected_commands = 0
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
            "transport_rejected_commands": self.run_rejected_commands,
            "search_encounters": self.run_encounters,
        }

    def decide(self) -> RealDecision:
        real = normalize_real(self.adapter.gs)
        if real is None:
            raise RuntimeError("native combat search called outside combat")
        if "seed" not in self.adapter.gs or "floor" not in self.adapter.gs:
            raise ValueError("real combat frame is missing seed or floor identity")
        identity = (int(self.adapter.gs["seed"]), int(self.adapter.gs["floor"]))
        if self.identity != identity or self.env.thick.bc is None or self.env.thick.bc.is_battle_over:
            self._attach_battle(real, identity)

        self._reconcile_and_sync(real)
        # One UI command is a transaction, including any unsent autoregressive
        # prefixes. Transport retries must reuse its NOSL random identity, while
        # run_decisions continues to account for all actual search work.
        rollback = self.env.thick.bc.clone()
        rollback_thin = {
            "decision_index": self.decision_index,
            "pending": deepcopy(self.env._pending),
            "multi": deepcopy(self.env._multi),
            "nosl_cpp_pending": deepcopy(self.env._nosl_cpp_pending),
            "real_multi_selected": set(self.adapter.multi_selected),
            "real_joint_multi_plan": deepcopy(self.adapter._joint_multi_plan),
        }
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
            self._awaiting_command_text = command
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
        character = str(global_state["character"])
        if character != "IRONCLAD":
            raise RuntimeError(
                "this checkpoint/native search bridge supports IRONCLAD only; "
                f"refusing to mirror {character!r} as Ironclad"
            )
        room_type = self.adapter.gs["room_type"]
        native_room = native_combat_room(room_type)
        if global_state['floor'] != floor:
            raise ValueError("combat identity floor differs from canonical global floor")

        def make_gc(seed_nonce: int):
            return make_native_combat_root(global_state, seed=seed,
                                           room_type=room_type, seed_nonce=seed_nonce)

        wanted = tuple(sorted(m["id"] for m in real["monsters"] if not m["gone"]))
        act = int(global_state["act"])
        pool_kind = "boss" if native_room == sts.Room.BOSS else (
            "elite" if native_room == sts.Room.ELITE else "normal"
        )
        preferred = list(ACT_POOLS.get(act, {}).get(pool_kind, ()))
        encounters = preferred + [
            name for name in sts.encounter_names() if name not in preferred
        ]
        gc, battle, encounter, seed_nonce = _find_shadow_battle(
            make_gc,
            wanted,
            encounters,
            allow_gremlin_leader_dynamic=(
                wanted.count("GremlinLeader") == 1
                and all(monster == "GremlinLeader"
                        or monster in _GREMLIN_LEADER_MINIONS
                        for monster in wanted)
            ),
        )
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
        self.card_identities = CombatCardIdentityMap()
        self.sim_to_real_targets = {}
        self.sim_to_real_targets = self._target_mapping(real)
        self.run_encounters += 1
        LOG.info(
            "attached native battle seed=%s shadow_seed_nonce=%s floor=%s "
            "encounter=%s monsters=%s",
            seed, seed_nonce, floor, encounter, wanted,
        )

    def _target_mapping(self, real: dict[str, Any]) -> dict[int, int]:
        sim = normalize_sim(self.env.thick.bc.to_dict())
        unused = {
            index for index, monster in enumerate(real["monsters"])
            if not monster["gone"] and monster["id"] != "PAD"
        }
        mapping: dict[int, int] = {}
        for sim_index, real_index in sorted(self.sim_to_real_targets.items()):
            if sim_index >= len(sim["monsters"]) or real_index not in unused:
                continue
            sim_monster = sim["monsters"][sim_index]
            real_monster = real["monsters"][real_index]
            # A predicted kill (or a user reload) must not erase a confirmed
            # entity mapping. The incoming live HP is authoritative and will
            # restore this slot during root injection.
            if (sim_monster["id"] != "PAD"
                    and sim_monster["id"] == real_monster["id"]):
                mapping[sim_index] = real_index
                unused.remove(real_index)
        # A confirmed entity may have died in Java while the shadow predicted
        # survival. Retain its identity across repeated mapping before injection.
        confirmed_dead = {
            si for si, ri in self.sim_to_real_targets.items()
            if si < len(sim["monsters"]) and ri < len(real["monsters"])
            and real["monsters"][ri]["gone"]
            and sim["monsters"][si]["id"] == real["monsters"][ri]["id"]
        }
        mapping.update({si: self.sim_to_real_targets[si] for si in confirmed_dead})
        unmatched_live_sim: list[int] = []
        for sim_index, monster in enumerate(sim["monsters"]):
            if (sim_index in mapping or sim_index in confirmed_dead or monster["gone"]
                    or monster["id"] == "PAD"):
                continue
            matches = [i for i in sorted(unused)
                       if not real["monsters"][i]["gone"]
                       and real["monsters"][i]["id"] == monster["id"]]
            if not matches:
                unmatched_live_sim.append(sim_index)
                continue
            real_index = min(
                matches,
                key=lambda i: (
                    real["monsters"][i]["max_hp"] != monster["max_hp"],
                    real["monsters"][i]["hp"] != monster["hp"],
                    i,
                ),
            )
            mapping[sim_index] = real_index
            unused.remove(real_index)

        # Gremlin Leader chooses summoned minion species through engine-private
        # RNG.  The real game is authoritative when that draw differs from the
        # native shadow.  Reuse the leader's three native minion slots, map the
        # public live entities onto them, and let root-state injection rebuild
        # any species that changed.  Other encounters still require exact IDs.
        leader_id = ("TheCollector" if any(
            m["id"] == "TheCollector" and not m["gone"]
            for m in real["monsters"]) else "GremlinLeader")
        minion_ids = ({"TorchHead"} if leader_id == "TheCollector"
                      else _GREMLIN_LEADER_MINIONS)
        real_leaders = [
            i for i, monster in enumerate(real["monsters"])
            if not monster["gone"] and monster["id"] == leader_id
        ]
        sim_leaders = [
            i for i, monster in enumerate(sim["monsters"])
            if not monster["gone"] and monster["id"] == leader_id
        ]
        dynamic_minions = (
            len(real_leaders) == 1
            and len(sim_leaders) == 1
            and all(real["monsters"][i]["id"] in minion_ids
                    for i in unused)
            and all(sim["monsters"][i]["id"] in minion_ids
                    for i in unmatched_live_sim)
        )
        if dynamic_minions and unused:
            # Dead Java summon slots remain in the transport list, while native
            # summon slots are reused. Release only confirmed minion corpses.
            for si in confirmed_dead:
                if sim["monsters"][si]["id"] in minion_ids:
                    mapping.pop(si, None)
            available = [
                i for i, monster in enumerate(sim["monsters"])
                if i not in mapping and monster["id"] != leader_id
            ]
            previous = dict(self.sim_to_real_targets)
            for sim_index, real_index in sorted(previous.items()):
                if (sim_index in available and real_index in unused
                        and real["monsters"][real_index]["id"]
                        in minion_ids):
                    mapping[sim_index] = real_index
                    available.remove(sim_index)
                    unused.remove(real_index)
            for real_index in sorted(tuple(unused)):
                exact = [
                    i for i in available
                    if sim["monsters"][i]["id"]
                    == real["monsters"][real_index]["id"]
                ]
                if exact:
                    sim_index = exact[0]
                    mapping[sim_index] = real_index
                    available.remove(sim_index)
                    unused.remove(real_index)
            for real_index, sim_index in zip(
                sorted(tuple(unused)),
                sorted(available, key=lambda i: (sim["monsters"][i]["gone"], i)),
            ):
                mapping[sim_index] = real_index
                unused.remove(real_index)
        if unmatched_live_sim and not dynamic_minions:
            sim_index = unmatched_live_sim[0]
            monster = sim["monsters"][sim_index]
            raise RuntimeError(
                f"cannot map native monster slot {sim_index} {monster['id']} to real slots"
            )
        if unused:
            raise RuntimeError(
                f"cannot map real monster slots {sorted(unused)!r} to native slots")
        return mapping

    def _sim_state_in_real_target_order(self) -> dict[str, Any]:
        sim = normalize_sim(self.env.thick.bc.to_dict())
        sim["monsters"] = [
            sim["monsters"][sim_index]
            for sim_index, _real_index in sorted(
                self.sim_to_real_targets.items(), key=lambda item: item[1]
            )
        ]
        return sim

    def _card_state(self, real: dict[str, Any]) -> dict[str, Any]:
        stasis = [None, None]
        covered = set()
        for sim_index, real_index in self.sim_to_real_targets.items():
            card = real['stasis_cards'][real_index]
            if card is not None:
                slot = min(sim_index, 1)  # CardManager's Bronze Orb physical slots.
                if stasis[slot] is not None:
                    raise ValueError('two real stasis cards map to one native slot')
                stasis[slot] = card
                covered.add(real_index)
        if covered != {i for i, card in enumerate(real['stasis_cards']) if card is not None}:
            raise ValueError('real stasis card has no native monster mapping')
        return self.card_identities.snapshot(self.adapter.gs['deck'], real['card_instances'], stasis)

    def _state_spec(self, real: dict[str, Any]) -> dict[str, Any]:
        self.sim_to_real_targets = self._target_mapping(real)
        native_count = len(self.env.thick.bc.to_dict()["monsters"])
        # Any unpaired native slot is absent from the authoritative real frame.
        # Mark it gone so a divergent summon cannot remain targetable.
        monsters: list[dict[str, Any]] = [
            {"gone": True} for _ in range(native_count)
        ]
        for sim_index, real_index in self.sim_to_real_targets.items():
            monster = real["monsters"][real_index]
            if monster["gone"]:
                monsters[sim_index] = {"gone": True}
                continue
            move = real_to_sim_monster_move(monster["id"], monster["real_move_id"])
            last_move = real_to_sim_monster_move(
                monster["id"], monster["real_last_move_id"]
            )
            second_last_move = real_to_sim_monster_move(
                monster["id"], monster["real_second_last_move_id"]
            )
            monsters[sim_index] = {
                "gone": False,
                "id": _REAL_TO_SIM_MONSTER[monster["id"]],
                "hp": int(monster["hp"]),
                "max_hp": int(monster["max_hp"]),
                "block": int(monster["block"]),
                "half_dead": bool(monster["half_dead"]),
                "unawakened": bool(monster["unawakened"]),
                "reactive": bool(monster["reactive"]),
                "move_history": [move, last_move, second_last_move],
                "base_damage": int(monster["base_damage"]),
                "hits": int(monster["hits"]),
                "powers": _power_rows(monster["powers"], _SUPPORTED_MONSTER_POWERS),
            }
            if move == "EXPLODER_EXPLODE":
                # Java exposes UNKNOWN intent (-1), not the explosion's hidden
                # 30 damage. Preserve the native move's transition semantics.
                monsters[sim_index].pop("base_damage")
                monsters[sim_index].pop("hits")
            if monster["intent_hidden"]:
                raise RuntimeError(
                    "Runic Dome current intent missing: install the simulator-parity "
                    "CommunicationMod exporter; shadow intent guesses are not allowed")
        player = real["player"]
        return {
            "strict_complete": True,
            "turn": max(0, int(real["turn"]) - 1),
            "player": {
                "hp": int(player["hp"]),
                "max_hp": int(player["max_hp"]),
                "block": int(player["block"]),
                "energy": int(player["energy"]),
                "energy_per_turn": int(player["energy_per_turn"]),
                "gold": int(player["gold"]),
                "counters": dict(player["combat_counters"]),
                "powers": _power_rows(player["powers"], _SUPPORTED_PLAYER_POWERS),
            },
            "monsters": monsters,
            "card_state": self._card_state(real),
            "potions": [_empty_potion_name(p) for p in self.adapter._potions()],
        }

    def _reconcile_and_sync(self, real: dict[str, Any]) -> None:
        selection = self.adapter.select_descriptor()
        if selection is not None:
            self._reconcile_selection(selection)
            return
        self.sim_to_real_selection_handles = {}
        self.sim_to_real_selection_actions = {}
        self.sim_to_real_targets = self._target_mapping(real)
        predicted = self._sim_state_in_real_target_order()
        before = diff(real, predicted)
        if before:
            LOG.warning("native shadow pre-sync differs count=%s sample=%s", len(before), before[:8])
        spec = self._state_spec(real)
        master = [dict(card, id=real_to_sim_card_enum(card['id']))
                  for card in real_master_deck_state(self.adapter.gs['deck'])]
        self.env.thick.gc.set_deck_state(master)
        self.env.thick.bc.apply_custom_state(spec)
        if self.env.thick.bc.card_state() != spec['card_state']:
            raise RuntimeError('native real-game card instance alignment failed')
        self.env.thick.gc.max_hp = int(real["player"]["max_hp"])
        self.env.thick.gc.cur_hp = int(real["player"]["hp"])
        self.env.thick.gc.gold = int(real["player"]["gold"])
        self.env._pending = None
        self.env._multi = None
        self.env._nosl_cpp_pending = None
        self.env._invalidate_backend_cache()
        after = diff(real, self._sim_state_in_real_target_order())
        if after:
            raise RuntimeError(f"native real-game state alignment failed: {after[:12]!r}")
        self.run_alignment_checks += 1

    @staticmethod
    def _selection_candidates(desc: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        if "candidates" not in desc or not isinstance(desc["candidates"], (list, tuple)):
            raise RuntimeError("combat selection descriptor is missing candidates")
        rows = []
        for index, candidate in enumerate(desc["candidates"]):
            if not isinstance(candidate, dict) or "native_index" not in candidate or "card" not in candidate:
                raise RuntimeError(f"malformed combat selection candidate {index}: {candidate!r}")
            card = candidate["card"]
            if card is not None and not {"id", "up", "cost"} <= set(card):
                raise RuntimeError(f"incomplete combat selection card {index}: {card!r}")
            rows.append((
                int(candidate["native_index"]),
                None if card is None else real_to_sim_card_enum(card["id"]),
                None if card is None else int(card["up"]),
                None if card is None else int(card["cost"]),
            ))
        return tuple(rows)

    def _reconcile_selection(self, real_desc: dict[str, Any]) -> None:
        sim_desc = self.env.thick.select_descriptor()
        if sim_desc is None:
            raise RuntimeError("real game entered combat selection but native shadow did not")
        required = {
            "context", "zone", "candidates", "selected", "pick_remaining",
            "pick_min", "pick_max", "can_pick_zero", "can_pick_any", "multi",
            "cards_in_hand",
        }
        for side, desc in (("real", real_desc), ("native", sim_desc)):
            missing = required - set(desc)
            if missing:
                raise RuntimeError(
                    f"{side} combat selection is missing {sorted(missing)!r}")
        real_context = str(real_desc["context"])
        sim_context = str(sim_desc["context"])
        if not real_context or not sim_context:
            raise RuntimeError(
                "combat selection context is missing: "
                f"real={real_context!r} sim={sim_context!r}"
            )
        if real_context != sim_context:
            raise RuntimeError(
                f"combat selection context mismatch real={real_context} sim={sim_context}")
        real_candidates = self._selection_candidates(real_desc)
        real_instances = []
        for candidate in real_desc["candidates"]:
            if "instance" not in candidate:
                raise ValueError("real selection candidate is missing its complete instance")
            if (candidate["card"] is None) != (candidate["instance"] is None):
                raise ValueError("real selection instance/skip mismatch")
            real_instances.append(candidate["instance"])
        battle = self.env.thick.bc
        if sim_context in {"CODEX", "DISCOVERY"}:
            generated = [card for _native, card, _up, _cost in real_candidates if card is not None]
            if len(generated) != 3:
                raise RuntimeError("generated combat selection did not expose three real cards")
            # Validate the full transaction before changing the live pending action.
            battle = battle.clone()
            battle.set_generated_selection_cards(generated)
            sim_desc = combat_select_options(battle.select_info())
            real_instances = [None if card is None else native_combat_card_state(card, None)
                              for card in real_instances]
        else:
            if any(card is None for card in real_instances):
                raise ValueError("physical selection cannot contain a skip template")
            real_instances = self.card_identities.selection_records(
                self.adapter.gs['deck'], real_instances)
        sim_candidates = self._selection_candidates(sim_desc)
        real_handles = tuple(row[0] for row in real_candidates)
        sim_handles = tuple(row[0] for row in sim_candidates)
        if len(set(real_handles)) != len(real_handles):
            raise RuntimeError(
                f"real combat selection contains duplicate handles: {real_handles!r}")
        if len(set(sim_handles)) != len(sim_handles):
            raise RuntimeError(
                f"native combat selection contains duplicate handles: {sim_handles!r}")
        sim_instances = battle.selection_card_state()
        if len(sim_instances) != len(sim_candidates) or len(sim_candidates) != len(real_candidates):
            raise RuntimeError("combat selection candidate counts differ")
        if sim_context in {"CODEX", "DISCOVERY"}:
            real_order = list(range(len(real_candidates)))
        else:
            real_by_uid = {card['uid']: i for i, card in enumerate(real_instances)}
            if len(real_by_uid) != len(real_instances):
                raise RuntimeError("real selection contains duplicate physical identities")
            try:
                real_order = [real_by_uid[card['uid']] for card in sim_instances]
            except KeyError as exc:
                raise RuntimeError("combat selection physical identities differ") from exc
            if len(set(real_order)) != len(real_order):
                raise RuntimeError("native selection contains duplicate physical identities")
        for si, ri in enumerate(real_order):
            if (sim_candidates[si][1:] != real_candidates[ri][1:]
                    or sim_instances[si] != real_instances[ri]):
                raise RuntimeError(
                    "combat selection instance mismatch: "
                    f"native={sim_instances[si]!r} real={real_instances[ri]!r}")
        real_multi = bool(real_desc["multi"])
        sim_multi = bool(sim_desc["multi"])
        if real_multi != sim_multi:
            raise RuntimeError(
                f"combat selection multi mismatch real={real_multi!r} sim={sim_multi!r}")
        def semantic_pick_max(desc: dict[str, Any]) -> int:
            raw = int(desc["pick_max"])
            if bool(desc["multi"]):
                return min(raw, len(desc["candidates"]))
            return raw

        def semantic_pick_remaining(desc: dict[str, Any]) -> int:
            if bool(desc["multi"]):
                return max(
                    0,
                    semantic_pick_max(desc)
                    - len(desc["selected"]),
                )
            return int(desc["pick_remaining"])

        for key in ("zone", "pick_min", "can_pick_zero", "can_pick_any"):
            if real_desc[key] != sim_desc[key]:
                raise RuntimeError(
                    f"combat selection {key} mismatch "
                    f"real={real_desc[key]!r} sim={sim_desc[key]!r}")

        if semantic_pick_remaining(real_desc) != semantic_pick_remaining(sim_desc):
            raise RuntimeError(
                "combat selection pick_remaining mismatch "
                f"real={real_desc.get('pick_remaining')!r} "
                f"sim={sim_desc.get('pick_remaining')!r}")
        if real_multi:
            # Autoregressive MULTI has an increasing-index prefix on both sides.
            # A permutation needs a whole-plan transport, not per-pick renumbering.
            if real_order != list(range(len(real_order))) or sim_handles != real_handles:
                raise RuntimeError("multi-selection physical order differs")
            if semantic_pick_max(real_desc) != semantic_pick_max(sim_desc):
                raise RuntimeError(
                    "combat selection pick_max mismatch "
                    f"real={real_desc.get('pick_max')!r} "
                    f"sim={sim_desc.get('pick_max')!r}")
            if int(real_desc["cards_in_hand"]) != int(sim_desc["cards_in_hand"]):
                raise RuntimeError(
                    "combat selection cards_in_hand mismatch "
                    f"real={real_desc.get('cards_in_hand')!r} "
                    f"sim={sim_desc.get('cards_in_hand')!r}")
        if sim_context in {"CODEX", "DISCOVERY"}:
            self.env.thick.bc.set_generated_selection_cards(generated)
        # Armaments/DualWield can remove ineligible cards from Java's displayed
        # hand, whereas native legal choices retain their original pile handles.
        # Match known physical identities, then translate dense policy indices.
        self.sim_to_real_selection_handles = {
            sim_handles[si]: real_handles[ri] for si, ri in enumerate(real_order)}
        self.sim_to_real_selection_actions = {} if real_multi else {
            C.AUTOREGRESSIVE_PICK + si: C.AUTOREGRESSIVE_PICK + ri
            for si, ri in enumerate(real_order)}
        self.env._invalidate_backend_cache()
        self.run_alignment_checks += 1

    def _map_action_to_real(self, action: int) -> int:
        aid = int(action)
        if C.AUTOREGRESSIVE_PICK <= aid < C.AUTOREGRESSIVE_MULTI:
            if aid not in self.sim_to_real_selection_actions:
                raise RuntimeError(f"unmapped native selection action {aid}")
            return self.sim_to_real_selection_actions[aid]
        bases = (C.AUTOREGRESSIVE_PLAY, C.AUTOREGRESSIVE_POTION)
        limits = (C.AUTOREGRESSIVE_POTION, C.AUTOREGRESSIVE_PICK)
        for base, limit in zip(bases, limits):
            if base <= aid < limit:
                source, target = divmod(aid - base, 6)
                if (base == C.AUTOREGRESSIVE_POTION and target == 5
                        and self._potion_uses_dummy_real_target(source)):
                    real_mask = np.asarray(self.adapter.action_mask(), dtype=bool)
                    real_targets = [
                        index for index in range(5)
                        if real_mask[base + source * 6 + index]
                    ]
                    if not real_targets:
                        raise RuntimeError(
                            f"real-game dummy-target potion slot {source} has no live target")
                    return base + source * 6 + real_targets[0]
                if target < 5:
                    if target not in self.sim_to_real_targets:
                        raise RuntimeError(f"searched action targets unmapped native slot {target}")
                    target = self.adapter.policy_target_index(
                        self.sim_to_real_targets[target]
                    )
                return base + source * 6 + target
        return aid

    def _potion_uses_dummy_real_target(self, slot: int) -> bool:
        records = self.adapter._potion_records()
        if not 0 <= int(slot) < len(records):
            return False
        potion = records[int(slot)]
        if not isinstance(potion, dict) or not bool(potion.get("requires_target")):
            return False
        potion_id = str(potion.get("id") or potion.get("name") or "")
        key = "".join(char for char in potion_id.lower() if char.isalnum())
        # CommunicationMod exposes these target-independent potions with
        # targetRequired=true and requires a target argument.  Keep the real
        # command contract while collapsing only the observed dummy UI target
        # to the simulator's untargeted potion action.
        return key in {"smokebomb", "explosivepotion"}

    def _remap_real_mask(self, real_mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(real_mask, dtype=bool)
        reverse = {real: sim for sim, real in self.sim_to_real_targets.items()}
        for aid in np.flatnonzero(real_mask):
            mapped = int(aid)
            if C.AUTOREGRESSIVE_PICK <= aid < C.AUTOREGRESSIVE_MULTI:
                selection_reverse = {real: sim for sim, real in self.sim_to_real_selection_actions.items()}
                if mapped not in selection_reverse:
                    raise RuntimeError(f"unmapped real selection action {mapped}")
                result[selection_reverse[mapped]] = True
                continue
            for base, limit in ((C.AUTOREGRESSIVE_PLAY, C.AUTOREGRESSIVE_POTION),
                                (C.AUTOREGRESSIVE_POTION, C.AUTOREGRESSIVE_PICK)):
                if base <= aid < limit:
                    source, target = divmod(int(aid) - base, 6)
                    if (base == C.AUTOREGRESSIVE_POTION
                            and self._potion_uses_dummy_real_target(source)):
                        mapped = base + source * 6 + 5
                        break
                    if target < 5:
                        real_target = self.adapter.real_target_index(target)
                        if real_target not in reverse:
                            continue
                        target = reverse[real_target]
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
