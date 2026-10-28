"""SimGameEnv: a whole *run* of Slay the Spire, backed by the sts_lightspeed sim.

One run = one episode: Neow -> map -> combats -> rewards -> rest/event/... ->
act boss. This is the thick, backend-specific layer; it OWNS the GameContext and
the per-combat BattleContext and is (with factories.py / combat_env.py /
spiremap.py) the only place that imports slaythespire.

It speaks the backend-agnostic semantic-action vocabulary (game_actions.py): at
each decision point it enumerates `legal_semantic_actions()` and `apply()`s the
one chosen. A future RealGameEnv will expose the same two methods over the
bridge, so the thin SlayTheSpireEnv (obs/action encoding for the policy) can sit
on either without change.

NOT a gym.Env yet (actions are variable-length semantic tuples, not a fixed
Discrete space). The thin layer adds the gym surface + fixed-length id/mask. See
doc/plan_simgameenv.md.

Scope: drives acts 1-3 end to end (stops before the act-4 heart). Events, campfires and
shops enumerate their real legal options via the sim's own action enumerator
(gc.legal_event_options / legal_campfire_options / legal_shop_options), so
multi-phase events (Golden Idol, Cursed Tome) resolve instead of looping and the
shop supports real buys/removes (gold/sold-out gated by the sim).
"""

import numpy as np

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts
from schemas.combat_state import (
    normalize_sim, normalize_sim_overworld, normalize_sim_global, sim_phase,
    combat_select_options, canonical_sim_potion,
)

from . import actions
from .boss_value import context_terminal_target
from .combat_value import combat_terminal_target, combat_terminal_outcome
from . import combat_env
from . import encode
from . import factories
from . import game_actions as ga
from . import spiremap

# gc.outcome is the module-level GameOutcome enum (UNDECIDED / PLAYER_VICTORY /
# PLAYER_LOSS). Unlike the combat-side Outcome enum (see combat_env), here we
# compare against the actual enum members, which is safe.
_GC_UNDECIDED = sts.GameOutcome.UNDECIDED

# campfire option name -> sim index (mirror of game_actions.CAMPFIRE_OPTIONS)
_CAMPFIRE = ga.CAMPFIRE_OPTIONS
# inverse: sim campfire index -> option name (0..5; 6=skip handled as proceed)
_CAMP_NAME_BY_IDX = {idx: name for name, idx in _CAMPFIRE.items()}

# sim InputState int for an in-combat card-select sub-screen (Armaments/Headbutt/
# Gamble/...). When the live bc is in this state, the combat canonical carries the
# select block so the obs exposes select_context/pick_remaining (obs_design §4.1).
_CARD_SELECT_INPUT = 2


# The exact encounter-name strings create_battle accepts. Most are the title-
# cased enum tail ("THREE_SENTRIES" -> "Three Sentries"), but the event-combat
# encounters (MASKED_BANDITS_EVENT, MUSHROOMS_EVENT, MYSTERIOUS_SPHERE_EVENT,
# LAGAVULIN_EVENT, COLOSSEUM_EVENT_*) are accepted ONLY in raw enum form. We
# resolve against this set so an unmatched name never silently builds an INVALID
# monster group (createMonsters' `case INVALID: assert(false)` -> native abort).
_ENCOUNTER_NAMES = frozenset(sts.encounter_names())


def _tail(gc):
    return gc.screen_state_name.split(".")[-1]


def _reward_card_count(gc):
    try:
        return int(gc.get_reward_info()["card_count"])
    except Exception:
        return 0


def _has_card_rewards(gc):
    return (
        gc.outcome == _GC_UNDECIDED
        and _tail(gc) == "REWARDS"
        and _reward_card_count(gc) > 0
    )


def _card_reward_state_error(gc, action):
    return RuntimeError(
        "card reward action in invalid GameContext state: "
        f"action={action!r} screen={gc.screen_state_name} "
        f"outcome={gc.outcome} card_count={_reward_card_count(gc)} "
        f"act={gc.act} floor={gc.floor_num}"
    )


class SimGameEnv:
    """Whole-run environment over the sim.

    reset(seed) -> drive to the first decision point.
    legal_semantic_actions() -> list of game_actions tuples currently legal.
    step(action) -> (canonical, reward, terminated, info).

    The Gym adapter builds its own observation after the transition. It opts
    out of the redundant canonical return with ``emit_step_observation=False``
    and backend legal list with ``emit_step_legal=False``. It constructs its
    own action mask; terminal snapshots and transition metadata stay intact.
    """

    def __init__(self, character=None, ascension=0, global_map_view=False,
                 interactive_reasoning=False,
                 nosl_is_az_boss_transition_target=False,
                 emit_step_observation=True, emit_step_legal=True):
        if not hasattr(sts, "nosl_compact_layout") or sts.nosl_compact_layout().get("version") != 9:
            raise RuntimeError("observation v20 requires native layout v9; rebuild the simulator")
        self._character = character or sts.CharacterClass.IRONCLAD
        self._ascension = ascension
        self._global_map_view = global_map_view
        self._interactive_reasoning = bool(interactive_reasoning)
        self._emit_step_observation = bool(emit_step_observation)
        self._emit_step_legal = bool(emit_step_legal)
        self._nosl_is_az_boss_transition_target = bool(
            nosl_is_az_boss_transition_target
        )
        if self._interactive_reasoning and not hasattr(
            sts.BattleContext, "execute_with_trace"
        ):
            raise RuntimeError(
                "interactive reasoning requires a trace-enabled simulator; "
                "rebuild slaythespire with -DSTS_ENABLE_RULE_TRACE=ON"
            )
        self._last_reasoning_trace = None
        self.gc = None
        self.bc = None            # live BattleContext while in combat, else None
        self.rmap = None          # RunMap for the current act
        self._np_random = None
        self._prev_hp = None
        self._chest_relic_pending = False

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None):
        if seed is None and self._np_random is None:
            self._np_random = np.random.default_rng()
        elif seed is not None:
            self._np_random = np.random.default_rng(seed)
        game_seed = int(self._np_random.integers(0, (1 << 63) - 1))

        self.gc = sts.GameContext(self._character, game_seed, self._ascension)
        self.bc = None
        self._chest_relic_pending = False
        self._rebuild_map()
        self._prev_hp = self.gc.cur_hp
        # GameContext starts on the Neow event; that IS the first decision point.
        return self._canonical(), self._info()

    def fork_combat(self):
        """Clone only the live combat state for inference-time lookahead."""
        if self.bc is None or self.bc.is_battle_over:
            raise RuntimeError("combat state can only be forked during a live battle")
        return _CombatBranchGameEnv(self)

    def fork_run(self, *, worldline=None):
        """Independent complete branch; optional sampled hidden combat future."""
        import copy
        branch = copy.copy(self)
        branch.gc = (self.gc.clone() if worldline is None else
                     self.gc.fork_combat_for_search(int(worldline)))
        branch.bc = self.bc.clone() if self.bc is not None else None
        # RunMap's native topology is read-only; the wrapper's act/seed fields
        # are independent. The mutable simulator map was deep-copied by clone.
        branch.rmap = copy.copy(self.rmap)
        branch._np_random = copy.deepcopy(self._np_random)
        branch._last_reasoning_trace = None
        return branch

    def settle_combat_leaf(self, battle):
        """Settle a supplied terminal BC on a fresh complete continuation copy."""
        branch = self.fork_run()
        return branch, branch._settle_combat_leaf(battle)

    def _settle_combat_leaf(self, battle):
        """Settle on an owned GameContext; never consumes reset RNG or map data."""
        from .outer_return import CombatExit, strategic_reward
        if not battle.is_battle_over:
            raise ValueError('combat exit requires a terminal battle')
        before_act, before_floor = self.gc.act, self.gc.floor_num
        from .finite_curves import exit_identity, settled_exit
        curve = exit_identity(self.gc, battle) if getattr(self, '_finite_curve_objective', False) else None
        self.bc = battle.clone()
        self._settle()
        reward = strategic_reward(before_act, before_floor, self.run_stats(),
                                  done=self.is_over, censored=False)
        kind = ('terminal' if self.is_over else
                'combat' if self.bc is not None else 'upper')
        return CombatExit(reward, kind, settled_exit(curve, self.gc))

    def _rebuild_map(self):
        self.rmap = spiremap.RunMap(self.gc.seed, self._ascension, self.gc.act,
                                    burning=False)

    # ------------------------------------------------------------ observation
    def phase(self):
        """Semantic phase. 'combat' whenever a BattleContext is live; otherwise
        the gc's screen phase."""
        if self.bc is not None:
            return "combat"
        return sim_phase(self.gc)

    def _canonical(self, *, for_encoding=False):
        """Backend-agnostic state dict. In combat: the combat canonical (+phase).
        Out of combat: the overworld canonical, with map options filled in from
        the rebuilt SpireMap (the schema can't do that without depending on env)."""
        if self.bc is not None:
            d = normalize_sim(self.bc.to_dict())
            d["phase"] = "combat"
            # an in-combat card-select sub-screen (Armaments/Headbutt/Gamble/...)
            # carries the select block so the obs can say WHAT it is for and how
            # many to pick (obs_design §4.1). Combat never accumulates picks, so
            # 'selected' is empty; the screen stays phase 'combat'.
            if self.bc.input_state == _CARD_SELECT_INPUT:
                d["options"] = {"select": combat_select_options(self.bc.select_info())}
            return d
        d = normalize_sim_overworld(self.gc, for_encoding=for_encoding)
        if d["phase"] == "event" and self.gc.cur_event_name == "NEOW":
            # Keep each generated reward visible to human-facing tools.  These
            # are the actual options stored by the sim, including their costs;
            # the observation encoder ignores this display-only metadata.
            get_options = getattr(self.gc, "get_neow_options", None)
            if get_options is not None:
                d["options"]["neow"] = [dict(option) for option in get_options()]
        elif d["phase"] == "event":
            # Normal events expose legal option indices via the sim.  The native
            # bindings do not currently export localized option text/effects, so
            # the observation carries the stable action-aligned option identity:
            # current event id + legal button index.  This lets the pointer head
            # score EVENT_OPT[i] from a per-option token instead of a flat column.
            d["options"]["event_options"] = list(self.gc.legal_event_options())
        if d["phase"] == "map":
            x, y = self.gc.cur_map_node_x, self.gc.cur_map_node_y
            opts = self.gc.legal_map_options()
            d["options"] = {
                "nodes": [(x2, "BOSS" if y == spiremap.MAX_Y else
                           self.rmap.room_type(x2, (y + 1) if y >= 0 else 0))
                          for x2 in opts]
            }
            if self._global_map_view:
                d["options"]["full_map"] = self.rmap.full_view(x, y, next_columns=opts)
        return d

    def _info(self):
        return {"phase": self.phase(), "legal": self.legal_semantic_actions()}

    def _step_info(self):
        # The Gym adapter builds its own (possibly autoregressive) legal mask.
        # Preserve reset/public backend defaults and all transition metadata.
        return self._info() if self._emit_step_legal else {"phase": self.phase()}

    def select_descriptor(self):
        """The live combat card-select 'select' block (context/pick_max/multi/
        cards_in_hand), or None when not on a combat card-select sub-screen. The
        thin layer reads this to drive the MULTI (Exhaust Many / Gamble) toggle+
        confirm machine — a RealGameEnv would expose the same. Overworld grids
        accumulate in the sim itself and are NOT reported here (they go through
        the SELECT verb / get_select_cards path)."""
        if self.bc is not None and self.bc.input_state == _CARD_SELECT_INPUT:
            return combat_select_options(self.bc.select_info())
        return None

    # --------------------------------------------------------------- obs
    def obs_canonical(self):
        """The assembled OBS canonical: the combat-or-overworld block from
        _canonical(), plus the run-global block (present every phase, §3/§4).
        This is what encode.encode() consumes."""
        return self._observation_canonical(for_encoding=False)

    def encoding_canonical(self):
        """Same tensor inputs, without unused differential-snapshot inventory."""
        return self._observation_canonical(for_encoding=True)

    def _observation_canonical(self, *, for_encoding):
        d = self._canonical(for_encoding=for_encoding)
        d["global"] = normalize_sim_global(self.gc, combat=self.bc is not None,
                                           for_encoding=for_encoding)
        if self.bc is not None:
            d["global"]["potions"] = [
                canonical_sim_potion(potion) for potion in self.bc.potions
            ]
        return d

    def obs(self):
        """The structured Dict observation (encode v2) for the current state."""
        return encode.encode(
            self.encoding_canonical(), global_map_view=self._global_map_view
        )

    @staticmethod
    def observation_space(global_map_view=False):
        """The gymnasium Dict space (shared with SimCombatEnv via encode)."""
        return encode.obs_space(global_map_view=global_map_view)

    # --------------------------------------------------------------- terminal
    @property
    def is_over(self):
        """Run ended: player died, or we cleared the act-3 boss into act 4 (we
        stop at the end of act 3, before the act-4 heart)."""
        if self.gc.outcome != _GC_UNDECIDED:
            return True
        return self.gc.act > 3

    def run_stats(self):
        """Ground-truth run-level diagnostics for logging (NOT obs/reward): the
        game owns these, so they ride in info rather than being decoded back out
        of the scaled observation. outcome is the module-level GameOutcome name
        (UNDECIDED / PLAYER_VICTORY / PLAYER_LOSS)."""
        return {
            "seed": int(self.gc.seed),
            "outcome": self.gc.outcome.name,
            "act": self.gc.act,
            "floor": self.gc.floor_num,
            "hp": self.gc.cur_hp,
            "deck_size": len(self.gc.deck_ids()),
        }


    # --------------------------------------------------- legal semantic actions
    def legal_semantic_actions(self):
        if self.is_over:
            return []
        if self.bc is not None:
            if int(self.bc.input_state) == 2 and not self.select_descriptor()["multi"]:
                return [ga.select(i) for i, _ in enumerate(self.select_descriptor()["candidates"])]
            # combat: reuse the verified 71-id mask, wrap each set id as COMBAT.
            mask = combat_env.legal_mask(self.bc)
            acts = [ga.combat(i) for i in np.flatnonzero(mask)]
            # Potions are NOT part of the 71-id combat space (the sim's
            # legal_actions omits them); they ride the dedicated USE_POTION verb so
            # the thin layer can give them their own reserved id segment. The sim's
            # own legal_potions() is the authority on which held slots are drinkable
            # now and whether each needs a target (Fire/Fear/Poison/Weak). For a
            # targeted potion we enumerate ONE action per targetable monster, exactly
            # like the targeted card-play path (_play_targeting in action_space.py),
            # so the autoregressive PICK_TARGET substep is uniform across cards and
            # potions. An untargeted potion is a single (slot, None) action.
            for pot in self.bc.legal_potions():
                slot = pot["slot"]
                if pot["requires_target"]:
                    # one action per sim-accepted target (already excludes dead/
                    # untargetable monsters), exactly like a targeted card play.
                    for t in pot["targets"]:
                        acts.append(ga.use_potion(slot, t))
                else:
                    acts.append(ga.use_potion(slot, None))
            return acts

        phase = sim_phase(self.gc)
        # Held potions are usable in MOST overworld phases too (Blood/Entropic
        # Brew/Fruit Juice drink out of combat). The sim's own validator
        # (legal_overworld_potions -> GameAction::isValidPotionAction) is the
        # authority on which slots are drinkable now — it gates on the potion type
        # (and blocks the WE_MEET_AGAIN event), NOT on the specific screen, so we
        # offer the same potion actions on every overworld screen and let the sim
        # decide. Overworld potions never need a target (no monsters), so they are
        # always single-step (target None). We PREPEND so a phase that returns []
        # (unknown/blocked screen) still surfaces the potion if the sim allows it.
        pots = [ga.use_potion(p["slot"], None)
                for p in self.gc.legal_overworld_potions()]

        if phase == "event":
            # Neow is an event screen too; its options are the 4 Neow rewards.
            if self.gc.cur_event_name == "NEOW":
                return [ga.neow(i) for i in range(4)] + pots
            # The sim's own action enumerator gives the phase/relic/gold-aware
            # valid option indices for THIS event (handles multi-phase events like
            # Golden Idol / Cursed Tome that the old option-0 stub looped on). May
            # be empty for unsupported events (MATCH_AND_KEEP / BONFIRE_SPIRITS);
            # the thin layer then has no legal action and the run stalls there
            # honestly rather than asserting in the sim.
            return [ga.event(i) for i in self.gc.legal_event_options()] + pots
        if phase == "map":
            return [ga.map_node(x2) for x2 in self.gc.legal_map_options()] + pots
        if phase == "combat_reward":
            info = self.gc.get_reward_info()
            legal_reward_options = {
                (str(kind), int(reward_idx))
                for kind, reward_idx, _item_idx in self.gc.legal_reward_options()
            }
            acts = []
            acts += [ga.claim(ga.CLAIM_GOLD, i) for i in range(len(info["gold"]))
                     if ("gold", i) in legal_reward_options]
            acts += [ga.claim(ga.CLAIM_RELIC, i) for i in range(len(info["relics"]))
                     if ("relic", i) in legal_reward_options]
            acts += [ga.claim(ga.CLAIM_POTION, i) for i in range(len(info["potions"]))
                     if ("potion", i) in legal_reward_options]
            key_actions = [
                (reward_idx, item_idx)
                for kind, reward_idx, item_idx in self.gc.legal_reward_options()
                if str(kind) == "key"
            ]
            if len(key_actions) > 1:
                raise RuntimeError(
                    f"simulator exposed multiple simultaneous key actions: {key_actions!r}"
                )
            if key_actions:
                if len(info["keys"]) != 1:
                    raise RuntimeError(
                        "simulator key action/details mismatch: "
                        f"actions={key_actions!r} keys={list(info['keys'])!r}"
                    )
                acts.append(ga.claim(ga.CLAIM_KEY, 0))
            if int(info["card_count"]) and _has_card_rewards(self.gc):
                n = len(self.gc.get_card_reward())
                acts += [ga.reward_card(j) for j in range(n)]
                acts.append(ga.skip_cards())
            acts.append(ga.proceed())
            return acts + pots
        if phase == "card_select":
            return [ga.select(k) for k in range(len(self.gc.get_select_cards()))] + pots
        if phase == "rest":
            # The sim enumerates the campfire options actually available here,
            # gated by relics/keys (rest/smith always; recall hidden once the ruby
            # key is held; lift only with Girya; dig only with Shovel; etc.).
            # Indices 0-5 are the named CAMPFIRE_OPTIONS; 6 (SKIP) == proceed.
            acts = []
            for i in self.gc.legal_campfire_options():
                if i == 6:
                    acts.append(ga.proceed())
                else:
                    acts.append(ga.campfire(_CAMP_NAME_BY_IDX[i]))
            return acts + pots
        if phase == "chest":
            return [ga.chest(True), ga.chest(False)] + pots
        if phase == "boss_relic":
            return [ga.boss_relic(i) for i in range(3)] + pots
        if phase == "shop":
            # The sim enumerates the affordable, in-stock shop actions as
            # (kind, idx) pairs (gold/sold-out/Sozu/potion-capacity gated).
            # kind maps 1:1 to the semantic shop verbs; "skip" is leaving the
            # shop.
            acts = []
            for kind, idx in self.gc.legal_shop_options():
                if kind == "skip":
                    acts.append(ga.shop_proceed())
                elif kind == "remove":
                    acts.append(ga.shop_remove())
                else:  # "card" / "relic" / "potion"
                    acts.append(ga.shop_buy(kind, idx))
            return acts + pots
        # unknown screen: nothing we can legally do EXCEPT maybe drink a potion.
        return pots

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.is_over:
            raise RuntimeError("run is over; call reset()")
        self._last_reasoning_trace = None
        reasoning_before = (
            self.obs_canonical() if self._interactive_reasoning else None
        )
        executed = False
        if self._interactive_reasoning and self.bc is not None:
            executed, self._last_reasoning_trace = self._apply_with_reasoning_trace(
                action
            )
        if not executed:
            self._apply(action)
        terminal_combat_state = self._winning_combat_value_state()
        terminal_combat_target = self._nosl_value_terminal_target()
        terminal_combat_outcome = self._nosl_value_terminal_outcome()
        from .finite_curves import exit_identity, settled_exit
        curve = exit_identity(self.gc, self.bc) if getattr(self, '_finite_curve_objective', False) else None
        self._settle()  # pump through any auto-resolving screens to next decision

        if self._interactive_reasoning and self._last_reasoning_trace is None:
            self._last_reasoning_trace = {
                "version": 1,
                "precision": "semantic_only",
                "semantic_action": tuple(action),
                "action_card": self._semantic_action_card(
                    action, reasoning_before
                ),
                "events": [{"type": "ACTION_COMPLETE"}],
            }

        terminated = self.is_over
        self._prev_hp = self.gc.cur_hp
        reward = 0.0
        if terminated:
            cleared_run = self.gc.outcome == _GC_UNDECIDED and self.gc.act > 3
            won = self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY
            reward = 1.0 if (cleared_run or won) else -1.0
        info = self._step_info()
        # One event per treasure opportunity, not per reward-screen click.
        if curve is not None:
            info['finite_curve_exit'] = settled_exit(curve, self.gc)
        # Journal replay reconstructs pending offers across checkpoint boundaries.
        if action[0] == ga.CHEST:
            info["chest_opened"] = bool(action[1])
            self._chest_relic_pending = bool(
                action[1] and self.gc.get_reward_info().get("relics")
            )
        elif self._chest_relic_pending:
            if action[0] == ga.CLAIM and action[1] == ga.CLAIM_RELIC:
                info["chest_relic_taken"] = True
                self._chest_relic_pending = False
            elif action[0] == ga.PROCEED:
                info["chest_relic_taken"] = False
                self._chest_relic_pending = False
        if terminal_combat_state is not None:
            info["nosl_value_terminal_state"] = terminal_combat_state
        if terminal_combat_target is not None:
            info["nosl_value_terminal_target"] = terminal_combat_target
            info["nosl_value_terminal_outcome"] = terminal_combat_outcome
        if self._last_reasoning_trace is not None:
            info["reasoning_trace"] = self._last_reasoning_trace
        observation = self._canonical() if self._emit_step_observation else None
        return observation, reward, terminated, info

    def step_native_combat(self, action_bits: int):
        """Apply one complete bit-packed native combat Action.

        This is used only by the C++ NOSL adapter.  Python semantic combat
        actions are 71-id indices and must continue through ``step``; decoding
        a native bit pattern as such an index would silently select the wrong
        card.  Keep the same settle, terminal reward, and info contract as
        ``step`` while bypassing that decoder.
        """
        if self.is_over or self.bc is None:
            raise RuntimeError("native combat action requires a live combat")
        self._last_reasoning_trace = None
        self.bc.execute(sts.Action(int(action_bits)))
        terminal_combat_state = self._winning_combat_value_state()
        terminal_combat_target = self._nosl_value_terminal_target()
        terminal_combat_outcome = self._nosl_value_terminal_outcome()
        from .finite_curves import exit_identity, settled_exit
        curve = exit_identity(self.gc, self.bc) if getattr(self, '_finite_curve_objective', False) else None
        self._settle()
        terminated = self.is_over
        self._prev_hp = self.gc.cur_hp
        reward = 0.0
        if terminated:
            cleared_run = self.gc.outcome == _GC_UNDECIDED and self.gc.act > 3
            won = self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY
            reward = 1.0 if (cleared_run or won) else -1.0
        info = self._step_info()
        if terminal_combat_state is not None:
            info["nosl_value_terminal_state"] = terminal_combat_state
        if curve is not None:
            info['finite_curve_exit'] = settled_exit(curve, self.gc)
        if terminal_combat_target is not None:
            info["nosl_value_terminal_target"] = terminal_combat_target
            info["nosl_value_terminal_outcome"] = terminal_combat_outcome
        observation = self._canonical() if self._emit_step_observation else None
        return observation, reward, terminated, info

    def preview_native_end_turn_boundary(self, action_bits: int):
        """Return the stable value boundary for a native End Turn action.

        The simulator performs this on a private clone, so the live combat and
        all of its RNG streams remain untouched.  Non-End-Turn and terminal
        transitions have no trainable boundary and return ``None``.
        """
        if self.is_over or self.bc is None:
            return None
        preview = getattr(self.bc, "preview_end_turn_boundary", None)
        if preview is None:
            return None
        return preview(int(action_bits))

    def _winning_combat_value_state(self):
        """Snapshot a just-won combat before settlement removes the battle."""
        if self.bc is None or not self.bc.is_battle_over:
            return None
        state = self.bc.to_dict()
        if state.get("outcome") != combat_env._OUTCOME_VICTORY:
            return None
        return state

    def _nosl_value_terminal_target(self):
        """Capture fight-local HP before any post-combat settlement.

        The historical boss objective remains explicit for old experiments.
        New standard targets exclude all post-combat healing.
        """
        if self.bc is None or not self.bc.is_battle_over:
            return None
        if self._nosl_is_az_boss_transition_target:
            return context_terminal_target(enabled=True, gc=self.gc, bc=self.bc)
        return combat_terminal_target(self.bc)

    def _nosl_value_terminal_outcome(self):
        if self.bc is None or not self.bc.is_battle_over:
            return None
        return combat_terminal_outcome(self.bc)

    @staticmethod
    def _semantic_action_card(action, before):
        """Card surface selected by a non-combat semantic action, if any."""
        if before is None:
            return None
        verb = action[0]
        options = before.get("options") or {}
        try:
            if verb == ga.REWARD_CARD:
                card = options["cards"][int(action[1])]
            elif verb == ga.SELECT:
                card = options["cards"][int(action[1])]
            elif verb == ga.SHOP_BUY and action[1] == ga.SHOP_CARD:
                card = options["shop_cards"][int(action[2])]
            else:
                return None
        except (IndexError, KeyError, TypeError):
            return None
        if isinstance(card, dict):
            return dict(card)
        if len(card) < 2:
            return None
        result = {"id": str(card[0]), "up": int(card[1])}
        if len(card) >= 3 and int(card[2]) >= 0:
            # Shop tuples carry price rather than runtime energy cost.  Do not
            # serialize it as RUNTIME_COST.
            result["price"] = int(card[2])
        return result

    def _apply_with_reasoning_trace(self, action):
        """Execute a combat action and retain atomic simulator snapshots.

        Reasoning training is deliberately tied to the trace-enabled simulator.
        A coarse before/after fallback would make the transcript look like a
        calculation while hiding the real execution order, so it is rejected.
        """
        verb = action[0]
        sts_action = None
        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            sts_action = combat_env.to_sts_action(kind, self.bc)
        elif verb == ga.SELECT and self.bc is not None:
            candidate = self.select_descriptor()["candidates"][int(action[1])]
            sts_action = sts.Action(sts.ActionType.SINGLE_CARD_SELECT, candidate["native_index"])
        elif verb == ga.MULTI_SELECT:
            mask = 0
            for hand_index in action[1]:
                mask |= 1 << int(hand_index)
            sts_action = sts.Action(sts.ActionType.MULTI_CARD_SELECT, mask)
        elif verb == ga.USE_POTION:
            slot, target = action[1], action[2]
            sts_action = sts.Action(
                sts.ActionType.POTION,
                int(slot),
                0 if target is None else int(target),
            )
        if sts_action is None:
            return False, None

        before = normalize_sim(self.bc.to_dict())
        action_card = None
        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            if kind[0] == actions.KIND_PLAY:
                hand_index = int(kind[1])
                if 0 <= hand_index < len(before.get("hand", ())):
                    action_card = dict(before["hand"][hand_index])

        if not hasattr(self.bc, "execute_with_trace"):
            # This should already have been caught in __init__, but keeping the
            # guard here protects custom backend factories and forked envs.
            raise RuntimeError(
                "interactive reasoning requires BattleContext.execute_with_trace()"
            )
        raw_trace = self.bc.execute_with_trace(sts_action)
        snapshots = [normalize_sim(raw_trace["before"])]
        snapshots.extend(normalize_sim(raw) for raw in raw_trace["steps"])
        precision = "atomic"
        return True, {
            "version": 1,
            "precision": precision,
            "semantic_action": tuple(action),
            "action_card": action_card,
            "snapshots": snapshots,
        }

    def _apply(self, action):
        """Dispatch one semantic action against the sim. Mirrors replay.Driver
        but driven by the chosen action rather than a log."""
        verb = action[0]
        gc = self.gc

        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            self.bc.execute(combat_env.to_sts_action(kind, self.bc))
            return
        if verb == ga.NEOW:
            gc.choose_neow_option(action[1]); return
        if verb == ga.EVENT:
            gc.choose_event_option(action[1]); return
        if verb == ga.MAP:
            gc.transition_to_map_node(action[1])
            # battle start is handled centrally in _settle (covers event combats
            # too), so we don't branch on the destination room type here.
            return
        if verb == ga.CLAIM:
            kind, idx = action[1], action[2]
            if kind == ga.CLAIM_GOLD:    gc.claim_gold(idx)
            elif kind == ga.CLAIM_RELIC: gc.claim_relic(idx)
            elif kind == ga.CLAIM_POTION: gc.claim_potion(idx)
            elif kind == ga.CLAIM_KEY:
                if idx != 0:
                    raise ValueError(f"key reward occurrence must be zero, got {idx}")
                gc.claim_key()
            else:
                raise ValueError(f"unknown reward claim kind: {kind!r}")
            return
        if verb == ga.REWARD_CARD:
            if not _has_card_rewards(gc):
                raise _card_reward_state_error(gc, action)
            cards = gc.get_card_reward()
            if action[1] >= len(cards):
                raise _card_reward_state_error(gc, action)
            gc.pick_reward_card(cards[action[1]]); return
        if verb == ga.SKIP_CARDS:
            if not _has_card_rewards(gc):
                raise _card_reward_state_error(gc, action)
            gc.skip_reward_cards(); return
        if verb == ga.SELECT:
            if self.bc is not None:
                candidate = self.select_descriptor()["candidates"][int(action[1])]
                self.bc.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, candidate["native_index"]))
            else:
                gc.choose_select_card_option(action[1])
            return
        if verb == ga.MULTI_SELECT:
            # combat-only: resolve an Exhaust Many / Gamble screen in one shot. The
            # sim reads the low 10 bits of the Action as a hand-index bitset
            # (getSelectedIdxs() -> chooseExhaust/GambleCards), so we OR the chosen
            # hand indices into a mask. An empty pick (mask 0) is valid and just
            # dismisses the screen. The thick layer never accumulates: the thin
            # layer hands over the complete index set on confirm.
            mask = 0
            for h in action[1]:
                mask |= (1 << int(h))
            self.bc.execute(sts.Action(sts.ActionType.MULTI_CARD_SELECT, mask))
            return
        if verb == ga.CAMPFIRE:
            gc.choose_campfire_option(_CAMPFIRE[action[1]]); return
        if verb == ga.CHEST:
            gc.choose_treasure_room(action[1]); return
        if verb == ga.BOSS_RELIC:
            gc.choose_boss_relic(action[1]); return
        if verb == ga.PROCEED:
            if self.phase() == 'rest':
                if 6 not in gc.legal_campfire_options():
                    raise ValueError('campfire skip is not legal')
                gc.choose_campfire_option(6)
            else:
                gc.reward_proceed()
            return
        if verb == ga.SHOP_PROCEED:
            gc.shop_proceed(); return
        if verb == ga.SHOP_BUY:
            kind, idx = action[1], action[2]
            if kind == ga.SHOP_CARD:    gc.shop_buy_card(idx)
            elif kind == ga.SHOP_RELIC: gc.shop_buy_relic(idx)
            elif kind == ga.SHOP_POTION: gc.shop_buy_potion(idx)
            else: raise ValueError(f"bad shop buy kind: {kind!r}")
            return
        if verb == ga.SHOP_REMOVE:
            gc.shop_remove(); return
        if verb == ga.USE_POTION:
            # (USE_POTION, slot, target). In combat we go through the SAME
            # bc.execute(Action) path as card plays so the action queue is pumped
            # (executeActions) and the input state transitions correctly — calling
            # bc.drinkPotion directly would enqueue the effect but never run it. A
            # POTION Action's targetIdx is the monster for a targeted potion
            # (Fire/Fear/Poison/Weak, resolved by the thin layer's PICK_TARGET
            # substep); untargeted potions ignore it, so we pass 0. Out of combat
            # the gc owns the belt and nothing is targeted, so we drink by slot via
            # the sim's GameContext::drinkPotionAtIdx (bound as gc.drink_potion).
            slot, target = action[1], action[2]
            if self.bc is not None:
                t = 0 if target is None else target
                self.bc.execute(sts.Action(sts.ActionType.POTION, slot, t))
            else:
                gc.drink_potion(slot)
            return
        raise ValueError(f"unhandled semantic action: {action!r}")

    def _start_battle(self):
        # The native boundary also rejects unresolved names. Validate here first
        # to include the whole-run floor/act/room in the error without attempting
        # construction or depending on the native build's assertion settings.
        name = self._encounter_name()
        if name not in _ENCOUNTER_NAMES:
            raise RuntimeError(
                f"unresolved encounter {name!r} (gc.encounter={self.gc.encounter}) "
                f"at floor={self.gc.floor_num} act={self.gc.act} "
                f"room={self.gc.cur_room}; refusing create_battle."
            )
        self.bc = sts.create_battle(self.gc, name)

    def _encounter_name(self):
        # Title-case the enum tail ("THREE_SENTRIES" -> "Three Sentries"); the
        # event-combat encounters are accepted only in raw enum form, so fall
        # back to the raw tail when the title-cased name isn't a known encounter.
        tail = str(self.gc.encounter).split(".")[-1]
        name = " ".join(w.capitalize() for w in tail.split("_"))
        return name if name in _ENCOUNTER_NAMES else tail

    def _settle(self):
        """After an action, advance through states that don't need an agent
        decision: start any battle we transitioned into (from the map OR from a
        combat event like Mushrooms / Dead Adventurer), end a finished combat
        (exit_battle opens rewards), and detect an act transition to rebuild the
        map. Stops at the next real decision."""
        # entered a battle by any path (map node or event) -> create the bc.
        if self.bc is None and _tail(self.gc) == "BATTLE":
            self._start_battle()
        if self.bc is not None and self.bc.is_battle_over:
            self.bc.exit_battle(self.gc)
            self.bc = None
            if not self.is_over and _tail(self.gc) == "BATTLE":
                self._start_battle()
        # act bumped (cleared the boss) -> rebuild the map for the new act.
        if self.gc.act != self.rmap.act and not self.is_over:
            self._rebuild_map()

    # ----------------------------------------------------------------- render
    def render(self):
        from . import render
        if self.bc is not None:
            return render.render_state(normalize_sim(self.bc.to_dict()))
        return render.render_overworld(self._canonical())

    def describe_action(self, action):
        canon = normalize_sim(self.bc.to_dict()) if self.bc is not None else self._canonical()
        return ga.describe(action, canon)


class _CombatBranchGameEnv(SimGameEnv):
    """A combat-only branch that never writes its result back to GameContext."""

    def __init__(self, source):
        self.__dict__ = source.__dict__.copy()
        self.bc = source.bc.clone()

    @property
    def is_over(self):
        return self.bc.is_battle_over

    def fork_combat(self):
        if self.is_over:
            raise RuntimeError("cannot fork a terminal combat branch")
        return _CombatBranchGameEnv(self)

    def step(self, action):
        if self.is_over:
            raise RuntimeError("combat branch is terminal")
        if action[0] not in (ga.COMBAT, ga.MULTI_SELECT, ga.USE_POTION, ga.SELECT):
            raise ValueError(f"combat branch received overworld action: {action!r}")
        self._apply(action)
        terminated = self.bc.is_battle_over
        info = self._step_info()
        target = self._nosl_value_terminal_target()
        if target is not None:
            info["nosl_value_terminal_target"] = target
            info["nosl_value_terminal_outcome"] = self._nosl_value_terminal_outcome()
        observation = self._canonical() if self._emit_step_observation else None
        return observation, 0.0, terminated, info

    def step_native_combat(self, action_bits: int):
        """Finish native actions inside this branch without settling the run.

        The branch shares read-only run metadata with its parent. Calling the
        whole-run implementation would exit_battle into that parent's gc and
        clear bc, violating both isolation and pre-settlement HP semantics.
        """
        if self.is_over:
            raise RuntimeError("combat branch is terminal")
        self.bc.execute(sts.Action(int(action_bits)))
        info = self._step_info()
        terminal = self._winning_combat_value_state()
        if terminal is not None:
            info['nosl_value_terminal_state'] = terminal
        target = self._nosl_value_terminal_target()
        if target is not None:
            info["nosl_value_terminal_target"] = target
            info["nosl_value_terminal_outcome"] = self._nosl_value_terminal_outcome()
        observation = self._canonical() if self._emit_step_observation else None
        return observation, 0.0, self.bc.is_battle_over, info

    def step_fast(self, action):
        """Apply a combat semantic action without rebuilding observations.

        Search rollouts only need the terminal flag and inspect the branch's
        BattleContext directly for terminal HP.  The regular ``step`` method is
        intentionally unchanged for Gym callers that need canonical state and
        info.  This method is restricted to combat branches, so ``_apply``
        dispatches directly to the native BattleContext and never needs the
        whole-run settle/observation path.
        """
        if self.is_over:
            raise RuntimeError("combat branch is terminal")
        if action[0] not in (ga.COMBAT, ga.MULTI_SELECT, ga.USE_POTION, ga.SELECT):
            raise ValueError(f"combat branch received overworld action: {action!r}")
        self._apply(action)
        return bool(self.bc.is_battle_over)
