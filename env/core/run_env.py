"""SlayTheSpireEnv: the THIN, backend-agnostic Gymnasium env for a whole run.

This is the model's only interface (env-two-layer-abstraction): it wraps a THICK
backend env (`SimGameEnv` today, a future `RealGameEnv`) that speaks variable-
length *semantic actions*, and exposes the RL surface the policy wants — a fixed
`Discrete(N_ACTIONS)` action space + a per-step legal mask, and the structured
Dict observation. OBS_VERSION / ACTION_VERSION discipline lives HERE, at the one
seam a checkpoint is tied to.

It is the consumer of `action_space.py`'s autoregressive codec: a single env
`step(id)` advances a tiny sub-state machine. A targeted card play spans TWO
steps —

    step PLAY_CARD[h]   -> codec returns (None, pending=('play', h))   # no backend call
    step PICK_TARGET[t] -> codec returns ((COMBAT, encode_play(h,t)), None)  # NOW apply

so the thick layer (and the sim) only ever sees a COMPLETE semantic action. The
in-between step yields the same obs with reward 0 and the next mask (only the
legal targets lit), and `obs['arg_step']` flips to `pick_target` so the model
knows why the menu changed.

A second multi-step shape is the combat MULTI card-select (Exhaust Many / Gamble,
action_design §1.1): one Action resolves the WHOLE screen with a hand-index
bitset, so the thin env SELECTS hand cards monotonically and CONFIRMS once —

    step PICK_CARD[h]      -> add hand card h to the pending set (no backend call;
                              arg_step = pick_hand; selected cards become illegal)
    step CONFIRM_SELECT    -> emit (MULTI_SELECT, chosen_idxs) to the backend

The pending set rides in obs as select_hand_mask (OBS_VERSION 6) so the
feed-forward policy sees what it has already selected (mask != obs discipline,
obs_design §0.5). The thin env reconciles its accumulator against the live screen
each step via the backend's select_descriptor(), so leaving the screen (the sim
dismissed it after confirm) clears the set.

Masking lib-agnostic, exactly like SimCombatEnv: we expose `action_masks()` and
let rl/train.py bridge it to sb3-contrib via ActionMasker.
"""

import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import _BUILD  # noqa: F401  (sys.path bootstrap for slaythespire/schemas)
from schemas.combat_state import (
    combat_select_options,
    normalize_sim,
    normalize_sim_global,
)

from . import action_space as A
from . import actions
from . import combat_env
from . import encode
from . import game_actions as ga
from .game_env import SimGameEnv


_UNCACHED = object()


class SlayTheSpireEnv(gym.Env):
    """Thin gym.Env over a thick whole-run backend.

    backend_factory() -> a fresh thick env exposing reset/phase/obs_canonical/
    legal_semantic_actions/step/is_over (the SimGameEnv surface). Defaults to a
    sim-backed SimGameEnv; pass your own to swap backends without touching this
    layer."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, backend_factory=None, character=None, ascension=0,
                 render_mode=None, global_map_view=False,
                 card_language=False, combat_language=False,
                 deck_size_observation=False, neow_observation=False,
                 event_option_observation=False,
                 interactive_reasoning=False):
        super().__init__()
        self._backend_factory = backend_factory or (
            lambda: SimGameEnv(
                character=character,
                ascension=ascension,
                global_map_view=global_map_view,
                interactive_reasoning=interactive_reasoning,
            )
        )
        self._global_map_view = global_map_view
        self._card_language = card_language
        self._combat_language = combat_language
        self._deck_size_observation = deck_size_observation
        self._neow_observation = neow_observation
        self._event_option_observation = event_option_observation
        self._interactive_reasoning = bool(interactive_reasoning)
        self.render_mode = render_mode
        self.thick = None
        # autoregressive sub-state: None at ROOT, else ('play', hand_idx) awaiting
        # a PICK_TARGET. Lives in the thin layer only — the backend never sees it.
        self._pending = None
        # combat MULTI card-select accumulator (Exhaust Many / Gamble): the set of
        # selected hand indices awaiting CONFIRM_SELECT. None when not on a MULTI
        # screen. Thin-layer only; reconciled against the live screen each step.
        self._multi = None
        self._cached_phase = _UNCACHED
        self._cached_legal = _UNCACHED
        self._cached_multi_desc = _UNCACHED
        self._cached_combat_mask = _UNCACHED
        self._cached_combat_potions = _UNCACHED
        self._cached_action_mask = None
        self._nosl_is_az_session = None
        # Native C++ NOSL returns a complete simulator action. Keep it beside
        # the thin root id until the collector submits that id through step().
        self._nosl_cpp_pending = None
        self._nosl_cpp_az_session = None
        # Completed session retained for the dashboard's lazy search-tree
        # inspection. It is replaced on the next combat decision/reset.
        self._nosl_cpp_last_session = None
        self._nosl_cpp_global_template = None
        self._nosl_cpp_compact_observation = False
        self._nosl_cpp_stochastic = False
        self._nosl_cpp_state_nonce = 0
        self._nosl_cpp_decision_index = 0

        self.observation_space = encode.obs_space(
            global_map_view=global_map_view,
            card_language=card_language,
            combat_language=combat_language,
            deck_size_observation=deck_size_observation,
            neow_observation=neow_observation,
            event_option_observation=event_option_observation,
            interactive_reasoning=interactive_reasoning,
        )
        self.action_space = spaces.Discrete(A.N_ACTIONS)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.thick = self._backend_factory()
        self.thick.reset(seed=seed)
        self._pending = None
        self._multi = None
        self._nosl_is_az_session = None
        self._nosl_cpp_pending = None
        self._nosl_cpp_az_session = None
        self._nosl_cpp_last_session = None
        self._nosl_cpp_global_template = None
        self._nosl_cpp_compact_observation = False
        self._nosl_cpp_stochastic = False
        self._nosl_cpp_state_nonce = 0
        self._nosl_cpp_decision_index = 0
        self._invalidate_backend_cache()
        return self._obs(), self._info()

    def fork_combat(self):
        """Return an isolated thin-env branch for combat-only lookahead."""
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("combat search requires a live combat state")
        branch = copy.copy(self)
        branch.thick = self.thick.fork_combat()
        branch._pending = self._pending
        branch._multi = set(self._multi) if self._multi is not None else None
        branch._nosl_is_az_session = None
        branch._nosl_cpp_pending = None
        branch._nosl_cpp_az_session = None
        branch._nosl_cpp_last_session = None
        branch._nosl_cpp_global_template = None
        branch._nosl_cpp_compact_observation = False
        branch._nosl_cpp_stochastic = False
        branch._nosl_cpp_state_nonce = 0
        branch._nosl_cpp_decision_index = 0
        branch._invalidate_backend_cache()
        return branch

    def combat_action_advances(self, action_id):
        """Whether a legal thin action completes one simulator decision."""
        aid = int(action_id)
        desc = self._multi_desc()
        if desc is not None:
            if aid not in set(A.multi_legal_ids(desc, self._multi)):
                raise ValueError(f"illegal combat search action: {aid}")
            return aid == A._CONFIRM

        legal = self._legal()
        phase = self._phase()
        if phase != "combat" or aid not in set(
            A.legal_ids(legal, self._pending, phase)
        ):
            raise ValueError(f"illegal combat search action: {aid}")
        semantic, _pending = A.resolve(aid, legal, self._pending, phase)
        return semantic is not None

    def _invalidate_thin_cache(self):
        self._cached_action_mask = None

    def _invalidate_backend_cache(self):
        self._cached_phase = _UNCACHED
        self._cached_legal = _UNCACHED
        self._cached_multi_desc = _UNCACHED
        self._cached_combat_mask = _UNCACHED
        self._cached_combat_potions = _UNCACHED
        self._invalidate_thin_cache()

    def _phase(self):
        if self._cached_phase is _UNCACHED:
            self._cached_phase = self.thick.phase()
        return self._cached_phase

    def _legal(self):
        if self._cached_legal is _UNCACHED:
            self._cached_legal = tuple(self.thick.legal_semantic_actions())
        return self._cached_legal

    def _combat_native_mask(self):
        """Return the cached native combat mask for search branches."""
        if self._cached_combat_mask is _UNCACHED:
            self._cached_combat_mask = np.asarray(
                combat_env.legal_mask(self.thick.bc), dtype=bool
            )
        return self._cached_combat_mask

    def _combat_potions(self):
        if self._cached_combat_potions is _UNCACHED:
            self._cached_combat_potions = tuple(self.thick.bc.legal_potions())
        return self._cached_combat_potions

    def _combat_action_mask_fast(self):
        """Build the thin mask directly from native combat ids.

        Search branches only need the fixed thin action mask.  Going through
        ``legal_semantic_actions`` first creates and wraps every native action,
        then ``legal_ids`` walks that list again.  This preserves the same
        autoregressive root/target semantics without those intermediate lists.
        """
        native = self._combat_native_mask()
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        for combat_id in np.flatnonzero(native):
            kind = actions.decode(int(combat_id))
            if kind[0] == actions.KIND_END:
                mask[A.END_TURN] = True
            elif kind[0] == actions.KIND_PLAY:
                mask[A._PLAY + int(kind[1])] = True
            elif kind[0] == actions.KIND_SELECT:
                mask[A._PICK + int(kind[1])] = True
        for potion in self._combat_potions():
            mask[A._POTION + int(potion["slot"])] = True
        return mask

    def _resolve_search_combat_action(self, aid):
        """Resolve a legal thin combat id without rebuilding semantic actions."""
        native = self._combat_native_mask()
        if self._pending is not None:
            if self._pending[0] == "play":
                target = aid - A._TARGET
                hand = int(self._pending[1])
                if 0 <= target < actions.MAX_MONSTERS:
                    native_id = actions.encode_play(hand, target)
                    if native_id < len(native) and native[native_id]:
                        return ga.combat(native_id)
                raise ValueError(f"illegal combat search action: {aid}")
            if self._pending[0] == "potion":
                target = aid - A._TARGET
                slot = int(self._pending[1])
                if 0 <= target < actions.MAX_MONSTERS:
                    for potion in self._combat_potions():
                        if (int(potion["slot"]) == slot
                                and target in potion.get("targets", ())):
                            return ga.use_potion(slot, target)
                raise ValueError(f"illegal combat search action: {aid}")

        if aid == A.END_TURN:
            if native[actions.END_TURN]:
                return ga.combat(actions.END_TURN)
            raise ValueError(f"illegal combat search action: {aid}")
        if A._PLAY <= aid < A._PLAY + A.HAND_SLOTS:
            hand = aid - A._PLAY
            untargeted = actions.encode_play(hand, None)
            if untargeted < len(native) and native[untargeted]:
                return ga.combat(untargeted)
            if any(
                native[actions.encode_play(hand, target)]
                for target in range(actions.MAX_MONSTERS)
            ):
                self._pending = ("play", hand)
                return None
            raise ValueError(f"illegal combat search action: {aid}")
        if A._POTION <= aid < A._POTION + A.P_MAX:
            slot = aid - A._POTION
            for potion in self._combat_potions():
                if int(potion["slot"]) != slot:
                    continue
                if potion.get("requires_target"):
                    self._pending = ("potion", slot)
                    return None
                return ga.use_potion(slot, None)
            raise ValueError(f"illegal combat search action: {aid}")
        if A._PICK <= aid < A._PICK + A.G_MAX:
            select = aid - A._PICK
            native_id = actions.encode_select(select)
            if native_id < len(native) and native[native_id]:
                return ga.combat(native_id)
        raise ValueError(f"illegal combat search action: {aid}")

    def _backend_step(self, semantic):
        result = self.thick.step(semantic)
        self._invalidate_backend_cache()
        return result

    def _backend_step_native_combat(self, action_bits):
        step_native = getattr(self.thick, "step_native_combat", None)
        if step_native is None:
            raise RuntimeError("native combat step requires the simulator backend")
        result = step_native(int(action_bits))
        self._invalidate_backend_cache()
        return result

    def _backend_step_fast(self, semantic):
        """Apply a combat branch action without constructing canonical state."""
        step_fast = getattr(self.thick, "step_fast", None)
        if step_fast is None:
            raise RuntimeError("search fast step requires a combat branch backend")
        terminated = bool(step_fast(semantic))
        self._invalidate_backend_cache()
        return terminated

    def _multi_desc(self):
        """The live combat MULTI card-select descriptor, or None. Also the single
        place that RECONCILES the accumulator with the screen: a MULTI screen ->
        ensure _multi is a set; anything else -> clear it (we left the screen)."""
        if self._cached_multi_desc is _UNCACHED:
            self._cached_multi_desc = self.thick.select_descriptor()
        desc = self._cached_multi_desc
        if A.is_multi_select(desc):
            if self._multi is None:
                self._multi = set()
                self._invalidate_thin_cache()
            return desc
        if self._multi is not None:
            self._multi = None
            self._invalidate_thin_cache()
        return None

    def _no_legal_action(self):
        """True when the live backend screen exposes ZERO legal thin actions and
        is not a combat MULTI select screen — a "dead screen" the policy cannot
        act on. The known cause is an event the sim's own enumerator does not
        support yet (MATCH_AND_KEEP / BONFIRE_SPIRITS -> legal_event_options
        returns []), so SimGameEnv.legal_semantic_actions() yields [] and the run
        would otherwise park there forever with an all-False mask (game_env's
        comment: "the run stalls there honestly"). Computed fresh from the live
        backend so it reflects whatever screen we are on right now.

        NOTE: only meaningful while the run is NOT over — a genuinely terminal run
        also has no legal action, but that is a `terminated`, not a dead screen."""
        if self._multi_desc() is not None:
            return False
        return not self.action_masks().any()

    def _truncate_dead(self):
        """The clean 5-tuple for truncating on a dead screen: a valid (current)
        obs, reward 0.0, terminated=False, truncated=True, with info["dead_screen"]
        set (mirrors the info["illegal"] convention) so callers can tell a dead-
        screen truncation apart from a step-cap (TimeLimit) truncation."""
        info = self._info()
        info["dead_screen"] = True
        return self._obs(), 0.0, False, True, info

    def _post_step(self, reward, terminated, transition_info=None,
                   action_id=None):
        """Assemble the 5-tuple AFTER a backend transition. If the run did not end
        but we LANDED on a dead screen (see _no_legal_action), truncate cleanly
        instead of returning a normal frame whose mask would be all-False.

        Ordering argument (why the policy never sees an all-False mask): the policy
        queries action_masks() to pick an action and THEN calls step(); so the
        dangerous moment is a sub-env that has *just transitioned into* a dead
        screen, which would be asked for a mask before its next step. By truncating
        ON the step that causes the dead landing, that step returns done=True; with
        VecMonitor autoreset the sub-env is reset before the next action_masks()
        query, so PPO's next mask is for the fresh run, never the dead screen. Non-
        vector consumers (eval/watch/test) loop `while not (terminated or
        truncated)` and break here too — so action_masks() is never handed to a
        policy as an all-False array for any consumer."""
        self._pending = None
        self._invalidate_thin_cache()
        if not terminated and self._no_legal_action():
            return self._truncate_dead()
        info = self._info()
        trace = (transition_info or {}).get("reasoning_trace")
        if trace is not None:
            trace = dict(trace)
            if action_id is not None:
                trace["thin_action_id"] = int(action_id)
            info["reasoning_trace"] = trace
        return self._obs(), reward, terminated, False, info

    def step(self, action_id):
        aid = int(action_id)

        # Dead-screen guard at the TOP of step (defense in depth): if we are parked
        # on a screen with no legal action, do NOT enter the illegal-id penalty loop
        # (the original bug: -0.1 forever, env never resets, NaN GAE). Truncate
        # cleanly. This also covers the pathological case where an all-False mask
        # somehow reached the policy and it sampled a garbage id. (In practice the
        # _post_step truncation below means we never re-enter step on a dead screen,
        # but reset() landing on one — Neow always offers 4 options, so it does not —
        # would be caught here on the first step.)
        if not self.thick.is_over and self._no_legal_action():
            return self._truncate_dead()

        legal = self._legal()
        phase = self._phase()
        desc = self._multi_desc()

        # Native C++ search returns a complete simulator action. Apply it when
        # the matching thin root id is submitted, bypassing the intermediate
        # PICK_TARGET state used by ordinary policy steps.
        pending_native = self._nosl_cpp_pending
        if pending_native is not None:
            self._nosl_cpp_pending = None
            expected_id, native_bits = pending_native
            if (aid == expected_id and phase == "combat" and desc is None
                    and aid in set(A.legal_ids(legal, self._pending, phase))):
                self._pending = None
                self._invalidate_thin_cache()
                _canon, reward, terminated, tinfo = self._backend_step_native_combat(
                    native_bits
                )
                return self._post_step(
                    reward, terminated, tinfo, action_id=aid
                )

        # ---- combat MULTI card-select (monotonic select + confirm) ----
        if desc is not None:
            if aid not in set(A.multi_legal_ids(desc, self._multi)):
                info = self._info()
                info["illegal"] = True
                return self._obs(), -0.1, False, False, info
            self._multi, commit = A.multi_toggle(aid, self._multi)
            self._invalidate_thin_cache()
            if commit is None:
                # Selected a hand card; stay on the screen with no backend call.
                return self._obs(), 0.0, False, False, self._info()
            _canon, reward, terminated, tinfo = self._backend_step(
                ga.multi_select(commit)
            )
            self._multi = None
            return self._post_step(
                reward, terminated, tinfo, action_id=aid
            )

        # Illegal id: no-op + small penalty (the mask should prevent this; guard
        # is defense in depth, mirroring SimCombatEnv). Flag it in info too: the
        # -0.1 reward is ambiguous (a legit -20 hp turn yields the same value), so
        # callers that need to detect the penalty must read info["illegal"].
        if aid not in set(A.legal_ids(legal, self._pending, phase)):
            info = self._info()
            info["illegal"] = True
            return self._obs(), -0.1, False, False, info

        semantic, self._pending = A.resolve(aid, legal, self._pending, phase)
        self._invalidate_thin_cache()

        if semantic is None:
            # Parameter step: a partial action (targeted play awaiting its target).
            # Do NOT touch the backend — just advance the sub-step and re-mask.
            return self._obs(), 0.0, False, False, self._info()

        # Complete semantic action -> drive the thick backend one decision.
        _canon, reward, terminated, tinfo = self._backend_step(semantic)
        return self._post_step(reward, terminated, tinfo, action_id=aid)

    def search_step(self, action_id, *, validate=True):
        """Advance one legal combat action for model-free search.

        Unlike :meth:`step`, this deliberately omits observation, reward, info,
        and dead-screen truncation assembly.  It preserves the autoregressive
        thin action machine, updates backend/thin caches, and returns the next
        legal mask when the branch remains live.  Search callers already have
        a legal mask and treat an empty next mask as a failed/truncated rollout.
        Internal callers can set ``validate=False`` after selecting from that
        mask; this avoids enumerating the same legal ids a second time.
        """
        aid = int(action_id)
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("search_step requires a live combat branch")

        desc = self._multi_desc()
        if desc is not None:
            if validate:
                legal = set(A.multi_legal_ids(desc, self._multi))
                if aid not in legal:
                    raise ValueError(f"illegal combat search action: {aid}")
            self._multi, commit = A.multi_toggle(aid, self._multi)
            self._invalidate_thin_cache()
            if commit is None:
                return False, self.action_masks()
            terminated = self._backend_step_fast(ga.multi_select(commit))
            self._multi = None
        else:
            if not validate:
                # The caller selected from the mask we just returned.  Resolve
                # directly against the cached native combat mask; rebuilding
                # the semantic-action list here was the hottest Python path in
                # fixed-prior search.
                semantic = self._resolve_search_combat_action(aid)
            else:
                legal_semantic = self._legal()
                phase = self._phase()
                legal = set(A.legal_ids(legal_semantic, self._pending, phase))
                if aid not in legal:
                    raise ValueError(f"illegal combat search action: {aid}")
                semantic, self._pending = A.resolve(
                    aid, legal_semantic, self._pending, phase
                )
            self._invalidate_thin_cache()
            if semantic is None:
                return False, self.action_masks()
            terminated = self._backend_step_fast(semantic)
            self._pending = None

        if terminated:
            return True, None
        return False, self.action_masks()

    def nosl_is_az_decide(
        self,
        prior_logits,
        *,
        state_nonce: int,
        master_nonce: int,
        simulations: int = 8,
        depth: int = 3,
        rollout_steps: int = 0,
        leaf_value: float | None = None,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
    ):
        """Run one NOSL-IS-AZ root search inside the live VecEnv worker.

        The learner only sends public policy logits.  The worker owns the live
        hidden simulator state, so cloning and redeterminization happen on the
        correct side of the process boundary and never expose hidden piles or
        the current world's future RNG to the learner.
        """
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("NOSL-IS-AZ requires a live combat state")
        from rl.nosl_is_az import NOSLISAZConfig, NOSLISAZSearch
        from rl.nosl_sampler import NOSLSampler

        config = NOSLISAZConfig(
            simulations=int(simulations),
            depth=int(depth),
            rollout_steps=int(rollout_steps),
            c_puct=float(c_puct),
            temperature=float(temperature),
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_epsilon=float(dirichlet_epsilon),
        )
        result = NOSLISAZSearch(config).choose(
            self,
            self.action_masks(),
            prior_logits=prior_logits,
            sampler=NOSLSampler(int(master_nonce)),
            state_nonce=int(state_nonce),
            decision_index=0,
            leaf_value=None if leaf_value is None else float(leaf_value),
        )
        return result.as_dict()

    def nosl_is_az_cpp_decide(
        self,
        *,
        state_nonce: int,
        simulations: int = 128,
        depth: int = 3,
        c_puct: float = 1.5,
        master_nonce: int = 0,
    ):
        """Run native C++ NOSL and adapt complete actions to thin root ids.

        C++ searches complete simulator actions, including card targets, while
        whole-run uses an autoregressive root action space. Root visits are
        aggregated by hand/potion/select slot; the selected native action is
        retained until the matching ``step`` submits the root id, so targeted
        cards are applied atomically. C++ performs root cloning and
        redeterminization for every simulation, preserving NOSL semantics.
        """
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("native NOSL search requires a live combat state")

        # A pending target is outside the native trainer's complete-action
        # contract. Keep the branch live and let the normal codec consume it.
        if self._pending is not None:
            legal = np.flatnonzero(self.action_masks())
            action = int(legal[0]) if legal.size else int(A.END_TURN)
            visit_policy = np.zeros(A.N_ACTIONS, dtype=np.float32)
            if legal.size:
                visit_policy[action] = 1.0
            self._nosl_cpp_pending = None
            return {
                "action": action,
                "visit_policy": visit_policy,
                "root_value": 0.0,
                "root_visits": np.zeros(A.N_ACTIONS, dtype=np.int32),
                "simulations": 0,
                "transitions": 0,
                "expanded_nodes": 0,
                "native": False,
            }

        result = self.thick.bc.nosl_native_search(
            int(simulations), int(depth), float(c_puct),
            int(state_nonce) & ((1 << 64) - 1),
        )

        def native_to_thin(bits: int) -> int | None:
            kind = (int(bits) >> 29) & 0xF
            source = int(bits) & 0xFFFF
            if kind == 0:  # CARD
                return A._PLAY + source if 0 <= source < A.HAND_SLOTS else None
            if kind == 1:  # POTION
                return A._POTION + source if 0 <= source < A.P_MAX else None
            if kind == 2:  # SINGLE_CARD_SELECT
                return A._PICK + source if 0 <= source < A.G_MAX else None
            if kind == 3:  # MULTI_CARD_SELECT: thin commits with CONFIRM
                return A._CONFIRM
            if kind == 4:  # END_TURN
                return A.END_TURN
            return None

        visit_policy = np.zeros(A.N_ACTIONS, dtype=np.float32)
        root_visit_counts = np.zeros(A.N_ACTIONS, dtype=np.int32)
        for raw_bits, raw_visits in result.get("root_visits", ()):
            thin = native_to_thin(int(raw_bits))
            if thin is not None and int(raw_visits) > 0:
                visit_policy[thin] += float(raw_visits)
                root_visit_counts[thin] += int(raw_visits)
        live_mask = np.asarray(self.action_masks(), dtype=bool)
        visit_policy[~live_mask] = 0.0
        visit_sum = float(visit_policy.sum())
        if visit_sum <= 0.0 or not np.isfinite(visit_sum):
            visit_policy[:] = 0.0
            legal = np.flatnonzero(live_mask)
            if legal.size:
                visit_policy[legal] = 1.0 / float(legal.size)
        else:
            visit_policy /= visit_sum

        selected_native = int(result.get("action_bits", 0xFFFFFFFF))
        action = native_to_thin(selected_native)
        native_valid = bool(result.get("action_valid", False))
        if action is None or not (0 <= action < A.N_ACTIONS) or not live_mask[action]:
            legal = np.flatnonzero(live_mask)
            action = int(legal[0]) if legal.size else int(A.END_TURN)
            native_valid = False
        self._nosl_cpp_pending = (
            (int(action), selected_native) if native_valid else None
        )
        return {
            "action": int(action),
            "visit_policy": visit_policy,
            "root_value": float(result.get("root_value", 0.0)),
            "root_visits": root_visit_counts,
            "simulations": int(result.get("simulations", simulations)),
            "transitions": int(result.get("transitions", 0)),
            "expanded_nodes": int(result.get("expanded_nodes", 0)),
            "native": True,
        }

    @staticmethod
    def _native_action_to_thin(bits: int) -> int | None:
        """Map a complete simulator action to the whole-run root slot."""
        kind = (int(bits) >> 29) & 0xF
        source = int(bits) & 0xFFFF
        if kind == 0:  # CARD
            return A._PLAY + source if 0 <= source < A.HAND_SLOTS else None
        if kind == 1:  # POTION
            return A._POTION + source if 0 <= source < A.P_MAX else None
        if kind == 2:  # SINGLE_CARD_SELECT
            return A._PICK + source if 0 <= source < A.G_MAX else None
        if kind == 3:  # MULTI_CARD_SELECT
            return A._CONFIRM
        if kind == 4:  # END_TURN
            return A.END_TURN
        return None

    def _native_leaf_observation(self, raw_state, *, combat_only=False):
        """Encode one C++ branch state without exposing hidden RNG/pile order."""
        canon = normalize_sim(raw_state)
        canon["phase"] = "combat"
        canon["arg_step"] = A.ROOT
        if int(raw_state.get("input_state", 1)) == 2:
            select = raw_state.get("select") or {}
            if select:
                canon["options"] = {
                    "select": combat_select_options(select)
                }
        global_state = dict(self._nosl_cpp_global_template or
                            normalize_sim_global(self.thick.gc))
        # The branch is a hypothetical world line.  Persistent metadata comes
        # from the live run, while combat HP/potions come from this branch.
        global_state["hp"] = int(raw_state["player"]["hp"])
        global_state["max_hp"] = int(raw_state["player"]["max_hp"])
        if "potions" in raw_state:
            global_state["potions"] = list(raw_state["potions"])
        canon["global"] = global_state
        if combat_only:
            return encode.encode_combat(
                canon,
                card_language=self._card_language,
                combat_language=self._combat_language,
            )
        return encode.encode(
            canon,
            global_map_view=self._global_map_view,
            card_language=self._card_language,
            combat_language=self._combat_language,
            deck_size_observation=self._deck_size_observation,
            neow_observation=self._neow_observation,
            event_option_observation=self._event_option_observation,
            interactive_reasoning=self._interactive_reasoning,
        )

    def _adapt_native_neural_result(
        self,
        result,
        *,
        stochastic: bool = False,
        state_nonce: int = 0,
        decision_index: int = 0,
    ):
        """Convert a C++ session result to the thin action/target contract."""
        # Keep raw visit frequencies separate from Full Gumbel's improved
        # policy.  The latter is the AlphaZero training target; visits remain
        # useful for UI/diagnostics and for the optional stochastic actor.
        root_visit_policy = np.zeros(A.N_ACTIONS, dtype=np.float32)
        root_visit_counts = np.zeros(A.N_ACTIONS, dtype=np.int32)
        raw_visits = []
        for raw_bits, visits in result.get("root_visits", ()):
            thin = self._native_action_to_thin(int(raw_bits))
            visits = int(visits)
            if thin is not None and visits > 0:
                root_visit_policy[thin] += float(visits)
                root_visit_counts[thin] += visits
                raw_visits.append((int(raw_bits), visits, int(thin)))
        # Full Gumbel returns the mctx-style improved policy
        # softmax(prior_logits + completed_Q), which may assign non-zero mass
        # to an action that received no simulation visit.  Legacy sessions do
        # not populate this field and continue to use visit frequencies.
        improved_policy = np.zeros(A.N_ACTIONS, dtype=np.float32)
        root_policy_entries = result.get("root_policy", ())
        if root_policy_entries:
            for raw_bits, probability in root_policy_entries:
                thin = self._native_action_to_thin(int(raw_bits))
                probability = float(probability)
                if (thin is None or not np.isfinite(probability) or
                        probability < 0.0):
                    continue
                improved_policy[thin] += probability
        live_mask = np.asarray(self.action_masks(), dtype=bool)
        root_visit_policy[~live_mask] = 0.0
        visit_total = float(root_visit_policy.sum())
        if visit_total <= 0.0 or not np.isfinite(visit_total):
            root_visit_policy[:] = 0.0
            legal = np.flatnonzero(live_mask)
            if legal.size:
                root_visit_policy[legal] = 1.0 / float(legal.size)
        else:
            root_visit_policy /= visit_total
        improved_policy[~live_mask] = 0.0
        improved_total = float(improved_policy.sum())
        has_improved_policy = bool(
            root_policy_entries and improved_total > 0.0 and
            np.isfinite(improved_total)
        )
        if has_improved_policy:
            improved_policy /= improved_total
        else:
            improved_policy[:] = root_visit_policy
        native_bits = int(result.get("action_bits", 0xFFFFFFFF))
        if stochastic and raw_visits:
            # Sample complete native actions, not the aggregated thin action.
            # This keeps card-target choices atomic while preserving the NOSL
            # hidden-state contract: only the root search statistics are used.
            weights = np.asarray([item[1] for item in raw_visits], dtype=np.float64)
            total = float(weights.sum())
            if total > 0.0 and np.isfinite(total):
                from rl.nosl_sampler import NOSLSampler

                nonce = NOSLSampler(0x4E4F534C_64617368).nonce(
                    state_nonce=int(state_nonce),
                    simulation_index=int(decision_index) * 1_000_000 + 999_998,
                )
                rng = np.random.default_rng(nonce)
                native_bits = int(
                    rng.choice(
                        np.asarray([item[0] for item in raw_visits], dtype=np.uint64),
                        p=weights / total,
                    )
                )
        action = self._native_action_to_thin(native_bits)
        native_valid = action is not None
        if action is None or not (0 <= action < A.N_ACTIONS) or not live_mask[action]:
            legal = np.flatnonzero(live_mask)
            action = int(legal[0]) if legal.size else int(A.END_TURN)
            native_valid = False
        self._nosl_cpp_pending = (
            (int(action), native_bits) if native_valid else None
        )
        return {
            "action": int(action),
            # Preserve the historical field name for the learner, but use the
            # completed-Q target whenever the native Full Gumbel session
            # supplied one.  `root_visit_policy` is the count-normalized view.
            "visit_policy": improved_policy,
            "root_visit_policy": root_visit_policy,
            "improved_policy": improved_policy,
            "used_completed_policy": has_improved_policy,
            "root_value": float(result.get("root_value", 0.0)),
            "root_visits": root_visit_counts,
            "native_action_bits": int(native_bits),
            "selected_target": (
                int((native_bits >> 16) & 0x1FFF)
                if self._native_action_targeted(native_bits) else -1
            ),
            "selected_action_targeted": self._native_action_targeted(native_bits),
            "simulations": int(result.get("simulations", 0)),
            "native": True,
        }

    def _native_action_targeted(self, bits: int) -> bool:
        """Whether a complete native action targets an enemy in the live root."""
        checker = getattr(getattr(self, "thick", None), "bc", None)
        checker = getattr(checker, "is_targeted_action", None)
        if checker is None:
            return False
        try:
            return bool(checker(int(bits)))
        except (TypeError, ValueError, RuntimeError):
            return False

    def nosl_is_az_cpp_begin(
        self,
        prior_logits,
        *,
        state_nonce: int,
        simulations: int = 128,
        depth: int = 3,
        c_puct: float = 1.5,
        leaf_batch_size: int = 64,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
        root_value: float = 0.0,
        prior_temperature: float = 1.0,
        root_search_mode: int = 0,
        gumbel_candidates: int = 0,
        gumbel_scale: float = 1.0,
        gumbel_value_scale: float = 0.1,
        gumbel_max_visit_init: float = 50.0,
        gumbel_rescale_q: bool = True,
        gumbel_use_mixed_value: bool = True,
        stochastic: bool = False,
        decision_index: int = 0,
        compact_observation: bool = False,
    ):
        """Start the batched neural C++ NOSL AlphaZero session."""
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("native NOSL search requires a live combat state")
        session_factory = getattr(self.thick.bc, "nosl_neural_search_begin", None)
        if session_factory is None:
            raise RuntimeError("simulator build lacks neural NOSL search")
        # Keep the adapter usable while Python and the native module are
        # updated independently. Current builds take root value and prior
        # temperature after the optional root-noise arguments; older builds
        # take either the root value, root-noise, or legacy five arguments.
        try:
            self._nosl_cpp_az_session = session_factory(
                np.asarray(prior_logits, dtype=np.float64),
                int(simulations), int(depth), float(c_puct),
                int(state_nonce) & ((1 << 64) - 1),
                float(dirichlet_alpha), float(dirichlet_epsilon),
                float(root_value), float(prior_temperature),
                int(root_search_mode), int(gumbel_candidates),
                float(gumbel_scale), float(gumbel_value_scale),
                float(gumbel_max_visit_init), bool(gumbel_rescale_q),
                bool(gumbel_use_mixed_value),
            )
        except TypeError as error:
            if "incompatible function arguments" not in str(error):
                raise
            # A legacy extension can still serve ordinary PUCT searches, but
            # it cannot honor Full-Gumbel arguments.  Do not silently run the
            # old algorithm while the dashboard/trainer reports the requested
            # newer mode; that makes the tree diagnostics and actual action
            # selection semantically false.
            if (
                int(root_search_mode) != 0
                or int(gumbel_candidates) != 0
                or float(gumbel_scale) != 1.0
                or float(gumbel_value_scale) != 0.1
                or float(gumbel_max_visit_init) != 50.0
                or not bool(gumbel_rescale_q)
                or not bool(gumbel_use_mixed_value)
            ):
                raise RuntimeError(
                    "loaded simulator extension does not support the requested "
                    "Full-Gumbel/NOSL search parameters; rebuild or set "
                    "STS_LIGHTSPEED_BUILD_DIR to the current extension"
                ) from error
            try:
                self._nosl_cpp_az_session = session_factory(
                    np.asarray(prior_logits, dtype=np.float64),
                    int(simulations), int(depth), float(c_puct),
                    int(state_nonce) & ((1 << 64) - 1),
                    float(dirichlet_alpha), float(dirichlet_epsilon),
                    float(root_value), float(prior_temperature),
                )
            except TypeError as legacy_error:
                if dirichlet_alpha != 0.0 or dirichlet_epsilon != 0.0:
                    raise
                if "incompatible function arguments" not in str(legacy_error):
                    raise
                self._nosl_cpp_az_session = session_factory(
                    np.asarray(prior_logits, dtype=np.float64),
                    int(simulations), int(depth), float(c_puct),
                    int(state_nonce) & ((1 << 64) - 1),
                )
        self._nosl_cpp_stochastic = bool(stochastic)
        self._nosl_cpp_state_nonce = int(state_nonce)
        self._nosl_cpp_decision_index = int(decision_index)
        self._nosl_cpp_compact_observation = bool(compact_observation)
        # Relics/deck/map metadata is invariant across hypothetical combat
        # branches.  Compute it once per root; only branch HP/potions are
        # overlaid in _native_leaf_observation().
        self._nosl_cpp_global_template = normalize_sim_global(self.thick.gc)
        return self.nosl_is_az_cpp_leaf_batch(leaf_batch_size)

    def nosl_is_az_cpp_leaf_batch(
        self,
        max_leaves: int = 64,
        *,
        stochastic: bool | None = None,
        state_nonce: int | None = None,
        decision_index: int | None = None,
    ):
        session = getattr(self, "_nosl_cpp_az_session", None)
        if session is None:
            raise RuntimeError("native NOSL AlphaZero session is not active")
        requests = session.next_batch(int(max_leaves))
        leaves = []
        for request in requests:
            raw = request["state"]
            mask = np.zeros(A.N_ACTIONS, dtype=bool)
            for bits in request.get("legal_action_bits", ()):
                thin = self._native_action_to_thin(int(bits))
                if thin is not None and 0 <= thin < A.N_ACTIONS:
                    mask[thin] = True
            leaves.append({
                "request_id": int(request["request_id"]),
                "observation": self._native_leaf_observation(
                    raw,
                    combat_only=self._nosl_cpp_compact_observation,
                ),
                "action_mask": mask,
                "needs_policy": bool(request.get("needs_policy", True)),
            })
        if not leaves and bool(session.done()):
            result = self._adapt_native_neural_result(
                session.result(),
                stochastic=(
                    self._nosl_cpp_stochastic
                    if stochastic is None else bool(stochastic)
                ),
                state_nonce=(
                    self._nosl_cpp_state_nonce
                    if state_nonce is None else int(state_nonce)
                ),
                decision_index=(
                    self._nosl_cpp_decision_index
                    if decision_index is None else int(decision_index)
                ),
            )
            # Keep the completed C++ object alive until the next decision so a
            # caller can request tree children by path without rerunning search.
            self._nosl_cpp_last_session = session
            self._nosl_cpp_az_session = None
            self._nosl_cpp_global_template = None
            self._nosl_cpp_compact_observation = False
            self._nosl_cpp_stochastic = False
            return {"done": True, "result": result, "leaves": []}
        return {"done": False, "result": None, "leaves": leaves}

    def nosl_is_az_cpp_backprop_batch(self, updates):
        session = getattr(self, "_nosl_cpp_az_session", None)
        if session is None:
            raise RuntimeError("native NOSL AlphaZero session is not active")
        for request_id, value, policy_logits in updates:
            session.backprop(
                int(request_id), float(value),
                np.asarray(policy_logits, dtype=np.float64),
            )
        return {"done": bool(session.done())}

    def nosl_is_az_cpp_backprop_and_leaf_batch(
        self, updates, max_leaves: int = 64
    ):
        """Backpropagate one neural batch and return the next leaves.

        Keeping both operations in one worker call removes one IPC round trip
        per leaf batch while preserving the two original public methods.
        """
        self.nosl_is_az_cpp_backprop_batch(updates)
        return self.nosl_is_az_cpp_leaf_batch(max_leaves)

    def nosl_is_az_cpp_tree_snapshot(self, path=(), max_children: int = 64):
        """Return one lazily requested node of the completed native search tree."""
        session = getattr(self, "_nosl_cpp_last_session", None)
        indexes = [int(index) for index in (path or ())]
        if session is None:
            return {
                "valid": False,
                "path": indexes,
                "visits": 0,
                "value": 0.0,
                "children": [],
            }
        raw = session.tree_snapshot(indexes, int(max_children))
        children = []
        for item in raw.get("children", ()):
            bits = int(item.get("action_bits", 0))
            action_type = (bits >> 29) & 0xF
            children.append({
                "edge_index": int(item.get("edge_index", -1)),
                "action_bits": bits,
                "action_type": action_type,
                "source_idx": bits & 0xFFFF,
                "target_idx": (bits >> 16) & 0x1FFF,
                "visits": int(item.get("visits", 0)),
                "child_visits": int(item.get("child_visits", 0)),
                "child_states": int(item.get("child_states", 0)),
                "prior": float(item.get("prior", 0.0)),
                "value": float(item.get("value", 0.0)),
                "raw_q": float(item.get("raw_q", item.get("value", 0.0))),
                "normalized_q": float(item.get("normalized_q", 0.0)),
                "completed_q": float(item.get("completed_q", 0.0)),
                "gumbel": float(item.get("gumbel", 0.0)),
                "gumbel_score": float(item.get("gumbel_score", 0.0)),
                "completed_score": float(item.get("completed_score", 0.0)),
                "candidate": bool(item.get("candidate", False)),
                "active_candidate": bool(item.get("active_candidate", False)),
                "expandable": bool(item.get("expandable", False)),
                "targeted": bool(item.get("targeted", False)),
                "label": str(item.get("label", "")),
            })
        return {
            "valid": bool(raw.get("valid", False)),
            "path": indexes,
            "visits": int(raw.get("visits", 0)),
            "value": float(raw.get("value", 0.0)),
            "children": children,
        }

    def nosl_is_az_begin(
        self,
        prior_logits,
        *,
        state_nonce: int,
        master_nonce: int,
        simulations: int = 8,
        depth: int = 3,
        rollout_steps: int = 0,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        """Start a staged NOSL-IS-AZ search and return its first leaf request."""
        if self.thick is None or self._phase() != "combat":
            raise RuntimeError("NOSL-IS-AZ requires a live combat state")
        from rl.nosl_is_az import NOSLISAZConfig, NOSLISAZSession
        from rl.nosl_sampler import NOSLSampler

        config = NOSLISAZConfig(
            simulations=int(simulations),
            depth=int(depth),
            rollout_steps=int(rollout_steps),
            c_puct=float(c_puct),
            temperature=float(temperature),
        )
        self._nosl_is_az_session = NOSLISAZSession(
            self,
            self.action_masks(),
            prior_logits=prior_logits,
            sampler=NOSLSampler(int(master_nonce)),
            state_nonce=int(state_nonce),
            decision_index=0,
            config=config,
        )
        result = self._nosl_is_az_session.next_request()
        if result.get("done"):
            self._nosl_is_az_session = None
        return result

    def nosl_is_az_backprop(self, value: float, policy_logits=None):
        """Evaluate one staged leaf and continue the worker-side PUCT session."""
        session = self._nosl_is_az_session
        if session is None:
            raise RuntimeError("NOSL-IS-AZ has no active staged session")
        result = session.backprop(float(value), policy_logits=policy_logits)
        if result.get("done"):
            self._nosl_is_az_session = None
        return result

    # ---------------------------------------------------------------- obs/mask
    def _arg_step(self):
        # On a combat MULTI screen the sub-step is pick_hand regardless of the
        # autoregressive play machine (they never overlap — different screens).
        if self._multi is not None:
            return A.PICK_HAND
        return A.arg_step_name(self._pending, self._legal(), self._phase())

    def _obs(self):
        """Structured Dict obs (encode v3): the thick layer's assembled canonical
        plus the thin layer's `arg_step` (which the backend cannot know). On a
        combat MULTI screen we also patch the thin-only progress into the select
        block: the selected hand indices (select_hand_mask) and pick_remaining =
        pick_max - already-selected (the sim can't know our pending set).

        Reconciles the MULTI accumulator FIRST so obs and the action mask never
        disagree within one step (e.g. the frame that just LANDED on the screen)."""
        self._multi_desc()
        canon = self.thick.obs_canonical()
        canon["arg_step"] = self._arg_step()
        if self._interactive_reasoning:
            canon["pending_card_slot"] = (
                int(self._pending[1]) + 1
                if self._pending is not None and self._pending[0] == "play"
                else 0
            )
        if self._multi is not None:
            sel = canon.setdefault("options", {}).setdefault("select", {})
            sel = dict(sel)                                   # don't mutate backend's
            sel["hand_selected"] = sorted(self._multi)
            cap = int(sel.get("pick_max", 0))
            sel["pick_remaining"] = max(0, cap - len(self._multi))
            canon["options"]["select"] = sel
        return encode.encode(
            canon,
            global_map_view=self._global_map_view,
            card_language=self._card_language,
            combat_language=self._combat_language,
            deck_size_observation=self._deck_size_observation,
            neow_observation=self._neow_observation,
            event_option_observation=self._event_option_observation,
            interactive_reasoning=self._interactive_reasoning,
        )

    def action_masks(self):
        """Bool[N_ACTIONS] legal mask for the CURRENT (phase, arg_step). Name
        matches sb3-contrib's ActionMasker convention."""
        if self._cached_action_mask is not None:
            return self._cached_action_mask.copy()
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        desc = self._multi_desc()
        if desc is not None:
            for i in A.multi_legal_ids(desc, self._multi):
                mask[i] = True
        elif (self.thick.bc is not None and self._phase() == "combat"
              and self._pending is None):
            # Combat search repeatedly asks for masks on cloned branches.  Use
            # the native fixed combat mask directly when the thin codec is at
            # its root; pending target selection still uses the general path.
            mask = self._combat_action_mask_fast()
        else:
            for i in A.legal_ids(self._legal(), self._pending, self._phase()):
                mask[i] = True
        self._cached_action_mask = mask
        return mask.copy()

    def _info(self):
        info = {
            "action_mask": self.action_masks(),
            "phase": self._phase(),
            "arg_step": self._arg_step(),
            "pending": self._pending,
            "multi_selected": (sorted(self._multi)
                               if self._multi is not None else None),
        }
        run_stats = getattr(self.thick, "run_stats", None)
        if run_stats is not None:
            info["run_stats"] = run_stats()
        return info

    # ---------------------------------------------------------------- render
    def render(self):
        if self.thick is None:
            return "(run not started — call reset())"
        out = self.thick.render()
        if self._pending is not None and self._pending[0] == "play":
            out += f"\n[awaiting target for hand card #{self._pending[1]}]"
        if self._pending is not None and self._pending[0] == "potion":
            out += f"\n[awaiting target for potion slot #{self._pending[1]}]"
        if self._multi is not None:
            out += f"\n[multi-select: picked hand {sorted(self._multi)}; CONFIRM to commit]"
        return out

    def describe_action(self, action_id):
        """Human-readable label for a thin action id in the current state — used
        by the interactive player and for debugging."""
        aid = int(action_id)
        legal = self._legal()
        phase = self._phase()

        # combat MULTI screen: CONFIRM commits; PICK_CARD adds an unselected card.
        if self._multi is not None:
            if aid == A._CONFIRM:
                return f"confirm multi-select {sorted(self._multi)}"
            seg, off = A.segment_of(aid)
            verb = "already selected (illegal)" if off in self._multi else "select"
            return f"{verb} {_card_label(self._combat_canon(), off)} (hand #{off})"

        # parameter step: a target pick (shared by a pending card play AND a
        # pending targeted potion — both fill a PICK_TARGET monster id).
        if self._pending is not None and self._pending[0] in ("play", "potion"):
            seg, off = A.segment_of(aid)
            return f"target monster #{off}"

        seg, off = A.segment_of(aid)
        # For ids that map 1:1 to a semantic action, reuse the thick describer.
        try:
            semantic, pend = A.resolve(aid, legal, self._pending, phase)
        except ValueError:
            return f"{seg}[{off}]"
        if semantic is None:                       # targeted action (root half)
            if pend[0] == "play":
                h = pend[1]                        # pend == ('play', hand_idx)
                return f"play {_card_label(self._combat_canon(), h)} (choose target next)"
            # pend == ('potion', slot): targeted potion, target chosen next step.
            return f"use potion (slot #{pend[1]}) (choose target next)"
        return self.thick.describe_action(semantic)

    def _combat_canon(self):
        if self.thick.bc is not None:
            return normalize_sim(self.thick.bc.to_dict())
        return None


def _card_label(canon, hand_idx):
    if not canon:
        return f"hand #{hand_idx}"
    try:
        c = canon["hand"][hand_idx]
    except (KeyError, IndexError, TypeError):
        return f"hand #{hand_idx}"
    return f"{c['id']}+" if c.get("up") else c["id"]
