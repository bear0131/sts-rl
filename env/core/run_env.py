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

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import _BUILD  # noqa: F401  (sys.path bootstrap for slaythespire/schemas)
from schemas.combat_state import normalize_sim

from . import action_space as A
from . import encode
from . import game_actions as ga
from .game_env import SimGameEnv


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
                 deck_size_observation=False, neow_observation=False):
        super().__init__()
        self._backend_factory = backend_factory or (
            lambda: SimGameEnv(
                character=character,
                ascension=ascension,
                global_map_view=global_map_view,
            )
        )
        self._global_map_view = global_map_view
        self._card_language = card_language
        self._combat_language = combat_language
        self._deck_size_observation = deck_size_observation
        self._neow_observation = neow_observation
        self.render_mode = render_mode
        self.thick = None
        # autoregressive sub-state: None at ROOT, else ('play', hand_idx) awaiting
        # a PICK_TARGET. Lives in the thin layer only — the backend never sees it.
        self._pending = None
        # combat MULTI card-select accumulator (Exhaust Many / Gamble): the set of
        # selected hand indices awaiting CONFIRM_SELECT. None when not on a MULTI
        # screen. Thin-layer only; reconciled against the live screen each step.
        self._multi = None

        self.observation_space = encode.obs_space(
            global_map_view=global_map_view,
            card_language=card_language,
            combat_language=combat_language,
            deck_size_observation=deck_size_observation,
            neow_observation=neow_observation,
        )
        self.action_space = spaces.Discrete(A.N_ACTIONS)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.thick = self._backend_factory()
        self.thick.reset(seed=seed)
        self._pending = None
        self._multi = None
        return self._obs(), self._info()

    def _multi_desc(self):
        """The live combat MULTI card-select descriptor, or None. Also the single
        place that RECONCILES the accumulator with the screen: a MULTI screen ->
        ensure _multi is a set; anything else -> clear it (we left the screen)."""
        desc = self.thick.select_descriptor()
        if A.is_multi_select(desc):
            if self._multi is None:
                self._multi = set()
            return desc
        self._multi = None
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
        legal = self.thick.legal_semantic_actions()
        return len(A.legal_ids(legal, self._pending, self.thick.phase())) == 0

    def _truncate_dead(self):
        """The clean 5-tuple for truncating on a dead screen: a valid (current)
        obs, reward 0.0, terminated=False, truncated=True, with info["dead_screen"]
        set (mirrors the info["illegal"] convention) so callers can tell a dead-
        screen truncation apart from a step-cap (TimeLimit) truncation."""
        info = self._info()
        info["dead_screen"] = True
        return self._obs(), 0.0, False, True, info

    def _post_step(self, reward, terminated):
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
        if not terminated and self._no_legal_action():
            return self._truncate_dead()
        return self._obs(), reward, terminated, False, self._info()

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

        legal = self.thick.legal_semantic_actions()
        phase = self.thick.phase()
        desc = self._multi_desc()

        # ---- combat MULTI card-select (monotonic select + confirm) ----
        if desc is not None:
            if aid not in set(A.multi_legal_ids(desc, self._multi)):
                info = self._info()
                info["illegal"] = True
                return self._obs(), -0.1, False, False, info
            self._multi, commit = A.multi_toggle(aid, self._multi)
            if commit is None:
                # Selected a hand card; stay on the screen with no backend call.
                return self._obs(), 0.0, False, False, self._info()
            _canon, reward, terminated, _tinfo = self.thick.step(ga.multi_select(commit))
            self._multi = None
            return self._post_step(reward, terminated)

        # Illegal id: no-op + small penalty (the mask should prevent this; guard
        # is defense in depth, mirroring SimCombatEnv). Flag it in info too: the
        # -0.1 reward is ambiguous (a legit -20 hp turn yields the same value), so
        # callers that need to detect the penalty must read info["illegal"].
        if aid not in set(A.legal_ids(legal, self._pending, phase)):
            info = self._info()
            info["illegal"] = True
            return self._obs(), -0.1, False, False, info

        semantic, self._pending = A.resolve(aid, legal, self._pending, phase)

        if semantic is None:
            # Parameter step: a partial action (targeted play awaiting its target).
            # Do NOT touch the backend — just advance the sub-step and re-mask.
            return self._obs(), 0.0, False, False, self._info()

        # Complete semantic action -> drive the thick backend one decision.
        _canon, reward, terminated, _tinfo = self.thick.step(semantic)
        return self._post_step(reward, terminated)

    # ---------------------------------------------------------------- obs/mask
    def _arg_step(self):
        # On a combat MULTI screen the sub-step is pick_hand regardless of the
        # autoregressive play machine (they never overlap — different screens).
        if self._multi is not None:
            return A.PICK_HAND
        legal = self.thick.legal_semantic_actions()
        return A.arg_step_name(self._pending, legal, self.thick.phase())

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
        )

    def action_masks(self):
        """Bool[N_ACTIONS] legal mask for the CURRENT (phase, arg_step). Name
        matches sb3-contrib's ActionMasker convention."""
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        desc = self._multi_desc()
        if desc is not None:
            for i in A.multi_legal_ids(desc, self._multi):
                mask[i] = True
            return mask
        legal = self.thick.legal_semantic_actions()
        for i in A.legal_ids(legal, self._pending, self.thick.phase()):
            mask[i] = True
        return mask

    def _info(self):
        info = {
            "action_mask": self.action_masks(),
            "phase": self.thick.phase(),
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
        legal = self.thick.legal_semantic_actions()
        phase = self.thick.phase()

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
