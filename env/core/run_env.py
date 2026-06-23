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
                 render_mode=None):
        super().__init__()
        self._backend_factory = backend_factory or (
            lambda: SimGameEnv(character=character, ascension=ascension)
        )
        self.render_mode = render_mode
        self.thick = None
        # autoregressive sub-state: None at ROOT, else ('play', hand_idx) awaiting
        # a PICK_TARGET. Lives in the thin layer only — the backend never sees it.
        self._pending = None

        self.observation_space = encode.obs_space()
        self.action_space = spaces.Discrete(A.N_ACTIONS)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.thick = self._backend_factory()
        self.thick.reset(seed=seed)
        self._pending = None
        return self._obs(), self._info()

    def step(self, action_id):
        aid = int(action_id)
        legal = self.thick.legal_semantic_actions()
        phase = self.thick.phase()

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
        self._pending = None
        return self._obs(), reward, terminated, False, self._info()

    # ---------------------------------------------------------------- obs/mask
    def _arg_step(self):
        legal = self.thick.legal_semantic_actions()
        return A.arg_step_name(self._pending, legal, self.thick.phase())

    def _obs(self):
        """Structured Dict obs (encode v3): the thick layer's assembled canonical
        plus the thin layer's `arg_step` (which the backend cannot know)."""
        canon = self.thick.obs_canonical()
        canon["arg_step"] = self._arg_step()
        return encode.encode(canon)

    def action_masks(self):
        """Bool[N_ACTIONS] legal mask for the CURRENT (phase, arg_step). Name
        matches sb3-contrib's ActionMasker convention."""
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        legal = self.thick.legal_semantic_actions()
        for i in A.legal_ids(legal, self._pending, self.thick.phase()):
            mask[i] = True
        return mask

    def _info(self):
        return {
            "action_mask": self.action_masks(),
            "phase": self.thick.phase(),
            "arg_step": self._arg_step(),
            "pending": self._pending,
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.thick is None:
            return "(run not started — call reset())"
        out = self.thick.render()
        if self._pending is not None and self._pending[0] == "play":
            out += f"\n[awaiting target for hand card #{self._pending[1]}]"
        return out

    def describe_action(self, action_id):
        """Human-readable label for a thin action id in the current state — used
        by the interactive player and for debugging."""
        aid = int(action_id)
        legal = self.thick.legal_semantic_actions()
        phase = self.thick.phase()

        # parameter step: a target pick
        if self._pending is not None and self._pending[0] == "play":
            seg, off = A.segment_of(aid)
            return f"target monster #{off}"

        seg, off = A.segment_of(aid)
        # For ids that map 1:1 to a semantic action, reuse the thick describer.
        try:
            semantic, pend = A.resolve(aid, legal, self._pending, phase)
        except ValueError:
            return f"{seg}[{off}]"
        if semantic is None:                       # targeted play (root half)
            h = pend[1]                            # pend == ('play', hand_idx)
            return f"play {_card_label(self._combat_canon(), h)} (choose target next)"
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
