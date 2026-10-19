"""SimCombatEnv: a Gymnasium env for a single Slay the Spire combat, backed by
the sts_lightspeed sim.

This is the ONLY file in env/ that imports slaythespire. It owns the sim/Gym
glue; encode.py and actions.py stay backend-agnostic so a future RealCombatEnv
(real game via bridge) can reuse them unchanged — only this file is swapped.

Status: skeleton. reset() needs a GameContext advanced to a real pre-combat
state (Neow + map progression push miscRng into place — a vanilla gc does not
reproduce real combats; see DEVNOTES). For now reset() takes a gc_factory the
caller supplies. Reward shaping is a placeholder (win/loss + hp delta).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# `import env.core` (the package __init__) puts the sim build dir and bridge/ on
# sys.path; that must run before importing slaythespire / schemas.
from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts
from schemas.combat_state import normalize_sim, normalize_sim_global, combat_select_options

from . import encode
from . import actions
from . import factories
from . import render

# sim InputState ints (see DEVNOTES / bindings): 1 = PLAYER_NORMAL, 2 = CARD_SELECT
_PLAYER_NORMAL = 1
_CARD_SELECT = 2
# BattleContext.to_dict()["outcome"] uses the combat-side Outcome enum
# (BattleContext.h): UNDECIDED=0, PLAYER_VICTORY=1, PLAYER_LOSS=2. This is NOT
# the same as the module-level sts.GameOutcome enum, whose ints are different
# (UNDECIDED=1, VICTORY=2, LOSS=0) — do not compare the two.
_OUTCOME_UNDECIDED = 0
_OUTCOME_VICTORY = 1
_OUTCOME_LOSS = 2


# ---- combat sts-glue, shared with SimGameEnv ----
# These two are the only combat helpers that touch the sim. SimGameEnv holds its
# own BattleContext mid-run and reuses them so the legal-action <-> 71-id mapping
# (RNG-verified) has a single source of truth.

def legal_mask(bc):
    """Return the fixed combat mask using the native path when available.

    The C++ binding and the fallback both use the simulator's legal-actions
    contract, so the abstract ids still line up with ``bc.execute``.
    """
    # Newer simulator builds perform the same mapping in C++ and return one
    # freshly allocated NumPy array. Keep the Python path for older builds and
    # alternate backends so the environment remains backwards compatible.
    native_mask = getattr(bc, "legal_combat_mask", None)
    if native_mask is not None:
        mask = np.asarray(native_mask(), dtype=bool)
        if mask.shape != (actions.N_ACTIONS,):
            raise RuntimeError(
                "legal_combat_mask returned unexpected shape "
                f"{mask.shape}; expected {(actions.N_ACTIONS,)}"
            )
        return mask

    mask = np.zeros(actions.N_ACTIONS, dtype=bool)
    for a in bc.legal_actions():
        t = a.action_type
        if t == sts.ActionType.END_TURN:
            mask[actions.END_TURN] = True
        elif t == sts.ActionType.CARD:
            tgt = a.target_idx
            # sim encodes "no target" as a sentinel; treat out-of-range as untargeted
            tgt = tgt if 0 <= tgt < actions.MAX_MONSTERS else None
            if a.source_idx < actions.HAND_SLOTS:
                mask[actions.encode_play(a.source_idx, tgt)] = True
        elif t == sts.ActionType.SINGLE_CARD_SELECT:
            if a.source_idx < actions.SELECT_SLOTS:
                mask[actions.encode_select(a.source_idx)] = True
        elif t == sts.ActionType.MULTI_CARD_SELECT:
            # The sim's single forced action for EXHAUST_MANY / GAMBLE card-select
            # tasks (it resolves the whole screen in one shot — no per-card choice
            # is modeled). Surface it as select slot 0; to_sts_action() consults bc
            # to emit MULTI rather than SINGLE for it.
            mask[actions.encode_select(0)] = True
    return mask


def _is_multi_select(bc):
    """True when the live card-select screen is the MULTI_CARD_SELECT catch-all
    (EXHAUST_MANY / GAMBLE) — its only legal action is one MULTI_CARD_SELECT."""
    if bc is None:
        return False
    return any(a.action_type == sts.ActionType.MULTI_CARD_SELECT
               for a in bc.legal_actions())


def to_sts_action(kind, bc=None):
    """Decoded abstract action tuple (from actions.decode) -> sts.Action.

    For KIND_SELECT the abstract id does not record whether the live screen is a
    SINGLE or MULTI card-select (they share the select-slot ids), so the caller
    passes the live `bc` and we read the screen type off it. Absent bc we default
    to SINGLE (the common case; MULTI only arises on EXHAUST_MANY / GAMBLE)."""
    if kind[0] == actions.KIND_END:
        return sts.Action(sts.ActionType.END_TURN)
    if kind[0] == actions.KIND_PLAY:
        _, h, t = kind
        if t is None:
            return sts.Action(sts.ActionType.CARD, h)
        return sts.Action(sts.ActionType.CARD, h, t)
    if kind[0] == actions.KIND_SELECT:
        if _is_multi_select(bc):
            return sts.Action(sts.ActionType.MULTI_CARD_SELECT, 0)
        return sts.Action(sts.ActionType.SINGLE_CARD_SELECT, kind[1])
    raise ValueError(kind)


class SimCombatEnv(gym.Env):
    """One combat = one episode.

    gc_factory() -> a fresh GameContext positioned at the combat to fight, plus
    the encounter name to force. Defaults to a random act-1 combat (vanilla
    Ironclad starter loadout); pass your own to fight specific combats or to
    replay a real pre-combat snapshot (cf. replay.py `_sync_gc`)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, gc_factory=None, pool="normal", ascension=0,
                 render_mode=None, combat_language=False):
        super().__init__()
        self._gc_factory = gc_factory
        self._pool = pool
        self._ascension = ascension
        self.render_mode = render_mode
        self._combat_language = combat_language
        self.gc = None
        self.bc = None
        self.encounter = None

        self.observation_space = encode.obs_space(combat_language=combat_language)
        self.action_space = spaces.Discrete(actions.N_ACTIONS)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Lazily build the default factory once self.np_random exists, so
        # reset(seed=...) drives the episode stream reproducibly.
        if self._gc_factory is None:
            self._gc_factory = factories.random_act1_factory(
                self.np_random, pool=self._pool, ascension=self._ascension
            )
        self.gc, self.encounter = self._gc_factory()
        self.bc = sts.create_battle(self.gc, self.encounter)
        self._prev_hp = self.bc.to_dict()["player"]["hp"]
        return self._obs(), self._info()

    def step(self, action_id):
        kind = actions.decode(int(action_id))
        # Illegal action: no-op with a small penalty rather than corrupting the
        # sim (the mask should prevent this; the guard is defense in depth).
        if not self.action_mask()[int(action_id)]:
            return self._obs(), -0.1, False, False, self._info()

        self.bc.execute(to_sts_action(kind, self.bc))

        terminated = self.bc.is_battle_over
        raw = self.bc.to_dict()
        hp = raw["player"]["hp"]
        reward = float(hp - self._prev_hp) / encode.HP_SCALE
        self._prev_hp = hp
        if terminated:
            reward += 1.0 if raw["outcome"] == _OUTCOME_VICTORY else -1.0
        return self._obs(), reward, terminated, False, self._info()

    # ---- helpers ----
    def _canon(self):
        """Current combat as the canonical combat dict (shared by obs + render)."""
        return normalize_sim(self.bc.to_dict())

    def _obs_canon(self):
        """The assembled OBS canonical: combat block + run-global block + phase.
        Combat decisions read the whole run (next boss, relics, deck) — see
        obs_design.md §3 — so the global block rides along even in combat."""
        d = self._canon()
        d["phase"] = "combat"
        d["global"] = normalize_sim_global(self.gc, combat=True)
        # in-combat card-select sub-screen -> carry the select block (obs_design
        # §4.1) so select_context/pick_remaining reach the obs.
        if self.bc.input_state == _CARD_SELECT:
            d["options"] = {"select": combat_select_options(self.bc.select_info())}
        return d

    def _obs(self):
        return encode.encode(
            self._obs_canon(), combat_language=self._combat_language
        )

    def _info(self):
        return {"action_mask": self.action_mask()}

    # ---- rendering ----
    def render(self):
        """Return a human-readable string of the current combat (render_mode
        'ansi'). Not the obs the policy sees — this is the debug/teaching view."""
        if self.bc is None:
            return "(combat not started — call reset())"
        return render.render_state(self._canon())

    def describe_action(self, action_id):
        """Human-readable label for an action id in the current state."""
        return render.render_action(action_id, self._canon() if self.bc else None)

    def action_mask(self):
        """Boolean mask over the fixed action space (shared helper; see
        combat_env.legal_mask)."""
        return legal_mask(self.bc)

    def _to_sts_action(self, kind):
        return to_sts_action(kind, self.bc)
