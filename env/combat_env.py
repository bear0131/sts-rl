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

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# env/ is flat and not a package; make sibling modules and the sim importable.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)                                   # encode, actions, cards_vocab
sys.path.insert(0, os.path.join(_HERE, "..", "bridge"))     # schemas
_BUILD = os.path.join(_HERE, "..", "sim", "sts_lightspeed", "build")
if hasattr(os, "add_dll_directory") and os.path.isdir(_BUILD):
    os.add_dll_directory(os.path.abspath(_BUILD))           # windows/mingw only
sys.path.insert(0, os.path.abspath(_BUILD))

import slaythespire as sts                                  # noqa: E402
from schemas.combat_state import normalize_sim             # noqa: E402

import encode                                               # noqa: E402
import actions                                              # noqa: E402
import factories                                            # noqa: E402

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


class SimCombatEnv(gym.Env):
    """One combat = one episode.

    gc_factory() -> a fresh GameContext positioned at the combat to fight, plus
    the encounter name to force. Defaults to a random act-1 combat (vanilla
    Ironclad starter loadout); pass your own to fight specific combats or to
    replay a real pre-combat snapshot (cf. replay.py `_sync_gc`)."""

    metadata = {"render_modes": []}

    def __init__(self, gc_factory=None, pool="normal", ascension=0):
        super().__init__()
        self._gc_factory = gc_factory
        self._pool = pool
        self._ascension = ascension
        self.gc = None
        self.bc = None

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(encode.OBS_SIZE,), dtype=np.float32
        )
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
        self.gc, encounter = self._gc_factory()
        self.bc = sts.create_battle(self.gc, encounter)
        self._prev_hp = self.bc.to_dict()["player"]["hp"]
        return self._obs(), self._info()

    def step(self, action_id):
        kind = actions.decode(int(action_id))
        # Illegal action: no-op with a small penalty rather than corrupting the
        # sim (the mask should prevent this; the guard is defense in depth).
        if not self.action_mask()[int(action_id)]:
            return self._obs(), -0.1, False, False, self._info()

        self.bc.execute(self._to_sts_action(kind))

        terminated = self.bc.is_battle_over
        raw = self.bc.to_dict()
        hp = raw["player"]["hp"]
        reward = float(hp - self._prev_hp) / encode.HP_SCALE
        self._prev_hp = hp
        if terminated:
            reward += 1.0 if raw["outcome"] == _OUTCOME_VICTORY else -1.0
        return self._obs(), reward, terminated, False, self._info()

    # ---- helpers ----
    def _obs(self):
        return encode.encode(normalize_sim(self.bc.to_dict()))

    def _info(self):
        return {"action_mask": self.action_mask()}

    def action_mask(self):
        """Boolean mask over the fixed action space, built from the sim's own
        legal_actions so abstract ids line up exactly with what bc.execute will
        accept."""
        mask = np.zeros(actions.N_ACTIONS, dtype=bool)
        for a in self.bc.legal_actions():
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
        return mask

    def _to_sts_action(self, kind):
        if kind[0] == actions.KIND_END:
            return sts.Action(sts.ActionType.END_TURN)
        if kind[0] == actions.KIND_PLAY:
            _, h, t = kind
            if t is None:
                return sts.Action(sts.ActionType.CARD, h)
            return sts.Action(sts.ActionType.CARD, h, t)
        if kind[0] == actions.KIND_SELECT:
            return sts.Action(sts.ActionType.SINGLE_CARD_SELECT, kind[1])
        raise ValueError(kind)
