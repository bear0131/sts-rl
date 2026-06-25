"""SimCombatBackend: a SimGameEnv scoped to a single random combat, for combat-
head pretraining (doc/night_0625/combat_pretrain_decision.md).

It REUSES the whole-run thick backend's combat handling verbatim — same
legal_semantic_actions / step / obs as SimGameEnv — so the thin SlayTheSpireEnv
on top exposes the SAME 149-action space + masks the whole-run policy trains on,
and combat-pretrained weights transfer to `rl.train --init-from` with no
remapping. (The standalone SimCombatEnv speaks the narrower 71-id combat space and
does NOT line up with the policy head; that is why pretraining goes through here.)

Only two things differ from SimGameEnv:
  - reset() drops straight into a randomized combat (no Neow / map / Spire run),
  - the episode ends when that combat ends (no reward screen / run continuation).
"""

import numpy as np

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts

from .game_env import SimGameEnv, _GC_UNDECIDED


class SimCombatBackend(SimGameEnv):
    """One random combat = one episode, in the whole-run 149-action space.

    factory_builder(np_random) -> a gc_factory() -> (GameContext, encounter_name)
    (see factories.random_combat_factory). The backend builds the factory once off
    its seeded np_random so the loadout stream is reproducible per seed."""

    def __init__(self, factory_builder, ascension=0):
        super().__init__(ascension=ascension)
        self._factory_builder = factory_builder
        self._factory = None

    def reset(self, *, seed=None):
        if seed is None and self._np_random is None:
            self._np_random = np.random.default_rng()
        elif seed is not None:
            self._np_random = np.random.default_rng(seed)
        if self._factory is None:
            self._factory = self._factory_builder(self._np_random)
        self.gc, encounter = self._factory()
        self.rmap = None  # no map in an isolated combat
        self.bc = sts.create_battle(self.gc, encounter)
        self._prev_hp = self.gc.cur_hp
        return self._canonical(), self._info()

    @property
    def is_over(self):
        # Combat-scoped terminal: the fight ended (victory or death). gc.outcome
        # stays UNDECIDED until exit_battle (which we never call), so the live bc
        # is the authority on whether the combat is done.
        if self.gc.outcome != _GC_UNDECIDED:
            return True
        return self.bc is None or self.bc.is_battle_over

    def _settle(self):
        # Intentionally a no-op: SimGameEnv._settle would exit_battle into the
        # reward screen and march the run on. Here the episode IS the one combat,
        # so we leave the finished bc in place for the terminal obs/reward.
        pass

    def combat_state(self):
        """(player_hp, outcome_int) off the live BattleContext, for
        CombatRewardWrapper. outcome: 0 undecided / 1 victory / 2 loss (the
        combat-side Outcome enum; see combat_env._OUTCOME_*)."""
        raw = self.bc.to_dict()
        return raw["player"]["hp"], raw["outcome"]
