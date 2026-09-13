"""SimCombatBackend: a SimGameEnv scoped to a random-combat GAUNTLET, for combat-
head pretraining (doc/night_0625/combat_pretrain_decision.md).

One episode = ONE random loadout (deck/relics/starting HP) fighting an ENDLESS
stream of random encounters. HP carries over between fights and there is no
healing, so a fight cleared is worth +1 (CombatGauntletRewardWrapper) but each
fight spends HP; the episode ends only when the player DIES. Surviving more fights
is the whole objective, so "spend HP wisely / block when it matters" is learned
implicitly from the structure — no explicit HP-delta or death-penalty reward term.

It REUSES the whole-run thick backend's combat handling verbatim — same
legal_semantic_actions / step / obs as SimGameEnv — so the thin SlayTheSpireEnv
on top exposes the SAME 181-action space + masks the whole-run policy trains on,
and combat-pretrained weights transfer to `rl.training.trainer --task wholerun
--init-from` with no
remapping. (The standalone SimCombatEnv speaks the narrower 71-id combat space and
does NOT line up with the policy head; that is why pretraining goes through here.)

Differences from SimGameEnv:
  - reset() drops straight into a randomized combat (no Neow / map / Spire run),
  - on a WON fight _settle() rolls the next random encounter on the same loadout
    (HP carried), instead of opening the reward screen and marching the run on,
  - is_over (episode terminal) is the player's DEATH, not a single fight ending.
"""

import numpy as np

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts

from .game_env import SimGameEnv, _GC_UNDECIDED

# BattleContext.to_dict()["outcome"] uses the combat-side Outcome enum:
# UNDECIDED=0, PLAYER_VICTORY=1, PLAYER_LOSS=2 (see combat_env._OUTCOME_*).
_OUTCOME_LOSS = 2


class SimCombatBackend(SimGameEnv):
    """A random-loadout combat gauntlet = one episode, in the whole-run 181-action
    space.

    factory_builder(np_random) -> (make_gc, next_encounter)
    (see factories.random_gauntlet_factory). The backend builds these once off its
    seeded np_random so the loadout/encounter stream is reproducible per seed."""

    def __init__(self, factory_builder, ascension=0):
        super().__init__(ascension=ascension)
        self._factory_builder = factory_builder
        self._make_gc = None
        self._next_encounter = None
        self.combats_won = 0

    def reset(self, *, seed=None):
        if seed is None and self._np_random is None:
            self._np_random = np.random.default_rng()
        elif seed is not None:
            self._np_random = np.random.default_rng(seed)
        if self._make_gc is None:
            self._make_gc, self._next_encounter = self._factory_builder(self._np_random)
        self.gc = self._make_gc()
        self.rmap = None  # no map in an isolated gauntlet
        self.combats_won = 0
        self.bc = sts.create_battle(self.gc, self._next_encounter(self.gc))
        self._prev_hp = self.gc.cur_hp
        return self._canonical(), self._info()

    @property
    def is_over(self):
        # Gauntlet terminal = the player is DEAD. A won fight does NOT end the
        # episode (_settle rolls the next one), and gc.outcome stays UNDECIDED
        # throughout (we never exit_battle), so the live bc is the authority — and
        # is_battle_over short-circuits, so mid-fight this never calls to_dict().
        if self.gc.outcome != _GC_UNDECIDED:
            return True
        return self.bc is not None and self.bc.is_battle_over and self._lost()

    def _lost(self):
        return self.bc.to_dict()["outcome"] == _OUTCOME_LOSS

    def _settle(self):
        # After a step: if the current fight just ended in VICTORY (player alive),
        # carry HP onto the gc and roll the NEXT random encounter on the same
        # loadout. On a loss we leave the finished bc in place so is_over -> True
        # and the terminal obs/reward read the death.
        if self.bc is None or not self.bc.is_battle_over or self._lost():
            return
        self.combats_won += 1
        # gc.cur_hp is stale mid-run; the bc owns the player's HP. Carry the live
        # post-fight HP onto the gc so create_battle starts the next fight there.
        self.gc.cur_hp = self.bc.to_dict()["player"]["hp"]
        self.bc = sts.create_battle(self.gc, self._next_encounter(self.gc))
        self._prev_hp = self.gc.cur_hp

    def combat_state(self):
        """(player_hp, outcome_int) off the live BattleContext. outcome: 0 undecided
        / 1 victory / 2 loss (combat-side Outcome enum; see combat_env._OUTCOME_*)."""
        raw = self.bc.to_dict()
        return raw["player"]["hp"], raw["outcome"]
