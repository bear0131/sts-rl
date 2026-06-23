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

v1 scope (narrow vertical slice): drives act 1 end to end. Shops are a stub
(only "leave"); events allow option 0 only (see _EVENT_* notes). Both gaps are
absorbed here, not leaked to the semantic action set's shape.
"""

import numpy as np

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts
from schemas.combat_state import (
    normalize_sim, normalize_sim_overworld, normalize_sim_global, sim_phase,
)

from . import actions
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

# Events whose option count we don't yet read from the sim get option 0 only.
# act-1 events are few; this keeps a run from stalling without a binding change.
# TODO: export event option count from the sim, then enumerate real choices.
_EVENT_MAX_OPTIONS = 1


def _tail(gc):
    return gc.screen_state_name.split(".")[-1]


class SimGameEnv:
    """Whole-run environment over the sim.

    reset(seed) -> drive to the first decision point.
    legal_semantic_actions() -> list of game_actions tuples currently legal.
    step(action) -> (canonical, reward, terminated, info).
    """

    def __init__(self, character=None, ascension=0):
        self._character = character or sts.CharacterClass.IRONCLAD
        self._ascension = ascension
        self.gc = None
        self.bc = None            # live BattleContext while in combat, else None
        self.rmap = None          # RunMap for the current act
        self._np_random = None
        self._prev_hp = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None):
        if seed is None and self._np_random is None:
            self._np_random = np.random.default_rng()
        elif seed is not None:
            self._np_random = np.random.default_rng(seed)
        game_seed = int(self._np_random.integers(0, (1 << 63) - 1))

        self.gc = sts.GameContext(self._character, game_seed, self._ascension)
        self.bc = None
        self._rebuild_map()
        self._prev_hp = self.gc.cur_hp
        # GameContext starts on the Neow event; that IS the first decision point.
        return self._canonical(), self._info()

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

    def _canonical(self):
        """Backend-agnostic state dict. In combat: the combat canonical (+phase).
        Out of combat: the overworld canonical, with map options filled in from
        the rebuilt SpireMap (the schema can't do that without depending on env)."""
        if self.bc is not None:
            d = normalize_sim(self.bc.to_dict())
            d["phase"] = "combat"
            return d
        d = normalize_sim_overworld(self.gc)
        if d["phase"] == "map":
            x, y = self.gc.cur_map_node_x, self.gc.cur_map_node_y
            opts = self.rmap.reachable(x, y)
            d["options"] = {
                "nodes": [(x2, self.rmap.room_type(x2, (y + 1) if y >= 0 else 0))
                          for x2 in opts]
            }
        return d

    def _info(self):
        return {"phase": self.phase(), "legal": self.legal_semantic_actions()}

    # --------------------------------------------------------------- obs
    def obs_canonical(self):
        """The assembled OBS canonical: the combat-or-overworld block from
        _canonical(), plus the run-global block (present every phase, §3/§4).
        This is what encode.encode() consumes."""
        d = self._canonical()
        d["global"] = normalize_sim_global(self.gc)
        return d

    def obs(self):
        """The structured Dict observation (encode v2) for the current state."""
        return encode.encode(self.obs_canonical())

    @staticmethod
    def observation_space():
        """The gymnasium Dict space (shared with SimCombatEnv via encode)."""
        return encode.obs_space()

    # --------------------------------------------------------------- terminal
    @property
    def is_over(self):
        """Run ended: player died, or we cleared the act-1 boss into act 2 (v1
        stops at the end of act 1)."""
        if self.gc.outcome != _GC_UNDECIDED:
            return True
        return self.gc.act > 1

    # --------------------------------------------------- legal semantic actions
    def legal_semantic_actions(self):
        if self.is_over:
            return []
        if self.bc is not None:
            # combat: reuse the verified 71-id mask, wrap each set id as COMBAT.
            mask = combat_env.legal_mask(self.bc)
            return [ga.combat(i) for i in np.flatnonzero(mask)]

        phase = sim_phase(self.gc)
        if phase == "event":
            # Neow is an event screen too; its options are the 4 Neow rewards.
            if self.gc.cur_event_name == "NEOW":
                return [ga.neow(i) for i in range(4)]
            return [ga.event(i) for i in range(_EVENT_MAX_OPTIONS)]
        if phase == "map":
            x, y = self.gc.cur_map_node_x, self.gc.cur_map_node_y
            return [ga.map_node(x2) for x2 in self.rmap.reachable(x, y)]
        if phase == "combat_reward":
            info = self.gc.get_reward_info()
            acts = []
            acts += [ga.claim(ga.CLAIM_GOLD, i) for i in range(len(info["gold"]))]
            acts += [ga.claim(ga.CLAIM_RELIC, i) for i in range(len(info["relics"]))]
            acts += [ga.claim(ga.CLAIM_POTION, i) for i in range(len(info["potions"]))]
            if info["card_count"]:
                n = len(self.gc.get_card_reward())
                acts += [ga.reward_card(j) for j in range(n)]
                acts.append(ga.skip_cards())
            acts.append(ga.proceed())
            return acts
        if phase == "card_select":
            return [ga.select(k) for k in range(len(self.gc.get_select_cards()))]
        if phase == "rest":
            # offer rest/smith always; the sim ignores options the campfire lacks,
            # but to stay honest we only surface the two universal ones in v1.
            # TODO: read available campfire options (recall/toke/dig relics).
            return [ga.campfire("rest"), ga.campfire("smith"), ga.proceed()]
        if phase == "chest":
            return [ga.chest(True), ga.chest(False)]
        if phase == "boss_relic":
            return [ga.boss_relic(i) for i in range(3)]
        if phase == "shop":
            return [ga.shop_proceed()]  # v1 stub
        # unknown screen: nothing we can legally do
        return []

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.is_over:
            raise RuntimeError("run is over; call reset()")
        self._apply(action)
        self._settle()  # pump through any auto-resolving screens to next decision

        terminated = self.is_over
        hp = self.gc.cur_hp
        reward = float(hp - self._prev_hp) / 200.0  # hp-delta shaping (placeholder)
        self._prev_hp = hp
        if terminated:
            # +1 cleared act 1, -1 died (v1 terminal signal; refine later)
            reward += 1.0 if self.gc.outcome == _GC_UNDECIDED and self.gc.act > 1 else -1.0
        return self._canonical(), reward, terminated, self._info()

    def _apply(self, action):
        """Dispatch one semantic action against the sim. Mirrors replay.Driver
        but driven by the chosen action rather than a log."""
        verb = action[0]
        gc = self.gc

        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            self.bc.execute(combat_env.to_sts_action(kind))
            return
        if verb == ga.NEOW:
            gc.choose_neow_option(action[1]); return
        if verb == ga.EVENT:
            gc.choose_event_option(action[1]); return
        if verb == ga.MAP:
            gc.transition_to_map_node(action[1])
            if _tail(gc) == "BATTLE":
                self._start_battle()
            return
        if verb == ga.CLAIM:
            kind, idx = action[1], action[2]
            if kind == ga.CLAIM_GOLD:    gc.claim_gold(idx)
            elif kind == ga.CLAIM_RELIC: gc.claim_relic(idx)
            elif kind == ga.CLAIM_POTION: gc.claim_potion(idx)
            return
        if verb == ga.REWARD_CARD:
            gc.pick_reward_card(gc.get_card_reward()[action[1]]); return
        if verb == ga.SKIP_CARDS:
            gc.skip_reward_cards(); return
        if verb == ga.SELECT:
            gc.choose_select_card_option(action[1]); return
        if verb == ga.CAMPFIRE:
            gc.choose_campfire_option(_CAMPFIRE[action[1]]); return
        if verb == ga.CHEST:
            gc.choose_treasure_room(action[1]); return
        if verb == ga.BOSS_RELIC:
            gc.choose_boss_relic(action[1]); return
        if verb == ga.PROCEED:
            gc.reward_proceed(); return
        if verb == ga.SHOP_PROCEED:
            gc.shop_proceed(); return
        raise ValueError(f"unhandled semantic action: {action!r}")

    def _start_battle(self):
        self.bc = sts.create_battle(self.gc, self._encounter_name())

    def _encounter_name(self):
        tail = str(self.gc.encounter).split(".")[-1]
        return " ".join(w.capitalize() for w in tail.split("_"))

    def _settle(self):
        """After an action, advance through states that don't need an agent
        decision: end a finished combat (exit_battle opens rewards), and detect
        an act transition to rebuild the map. Stops at the next real decision."""
        if self.bc is not None and self.bc.is_battle_over:
            self.bc.exit_battle(self.gc)
            self.bc = None
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
