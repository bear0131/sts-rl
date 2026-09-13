"""Whole-run training constrained to Strike and Defend cards."""

from __future__ import annotations

from env.core import game_actions as ga
from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv
from rl.training.reward import ShapedRewardWrapper
from rl.tasks.wholerun import WholeRunTask, _MAX_EPISODE_STEPS

from gymnasium.wrappers import TimeLimit


_BASIC_DECK = [
    *([("STRIKE_RED", 0)] * 5),
    *([("DEFEND_RED", 0)] * 4),
]
_ALLOWED_CARDS = frozenset({"STRIKE_RED", "DEFEND_RED"})


class BasicCardsGameEnv(SimGameEnv):
    """Full-run backend that permits only Strike and Defend in the deck.

    Card removals and upgrades are retained. Cards granted or transformed by
    Neow, events, relics, or other overworld effects are removed before the next
    observation or battle.
    """

    def _sanitize_deck(self) -> None:
        current = [
            (card_id, int(upgrades))
            for card_id, upgrades in self.gc.deck_ids()
        ]
        allowed = [card for card in current if card[0] in _ALLOWED_CARDS]
        if allowed != current:
            self.gc.set_deck(allowed)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.gc.set_deck(_BASIC_DECK)
        return self._canonical(), self._info()

    def legal_semantic_actions(self):
        actions = super().legal_semantic_actions()
        return [
            action
            for action in actions
            if action[0] != ga.REWARD_CARD
            and not (
                action[0] == ga.SHOP_BUY
                and action[1] == ga.SHOP_CARD
            )
        ]

    def step(self, action):
        _canonical, reward, terminated, _info = super().step(action)
        # Do not mutate the deck while the simulator is holding indices into a
        # live upgrade/remove/transform selection. Sanitize as soon as that
        # screen resolves; _start_battle is the final guard for event combats.
        if self.bc is None and self.phase() != "card_select":
            self._sanitize_deck()
        return self._canonical(), reward, terminated, self._info()

    def _start_battle(self):
        self._sanitize_deck()
        super()._start_battle()


class BasicWholeRunTask(WholeRunTask):
    name = "wholerun-basic"
    ckpt_prefix = "basic_wholerun_ppo"
    defaults = {
        "steps": 100_000,
        "metrics_window": 100,
        "out": "rl/runs/whole-run-basic",
    }

    def make_env_fn(self, args):
        ascension = args.ascension

        def _thunk():
            env = SlayTheSpireEnv(
                backend_factory=lambda: BasicCardsGameEnv(
                    ascension=ascension,
                    global_map_view=args.global_map_view,
                ),
                global_map_view=args.global_map_view,
                card_language=args.card_language,
                combat_language=args.combat_language,
                deck_size_observation=args.deck_size_observation,
            )
            env = ShapedRewardWrapper(env)
            return TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)

        return _thunk

    def tag(self, args, run_id):
        map_tag = "-globalmap" if args.global_map_view else ""
        card_tag = "-cardlang" if args.card_language else ""
        if args.combat_language:
            card_tag += "-combatlang"
        deck_tag = "-decksize" if args.deck_size_observation else ""
        return (
            f"whole-run-basic{map_tag}{card_tag}{deck_tag}-asc{args.ascension}"
            f"-s{args.seed}-{run_id}"
        )
