"""Real-engine STS2 Gym environment. Every emitted candidate is policy-controlled."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .backend import load_backend
from .encoding import ACTION_DIM, ACTION_VERSION, OBS_VERSION, STATE_DIM, canonical, features, public_card, visible_combat

REWARD_VERSION = "sts2-sparse-win-v1"


@dataclass(frozen=True)
class EnvConfig:
    task: str = "combat"
    character: str = "Ironclad"
    ascension: int = 0
    max_steps: int = 3000
    max_actions: int = 512

    def __post_init__(self):
        if self.task not in ("combat", "run") or self.character != "Ironclad":
            raise ValueError("Initial supported contract: Ironclad combat/run")
        if not 0 <= self.ascension <= 10 or min(self.max_steps, self.max_actions) < 1:
            raise ValueError("Invalid ascension or budget")


class Sts2Env(gym.Env):
    metadata = {}

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config
        self.native, self.manifest = load_backend()
        self.action_space = spaces.Discrete(config.max_actions)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "candidates": spaces.Box(-np.inf, np.inf, (config.max_actions, ACTION_DIM), np.float32),
            "mask": spaces.MultiBinary(config.max_actions),
        })
        self.rm = None
        self._done = True

    @property
    def contract(self):
        return dict(config=asdict(self.config), observation=OBS_VERSION, action=ACTION_VERSION,
                    reward=REWARD_VERSION, backend=self.manifest["pin"],
                    source_hash=self.manifest["source_hash"], binary_sha256=self.manifest["binary_sha256"])

    def _new_run(self, seed):
        return self.native.RunManager(seed=seed, character=self.config.character,
                    ascension_level=self.config.ascension, neow_start=True,
                    interactive_card_removal=True, interactive_card_upgrade=True,
                    interactive_card_transform=True, interactive_potion_swap=True,
                    direct_event_relic_grant=True, act1="overgrowth")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed_value = int(seed if seed is not None else self.np_random.integers(0, 2**31-1))
        if not 0 <= self.seed_value < 2**31:
            raise ValueError("Seed must fit positive i32")
        self.rm = self._new_run(self.seed_value)
        self.actions_taken = []
        self.prelude = []
        if self.config.task == "combat":
            # Use a real run's initial encounter; no fabricated rule state.
            for _ in range(100):
                if self.rm.phase() == "combat":
                    break
                if self.rm.phase() == "event":
                    self.rm.materialize_event()
                actions = self.rm.available_actions()
                if not actions or self.rm.is_over():
                    raise RuntimeError("Failed to reach initial combat")
                action = actions[0]
                self._execute(action)
                self.prelude.append(action)
            else:
                raise RuntimeError("Prelude exceeded 100 actions")
        self.steps = 0
        self._done = False
        self.outcome = "ongoing"
        self._refresh()
        return self._observation(), self._info()

    def _execute(self, action):
        result = self.rm.take_action(action)
        if result not in ("advanced", "run_over"):
            raise RuntimeError(f"Engine rejected legal candidate: {result}: {action}")

    def _refresh(self):
        if self.rm.phase() == "event" and not self.rm.is_over():
            self.rm.materialize_event()
        cs = self.rm.combat_state()
        self.visible = dict(phase=self.rm.phase(), hp=self.rm.current_hp,
            max_hp=self.rm.max_hp, gold=self.rm.gold, act=self.rm.act_index,
            floor=self.rm.total_floor(), ascension=self.config.ascension,
            deck=self.rm.master_deck_detailed(), relics=self.rm.relics(), potions=self.rm.potions())
        pending = None
        if cs is not None:
            violation = cs.check_invariants("sts2-env")
            if violation:
                raise RuntimeError(f"Engine invariant: {violation}")
            pending = cs.pending_choice()
            self.visible["combat"] = visible_combat(self.native.state_to_dict(cs), pending)
        if self.config.task == "run":
            self.visible.update(map=self.rm.full_map(), map_coord=self.rm.map_coord(),
                card_rewards=self.rm.card_reward_option_ids(),
                card_reward_upgraded=self.rm.card_reward_option_upgraded(),
                card_reward_bundles=self.rm.card_reward_bundles(),
                relic_rewards=self.rm.relic_reward_ids(), boss_relics=self.rm.boss_relic_options(),
                shop=self.rm.shop_inventory(), rest=self.rm.rest_site_options(),
                smith=self.rm.smith_options(), removal=self.rm.card_removal_options(),
                upgrade_indices=self.rm.upgrade_select_deck_indices(),
                transform_indices=self.rm.transform_select_deck_indices(),
                enchant_indices=self.rm.enchant_select_deck_indices(),
                enchantment=self.rm.pending_enchantment, grants=self.rm.card_grant_options(),
                potion_offer=self.rm.pending_potion_swap)
            if self.rm.phase() == "event":
                self.visible.update(event=self.rm.current_event_name(), event_options=self.rm.event_option_infos())
        actions = [] if self._done and self.outcome != "truncated" else self.rm.available_actions()
        # RunManager's convenience multi-pick actions take the FIRST k cards.
        # Enumerate actual subsets instead, preserving native indices privately.
        if pending is not None and actions:
            actions = []
            n = len(pending["cards"])
            lo = 0 if pending.get("can_skip") else pending["min_select"]
            for k in range(lo, min(n, pending["max_select"])+1):
                for selection in combinations(range(n), k):
                    actions.append(dict(action="combat_resolve_choice_selection", selection=list(selection)))
                    if len(actions) > self.config.max_actions:
                        raise RuntimeError("Choice set exceeds declared max_actions; no candidates were dropped")
        if len(actions) > self.config.max_actions:
            raise RuntimeError("Legal action capacity exceeded; no candidates were dropped")
        self.candidates = []
        for action in actions:
            desc = deepcopy(action)
            if action["action"] == "combat_resolve_choice_selection":
                desc = dict(action=action["action"], kind=pending["kind"],
                            selected=sorted([public_card(pending["cards"][i]) for i in action["selection"]], key=canonical))
            elif action["action"] == "play_card":
                desc["card"] = self.visible["combat"]["deck"]["hand"][action["hand_idx"]]
            self.candidates.append((action, desc))
        # Selection order must not expose native draw-pile indices.
        if pending is not None:
            self.candidates.sort(key=lambda pair: canonical(pair[1]))
        if not self.candidates and not self._done:
            raise RuntimeError(f"Nonterminal state has no legal actions ({self.rm.phase()})")

    def _observation(self):
        action_features = np.zeros((self.config.max_actions, ACTION_DIM), dtype=np.float32)
        for i, (_, desc) in enumerate(self.candidates):
            action_features[i] = features(desc, ACTION_DIM)
        return dict(state=features(self.visible, STATE_DIM), candidates=action_features,
                    mask=self.action_masks().astype(np.int8))

    def action_masks(self):
        mask = np.zeros(self.config.max_actions, dtype=bool)
        mask[:len(self.candidates)] = True
        return mask

    def _info(self):
        return dict(outcome=self.outcome, phase=self.rm.phase(), floor=self.rm.total_floor(),
                    act=self.rm.act_index, hp=self.rm.current_hp, steps=self.steps)

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required after episode end")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(self.candidates):
            raise ValueError("Action is not in the current candidate set")
        native_action = deepcopy(self.candidates[int(action)][0])
        self._execute(native_action)
        self.actions_taken.append(native_action)
        self.steps += 1
        terminated = self.rm.is_over() or (self.config.task == "combat" and self.rm.phase() != "combat")
        won = self.rm.player_won if self.config.task == "run" else self.rm.current_hp > 0 and terminated
        truncated = self.steps >= self.config.max_steps and not terminated
        self.outcome = "win" if terminated and won else "death" if terminated else "truncated" if truncated else "ongoing"
        self._done = terminated or truncated
        self._refresh()
        return self._observation(), (1.0 if won else -1.0) if terminated else 0.0, terminated, truncated, self._info()

    def clone(self):
        twin = object.__new__(type(self))
        twin.__dict__ = dict(self.__dict__)
        twin.rm = self.rm.clone()
        twin.actions_taken = deepcopy(self.actions_taken)
        twin.prelude = deepcopy(self.prelude)
        twin._np_random = deepcopy(self.np_random)
        twin._refresh()
        return twin

    def fingerprint(self):
        # Observation alone is insufficient for recovery validation: include
        # ordered internal card piles and every exposed RNG stream cursor.
        cs = self.rm.combat_state()
        value = dict(visible=self.visible, rng=self.rm.rng_counters(),
                     combat=self.native.state_to_dict(cs) if cs else None,
                     pending=cs.pending_choice() if cs else None,
                     candidates=self.candidates, outcome=self.outcome)
        return hashlib.sha256(canonical(value).encode()).hexdigest()

    def snapshot(self):
        return dict(contract=self.contract, seed=self.seed_value, actions=deepcopy(self.actions_taken),
                    fingerprint=self.fingerprint())

    def restore(self, snapshot):
        if snapshot["contract"] != self.contract:
            raise ValueError("Snapshot contract/backend mismatch")
        self.reset(seed=snapshot["seed"])
        for action in snapshot["actions"]:
            indices = [i for i, (candidate, _) in enumerate(self.candidates) if candidate == action]
            if len(indices) != 1:
                raise RuntimeError("Recovery action not uniquely legal")
            self.step(indices[0])
        if self.fingerprint() != snapshot["fingerprint"]:
            raise RuntimeError("Replay recovery diverged")
        return self._observation(), self._info()

    def close(self):
        self.rm = None
