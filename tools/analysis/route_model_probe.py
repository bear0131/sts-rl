"""Probe the current whole-run route-planning path on synthetic maps.

This keeps the production path intact:

    synthetic canonical map
      -> env.core.encode.encode(global_map_view=True)
      -> rl.policies.pointer_policy.StSPolicyCore.action_logits()
      -> MAP_NODE[x] logits

The only probe-specific pieces are the synthetic map generator and the oracle
targets.  Elites receive a fixed negative reward; all non-elite nodes receive
unobserved random rewards, so the model can only learn their expectation and the
visible elite penalty.  This is intentional: it tests whether the existing map
DAG encoder and MAP_NODE pointer head can react to elites several floors ahead.

Example:

    python -m tools.analysis.route_model_probe --updates 80 --batch-size 32 --d-model 128 \
        --n-layers 2 --n-heads 4
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from env.core import action_space as A
from env.core import encode
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.pointer_policy import StSPointerPolicy, StSPolicyCore


ROWS = encode.MAP_ROWS
COLS = encode.MAP_COLS
NON_ELITE_ROOMS = ("MONSTER", "EVENT", "REST", "SHOP", "TREASURE")
MAP_BASE = A.seg_base("MAP_NODE")


@dataclass
class RouteMap:
    rooms: np.ndarray                 # [ROWS, COLS], dtype object, room names
    children: list[list[list[int]]]    # children[y][x] -> child columns in y + 1
    rewards: np.ndarray               # [ROWS, COLS], oracle-only scalar reward


@dataclass
class RouteCase:
    route: RouteMap
    current_x: int
    current_y: int
    target_x: int
    legal: tuple[int, ...]
    q_values: np.ndarray              # [COLS], -inf outside legal choices


def _non_elite_reward(rng: random.Random, reward_low: float, reward_high: float) -> float:
    return rng.uniform(reward_low, reward_high)


def _room_reward(
    room: str,
    rng: random.Random,
    *,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
) -> float:
    if room == "ELITE":
        return elite_penalty
    return _non_elite_reward(rng, reward_low, reward_high)


def _random_children(rng: random.Random, x: int) -> list[int]:
    candidates = [c for c in (x - 1, x, x + 1) if 0 <= c < COLS]
    n = rng.randint(1, len(candidates))
    return sorted(rng.sample(candidates, n))


def generate_random_map(
    rng: random.Random,
    *,
    elite_prob: float,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
) -> RouteMap:
    rooms = np.empty((ROWS, COLS), dtype=object)
    rewards = np.zeros((ROWS, COLS), dtype=np.float32)
    for y in range(ROWS):
        for x in range(COLS):
            room = "ELITE" if rng.random() < elite_prob else rng.choice(NON_ELITE_ROOMS)
            rooms[y, x] = room
            rewards[y, x] = _room_reward(
                room,
                rng,
                elite_penalty=elite_penalty,
                reward_low=reward_low,
                reward_high=reward_high,
            )

    children: list[list[list[int]]] = [[[] for _ in range(COLS)] for _ in range(ROWS)]
    for y in range(ROWS - 1):
        for x in range(COLS):
            children[y][x] = _random_children(rng, x)
    return RouteMap(rooms=rooms, children=children, rewards=rewards)


def oracle_values(route: RouteMap, discount: float) -> np.ndarray:
    values = np.array(route.rewards, dtype=np.float32)
    for y in range(ROWS - 2, -1, -1):
        for x in range(COLS):
            future = max(values[y + 1, child] for child in route.children[y][x])
            values[y, x] = route.rewards[y, x] + discount * future
    return values


def make_case(
    route: RouteMap,
    rng: random.Random,
    *,
    discount: float,
    min_current_y: int,
    max_current_y: int,
) -> RouteCase:
    max_current_y = min(max_current_y, ROWS - 2)
    for _ in range(100):
        y = rng.randint(min_current_y, max_current_y)
        x = rng.randrange(COLS)
        legal = tuple(route.children[y][x])
        if len(legal) >= 2:
            break
    else:
        y, x = 0, COLS // 2
        legal = tuple(route.children[y][x])
        if len(legal) < 2:
            legal = tuple(sorted(set(legal) | {max(0, x - 1), min(COLS - 1, x + 1)}))
            route.children[y][x] = list(legal)

    values = oracle_values(route, discount)
    q = np.full(COLS, -np.inf, dtype=np.float32)
    for child in legal:
        q[child] = values[y + 1, child]
    target = int(np.argmax(q))
    return RouteCase(route=route, current_x=x, current_y=y, target_x=target,
                     legal=legal, q_values=q)


def _downstream(route: RouteMap, choices: set[tuple[int, int]]) -> set[tuple[int, int]]:
    seen = set(choices)
    frontier = list(choices)
    while frontier:
        x, y = frontier.pop()
        if y >= ROWS - 1:
            continue
        for child_x in route.children[y][x]:
            child = (child_x, y + 1)
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def full_view(route: RouteMap, current_x: int, current_y: int) -> list[dict]:
    if current_y >= ROWS - 1:
        choices = set()
    else:
        choices = {
            (child_x, current_y + 1)
            for child_x in route.children[current_y][current_x]
        }
    reachable = _downstream(route, choices)
    nodes = []
    for y in range(ROWS):
        for x in range(COLS):
            nodes.append({
                "x": x,
                "y": y,
                "room": str(route.rooms[y, x]),
                "children": route.children[y][x] if y < ROWS - 1 else [],
                "current": x == current_x and y == current_y,
                "choice": (x, y) in choices,
                "reachable": (x, y) in reachable,
            })
    return nodes


def canonical(case: RouteCase) -> dict:
    route = case.route
    next_y = case.current_y + 1
    nodes = (
        [(x, str(route.rooms[next_y, x])) for x in case.legal]
        if next_y < ROWS else []
    )
    return {
        "phase": "map",
        "arg_step": "root",
        "options": {
            "nodes": nodes,
            "full_map": full_view(route, case.current_x, case.current_y),
        },
        "global": {
            "character": "IRONCLAD",
            "ascension": 0,
            "act": 1,
            "floor": case.current_y + 1,
            "boss": "SLIME_BOSS",
            "relics": [],
            "potions": [],
            "gold": 0,
            "deck": [],
        },
    }


def state_case(
    route: RouteMap,
    current_x: int,
    current_y: int,
    *,
    discount: float,
) -> RouteCase:
    if current_y >= ROWS - 1:
        q = np.full(COLS, -np.inf, dtype=np.float32)
        return RouteCase(
            route=route,
            current_x=current_x,
            current_y=current_y,
            target_x=current_x,
            legal=(),
            q_values=q,
        )
    legal = tuple(route.children[current_y][current_x])
    values = oracle_values(route, discount)
    q = np.full(COLS, -np.inf, dtype=np.float32)
    for child in legal:
        q[child] = values[current_y + 1, child]
    target = int(np.argmax(q)) if legal else current_x
    return RouteCase(route=route, current_x=current_x, current_y=current_y,
                     target_x=target, legal=legal, q_values=q)


def encode_cases(cases: Iterable[RouteCase], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = [
        encode.encode(
            canonical(case),
            global_map_view=True,
            deck_size_observation=True,
            neow_observation=True,
            event_option_observation=True,
        )
        for case in cases
    ]
    keys = encoded[0].keys()
    return {
        key: torch.as_tensor(np.stack([obs[key] for obs in encoded]), device=device)
        for key in keys
    }


def make_batch(
    rng: random.Random,
    batch_size: int,
    *,
    elite_prob: float,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
    discount: float,
    min_current_y: int,
    max_current_y: int,
) -> list[RouteCase]:
    cases = []
    for _ in range(batch_size):
        route = generate_random_map(
            rng,
            elite_prob=elite_prob,
            elite_penalty=elite_penalty,
            reward_low=reward_low,
            reward_high=reward_high,
        )
        cases.append(
            make_case(
                route,
                rng,
                discount=discount,
                min_current_y=min_current_y,
                max_current_y=max_current_y,
            )
        )
    return cases


def masked_map_logits(core: StSPolicyCore, obs: dict[str, torch.Tensor],
                      cases: list[RouteCase]) -> torch.Tensor:
    logits = core.action_logits(obs)[:, MAP_BASE:MAP_BASE + COLS]
    mask = torch.full_like(logits, -1.0e9)
    for b, case in enumerate(cases):
        mask[b, list(case.legal)] = 0.0
    return logits + mask


def evaluate_random(
    core: StSPolicyCore,
    rng: random.Random,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
    elite_prob: float,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
    discount: float,
    min_current_y: int,
    max_current_y: int,
) -> dict[str, float]:
    total = 0
    exact = 0
    regret = 0.0
    elite_rate = 0.0
    with torch.no_grad():
        for _ in range(batches):
            cases = make_batch(
                rng,
                batch_size,
                elite_prob=elite_prob,
                elite_penalty=elite_penalty,
                reward_low=reward_low,
                reward_high=reward_high,
                discount=discount,
                min_current_y=min_current_y,
                max_current_y=max_current_y,
            )
            obs = encode_cases(cases, device)
            logits = masked_map_logits(core, obs, cases)
            chosen = torch.argmax(logits, dim=1).cpu().numpy()
            for pick, case in zip(chosen, cases):
                target = case.target_x
                exact += int(int(pick) == target)
                regret += float(case.q_values[target] - case.q_values[int(pick)])
                next_y = case.current_y + 1
                elite_rate += float(case.route.rooms[next_y, int(pick)] == "ELITE")
                total += 1
    return {
        "exact": exact / max(1, total),
        "regret": regret / max(1, total),
        "chosen_next_elite": elite_rate / max(1, total),
    }


def make_blocked_case(
    rng: random.Random,
    *,
    elite_distance: int,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
    discount: float,
) -> RouteCase:
    """Two visible branches; the right branch has an unavoidable future elite.

    ``elite_distance`` is measured in map rows ahead of the current node.  With
    current_y=0, distance=1 means the immediate choice node is elite; distance=4
    means the elite is four floors ahead.
    """
    current_x, current_y = 3, 0
    safe_x, blocked_x = 2, 4
    elite_row = max(1, min(ROWS - 1, elite_distance))

    rooms = np.empty((ROWS, COLS), dtype=object)
    rewards = np.zeros((ROWS, COLS), dtype=np.float32)
    for y in range(ROWS):
        for x in range(COLS):
            room = rng.choice(NON_ELITE_ROOMS)
            rooms[y, x] = room
            rewards[y, x] = _non_elite_reward(rng, reward_low, reward_high)
    # Mirror the two relevant corridors so the future elite is the only
    # systematic difference between the legal choices; other nodes remain random.
    for y in range(1, ROWS):
        room = rng.choice(NON_ELITE_ROOMS)
        reward = _non_elite_reward(rng, reward_low, reward_high)
        rooms[y, safe_x] = room
        rooms[y, blocked_x] = room
        rewards[y, safe_x] = reward
        rewards[y, blocked_x] = reward
    rooms[elite_row, blocked_x] = "ELITE"
    rewards[elite_row, blocked_x] = elite_penalty

    children: list[list[list[int]]] = [[[] for _ in range(COLS)] for _ in range(ROWS)]
    for y in range(ROWS - 1):
        for x in range(COLS):
            children[y][x] = [x]
    children[current_y][current_x] = [safe_x, blocked_x]
    route = RouteMap(rooms=rooms, children=children, rewards=rewards)
    values = oracle_values(route, discount=discount)
    q = np.full(COLS, -np.inf, dtype=np.float32)
    q[safe_x] = values[current_y + 1, safe_x]
    q[blocked_x] = values[current_y + 1, blocked_x]
    return RouteCase(route=route, current_x=current_x, current_y=current_y,
                     target_x=safe_x, legal=(safe_x, blocked_x), q_values=q)


class RouteProbeEnv(gym.Env):
    """Synthetic route MDP with the same observation/action surface as wholerun.

    Each step exposes only MAP_NODE actions.  The policy receives the full map
    through encode.encode(global_map_view=True), chooses a child column, receives
    that node's scalar reward, and continues until row 14.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int,
        elite_prob: float,
        elite_penalty: float,
        reward_low: float,
        reward_high: float,
        discount: float,
        blocked_episode_frac: float,
        blocked_distances: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.observation_space = encode.obs_space(
            global_map_view=True,
            deck_size_observation=True,
            neow_observation=True,
            event_option_observation=True,
        )
        self.action_space = spaces.Discrete(A.N_ACTIONS)
        self._base_seed = int(seed)
        self._rng = random.Random(seed)
        self.elite_prob = elite_prob
        self.elite_penalty = elite_penalty
        self.reward_low = reward_low
        self.reward_high = reward_high
        self.discount = discount
        self.blocked_episode_frac = blocked_episode_frac
        self.blocked_distances = blocked_distances
        self.route: RouteMap | None = None
        self.current_x = COLS // 2
        self.current_y = 0
        self._episode = 0
        self._kind = "random"
        self._distance = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(int(seed))
        else:
            self._rng.seed(self._base_seed + self._episode * 1009)
        self._episode += 1
        if self._rng.random() < self.blocked_episode_frac:
            self._kind = "blocked"
            self._distance = int(self._rng.choice(self.blocked_distances))
            case = make_blocked_case(
                self._rng,
                elite_distance=self._distance,
                elite_penalty=self.elite_penalty,
                reward_low=self.reward_low,
                reward_high=self.reward_high,
                discount=self.discount,
            )
            self.route = case.route
            self.current_x = case.current_x
            self.current_y = case.current_y
        else:
            self._kind = "random"
            self._distance = None
            self.route = generate_random_map(
                self._rng,
                elite_prob=self.elite_prob,
                elite_penalty=self.elite_penalty,
                reward_low=self.reward_low,
                reward_high=self.reward_high,
            )
            self.current_x = COLS // 2
            self.current_y = 0
        return self._obs(), self._info()

    def _case(self) -> RouteCase:
        assert self.route is not None
        return state_case(
            self.route,
            self.current_x,
            self.current_y,
            discount=self.discount,
        )

    def _obs(self):
        return encode.encode(
            canonical(self._case()),
            global_map_view=True,
            deck_size_observation=True,
            neow_observation=True,
            event_option_observation=True,
        )

    def _info(self):
        case = self._case()
        return {
            "route_kind": self._kind,
            "blocked_distance": self._distance,
            "current_y": self.current_y,
            "current_x": self.current_x,
            "oracle_next": case.target_x,
        }

    def action_masks(self):
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        if self.route is None or self.current_y >= ROWS - 1:
            return mask
        for x in self.route.children[self.current_y][self.current_x]:
            mask[MAP_BASE + x] = True
        return mask

    def step(self, action):
        assert self.route is not None
        aid = int(action)
        legal = set(np.flatnonzero(self.action_masks()).tolist())
        if aid not in legal:
            info = self._info()
            info["illegal"] = True
            return self._obs(), -1.0, False, True, info
        self.current_x = aid - MAP_BASE
        self.current_y += 1
        reward = float(self.route.rewards[self.current_y, self.current_x])
        terminated = self.current_y >= ROWS - 1
        return self._obs(), reward, terminated, False, self._info()


def evaluate_blocked(
    core: StSPolicyCore,
    rng: random.Random,
    *,
    distances: list[int],
    cases_per_distance: int,
    device: torch.device,
    elite_penalty: float,
    reward_low: float,
    reward_high: float,
    discount: float,
) -> list[dict[str, float]]:
    rows = []
    with torch.no_grad():
        for distance in distances:
            cases = [
                make_blocked_case(
                    rng,
                    elite_distance=distance,
                    elite_penalty=elite_penalty,
                    reward_low=reward_low,
                    reward_high=reward_high,
                    discount=discount,
                )
                for _ in range(cases_per_distance)
            ]
            obs = encode_cases(cases, device)
            logits = masked_map_logits(core, obs, cases)
            probs = torch.softmax(logits[:, [2, 4]], dim=1).cpu().numpy()
            raw = logits[:, [2, 4]].cpu().numpy()
            chosen = np.argmax(raw, axis=1)
            rows.append({
                "distance": float(distance),
                "avoid_rate": float(np.mean(chosen == 0)),
                "safe_prob": float(np.mean(probs[:, 0])),
                "blocked_prob": float(np.mean(probs[:, 1])),
                "logit_margin": float(np.mean(raw[:, 0] - raw[:, 1])),
            })
    return rows


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the production map encoder + MAP_NODE head "
                    "on synthetic elite-avoidance route maps.",
    )
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument(
        "--mode",
        choices=["supervised", "ppo"],
        default="supervised",
        help="supervised = oracle cross-entropy on MAP_NODE logits; ppo = "
             "MaskablePPO over a synthetic route MDP.",
    )
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--blocked-cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--elite-prob", type=float, default=0.18)
    parser.add_argument("--elite-penalty", type=float, default=-5.0)
    parser.add_argument("--reward-low", type=float, default=0.0)
    parser.add_argument("--reward-high", type=float, default=1.0)
    parser.add_argument(
        "--blocked-train-frac",
        type=float,
        default=0.5,
        help="Fraction of each training batch made of mirrored two-branch maps "
             "where one branch has an unavoidable elite several rows ahead. "
             "Set 0 for purely random-map training.",
    )
    parser.add_argument("--min-current-y", type=int, default=0)
    parser.add_argument("--max-current-y", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--blocked-distances",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 10],
        help="Rows ahead of the current node where the unavoidable elite appears.",
    )
    return parser


def run_ppo(args: argparse.Namespace, device: torch.device) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def make_env(i: int):
        def _thunk():
            return RouteProbeEnv(
                seed=args.seed + i * 97,
                elite_prob=args.elite_prob,
                elite_penalty=args.elite_penalty,
                reward_low=args.reward_low,
                reward_high=args.reward_high,
                discount=args.discount,
                blocked_episode_frac=args.blocked_train_frac,
                blocked_distances=tuple(args.blocked_distances),
            )

        return _thunk

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    venv = VecMonitor(venv)
    policy_kwargs = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        map_dag=True,
    )
    model = AccumulatingMaskablePPO(
        policy=StSPointerPolicy,
        env=venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.discount,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
        device=str(device),
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    print(
        "[route-probe-ppo] production RL path: RouteProbeEnv.action_masks "
        "-> AccumulatingMaskablePPO -> StSPointerPolicy",
        flush=True,
    )
    before_core = model.policy.core
    print_random_metrics(
        "[before PPO] random-map oracle",
        evaluate_random(
            before_core,
            random.Random(args.seed + 5000),
            batches=args.eval_batches,
            batch_size=args.eval_batch_size,
            device=device,
            elite_prob=args.elite_prob,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
            min_current_y=args.min_current_y,
            max_current_y=args.max_current_y,
        ),
    )
    print_blocked(
        "[before PPO] blocked-elite sensitivity",
        evaluate_blocked(
            before_core,
            random.Random(args.seed + 6000),
            distances=args.blocked_distances,
            cases_per_distance=args.blocked_cases,
            device=device,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
        ),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    after_core = model.policy.core
    print_random_metrics(
        "[after PPO] random-map oracle",
        evaluate_random(
            after_core,
            random.Random(args.seed + 7000),
            batches=args.eval_batches,
            batch_size=args.eval_batch_size,
            device=device,
            elite_prob=args.elite_prob,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
            min_current_y=args.min_current_y,
            max_current_y=args.max_current_y,
        ),
    )
    print_blocked(
        "[after PPO] blocked-elite sensitivity",
        evaluate_blocked(
            after_core,
            random.Random(args.seed + 8000),
            distances=args.blocked_distances,
            cases_per_distance=args.blocked_cases,
            device=device,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
        ),
    )
    venv.close()


def print_random_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: exact={metrics['exact'] * 100:5.1f}% "
        f"regret={metrics['regret']:+.3f} "
        f"chosen_next_elite={metrics['chosen_next_elite'] * 100:4.1f}%",
        flush=True,
    )


def print_blocked(prefix: str, rows: list[dict[str, float]]) -> None:
    print(prefix, flush=True)
    print("  dist  avoid   p_safe  p_block  margin", flush=True)
    for row in rows:
        print(
            f"  {int(row['distance']):>4}  "
            f"{row['avoid_rate'] * 100:5.1f}%  "
            f"{row['safe_prob']:.3f}   "
            f"{row['blocked_prob']:.3f}   "
            f"{row['logit_margin']:+.3f}",
            flush=True,
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.reward_low > args.reward_high:
        raise ValueError("--reward-low must be <= --reward-high")
    if args.min_current_y < 0 or args.max_current_y < args.min_current_y:
        raise ValueError("bad current-y range")
    if not (0.0 <= args.blocked_train_frac <= 1.0):
        raise ValueError("--blocked-train-frac must be in [0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = _device(args.device)

    if args.mode == "ppo":
        print(
            f"[route-probe] mode=ppo device={device} timesteps={args.timesteps} "
            f"n_envs={args.n_envs} n_steps={args.n_steps} "
            f"batch={args.batch_size} d_model={args.d_model} "
            f"layers={args.n_layers} heads={args.n_heads} "
            f"elite_penalty={args.elite_penalty} nonelite_reward="
            f"[{args.reward_low},{args.reward_high}]",
            flush=True,
        )
        run_ppo(args, device)
        return

    obs_space = encode.obs_space(
        global_map_view=True,
        deck_size_observation=True,
        neow_observation=True,
        event_option_observation=True,
    )
    core = StSPolicyCore(
        obs_space,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        map_dag=True,
    ).to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.learning_rate)

    print(
        "[route-probe] production path: encode.encode(global_map_view=True) "
        "-> StSPolicyCore.action_logits -> MAP_NODE logits",
        flush=True,
    )
    print(
        f"[route-probe] device={device} updates={args.updates} "
        f"batch={args.batch_size} d_model={args.d_model} "
        f"layers={args.n_layers} heads={args.n_heads} "
        f"elite_penalty={args.elite_penalty} nonelite_reward="
        f"[{args.reward_low},{args.reward_high}]",
        flush=True,
    )

    before = evaluate_random(
        core,
        random.Random(args.seed + 1000),
        batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        device=device,
        elite_prob=args.elite_prob,
        elite_penalty=args.elite_penalty,
        reward_low=args.reward_low,
        reward_high=args.reward_high,
        discount=args.discount,
        min_current_y=args.min_current_y,
        max_current_y=args.max_current_y,
    )
    print_random_metrics("[before] random-map oracle", before)
    print_blocked(
        "[before] blocked-elite sensitivity",
        evaluate_blocked(
            core,
            random.Random(args.seed + 2000),
            distances=args.blocked_distances,
            cases_per_distance=args.blocked_cases,
            device=device,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
        ),
    )

    for update in range(1, args.updates + 1):
        n_blocked = round(args.batch_size * args.blocked_train_frac)
        n_random = args.batch_size - n_blocked
        cases = make_batch(
            rng,
            n_random,
            elite_prob=args.elite_prob,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
            min_current_y=args.min_current_y,
            max_current_y=args.max_current_y,
        )
        cases.extend(
            make_blocked_case(
                rng,
                elite_distance=rng.choice(args.blocked_distances),
                elite_penalty=args.elite_penalty,
                reward_low=args.reward_low,
                reward_high=args.reward_high,
                discount=args.discount,
            )
            for _ in range(n_blocked)
        )
        rng.shuffle(cases)
        obs = encode_cases(cases, device)
        logits = masked_map_logits(core, obs, cases)
        targets = torch.as_tensor([case.target_x for case in cases], device=device)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimizer.step()

        if update == 1 or update % max(1, args.updates // 4) == 0:
            with torch.no_grad():
                chosen = torch.argmax(logits, dim=1)
                acc = (chosen == targets).float().mean().item()
            print(
                f"[train] update={update:>4}/{args.updates} "
                f"loss={float(loss.detach()):.4f} batch_exact={acc * 100:5.1f}%",
                flush=True,
            )

    after = evaluate_random(
        core,
        random.Random(args.seed + 3000),
        batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        device=device,
        elite_prob=args.elite_prob,
        elite_penalty=args.elite_penalty,
        reward_low=args.reward_low,
        reward_high=args.reward_high,
        discount=args.discount,
        min_current_y=args.min_current_y,
        max_current_y=args.max_current_y,
    )
    print_random_metrics("[after] random-map oracle", after)
    print_blocked(
        "[after] blocked-elite sensitivity",
        evaluate_blocked(
            core,
            random.Random(args.seed + 4000),
            distances=args.blocked_distances,
            cases_per_distance=args.blocked_cases,
            device=device,
            elite_penalty=args.elite_penalty,
            reward_low=args.reward_low,
            reward_high=args.reward_high,
            discount=args.discount,
        ),
    )


if __name__ == "__main__":
    main()
