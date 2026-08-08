"""Independent route-DAG experiment.

This is deliberately separate from the game environment.  It tests whether a
node scorer with bottom-up, edge-aware route propagation can learn a good path
on random Slay-the-Spire-like maps, compared with a flat whole-map MLP.

Run:
    uv run python -m rl.route_dag_probe --episodes 512 --batch-size 32 --eval-episodes 200
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


ROWS = 10
COLS = 7
ROOMS = ("monster", "elite", "event", "rest")
ROOM_INDEX = {name: i for i, name in enumerate(ROOMS)}
EXPECTED_REWARD = {"monster": 1.0, "elite": -1.0, "event": 0.5, "rest": 2.0}


@dataclass
class RouteMap:
    rooms: np.ndarray  # [ROWS, COLS], integer room ids
    children: list[list[list[int]]]  # children[y][x] -> columns in y + 1


def generate_map(rng: random.Random) -> RouteMap:
    rooms = np.empty((ROWS, COLS), dtype=np.int64)
    # Fireplaces are valuable but uncommon; elites are dangerous but useful as
    # a negative test signal.  Events have an expected reward of 0.5.
    for y in range(ROWS):
        for x in range(COLS):
            p = rng.random()
            room = "rest" if p < 0.14 else "elite" if p < 0.27 else "event" if p < 0.57 else "monster"
            rooms[y, x] = ROOM_INDEX[room]

    children: list[list[list[int]]] = [[[] for _ in range(COLS)] for _ in range(ROWS)]
    for y in range(ROWS - 1):
        for x in range(COLS):
            choices = [c for c in (x - 1, x, x + 1) if 0 <= c < COLS]
            # At least two exits keeps route choices meaningful.  The random
            # extra edge creates merges and forks like the real map.
            children[y][x] = sorted(rng.sample(choices, min(2 + rng.randrange(2), len(choices))))
    return RouteMap(rooms, children)


def room_name(room_id: int) -> str:
    return ROOMS[int(room_id)]


def expected_optimal(route: RouteMap, start_x: int = 3) -> float:
    value = np.zeros((ROWS, COLS), dtype=np.float32)
    for y in range(ROWS - 1, 0, -1):
        for x in range(COLS):
            future = max((value[y + 1, child] for child in route.children[y][x]), default=0.0)
            value[y, x] = EXPECTED_REWARD[room_name(route.rooms[y, x])] + future
    # The first move enters row 1, so select the best child of the virtual root.
    return max(value[1, child] for child in route.children[0][start_x])


def sample_reward(room_id: int, rng: random.Random) -> float:
    name = room_name(room_id)
    if name == "monster":
        return float(rng.random() < 1.0)
    if name == "event":
        return float(rng.random() < 0.5)
    return EXPECTED_REWARD[name]


def map_tensor(route: RouteMap, current_x: int, current_y: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rooms = torch.as_tensor(route.rooms, dtype=torch.long)
    # Directional edge tensor: [ROWS, COLS, COLS], edge[y,x,c] == 1 if c is a child.
    edges = torch.zeros((ROWS, COLS, COLS), dtype=torch.float32)
    for y in range(ROWS - 1):
        for x in range(COLS):
            edges[y, x, route.children[y][x]] = 1.0
    current = torch.tensor([current_x / (COLS - 1), current_y / (ROWS - 1)], dtype=torch.float32)
    return rooms, edges, current


class FlatPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.room = nn.Embedding(len(ROOMS), 12)
        self.net = nn.Sequential(nn.Linear(ROWS * COLS * 12 + 2, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())
        self.actor = nn.Linear(64, COLS)
        self.critic = nn.Linear(64, 1)

    def forward(self, rooms, edges, current):
        if rooms.ndim == 2:
            rooms, current = rooms.unsqueeze(0), current.unsqueeze(0)
        z = self.net(torch.cat([self.room(rooms).flatten(start_dim=1), current], dim=-1))
        logits, value = self.actor(z), self.critic(z).squeeze(-1)
        return (logits[0], value[0]) if current.shape[0] == 1 else (logits, value)


class DAGPolicy(nn.Module):
    """A small learned DP: each node receives messages from its children."""

    def __init__(self):
        super().__init__()
        self.room = nn.Embedding(len(ROOMS), 16)
        self.node = nn.Sequential(nn.Linear(18, 64), nn.Tanh())
        self.child_message = nn.Linear(64, 64, bias=False)
        self.future = nn.GRUCell(64, 64)
        self.context = nn.Sequential(nn.Linear(66, 64), nn.Tanh())
        self.actor = nn.Linear(128, 1)
        self.critic = nn.Linear(64, 1)

    def encode_nodes(self, rooms, edges):
        single = rooms.ndim == 2
        if single:
            rooms, edges = rooms.unsqueeze(0), edges.unsqueeze(0)
        y = torch.arange(ROWS, dtype=torch.float32, device=rooms.device).view(ROWS, 1).expand(ROWS, COLS) / (ROWS - 1)
        x = torch.arange(COLS, dtype=torch.float32, device=rooms.device).view(1, COLS).expand(ROWS, COLS) / (COLS - 1)
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).expand(rooms.shape[0], -1, -1, -1)
        h = self.node(torch.cat([self.room(rooms), coords], dim=-1))
        future = torch.zeros_like(h)
        for row in range(ROWS - 1, -1, -1):
            if row == ROWS - 1:
                future[:, row] = h[:, row]
                continue
            degree = edges[:, row].sum(dim=-1, keepdim=True).clamp_min(1.0)
            msg = torch.bmm(edges[:, row], future[:, row + 1]) / degree
            b, c, d = msg.shape
            future[:, row] = self.future(
                self.child_message(msg).reshape(b * c, d),
                h[:, row].reshape(b * c, d),
            ).reshape(b, c, d)
        return (h[:, 0], future[:, 0]) if single else (h, future)

    def forward(self, rooms, edges, current):
        single = current.ndim == 1
        if single:
            rooms, edges, current = rooms.unsqueeze(0), edges.unsqueeze(0), current.unsqueeze(0)
        _, future = self.encode_nodes(rooms, edges)
        cx = (current[:, 0] * (COLS - 1)).round().long()
        cy = (current[:, 1] * (ROWS - 1)).round().long()
        batch = torch.arange(rooms.shape[0], device=rooms.device)
        root = future[batch, cy, cx]
        ctx = self.context(torch.cat([root, current], dim=-1))
        row = future[batch, cy + 1]
        logits = self.actor(torch.cat([row, ctx.unsqueeze(1).expand(-1, COLS, -1)], dim=-1)).squeeze(-1)
        value = self.critic(root).squeeze(-1)
        return (logits[0], value[0]) if single else (logits, value)


def choose(logits: torch.Tensor, route: RouteMap, x: int, y: int, rng: random.Random, stochastic: bool):
    mask = torch.full((COLS,), -1e9)
    mask[route.children[y][x]] = 0.0
    dist = torch.distributions.Categorical(logits=logits + mask)
    action = dist.sample() if stochastic else torch.argmax(dist.logits)
    return int(action), dist.log_prob(action), dist.entropy()


def run_episode(policy, rng: random.Random, train: bool, stochastic: bool):
    route = generate_map(rng)
    x, y = 3, 0
    log_probs, values, entropies, rewards = [], [], [], []
    while y < ROWS - 1:
        rooms, edges, current = map_tensor(route, x, y)
        logits, value = policy(rooms, edges, current)
        x, log_prob, entropy = choose(logits, route, x, y, rng, stochastic)
        y += 1
        reward = sample_reward(route.rooms[y, x], rng)
        if train:
            log_probs.append(log_prob); values.append(value); entropies.append(entropy)
        rewards.append(reward)
    if not train:
        return sum(rewards), expected_optimal(route)
    returns, running = [], 0.0
    for reward in reversed(rewards):
        running = reward + 0.98 * running
        returns.append(running)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    values_t = torch.stack(values)
    advantage = returns - values_t.detach()
    loss = -(torch.stack(log_probs) * advantage).mean() + 0.5 * (returns - values_t).pow(2).mean() - 0.01 * torch.stack(entropies).mean()
    return loss


def train_batch(policy, batch_size: int, updates: int, seed: int):
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    rng = random.Random(seed)
    for update in range(1, updates + 1):
        routes = [generate_map(rng) for _ in range(batch_size)]
        room_batch = torch.as_tensor(np.stack([r.rooms for r in routes]), dtype=torch.long)
        edge_batch = torch.zeros((batch_size, ROWS, COLS, COLS), dtype=torch.float32)
        for b, route in enumerate(routes):
            for y in range(ROWS - 1):
                for x in range(COLS):
                    edge_batch[b, y, x, route.children[y][x]] = 1.0
        xs = np.full(batch_size, 3, dtype=np.int64)
        ys = np.zeros(batch_size, dtype=np.int64)
        log_probs, values, entropies, rewards = [], [], [], []
        for _ in range(ROWS - 1):
            current = torch.as_tensor(np.stack((xs / (COLS - 1), ys / (ROWS - 1)), axis=1), dtype=torch.float32)
            logits, value = policy(room_batch, edge_batch, current)
            mask = torch.full((batch_size, COLS), -1e9)
            for b, route in enumerate(routes):
                mask[b, route.children[ys[b]][xs[b]]] = 0.0
            dist = torch.distributions.Categorical(logits=logits + mask)
            next_x = dist.sample()
            log_probs.append(dist.log_prob(next_x)); entropies.append(dist.entropy()); values.append(value)
            next_np = next_x.detach().numpy()
            reward = np.array([sample_reward(route.rooms[ys[b] + 1, next_np[b]], rng) for b, route in enumerate(routes)], dtype=np.float32)
            rewards.append(torch.as_tensor(reward))
            xs = next_np; ys += 1
        returns=[]; running=torch.zeros(batch_size)
        for reward in reversed(rewards):
            running = reward + 0.98 * running; returns.append(running)
        returns=torch.stack(list(reversed(returns)))
        values_t=torch.stack(values); advantages=returns-values_t.detach()
        optimizer.zero_grad()
        loss = -(torch.stack(log_probs) * advantages).mean() + 0.5 * (returns-values_t).pow(2).mean() - 0.01 * torch.stack(entropies).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        if update % max(1, updates // 4) == 0:
            print(f"  update {update}/{updates} ({update * batch_size} episodes) loss={float(loss.detach()):+.4f}", flush=True)


def evaluate(policy, episodes: int, seed: int):
    rng = random.Random(seed); ratios=[]; gaps=[]
    for _ in range(episodes):
        actual, optimum = run_episode(policy, rng, train=False, stochastic=False)
        ratios.append(actual / max(1e-6, optimum)); gaps.append(optimum - actual)
    return float(np.mean(ratios)), float(np.mean(gaps)), float(np.mean(np.array(gaps) < 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("training flat baseline")
    updates = max(1, math.ceil(args.episodes / args.batch_size))
    flat = FlatPolicy(); train_batch(flat, args.batch_size, updates, args.seed)
    print("training topological DAG policy")
    dag = DAGPolicy(); train_batch(dag, args.batch_size, updates, args.seed + 1)
    print("evaluation")
    for name, policy in (("flat", flat), ("dag", dag)):
        ratio, gap, exact = evaluate(policy, args.eval_episodes, args.seed + 100)
        print(f"{name}: mean_actual_over_optimal={ratio:.3f} mean_expected_gap={gap:.3f} exact_route_rate={exact*100:.1f}%")


if __name__ == "__main__":
    main()
