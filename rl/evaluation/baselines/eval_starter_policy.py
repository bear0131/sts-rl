"""Evaluate a no-card-reward Ironclad starter-deck heuristic.

Combat priority is Bash, then Strike, then Defend; each card is played whenever
legal, and targeted attacks choose the lowest-HP legal monster. The policy ends
the turn when none of those starter cards can be played. Map edges are sampled
uniformly, card rewards and shops are skipped, and campfires are used to rest.

The evaluator has no step limit and reports run, combat, card-use, damage, and
deck-correction statistics. An experiment-only backend restores the exact ten
unupgraded starter cards after overworld decisions and before every battle, so
Neow, events, and boss relics cannot change the deck used in combat.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from env.core import action_space as A
from env.core import game_actions as ga
from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv


CARD_PRIORITY = ("Bash", "Strike_R", "Defend_R")
STARTER_DECK = Counter({"Strike_R": 5, "Defend_R": 4, "Bash": 1})
STARTER_SIM_DECK = (
    [("STRIKE_RED", 0)] * 5
    + [("DEFEND_RED", 0)] * 4
    + [("BASH", 0)]
)
DECK_CHANGING_BOSS_RELICS = {
    "Astrolabe",
    "Calling Bell",
    "Empty Cage",
    "Pandora's Box",
    "Coffee Dripper",
}


class StarterDeckGameEnv(SimGameEnv):
    """Whole-run backend that enforces the starter deck as an experiment."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deck_corrections = 0

    def _restore_starter_deck(self) -> None:
        expected = Counter(STARTER_SIM_DECK)
        current = Counter(
            (card_id, int(upgrades))
            for card_id, upgrades in self.gc.deck_ids()
        )
        if current != expected:
            self.gc.set_deck(STARTER_SIM_DECK)
            self.deck_corrections += 1

    def reset(self, *, seed=None):
        result = super().reset(seed=seed)
        self._restore_starter_deck()
        return result

    def _start_battle(self) -> None:
        self._restore_starter_deck()
        super()._start_battle()

    def step(self, action):
        self._restore_starter_deck()
        result = super().step(action)
        if self.bc is None:
            self._restore_starter_deck()
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-run starter-deck baseline: Bash then Strike then Defend, "
            "lowest-HP target, configurable route policy, and no card rewards."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument(
        "--route",
        choices=("random", "min-elite-max-rest"),
        default="random",
        help=(
            "random samples reachable nodes uniformly; min-elite-max-rest "
            "uses full-map DP to minimize elites first, then maximize rests."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def _legal_by_segment(info: dict) -> dict[str, list[tuple[int, int]]]:
    mask = np.asarray(info["action_mask"], dtype=bool)
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        raise RuntimeError("environment returned an all-false action mask")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for action_id in legal:
        segment, offset = A.segment_of(action_id)
        grouped[segment].append((offset, action_id))
    return grouped


def _lowest_hp_target(
    choices: list[tuple[int, int]], canon: dict
) -> int:
    monsters = canon.get("monsters", [])

    def key(choice: tuple[int, int]) -> tuple[float, int]:
        offset, _action_id = choice
        hp = monsters[offset]["hp"] if offset < len(monsters) else math.inf
        return hp, offset

    return min(choices, key=key)[1]


from rl.evaluation.baselines.routes import optimal_route_choices as _dp_route_choices


def choose_action(
    info: dict,
    canon: dict,
    route_rng: np.random.Generator,
    route_mode: str = "random",
    rmap=None,
    current_map_y: int | None = None,
) -> int:
    """Choose one legal thin action according to the starter policy."""
    by_segment = _legal_by_segment(info)

    if info.get("arg_step") == A.PICK_TARGET:
        return _lowest_hp_target(by_segment["PICK_TARGET"], canon)
    if info.get("arg_step") == A.PICK_HAND:
        if not info.get("multi_selected") and by_segment.get("PICK_CARD"):
            return min(by_segment["PICK_CARD"])[1]
        return by_segment["CONFIRM_SELECT"][0][1]
    if info.get("arg_step") == A.PICK_CARD:
        return min(by_segment["PICK_CARD"])[1]

    phase = info.get("phase")
    if phase == "combat":
        hand = canon.get("hand", [])
        playable = dict(by_segment.get("PLAY_CARD", []))
        for card_id in CARD_PRIORITY:
            candidates = [
                (slot, playable[slot])
                for slot, card in enumerate(hand)
                if slot in playable and card.get("id") == card_id
            ]
            if candidates:
                return min(candidates)[1]
        return by_segment["END_TURN"][0][1]

    if phase == "map" and by_segment.get("MAP_NODE"):
        choices = [action_id for _offset, action_id in by_segment["MAP_NODE"]]
        if route_mode == "min-elite-max-rest":
            if rmap is None or current_map_y is None:
                raise ValueError("DP route requires rmap and current_map_y")
            choices, _score = _dp_route_choices(
                by_segment["MAP_NODE"], rmap, current_map_y
            )
        return int(route_rng.choice(choices))

    # Take non-card combat rewards before skipping the card offer.
    if phase == "combat_reward":
        for segment in ("REWARD_GOLD", "REWARD_RELIC", "REWARD_POTION"):
            if by_segment.get(segment):
                return min(by_segment[segment])[1]
        if by_segment.get("REWARD_SKIP"):
            return by_segment["REWARD_SKIP"][0][1]
        if by_segment.get("PROCEED"):
            return by_segment["PROCEED"][0][1]

    if phase == "rest":
        rest_id = A.seg_base("CAMPFIRE") + ga.CAMPFIRE_OPTIONS["rest"]
        legal_ids = {aid for choices in by_segment.values() for _off, aid in choices}
        if rest_id in legal_ids:
            return rest_id
        if by_segment.get("PROCEED"):
            return by_segment["PROCEED"][0][1]

    if phase == "shop" and by_segment.get("PROCEED"):
        return by_segment["PROCEED"][0][1]

    if phase == "chest" and by_segment.get("CHEST"):
        # CHEST offset 0 is open, offset 1 is skip.
        return min(by_segment["CHEST"])[1]

    if phase == "boss_relic" and by_segment.get("BOSS_RELIC"):
        offered = (canon.get("global") or {}).get("boss_relics", [])
        for offset, action_id in sorted(by_segment["BOSS_RELIC"]):
            relic = offered[offset] if offset < len(offered) else None
            if relic not in DECK_CHANGING_BOSS_RELICS:
                return action_id
        return min(by_segment["BOSS_RELIC"])[1]

    if phase == "event" and by_segment.get("EVENT_OPT"):
        # Event exits are generally the last option. This minimizes voluntary
        # deck changes; mandatory event changes are measured as deck drift.
        return max(by_segment["EVENT_OPT"])[1]

    if by_segment.get("PROCEED"):
        return by_segment["PROCEED"][0][1]

    # Neow and unavoidable card-select/event choices are deterministic. The
    # experiment backend restores any resulting deck change before combat.
    return min(
        action_id
        for choices in by_segment.values()
        for _offset, action_id in choices
    )


def _deck_counter(canon: dict) -> Counter[str]:
    deck = (canon.get("global") or {}).get("deck", [])
    return Counter(card_id for card_id, _upgrades in deck)


def _deck_upgrades(canon: dict) -> int:
    deck = (canon.get("global") or {}).get("deck", [])
    return sum(int(upgrades > 0) for _card_id, upgrades in deck)


def _monster_hp(canon: dict) -> dict[int, int]:
    return {
        index: int(monster["hp"])
        for index, monster in enumerate(canon.get("monsters", []))
        if not monster.get("gone", False)
    }


def _player_hp(canon: dict, default: int = 0) -> int:
    if "player" in canon:
        return int(canon["player"].get("hp", default))
    return int(canon.get("hp", default))


def _hp_loss(before: dict[int, int], after_canon: dict) -> int:
    after = _monster_hp(after_canon)
    return sum(max(hp - after.get(index, 0), 0) for index, hp in before.items())


def _mean_std(values: list[int | float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[starter] CSV written to {path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be non-negative")

    env = SlayTheSpireEnv(
        ascension=args.ascension,
        backend_factory=lambda: StarterDeckGameEnv(ascension=args.ascension),
    )
    rows: list[dict[str, object]] = []
    death_floors: Counter[int] = Counter()
    started = time.monotonic()
    last_report = started

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            route_rng = np.random.default_rng(episode_seed ^ 0x51A7_7E12)
            _obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            steps = 0
            raw_return = 0.0
            combats = combat_wins = 0
            in_combat = False
            end_turns = route_choices = rests = card_rewards_skipped = 0
            elite_nodes = rest_nodes = rest_unavailable = 0
            route_dp_violations = 0
            card_plays: Counter[str] = Counter()
            upgraded_plays = 0
            nonstarter_plays = 0
            target_priority_violations = 0
            energy_spent = 0
            monster_hp_loss = attack_hp_loss = 0
            player_hp_loss = defend_block_gained = 0
            deck_invariant_violations = 0
            max_deck_size = 10

            while not (terminated or truncated):
                canon = env.unwrapped.thick.obs_canonical()
                phase = info.get("phase")
                if phase == "combat" and not in_combat:
                    combats += 1
                    in_combat = True

                deck = _deck_counter(canon)
                max_deck_size = max(max_deck_size, sum(deck.values()))
                if deck != STARTER_DECK or _deck_upgrades(canon):
                    deck_invariant_violations += 1

                thick = env.unwrapped.thick
                route_kwargs = {}
                if phase == "map":
                    route_kwargs = {
                        "route_mode": args.route,
                        "rmap": thick.rmap,
                        "current_map_y": int(thick.gc.cur_map_node_y),
                    }
                action = choose_action(
                    info, canon, route_rng, **route_kwargs
                )
                segment, offset = A.segment_of(action)
                if phase == "combat" and segment == "PICK_TARGET":
                    legal_targets = _legal_by_segment(info)["PICK_TARGET"]
                    expected_target = _lowest_hp_target(legal_targets, canon)
                    target_priority_violations += int(action != expected_target)
                selected_card = None
                if phase == "combat" and segment == "PLAY_CARD":
                    hand = canon.get("hand", [])
                    if offset < len(hand):
                        selected_card = hand[offset]
                        card_id = selected_card["id"]
                        card_plays[card_id] += 1
                        nonstarter_plays += int(card_id not in CARD_PRIORITY)
                        upgraded_plays += int(selected_card.get("up", 0) > 0)
                        energy_spent += max(int(selected_card.get("cost", 0)), 0)
                if phase == "combat" and segment == "END_TURN":
                    end_turns += 1
                if phase == "map" and segment == "MAP_NODE":
                    route_choices += 1
                    rooms = dict(canon.get("options", {}).get("nodes", []))
                    room = rooms.get(offset)
                    elite_nodes += int(room == "ELITE")
                    rest_nodes += int(room == "REST")
                    if args.route == "min-elite-max-rest":
                        optimal, _score = _dp_route_choices(
                            _legal_by_segment(info)["MAP_NODE"],
                            thick.rmap,
                            int(thick.gc.cur_map_node_y),
                        )
                        route_dp_violations += int(action not in optimal)
                if phase == "rest" and segment == "CAMPFIRE":
                    if offset == ga.CAMPFIRE_OPTIONS["rest"]:
                        rests += 1
                if phase == "rest" and segment != "CAMPFIRE":
                    rest_unavailable += 1
                if phase == "combat_reward" and segment == "REWARD_SKIP":
                    card_rewards_skipped += 1

                # A targeted attack executes on PICK_TARGET, not PLAY_CARD.
                executed_card = selected_card
                if phase == "combat" and info.get("arg_step") == A.PICK_TARGET:
                    pending = info.get("pending")
                    hand = canon.get("hand", [])
                    if pending and pending[0] == "play" and pending[1] < len(hand):
                        executed_card = hand[pending[1]]

                before_monsters = _monster_hp(canon) if phase == "combat" else {}
                before_player = canon.get("player", {})
                before_hp = _player_hp(canon)
                before_block = int(before_player.get("block", 0))

                _obs, reward, terminated, truncated, info = env.step(action)
                raw_return += float(reward)
                steps += 1
                after_canon = env.unwrapped.thick.obs_canonical()
                after_phase = info.get("phase")

                if phase == "combat":
                    hp_loss = _hp_loss(before_monsters, after_canon)
                    monster_hp_loss += hp_loss
                    if executed_card and executed_card.get("id") in ("Bash", "Strike_R"):
                        attack_hp_loss += hp_loss
                    after_player = after_canon.get("player", {})
                    after_hp = _player_hp(after_canon, before_hp)
                    player_hp_loss += max(before_hp - after_hp, 0)
                    if executed_card and executed_card.get("id") == "Defend_R":
                        after_block = int(after_player.get("block", before_block))
                        defend_block_gained += max(after_block - before_block, 0)

                if in_combat and after_phase != "combat":
                    transition_stats = env.unwrapped.thick.run_stats()
                    if transition_stats["outcome"] != "PLAYER_LOSS":
                        combat_wins += 1
                    in_combat = False

                now = time.monotonic()
                if (
                    args.heartbeat_seconds > 0
                    and now - last_report >= args.heartbeat_seconds
                ):
                    stats = env.unwrapped.thick.run_stats()
                    print(
                        f"[starter] run {episode_index + 1}/{args.episodes}: "
                        f"floor={stats['floor']} step={steps} combats={combats} "
                        "in progress",
                        flush=True,
                    )
                    last_report = now

            stats = env.unwrapped.thick.run_stats()
            floor = int(stats["floor"])
            act = int(stats["act"])
            won = act > 3 or stats["outcome"] == "PLAYER_VICTORY"
            if won:
                result = "win"
            elif truncated and info.get("dead_screen"):
                result = "dead_screen"
            elif truncated:
                result = "other_truncation"
            else:
                result = "death"
                death_floors[floor] += 1

            final_canon = env.unwrapped.thick.obs_canonical()
            final_deck = _deck_counter(final_canon)
            rows.append(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "result": result,
                    "final_floor": floor,
                    "final_act": act,
                    "final_hp": int(stats["hp"]),
                    "steps": steps,
                    "raw_return": raw_return,
                    "combats": combats,
                    "combat_wins": combat_wins,
                    "end_turns": end_turns,
                    "bash_plays": card_plays["Bash"],
                    "strike_plays": card_plays["Strike_R"],
                    "defend_plays": card_plays["Defend_R"],
                    "upgraded_plays": upgraded_plays,
                    "nonstarter_plays": nonstarter_plays,
                    "target_priority_violations": target_priority_violations,
                    "energy_spent": energy_spent,
                    "monster_hp_loss": monster_hp_loss,
                    "attack_hp_loss": attack_hp_loss,
                    "player_hp_loss": player_hp_loss,
                    "defend_block_gained": defend_block_gained,
                    "route_choices": route_choices,
                    "elite_nodes": elite_nodes,
                    "rest_nodes": rest_nodes,
                    "rests": rests,
                    "rest_unavailable": rest_unavailable,
                    "route_dp_violations": route_dp_violations,
                    "card_rewards_skipped": card_rewards_skipped,
                    "deck_corrections": int(env.unwrapped.thick.deck_corrections),
                    "deck_invariant_violations": deck_invariant_violations,
                    "final_deck_size": sum(final_deck.values()),
                    "max_deck_size": max_deck_size,
                    "final_upgraded_cards": _deck_upgrades(final_canon),
                }
            )

            done = episode_index + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / done * (args.episodes - done)
                print(
                    f"[starter] {done}/{args.episodes}: result={result} "
                    f"floor={floor} ({done / elapsed:.2f} runs/s, "
                    f"ETA={remaining / 60:.1f} min)",
                    flush=True,
                )
                last_report = time.monotonic()
    finally:
        env.close()

    result_counts = Counter(str(row["result"]) for row in rows)
    deaths = result_counts["death"]
    elapsed = time.monotonic() - started

    print()
    print(
        f"[starter] episodes={args.episodes} seed={args.seed} "
        f"ascension={args.ascension} route={args.route} no_step_limit=True"
    )
    print(
        "[starter] policy=Bash > Strike_R > Defend_R > end_turn; "
        f"lowest_hp_target; route={args.route}; skip_cards; rest; leave_shop"
    )
    print(
        f"[starter] wins={result_counts['win']}/{args.episodes} "
        f"win_rate={result_counts['win'] / args.episodes:.4f} "
        f"deaths={deaths} dead_screens={result_counts['dead_screen']} "
        f"other_truncations={result_counts['other_truncation']}"
    )
    floors = [int(row["final_floor"]) for row in rows]
    floor_mean, floor_std = _mean_std(floors)
    print(
        f"[starter] final_floor={floor_mean:.2f} +/- {floor_std:.2f} "
        f"median={np.median(floors):.1f} p25={np.percentile(floors, 25):.1f} "
        f"p75={np.percentile(floors, 75):.1f}"
    )
    for cleared_act in (1, 2, 3):
        cleared = sum(int(row["final_act"]) > cleared_act for row in rows)
        print(
            f"[starter] cleared_act{cleared_act}="
            f"{cleared / args.episodes:.4f} ({cleared}/{args.episodes})"
        )

    total_combats = sum(int(row["combats"]) for row in rows)
    total_combat_wins = sum(int(row["combat_wins"]) for row in rows)
    print(
        f"[starter] combats={total_combats} wins={total_combat_wins} "
        f"combat_win_rate="
        f"{total_combat_wins / total_combats if total_combats else 0.0:.4f}"
    )
    for field in (
        "end_turns",
        "bash_plays",
        "strike_plays",
        "defend_plays",
        "energy_spent",
        "monster_hp_loss",
        "attack_hp_loss",
        "player_hp_loss",
        "defend_block_gained",
        "route_choices",
        "elite_nodes",
        "rest_nodes",
        "rests",
        "rest_unavailable",
        "card_rewards_skipped",
    ):
        mean, std = _mean_std([int(row[field]) for row in rows])
        print(f"[starter] {field}_per_run={mean:.2f} +/- {std:.2f}")

    total_bash = sum(int(row["bash_plays"]) for row in rows)
    total_strike = sum(int(row["strike_plays"]) for row in rows)
    total_defend = sum(int(row["defend_plays"]) for row in rows)
    total_attack_loss = sum(int(row["attack_hp_loss"]) for row in rows)
    total_defend_block = sum(int(row["defend_block_gained"]) for row in rows)
    total_player_loss = sum(int(row["player_hp_loss"]) for row in rows)
    print(
        f"[starter] end_turns_per_combat="
        f"{sum(int(row['end_turns']) for row in rows) / total_combats:.2f} "
        f"cards_per_combat="
        f"{(total_bash + total_strike + total_defend) / total_combats:.2f} "
        f"player_hp_loss_per_combat={total_player_loss / total_combats:.2f}"
    )
    print(
        f"[starter] attack_hp_loss_per_attack="
        f"{total_attack_loss / (total_bash + total_strike):.2f} "
        f"block_gained_per_defend="
        f"{total_defend_block / total_defend if total_defend else 0.0:.2f}"
    )

    correction_runs = sum(int(row["deck_corrections"]) > 0 for row in rows)
    total_corrections = sum(int(row["deck_corrections"]) for row in rows)
    invariant_violations = sum(
        int(row["deck_invariant_violations"]) for row in rows
    )
    print(
        f"[starter] deck_correction_runs={correction_runs}/{args.episodes} "
        f"corrections={total_corrections} "
        f"observed_invariant_violations={invariant_violations}"
    )
    print(
        f"[starter] nonstarter_plays="
        f"{sum(int(row['nonstarter_plays']) for row in rows)} "
        f"target_priority_violations="
        f"{sum(int(row['target_priority_violations']) for row in rows)} "
        f"route_dp_violations="
        f"{sum(int(row['route_dp_violations']) for row in rows)}"
    )
    print("[starter] death_floor_distribution:")
    if not death_floors:
        print("  (no deaths)")
    for floor in sorted(death_floors):
        count = death_floors[floor]
        print(
            f"  floor={floor:02d} count={count:4d} "
            f"pct_of_deaths={count / deaths:7.2%} "
            f"pct_of_all_runs={count / args.episodes:7.2%}"
        )
    print(f"[starter] elapsed={elapsed:.1f}s")

    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
