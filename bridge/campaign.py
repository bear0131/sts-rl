"""Distinct-seed real-game campaign bookkeeping for CommunicationMod."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


_SEED_ALPHABET = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def seed_string_to_long(seed: str) -> int:
    """Return the game's integral representation of a displayed seed string."""
    text = str(seed).strip().upper()
    if not text:
        raise ValueError("Slay the Spire seed must not be empty")
    value = 0
    for char in text:
        try:
            digit = _SEED_ALPHABET.index(char)
        except ValueError as exc:
            raise ValueError(f"invalid Slay the Spire seed character {char!r}") from exc
        value = value * len(_SEED_ALPHABET) + digit
    return value


class SeedCampaign:
    def __init__(self, seeds: list[str], results_path: str | Path | None = None) -> None:
        if not seeds or len({seed_string_to_long(seed) for seed in seeds}) != len(seeds):
            raise ValueError("campaign seeds must be non-empty and distinct")
        self.seeds = tuple(str(seed) for seed in seeds)
        self.results_path = None if results_path is None else Path(results_path)
        self.completed: set[str] = set()
        if self.results_path is not None and self.results_path.is_file():
            for line in self.results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.completed.add(str(json.loads(line)["requested_seed"]))
        self.index = next(
            (i for i, seed in enumerate(self.seeds) if seed not in self.completed),
            len(self.seeds),
        )
        self.active = False
        self.recorded = False
        self.start_sent = False

    @property
    def done(self) -> bool:
        return self.index >= len(self.seeds)

    @property
    def current_seed(self) -> str | None:
        return None if self.done else self.seeds[self.index]

    def observe(
        self, frame: dict[str, Any], *, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        in_game = frame.get("in_game")
        gs = frame.get("game_state") or {}
        if in_game is True:
            self.active = True
        if in_game is True and gs.get("screen_type") == "GAME_OVER" and not self.recorded:
            if frame.get("ready_for_command") is not True:
                return None  # A partially constructed terminal is not a result.
            ss = gs.get("screen_state")
            if (not isinstance(ss, dict) or type(ss.get("victory")) is not bool
                    or type(ss.get("score")) is not int):
                raise ValueError("ready GAME_OVER requires actual boolean victory and integer score")
            if (type(gs.get("seed")) is not int or type(gs.get("floor")) is not int
                    or gs["floor"] < 0 or type(gs.get("act")) is not int or gs["act"] < 1):
                raise ValueError("ready GAME_OVER requires actual integer seed/floor/act identity")
            requested_seed = self.current_seed
            if requested_seed is None:
                raise RuntimeError("game ended after the configured seed campaign completed")
            row = {
                "requested_seed": requested_seed,
                "requested_seed_long": seed_string_to_long(requested_seed),
                "actual_seed": str(gs["seed"]),
                "victory": ss["victory"],
                "score": ss["score"],
                "floor": gs["floor"],
                "act": gs["act"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            if set(metadata or {}) & set(row):
                raise ValueError("campaign metadata must not replace actual terminal fields")
            row.update(metadata or {})
            if row["actual_seed"] != str(row["requested_seed_long"]):
                raise RuntimeError(
                    "game seed mismatch "
                    f"requested={row['requested_seed']} "
                    f"requested_long={row['requested_seed_long']} "
                    f"actual={row['actual_seed']}"
                )
            if self.results_path is not None:
                self.results_path.parent.mkdir(parents=True, exist_ok=True)
                with self.results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self.completed.add(str(row["requested_seed"]))
            self.recorded = True
            return row
        if (in_game is False and frame.get("ready_for_command") is True
                and self.active and self.recorded):
            self.index += 1
            while self.index < len(self.seeds) and self.seeds[self.index] in self.completed:
                self.index += 1
            self.active = False
            self.recorded = False
            self.start_sent = False
        return None

    def start_command(self, character: str, ascension: int) -> str | None:
        seed = self.current_seed
        if seed is None or self.start_sent or self.active:
            return None
        self.start_sent = True
        return f"START {character} {int(ascension)} {seed}"
