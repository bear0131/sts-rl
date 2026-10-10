"""Distinct-seed real-game campaign bookkeeping for CommunicationMod."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class SeedCampaign:
    def __init__(self, seeds: list[str], results_path: str | Path | None = None) -> None:
        if not seeds or len(set(seeds)) != len(seeds):
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
        in_game = bool(frame.get("in_game"))
        gs = frame.get("game_state") or {}
        if in_game:
            self.active = True
        if in_game and str(gs.get("screen_type") or "") == "GAME_OVER" and not self.recorded:
            ss = gs.get("screen_state") or {}
            row = {
                "requested_seed": self.current_seed,
                "actual_seed": str(gs.get("seed")),
                "victory": bool(ss.get("victory")),
                "score": int(ss.get("score", 0) or 0),
                "floor": int(gs.get("floor", 0) or 0),
                "act": int(gs.get("act", 0) or 0),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            row.update(metadata or {})
            if row["requested_seed"] is None:
                raise RuntimeError("game ended after the configured seed campaign completed")
            if row["actual_seed"] != str(row["requested_seed"]):
                raise RuntimeError(
                    f"game seed mismatch requested={row['requested_seed']} actual={row['actual_seed']}"
                )
            if self.results_path is not None:
                self.results_path.parent.mkdir(parents=True, exist_ok=True)
                with self.results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self.completed.add(str(row["requested_seed"]))
            self.recorded = True
            return row
        if not in_game and self.active and self.recorded:
            self.index += 1
            while self.index < len(self.seeds) and self.seeds[self.index] in self.completed:
                self.index += 1
            self.active = False
            self.recorded = False
            self.start_sent = False
        return None

    def start_command(self, character: str, ascension: int) -> str | None:
        seed = self.current_seed
        if seed is None or self.start_sent:
            return None
        self.start_sent = True
        return f"START {character} {int(ascension)} {seed}"
