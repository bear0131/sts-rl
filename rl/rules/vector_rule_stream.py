"""One-pass schema-v5 record streams used by online vector-rule training.

The producer protocol is deliberately small: write exactly one JSON object per
line to stdout, and write diagnostics to stderr.  The consumer validates each
record before putting it into a batch.  A subprocess is terminated after the
requested number of optimization steps, so records generated beyond that point
are never silently replayed.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, IO, Iterable, Iterator, Mapping

from rl.rules.rule_vector_model import StateVectorConfig
from rl.rules.vector_rule_data import (
    VectorRuleV6Example,
    example_from_v6_record,
)


def iter_jsonl_records(stream: IO[str], *, source: str = "<stream>") -> Iterator[Mapping[str, Any]]:
    """Yield JSON objects from a producer stdout stream exactly once."""

    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"record at {source}:{line_number} must be an object")
        yield value


def iter_v6_examples(
    records: Iterable[Mapping[str, Any]],
    config: StateVectorConfig,
    *,
    card_vocab_size: int,
    relic_vocab_size: int,
    max_cards: int,
    max_relics: int,
    max_tokens: int,
) -> Iterator[VectorRuleV6Example]:
    """Validate a record iterable without caching or replaying it."""

    for record in records:
        yield example_from_v6_record(
            record,
            config,
            card_vocab_size=card_vocab_size,
            relic_vocab_size=relic_vocab_size,
            max_cards=max_cards,
            max_relics=max_relics,
            max_tokens=max_tokens,
        )


def take_batch(
    examples: Iterator[VectorRuleV6Example], batch_size: int,
) -> list[VectorRuleV6Example]:
    """Consume at most one batch from an example iterator."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch: list[VectorRuleV6Example] = []
    for _ in range(batch_size):
        try:
            batch.append(next(examples))
        except StopIteration:
            break
    return batch


class SubprocessV6RecordStream:
    """Context-managed one-pass JSONL producer.

    ``command`` is intentionally passed to the platform shell so callers can
    use a virtualenv command and its arguments.  The command must reserve
    stdout for JSONL records; its logs belong on stderr.
    """

    def __init__(self, command: str, *, cwd: str | Path | None = None) -> None:
        if not command.strip():
            raise ValueError("stream command cannot be empty")
        self.command = command
        self.cwd = None if cwd is None else str(cwd)
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "SubprocessV6RecordStream":
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdout is None:  # pragma: no cover - Popen invariant
            raise RuntimeError("failed to open producer stdout")
        return self

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("stream must be entered before iteration")
        yield from iter_jsonl_records(self.process.stdout, source=self.command)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        process = self.process
        if process is None:
            return
        terminated_by_consumer = process.poll() is None
        if terminated_by_consumer:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        # A live stream is intentionally stopped by the consumer after its
        # budget. Shell wrappers commonly report that termination as status 1
        # (especially on Windows), which is not a producer failure.
        if (
            exc_type is None
            and not terminated_by_consumer
            and process.returncode != 0
        ):
            raise RuntimeError(
                f"v6 producer exited with status {process.returncode}: {self.command}"
            )
        if process.stdout is not None:
            process.stdout.close()
        self.process = None


__all__ = [
    "iter_jsonl_records", "iter_v6_examples", "take_batch",
    "SubprocessV6RecordStream",
]
