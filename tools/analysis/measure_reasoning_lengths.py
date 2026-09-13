"""Measure lengths of simulator-backed whole-run reasoning transcripts.

The sampler uses the same environment and transcript compiler as training, but
does not construct a policy. It randomly selects legal thin actions and reports
length statistics for completed transitions with a reasoning trace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _configure_sim_build() -> None:
    """Prefer a local build compiled with atomic rule tracing."""
    if "STS_LIGHTSPEED_BUILD_DIR" not in os.environ:
        major, minor = sys.version_info[:2]
        if os.name == "nt":
            suffix = f"cp{major}{minor}-win_amd64.pyd"
        else:
            suffix = f"cpython-{major}{minor}*.so"
        for candidate in (
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1-cp311",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1",
        ):
            if list(candidate.glob(f"slaythespire.*{suffix}")):
                os.environ["STS_LIGHTSPEED_BUILD_DIR"] = str(candidate)
                break
    if os.name == "nt" and "STS_LIGHTSPEED_DLL_DIRS" not in os.environ:
        runtime_dirs = [
            str(candidate)
            for candidate in (
                REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1-cp311",
                REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1",
                REPO_ROOT / "sim" / "sts_lightspeed" / "build",
            )
            if candidate.is_dir()
        ]
        if runtime_dirs:
            os.environ["STS_LIGHTSPEED_DLL_DIRS"] = os.pathsep.join(runtime_dirs)


_configure_sim_build()

from env.core import action_space as A  # noqa: E402
from env.core.run_env import SlayTheSpireEnv  # noqa: E402
from rl.policies.wholerun_reasoning import (  # noqa: E402
    InteractiveTranscript,
    build_interactive_transcript,
    card_stochastic_flags,
    events_from_snapshots,
)


def _compile_trace(
    trace: dict, action_id: int
) -> tuple[InteractiveTranscript, int, str] | None:
    snapshots = trace.get("snapshots") or ()
    events = trace.get("events")
    if events is None:
        if len(snapshots) < 2:
            return None
        events = events_from_snapshots(snapshots)
    segment, offset = A.segment_of(action_id)
    transcript = build_interactive_transcript(
        events=events,
        action_segment=segment,
        action_offset=offset,
        action_card=trace.get("action_card"),
        **card_stochastic_flags(trace.get("action_card")),
    )
    return transcript, len(events), str(trace.get("precision", "unknown"))


def _trace_length(trace: dict, action_id: int) -> tuple[int, int, str] | None:
    compiled = _compile_trace(trace, action_id)
    if compiled is None:
        return None
    transcript, event_count, precision = compiled
    inputs, _labels = transcript.teacher_labels()
    return len(inputs), event_count, precision


def _token_category(token: str) -> str:
    """Assign one exclusive accounting category to a protocol token."""

    if token.startswith("BYTE_"):
        return "text_bytes"
    if token in ("TEXT_BEGIN", "TEXT_END"):
        return "text_delimiters"
    if token.startswith("DIGIT_"):
        return "number_digits"
    if token.startswith("INT_"):
        return "number_atoms"
    if token in ("NUM_BEGIN", "NUM_END"):
        return "number_delimiters"
    if token in ("SIGN_POS", "SIGN_NEG"):
        return "number_signs"
    if token in ("EVENT_BEGIN", "EVENT_END"):
        return "event_delimiters"
    if token.startswith("EVENT_"):
        return "event_types"
    if token.startswith("CALC_"):
        return "calculation_structure"
    if token.startswith("FIELD_"):
        return "field_structure"
    if token.startswith("QUERY_"):
        return "oracle_queries"
    if token.startswith("OBS_") or token in (
        "OBSERVATION_BEGIN", "OBSERVATION_END"
    ):
        return "oracle_observations"
    if token in (
        "TRACE_BEGIN", "TRACE_END", "ACTION_BEGIN", "ACTION_END",
        "ACTION_SEGMENT", "ACTION_OFFSET",
    ):
        return "trace_action_structure"
    if token in (
        "CARD_INSTANCE_BEGIN", "CARD_INSTANCE_END", "CARD_REF",
        "RULE_BEGIN", "RULE_END", "RUNTIME_COST", "CARD_RULE_UNAVAILABLE",
    ):
        return "card_structure"
    if token.startswith("VALUE_"):
        return "value_structure"
    if token in ("<PAD>", "<BOS>", "<EOS>"):
        return "sequence_boundaries"
    return "rule_or_event_semantics"


def _read_text_span(tokens: Sequence[str], start: int) -> tuple[str, int] | None:
    if tokens[start] != "TEXT_BEGIN":
        return None
    raw = bytearray()
    cursor = start + 1
    while cursor < len(tokens) and tokens[cursor].startswith("BYTE_"):
        raw.append(int(tokens[cursor][5:], 16))
        cursor += 1
    if cursor >= len(tokens) or tokens[cursor] != "TEXT_END":
        return None
    return raw.decode("ascii", errors="replace"), cursor + 1


def _read_number_span(
    tokens: Sequence[str], start: int
) -> tuple[int, int] | None:
    if tokens[start] != "NUM_BEGIN":
        return None
    cursor = start + 1
    sign = 1
    if cursor < len(tokens) and tokens[cursor] in ("SIGN_POS", "SIGN_NEG"):
        sign = -1 if tokens[cursor] == "SIGN_NEG" else 1
        cursor += 1
    digits: list[str] = []
    while cursor < len(tokens) and tokens[cursor].startswith("DIGIT_"):
        digits.append(tokens[cursor][6:])
        cursor += 1
    if not digits or cursor >= len(tokens) or tokens[cursor] != "NUM_END":
        return None
    return sign * int("".join(digits)), cursor + 1


def _replace_fixed_text(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        span = _read_text_span(tokens, cursor)
        if span is None:
            out.append(tokens[cursor])
            cursor += 1
            continue
        value, cursor = span
        out.append(f"ATOM_{value}")
    return out


_ENUM_TEXT_CONTEXTS = frozenset({
    "ACTION_SEGMENT",
    "FIELD_NAME",
    "FIELD_OWNER",
    "FIELD_STATUS",
    "FIELD_COUNTER",
    "FIELD_SOURCE",
    "FIELD_DESTINATION",
})


def _replace_enum_text(tokens: Sequence[str]) -> list[str]:
    """Atomize schema-enumerated text while retaining free-form move labels."""

    out: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        span = _read_text_span(tokens, cursor)
        previous = tokens[cursor - 1] if cursor else "<START>"
        if span is None or previous not in _ENUM_TEXT_CONTEXTS:
            out.append(tokens[cursor])
            cursor += 1
            continue
        _value, cursor = span
        out.append("ENUM_VALUE")
    return out


def _replace_schema_text_pairs(tokens: Sequence[str]) -> list[str]:
    """Fold ``FIELD_X TEXT_BEGIN ... TEXT_END`` into one schema atom."""

    out: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor] not in _ENUM_TEXT_CONTEXTS or cursor + 1 >= len(tokens):
            out.append(tokens[cursor])
            cursor += 1
            continue
        span = _read_text_span(tokens, cursor + 1)
        if span is None:
            out.append(tokens[cursor])
            cursor += 1
            continue
        value, cursor = span
        out.append("SCHEMA_VALUE")
    return out


def _replace_small_integers(
    tokens: Sequence[str], *, minimum: int = -32, maximum: int = 255
) -> list[str]:
    # Kept as a compatibility hook for older reports. Numeric spans are no
    # longer collapsed into literal-value tokens in the protocol.
    return list(tokens)


def _remove_positive_signs(tokens: Sequence[str]) -> list[str]:
    """Use an omitted sign for non-negative integer literals."""

    return [token for token in tokens if token != "SIGN_POS"]


def _compact_local_calculations(tokens: Sequence[str]) -> list[str]:
    """Remove wrappers and the second copy of a numeric calculation result."""

    out: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor] != "CALC_BEGIN":
            out.append(tokens[cursor])
            cursor += 1
            continue
        try:
            calc_end = tokens.index("CALC_END", cursor + 1)
        except ValueError:
            out.append(tokens[cursor])
            cursor += 1
            continue
        result_marker = next(
            (
                index for index in range(cursor + 1, calc_end)
                if tokens[index] == "CALC_RESULT"
            ),
            None,
        )
        if result_marker is None:
            out.extend(tokens[cursor + 1:calc_end])
            cursor = calc_end + 1
            continue
        result_span = _read_number_span(tokens, result_marker + 1)
        visible_marker = calc_end + 1
        visible_span = (
            _read_number_span(tokens, visible_marker + 1)
            if visible_marker < len(tokens)
            and tokens[visible_marker] == "VALUE_AFTER"
            else None
        )
        out.extend(tokens[cursor + 1:calc_end])
        cursor = calc_end + 1
        if result_span is not None and visible_span is not None:
            result_value, _result_end = result_span
            visible_value, visible_end = visible_span
            if result_value == visible_value:
                cursor = visible_end
    return out


def _remove_calculation_result_copy(tokens: Sequence[str]) -> list[str]:
    """Keep the visible ``VALUE_AFTER`` result and drop ``CALC_RESULT`` copy."""

    out: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor] != "CALC_RESULT":
            out.append(tokens[cursor])
            cursor += 1
            continue
        span = _read_number_span(tokens, cursor + 1)
        if span is None:
            out.append(tokens[cursor])
            cursor += 1
            continue
        cursor = span[1]
    return out


def _remove_action_complete(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    cursor = 0
    pattern = ("EVENT_BEGIN", "EVENT_ACTION_COMPLETE", "EVENT_END")
    while cursor < len(tokens):
        if tuple(tokens[cursor:cursor + len(pattern)]) == pattern:
            cursor += len(pattern)
        else:
            out.append(tokens[cursor])
            cursor += 1
    return out


def _remove_action_card_reference(tokens: Sequence[str]) -> list[str]:
    """Make the selected action card's reference zero implicit."""

    out = list(tokens)
    try:
        action_end = out.index("ACTION_END")
        reference = out.index("CARD_REF", 0, action_end)
    except ValueError:
        return out
    span = _read_number_span(out, reference + 1)
    if span is None or span[0] != 0:
        return out
    return out[:reference] + out[span[1]:]


def _project_compact(tokens: Sequence[str], *, aggressive: bool) -> list[str]:
    out = _remove_action_card_reference(tokens)
    out = (
        _compact_local_calculations(out)
        if aggressive
        else _remove_calculation_result_copy(out)
    )
    out = _remove_action_complete(out)
    out = _replace_fixed_text(out) if aggressive else _replace_schema_text_pairs(out)
    out = _replace_small_integers(out)
    removable = {
        "EVENT_BEGIN",
        "CARD_INSTANCE_BEGIN",
        "CARD_INSTANCE_END",
    }
    if aggressive:
        removable.update({
            "EVENT_END",
            "TRACE_BEGIN",
            "TRACE_END",
            "ACTION_BEGIN",
            "ACTION_END",
        })
    return [token for token in out if token not in removable]


def _candidate_transforms(tokens: Sequence[str]) -> dict[str, list[str]]:
    """Return independently measurable protocol transformations."""

    source = list(tokens)
    no_action_ref = _remove_action_card_reference(source)
    no_calc_wrappers = _compact_local_calculations(source)
    no_action_complete = _remove_action_complete(source)
    no_event_begin = [token for token in source if token != "EVENT_BEGIN"]
    no_event_wrappers = [
        token for token in source if token not in ("EVENT_BEGIN", "EVENT_END")
    ]
    no_card_wrappers = [
        token for token in source
        if token not in ("CARD_INSTANCE_BEGIN", "CARD_INSTANCE_END")
    ]
    return {
        "text_atoms": _replace_fixed_text(source),
        "enum_text_atoms": _replace_enum_text(source),
        "schema_text_pairs": _replace_schema_text_pairs(source),
        "small_integer_atoms": _replace_small_integers(source),
        "omit_positive_sign": _remove_positive_signs(source),
        "calculation_compact": no_calc_wrappers,
        "calculation_keep_visible_result": _remove_calculation_result_copy(source),
        "remove_action_complete": no_action_complete,
        "remove_event_begin": no_event_begin,
        "remove_event_wrappers": no_event_wrappers,
        "remove_card_instance_wrappers": no_card_wrappers,
        "implicit_played_card_reference": no_action_ref,
        "conservative_combined": _project_compact(source, aggressive=False),
        "aggressive_combined": _project_compact(source, aggressive=True),
    }


def analyze_transcripts(
    transcripts: Iterable[InteractiveTranscript],
) -> dict[str, object]:
    """Measure protocol overhead and lossless/local compression candidates."""

    transcript_list = list(transcripts)
    all_tokens = [token for item in transcript_list for token in item.tokens[:-1]]
    total = len(all_tokens)
    categories = Counter(map(_token_category, all_tokens))
    token_counts = Counter(all_tokens)
    owner_counts = Counter(
        owner.name.lower()
        for item in transcript_list
        for owner in item.owners[:-1]
    )

    texts: Counter[str] = Counter()
    text_contexts: dict[str, Counter[str]] = {}
    numbers: Counter[int] = Counter()
    event_lengths: dict[str, list[int]] = {}
    for item in transcript_list:
        tokens = item.tokens[:-1]
        cursor = 0
        while cursor < len(tokens):
            span_start = cursor
            text_span = _read_text_span(tokens, cursor)
            if text_span is not None:
                value, cursor = text_span
                texts[value] += 1
                previous = tokens[span_start - 1] if span_start else "<START>"
                text_contexts.setdefault(previous, Counter())[value] += 1
                continue
            number_span = _read_number_span(tokens, cursor)
            if number_span is not None:
                value, cursor = number_span
                numbers[value] += 1
                continue
            if tokens[cursor] == "EVENT_BEGIN":
                try:
                    end = tokens.index("EVENT_END", cursor + 1)
                except ValueError:
                    cursor += 1
                    continue
                event_type = next(
                    (token for token in tokens[cursor + 1:end]
                     if token.startswith("EVENT_")),
                    "EVENT_UNKNOWN",
                )
                event_lengths.setdefault(event_type, []).append(end - cursor + 1)
            cursor += 1

    baseline_lengths = [len(item.tokens) - 1 for item in transcript_list]
    def projection(lengths: Sequence[int]) -> dict[str, float | int]:
        projected = sum(lengths)
        return {
            "tokens": projected,
            "mean": float(statistics.fmean(lengths)),
            "saved_tokens": total - projected,
            "saved_fraction": (total - projected) / total,
        }

    candidate_lengths: dict[str, list[int]] = {}
    for item in transcript_list:
        for name, transformed in _candidate_transforms(item.tokens[:-1]).items():
            candidate_lengths.setdefault(name, []).append(len(transformed))

    return {
        "tokens": total,
        "top_tokens": token_counts.most_common(50),
        "exclusive_categories": {
            category: {
                "tokens": count,
                "fraction": count / total,
            }
            for category, count in categories.most_common()
        },
        "owner_counts": dict(owner_counts),
        "fixed_text": {
            "spans": sum(texts.values()),
            "unique_values": len(texts),
            "top_values": texts.most_common(30),
            "by_context": {
                context: {
                    "spans": sum(values.values()),
                    "unique_values": len(values),
                    "top_values": values.most_common(20),
                }
                for context, values in sorted(text_contexts.items())
            },
            "projected_saved_tokens": (
                token_counts["TEXT_BEGIN"]
                + token_counts["TEXT_END"]
                + sum(token_counts[f"BYTE_{value:02X}"] for value in range(128))
                - sum(texts.values())
            ),
        },
        "numbers": {
            "spans": sum(numbers.values()),
            "unique_values": len(numbers),
            "minimum": min(numbers) if numbers else None,
            "maximum": max(numbers) if numbers else None,
            "small_integer_fraction": (
                sum(count for value, count in numbers.items() if -32 <= value <= 255)
                / sum(numbers.values())
                if numbers else 0.0
            ),
            "negative_sign_tokens": token_counts["SIGN_NEG"],
            "top_values": numbers.most_common(30),
        },
        "event_types": {
            event_type: {
                "count": len(lengths),
                "mean_tokens": float(statistics.fmean(lengths)),
                "max_tokens": max(lengths),
            }
            for event_type, lengths in sorted(event_lengths.items())
        },
        "specific_redundancy": {
            "action_complete_tokens": 3 * token_counts["EVENT_ACTION_COMPLETE"],
            "event_begin_tokens": token_counts["EVENT_BEGIN"],
            "event_end_tokens": token_counts["EVENT_END"],
            "card_instance_wrapper_tokens": (
                token_counts["CARD_INSTANCE_BEGIN"]
                + token_counts["CARD_INSTANCE_END"]
            ),
            "calculation_wrapper_tokens": (
                token_counts["CALC_BEGIN"] + token_counts["CALC_END"]
            ),
        },
        "projections": {
            "baseline": {
                "tokens": total,
                "mean": float(statistics.fmean(baseline_lengths)),
                "saved_tokens": 0,
                "saved_fraction": 0.0,
            },
            **{
                name: projection(lengths)
                for name, lengths in candidate_lengths.items()
            },
        },
    }


def measure(*, samples: int, seed: int, max_steps: int) -> dict[str, object]:
    if samples < 1 or max_steps < 1:
        raise ValueError("samples and max_steps must be positive")
    rng = np.random.default_rng(seed)
    env = SlayTheSpireEnv(
        card_language=True,
        interactive_reasoning=True,
    )
    lengths: list[int] = []
    event_counts: list[int] = []
    precisions: Counter[str] = Counter()
    lengths_by_precision: dict[str, list[int]] = {}
    transcripts: list[InteractiveTranscript] = []
    transcripts_by_precision: dict[str, list[InteractiveTranscript]] = {}
    steps = 0
    resets = 0
    obs, info = env.reset(seed=seed)
    try:
        while len(lengths) < samples and steps < max_steps:
            legal = np.flatnonzero(env.action_masks())
            if not len(legal):
                obs, info = env.reset(seed=seed + resets + 1)
                resets += 1
                continue
            action = int(rng.choice(legal))
            result = env.step(action)
            obs, _reward, terminated, truncated, info = result
            steps += 1
            trace = info.get("reasoning_trace")
            if trace is not None:
                compiled = _compile_trace(trace, action)
                if compiled is not None:
                    transcript, event_count, precision = compiled
                    inputs, _labels = transcript.teacher_labels()
                    length = len(inputs)
                    lengths.append(length)
                    event_counts.append(event_count)
                    transcripts.append(transcript)
                    transcripts_by_precision.setdefault(precision, []).append(transcript)
                    precisions[precision] += 1
                    lengths_by_precision.setdefault(precision, []).append(length)
            if terminated or truncated:
                obs, info = env.reset(seed=seed + resets + 1)
                resets += 1
    finally:
        env.close()

    if not lengths:
        raise RuntimeError("no complete reasoning traces were observed")
    values = np.asarray(lengths, dtype=np.float64)
    events = np.asarray(event_counts, dtype=np.float64)
    by_precision = {
        precision: {
            "samples": len(values),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": int(max(values)),
        }
        for precision, values in sorted(lengths_by_precision.items())
    }
    return {
        "requested_samples": samples,
        "measured_samples": len(lengths),
        "environment_steps": steps,
        "resets": resets,
        "precision_counts": dict(sorted(precisions.items())),
        "sequence_length_by_precision": by_precision,
        "sequence_length_mean": float(values.mean()),
        "sequence_length_median": float(np.percentile(values, 50)),
        "sequence_length_p95": float(np.percentile(values, 95)),
        "sequence_length_p99": float(np.percentile(values, 99)),
        "sequence_length_max": int(values.max()),
        "event_count_mean": float(events.mean()),
        "token_analysis": analyze_transcripts(transcripts),
        "token_analysis_by_precision": {
            precision: analyze_transcripts(items)
            for precision, items in sorted(transcripts_by_precision.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(measure(
        samples=args.samples, seed=args.seed, max_steps=args.max_steps
    ), ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
