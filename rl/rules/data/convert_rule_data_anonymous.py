"""Convert collected rule samples to the anonymous-card v5 schema."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from rl.rules.rule_data import RULE_DATA_VERSION, SUPPORTED_RULE_DATA_VERSIONS


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", compresslevel=3)
    return path.open(mode, encoding="utf-8")


def convert(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open(source, "r") as reader, _open(destination, "w") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("data_version") not in SUPPORTED_RULE_DATA_VERSIONS:
                raise ValueError(f"{source}:{line_number}: unsupported data version")
            sample["data_version"] = RULE_DATA_VERSION
            sample.setdefault("metadata", {})["card_identity_schema"] = "anonymous_ref_v1"
            writer.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert(args.source, args.destination)
    print(f"[anonymous-rule-data] samples={count} out={args.destination.resolve()}")


if __name__ == "__main__":
    main()
