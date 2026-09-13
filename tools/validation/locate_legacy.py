"""Locate a historical module/path or verify archived files against their hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_archive(root=ROOT):
    entries = json.loads((root / "docs/path-migration.json").read_text(encoding="utf-8"))
    for old, entry in entries.items():
        path = (root / entry["path"]).resolve()
        assert path.is_relative_to(root.resolve()), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], old
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_path", nargs="?")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(f"verified {verify_archive()} archived files")
    if args.old_path:
        key = args.old_path.replace("\\", "/")
        for name in ("path-migration", "launcher-migration", "module-migration"):
            mapping = json.loads((ROOT / f"docs/{name}.json").read_text(encoding="utf-8"))
            if key in mapping:
                entry = mapping[key]
                print(entry["path"] if isinstance(entry, dict) else entry)
                break
        else:
            raise SystemExit(f"no migration entry for {key}")
    elif not args.verify:
        parser.error("provide OLD_PATH or --verify")


if __name__ == "__main__":
    main()
