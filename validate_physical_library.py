"""Validate generated conductivity physical-library YAML records."""

from __future__ import annotations

import argparse
from pathlib import Path

from conductivity.physical_library import load_physical_library


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Validate conductivity physical-library records."
    )
    argument_parser.add_argument(
        "library_root",
        type=Path,
        nargs="?",
        default=Path("conductivity/physical_library"),
    )
    parsed_arguments = argument_parser.parse_args()
    records = load_physical_library(parsed_arguments.library_root)
    print(f"library_root={records.root}")
    print(f"species_count={len(records.species_records)}")
    print(f"pair_count={len(records.pair_records)}")
    print("species=" + ",".join(sorted(records.species_records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
