"""Write a same-row structured-brief SQL versus direct-SQL comparison manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.compare_structured_brief_direct_sql import (
    compare_structured_brief_manifest_files,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_structured_brief_manifest_files(
        structured_manifest_path=args.structured_manifest,
        direct_manifest_path=args.direct_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote structured-brief comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['structured_brief_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
