#!/usr/bin/env python3
"""
DEPRECATED: this script has been archived.

Archive copy:
  archive/retired_ideas/scripts/create_fig1_uai_fixed.py

Use scripts/create_fig1_minimal.py (canonical Figure 1 generator) instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from create_fig1_minimal import main as create_fig1_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper. Delegates to create_fig1_minimal.py."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Forwarded to create_fig1_minimal.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Forwarded to create_fig1_minimal.py",
    )
    args, _unknown = parser.parse_known_args()

    print(
        "WARNING: scripts/create_fig1_uai_fixed.py is deprecated. "
        "Delegating to scripts/create_fig1_minimal.py."
    )

    forwarded = ["create_fig1_minimal.py"]
    if args.metrics is not None:
        forwarded.extend(["--metrics", str(args.metrics)])
    if args.out is not None:
        forwarded.extend(["--out", str(args.out)])

    import sys

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        create_fig1_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
