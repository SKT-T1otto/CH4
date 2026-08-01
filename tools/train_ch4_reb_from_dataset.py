# -*- coding: utf-8 -*-
"""Unified offline training entry point for Chapter-4 REB models."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._internal import dispatch


COMMANDS = {
    "reb-only": (
        "tools._internal.train_ch4_reb_only_from_dataset",
        "Train the original REB-only model from a boundary dataset.",
    ),
    "found-aware": (
        "tools._internal.train_ch4_found_aware_reb_from_dataset",
        "Train the found-aware REB model from a boundary dataset.",
    ),
}


def main(argv=None):
    return dispatch(
        "Train a Chapter-4 REB model from an offline boundary dataset.",
        COMMANDS,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
