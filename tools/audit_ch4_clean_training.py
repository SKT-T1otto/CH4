# -*- coding: utf-8 -*-
"""Unified audit entry point for clean Chapter-4 training assets."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._internal import dispatch


COMMANDS = {
    "baseline-training": (
        "tools._internal.summarize_ch4_baseline_training",
        "Summarize and audit a clean baseline training run.",
    ),
    "initialization-smoke": (
        "tools._internal.summarize_ch4_initialization_smoke",
        "Audit the explicit-initialization smoke suite.",
    ),
}


def main(argv=None):
    return dispatch(
        "Audit clean Chapter-4 training or initialization smoke assets.",
        COMMANDS,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
