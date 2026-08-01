# -*- coding: utf-8 -*-
"""Unified finalization entry point for the clean Chapter-4 baseline."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._internal import dispatch


COMMANDS = {
    "checkpoint-selection": (
        "tools._internal.finalize_ch4_clean_checkpoint_selection",
        "Audit a fine checkpoint sweep and freeze the selected clean model.",
    ),
    "nominal-test": (
        "tools._internal.finalize_ch4_clean_nominal_test",
        "Audit and finalize the frozen clean model's nominal test.",
    ),
}


def main(argv=None):
    return dispatch(
        "Finalize clean Chapter-4 checkpoint selection or nominal testing.",
        COMMANDS,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
