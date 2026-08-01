#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused static tests for Perf-RBE ablation protocol support."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import _rbe_protocol_contract
from tools._internal.prepare_ch4_rbe_perf_ablation_protocols import VARIANTS


def main() -> int:
    checks = []

    def add(name, passed, detail=""):
        checks.append({"name": name, "pass": bool(passed), "detail": str(detail)})

    name_map = {
        "no_boundary": "ch4_rbe_perf_ablation_no_boundary_protocol_v1",
        "all_actors": "ch4_rbe_perf_ablation_all_actors_protocol_v1",
        "no_nominal": "ch4_rbe_perf_ablation_no_nominal_protocol_v1",
    }
    for variant, protocol_name in name_map.items():
        contract = _rbe_protocol_contract(protocol_name)
        spec = VARIANTS[variant]
        add(f"{variant}_ablation_id", contract.get("ablation_id") == spec["ablation_id"], contract)
        add(f"{variant}_ratios", contract.get("sampling_ratios") == spec["sampling_ratios"], contract)
        add(f"{variant}_freeze", contract.get("search_actors_frozen") is spec["search_actors_frozen"], contract)
        add(f"{variant}_executor_only", contract.get("executor_only_actor_training") is spec["executor_only_actor_training"], contract)
        add(f"{variant}_scope", contract.get("optimization_scope") == "mechanism_ablation_only", contract)
        add(f"{variant}_ratio_sum", math.isclose(sum(contract["sampling_ratios"].values()), 1.0, abs_tol=1e-12), contract)

    try:
        _rbe_protocol_contract("unknown_protocol")
        add("unknown_protocol_rejected", False, "no exception")
    except RuntimeError as exc:
        add("unknown_protocol_rejected", True, exc)

    result = {
        "overall_pass": all(item["pass"] for item in checks),
        "case_count": len(checks),
        "passed_case_count": sum(item["pass"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
