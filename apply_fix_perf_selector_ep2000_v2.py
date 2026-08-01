#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Apply the ep2000/final-model payload-comparison fix to the perf selector.

from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path


OLD_VERSION = 'SELECTOR_VERSION = "20260724-v2-perf-fresh-validation-policy-lock"'
NEW_VERSION = 'SELECTOR_VERSION = "20260724-v3-perf-payload-equality"'

SHA_ANCHOR = """def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


"""

HELPERS = """def _torch_load_cpu(path: Path | str) -> Any:
    # Load a checkpoint payload without using serialized bytes as identity.
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _nested_payload_equal(left: Any, right: Any) -> bool:
    # Exact recursive equality for torch-save checkpoint payloads.
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right, equal_nan=True))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_nested_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_payload_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    try:
        return bool(left == right)
    except Exception:
        return False


def checkpoint_payload_equal(left_path: Path | str, right_path: Path | str) -> bool:
    return _nested_payload_equal(
        _torch_load_cpu(left_path),
        _torch_load_cpu(right_path),
    )


"""

OLD_COMPARE = """        final_same_as_ep2000 = sha256(model_dir(seed) / "snapshot_ep2000.pt") == sha256(final_path)
        require(final_same_as_ep2000, f"final model differs from snapshot_ep2000 seed {seed}")
"""

NEW_COMPARE = """        snapshot_ep2000_path = model_dir(seed) / "snapshot_ep2000.pt"
        final_binary_sha_same_as_ep2000 = (
            sha256(snapshot_ep2000_path) == sha256(final_path)
        )
        final_payload_same_as_ep2000 = checkpoint_payload_equal(
            snapshot_ep2000_path,
            final_path,
        )
        require(
            final_payload_same_as_ep2000,
            f"final model payload differs from snapshot_ep2000 seed {seed}",
        )
"""

OLD_AUDIT_FIELD = """                "final_model_same_as_snapshot_ep2000": True,
"""

NEW_AUDIT_FIELDS = """                "final_model_same_as_snapshot_ep2000": final_payload_same_as_ep2000,
                "final_model_payload_equal_snapshot_ep2000": final_payload_same_as_ep2000,
                "final_model_binary_sha_equal_snapshot_ep2000": final_binary_sha_same_as_ep2000,
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}. "
            "The selector may already be modified or differ from the expected version."
        )
    return text.replace(old, new, 1)


def main() -> int:
    project_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    target = (
        project_root
        / "tools"
        / "_internal"
        / "select_ch4_rbe_boundary_core_perf_checkpoints.py"
    )
    if not target.is_file():
        raise FileNotFoundError(f"selector not found: {target}")

    original = target.read_text(encoding="utf-8")

    if NEW_VERSION in original and "def checkpoint_payload_equal(" in original:
        print(f"[ApplyFix] already applied: {target}")
        py_compile.compile(str(target), doraise=True)
        return 0

    updated = replace_once(original, OLD_VERSION, NEW_VERSION, "selector version")
    updated = replace_once(
        updated,
        SHA_ANCHOR,
        SHA_ANCHOR + HELPERS,
        "payload helper insertion",
    )
    updated = replace_once(
        updated,
        OLD_COMPARE,
        NEW_COMPARE,
        "ep2000/final comparison",
    )
    updated = replace_once(
        updated,
        OLD_AUDIT_FIELD,
        NEW_AUDIT_FIELDS,
        "training audit fields",
    )

    backup = target.with_suffix(target.suffix + ".before_ep2000_payload_fix")
    if backup.exists():
        raise FileExistsError(
            f"backup already exists; refusing to overwrite it: {backup}"
        )
    backup.write_text(original, encoding="utf-8")

    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(updated, encoding="utf-8")
        py_compile.compile(str(temporary), doraise=True)
        os.replace(temporary, target)
        py_compile.compile(str(target), doraise=True)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"[ApplyFix] PASS target={target}")
    print(f"[ApplyFix] backup={backup}")
    print("[ApplyFix] selector_version=20260724-v3-perf-payload-equality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
