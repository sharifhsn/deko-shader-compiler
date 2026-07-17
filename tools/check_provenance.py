#!/usr/bin/env python3
"""Validate imported-source manifests, retained licenses, and exact snapshots."""

from __future__ import annotations

import filecmp
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def fail(message: str) -> None:
    print(f"provenance error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_spdx(path: Path, license_name: str) -> None:
    prefix = path.read_text(encoding="utf-8")[:1024]
    if f"SPDX-License-Identifier: {license_name}" not in prefix:
        fail(f"{path.relative_to(ROOT)} does not retain SPDX {license_name}")


def main() -> None:
    third_party = tomllib.loads((ROOT / "THIRD_PARTY.toml").read_text(encoding="utf-8"))
    for upstream in third_party["upstreams"]:
        license_name = upstream["license"]
        for record in upstream.get("files", []):
            local = ROOT / record["local"]
            if not local.is_file():
                fail(f"missing declared local file {record['local']}")
            if license_name == "MIT":
                require_spdx(local, license_name)

    nak = ROOT / "crates/deko-nak"
    active = tomllib.loads((nak / "ACTIVE_FILES.toml").read_text(encoding="utf-8"))
    upstream_files = {
        line
        for line in (nak / "UPSTREAM_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    for name in [*active["exact"], *active["modified"]]:
        if name not in upstream_files:
            fail(f"active NAK file {name} is absent from UPSTREAM_FILES.txt")
        local = nak / "src" / name
        staged = nak / "upstream/mesa" / name
        if not local.is_file() or not staged.is_file():
            fail(f"missing active or staged NAK file {name}")
        require_spdx(local, "MIT")
        require_spdx(staged, "MIT")
    for name in active["exact"]:
        if not filecmp.cmp(nak / "src" / name, nak / "upstream/mesa" / name, shallow=False):
            fail(f"exact NAK file differs from its staged source: {name}")

    print("provenance manifests and retained SPDX headers are valid")


if __name__ == "__main__":
    main()
