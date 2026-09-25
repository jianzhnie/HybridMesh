"""Run the integration tests: one torchrun invocation per script.

Each ``*_equivalence.py`` (and sibling) script in this directory carries its
own launch command in its module docstring (``PYTHONPATH=. torchrun
--nproc_per_node=N <script>``). This runner reads that command rather than
duplicating it, so a script that needs a different rank count keeps a single
source of truth.

Usage (from the repo root):

    python tests/integration_tests/run_all.py --list          # just enumerate
    python tests/integration_tests/run_all.py                  # run everything
    python tests/integration_tests/run_all.py ac_equivalence   # one script
    python tests/integration_tests/run_all.py ac_equivalence.py tp_equivalence

Every script runs with ``PYTHONPATH=.`` and its own torchrun process; the
exit code is non-zero if any script fails, and a pass/fail table is printed
at the end either way.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Helpers, not runnable scripts: no ``__main__`` driver, imported by hand.
_HELPERS = {"full_precision_equivalence.py"}

_TORCHRUN_RE = re.compile(r"torchrun\s+(--nproc_per_node=\d+)\s+\\\s*\n?\s*(\S+)")


def _torchrun_args(script: Path) -> list[str]:
    """The torchrun flags from the script's docstring command, or the default."""
    docstring = ast.get_docstring(ast.parse(script.read_text())) or ""
    match = _TORCHRUN_RE.search(docstring)
    if match:
        return [match.group(1)]
    return ["--nproc_per_node=2"]


def discover() -> list[Path]:
    """Every runnable script in this directory, sorted by name."""
    return sorted(
        p
        for p in SCRIPT_DIR.glob("*.py")
        if p.name not in ("__init__.py", "run_all.py") and p.name not in _HELPERS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Script names (with or without .py) to run; default is all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the discovered scripts and their torchrun args, then exit.",
    )
    args = parser.parse_args()

    scripts = discover()
    if args.scripts:
        wanted = {s.removesuffix(".py") for s in args.scripts}
        scripts = [p for p in scripts if p.stem in wanted]
        missing = wanted - {p.stem for p in scripts}
        if missing:
            print(f"no such integration script(s): {sorted(missing)}")
            return 2

    if args.list:
        for script in scripts:
            print(
                f"{script.name:45s} torchrun {' '.join(_torchrun_args(script))}"
            )
        print(f"{len(scripts)} scripts")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    results: list[tuple[str, bool, float]] = []
    for script in scripts:
        cmd = [
            "torchrun",
            *_torchrun_args(script),
            str(script.relative_to(REPO_ROOT)),
        ]
        print(f"=== {' '.join(cmd)}", flush=True)
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        elapsed = time.monotonic() - start
        results.append((script.name, proc.returncode == 0, elapsed))

    width = max(len(name) for name, _, _ in results)
    print("\n=== integration test summary ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {elapsed:6.1f}s")
    failed = [name for name, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
