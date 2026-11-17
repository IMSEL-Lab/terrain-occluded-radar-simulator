"""Record the source checkout revision when Git metadata is available."""

import subprocess
from pathlib import Path


def checkout_revision(source_root: Path) -> str | None:
    source_root = source_root.resolve()
    if not (source_root / ".git").exists():
        return None

    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel", "--verify", "HEAD"],
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    top_level, _, revision = output.rstrip("\n").rpartition("\n")
    if top_level and Path(top_level).resolve() == source_root:
        return revision
    return None
