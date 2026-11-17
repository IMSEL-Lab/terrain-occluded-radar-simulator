"""Verify the two byte-exact editions of the frozen CANOE source manifest."""

import hashlib
from pathlib import Path

HISTORICAL_SOURCE_SHA256 = (
    "443f83491158ef116f7dd4928c3541058c7e8b96374a4564f8604e53568f0e27"
)
PORTABLE_SOURCE_SHA256 = (
    "8b74491b34e96434aae2692a952793446429d871ec69ec6e360b980afdaa5572"
)


def verify_source_manifest(path: Path, expected: str) -> str:
    """Accept the portable edition only when the historical edition is pinned."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    allowed = {expected}
    if expected == HISTORICAL_SOURCE_SHA256:
        allowed.add(PORTABLE_SOURCE_SHA256)
    if actual not in allowed:
        raise ValueError(
            f"source manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual
