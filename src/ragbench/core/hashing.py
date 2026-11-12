"""Stable hashes for cache keys, configuration fingerprints and identifiers."""

from __future__ import annotations

import hashlib
import json


def canonical_json(value: object) -> str:
    """Serialise ``value`` to JSON with sorted keys and no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: object, *, length: int = 64) -> str:
    """Return the first ``length`` hex characters of the SHA-256 of ``value`` as JSON.

    Example:
        >>> stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
        True
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]
