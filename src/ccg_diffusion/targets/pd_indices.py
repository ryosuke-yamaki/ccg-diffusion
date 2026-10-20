"""Guidance targets I = (E, C, K, P) as CLIP token indices, and their program hash.

The module uses only the standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from ccg_diffusion.provenance import canonical_json, sha256_text

PD_FIELDS = ("exist", "corr", "leak", "possession")


@dataclass(frozen=True)
class PDIndices:
    """Guidance targets I=(E,C,K,P) as CLIP token indices: E=exist, C=corr, K=leak, P=possession (paper Table 1)."""

    exist: List[int] = field(default_factory=list)
    corr: List[List[int]] = field(default_factory=list)
    leak: List[List[int]] = field(default_factory=list)
    possession: List[List[int]] = field(default_factory=list)

    def has_any(self) -> bool:
        """True if any of the four index lists carries at least one entry."""
        return bool(self.exist or self.corr or self.leak or self.possession)


def pd_indices_payload(indices: PDIndices) -> dict[str, list[Any]]:
    """Return the canonical JSON representation used by all four arms."""

    return {
        "exist": [int(index) for index in indices.exist],
        "corr": [[int(a), int(b)] for a, b in indices.corr],
        "leak": [[int(a), int(b)] for a, b in indices.leak],
        "possession": [[int(a), int(b)] for a, b in indices.possession],
    }


def program_sha256(indices: PDIndices) -> str:
    """SHA-256 of the canonical JSON of ``pd_indices_payload(indices)``."""

    return sha256_text(canonical_json(pd_indices_payload(indices)))
