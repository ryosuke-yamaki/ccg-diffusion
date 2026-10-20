"""Pinned spaCy lemma lookups for neutral noun-compound context.

Only the installed ``en_core_web_sm`` lookup tables are read, after their
bytes are checked against the pinned sha256. No spaCy tagger, parser, neural
pipeline or benchmark vocabulary is used. spaCy is imported only by the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata, util
from pathlib import Path
from typing import Any

from ccg_diffusion.provenance import sha256_file

LEXICAL_RESOURCE_SHA256 = "eb64f40c0f8396d1762730c0ddf4dad2a52d138f5a389f71a1a1d088173b7737"
LEXICAL_RESOURCE_SIZE = 972893
LEXICAL_RESOURCE_VERSION = "holsd.spacy_english_lemma_indices.v1"
MODEL_VERSION = "3.8.0"
SPACY_VERSION = "3.8.14"
RESOURCE_RELATIVE_PATH = f"en_core_web_sm-{MODEL_VERSION}/lemmatizer/lookups/lookups.bin"


def lexical_resource_definition() -> dict[str, Any]:
    """Return the fixed resource identity included in the routing definition."""

    return {
        "schema": LEXICAL_RESOURCE_VERSION,
        "package": "en_core_web_sm",
        "package_version": MODEL_VERSION,
        "relative_path": RESOURCE_RELATIVE_PATH,
        "sha256": LEXICAL_RESOURCE_SHA256,
        "size_bytes": LEXICAL_RESOURCE_SIZE,
        "loader": "spacy.lookups.Lookups",
        "loader_package_version": SPACY_VERSION,
        "table": "lemma_index",
        "noun_entry_count": 57528,
        "adjective_entry_count": 20993,
        "classifier": "exact_lowercase_noun_entry_and_no_adjective_entry",
        "use": "neutral_compound_context_only_never_positive_visual_binding_evidence",
        "inference": "none",
    }


def lexical_resource_path() -> Path:
    """Find the installed pinned lookup bytes and fail closed on any drift."""

    spec = util.find_spec("en_core_web_sm")
    if spec is None or spec.origin is None:
        raise ValueError("neutral noun context requires the pinned en_core_web_sm lookup resource")
    if metadata.version("en-core-web-sm") != MODEL_VERSION or metadata.version("spacy") != SPACY_VERSION:
        raise ValueError("neutral noun context lexical resource/loader package version mismatch")
    path = Path(spec.origin).parent / RESOURCE_RELATIVE_PATH
    if (
        not path.is_file() or path.stat().st_size != LEXICAL_RESOURCE_SIZE
        or sha256_file(path) != LEXICAL_RESOURCE_SHA256
    ):
        raise ValueError("neutral noun context lexical resource bytes differ from the pinned lookup")
    return path.resolve()


@dataclass(frozen=True)
class NominalModifierLexicon:
    """Generic noun/adjective inventories; the semantic role still comes from the CCG derivation."""

    nouns: frozenset[str]
    adjectives: frozenset[str]

    def is_noun_only(self, word: str) -> bool:
        lowered = word.lower()
        return lowered in self.nouns and lowered not in self.adjectives


@lru_cache(maxsize=1)
def load_nominal_modifier_lexicon() -> NominalModifierLexicon:
    """Load only two immutable lexical tables after checking exact resource bytes."""

    from spacy.lookups import Lookups

    path = lexical_resource_path()
    index = Lookups().from_disk(path.parent).get_table("lemma_index")
    nouns, adjectives = frozenset(index["noun"]), frozenset(index["adj"])
    if len(nouns) != 57528 or len(adjectives) != 20993:
        raise ValueError("neutral noun context lexical table cardinality mismatch")
    return NominalModifierLexicon(nouns=nouns, adjectives=adjectives)
