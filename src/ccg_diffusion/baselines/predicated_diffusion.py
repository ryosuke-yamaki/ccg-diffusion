"""Predicated Diffusion baseline: guidance targets from a spaCy parse (original paper, App. B.3 rules).

``SpacyProvider`` parses one prompt with spaCy ``en_core_web_sm`` and derives the
guidance targets I = (E, C, K, P) on SD1.4 CLIP token indices with the
hand-written rules of the Predicated Diffusion paper:

* E: every ``NOUN``/``PROPN`` token;
* C: every ``amod``/``nmod``/``compound``/``acomp`` modifier with its noun head;
* K: every modifier with every other entity (leak);
* P: the subject and the single direct object of ``have``/``wear``/``grasp``/``hold``.

A prompt whose entities or modifiers cannot be aligned to CLIP tokens gets the
status ``alignment_error`` and is generated as stock Vanilla by the SD1.4
driver (270 of 24,000 T2I-CompBench++ images). ``build_input_file`` writes the
seed-expanded guidance-target file of the ``sd14_pd_spacy`` arm; the Sana
``sana_pd_spacy`` baseline and the TAB Predicated Diffusion rows read the same
per-prompt rows. The loss and the denoising loop are the ones of CCG Diffusion
(``guidance.predicated_diffusion_loss``, ``backbones.sd14.pipeline``).

spaCy is imported by ``SpacyProvider`` only; every failure string below is
written into the ``error`` field of a recorded row and must not change.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ccg_diffusion.config import (
    SD14_GENERATION,
    SD14_PREDICATED_DIFFUSION_ARM,
    SD14_PREDICATED_DIFFUSION_PROVIDER,
)
from ccg_diffusion.provenance import read_jsonl, sha256_text
from ccg_diffusion.targets.guidance_targets import expand_to_image_rows, write_image_rows_file
from ccg_diffusion.targets.pd_indices import PDIndices, pd_indices_payload, program_sha256
from ccg_diffusion.targets.tokenizer import require_clip_tokenizer_identity

SPACY_MODEL = "en_core_web_sm"
PROVIDER = SD14_PREDICATED_DIFFUSION_PROVIDER  # "spacy_sd14" (recorded value)
ARM = SD14_PREDICATED_DIFFUSION_ARM  # "sd14_pd_spacy" (recorded value)
MODIFIER_DEPENDENCIES = frozenset({"AMOD", "NMOD", "COMPOUND", "ACOMP"})
POSSESSION_LEMMAS = frozenset({"have", "wear", "grasp", "hold"})
PROMPT_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})

_NORMALISE_RE = re.compile(r"\s|_")


class PredicatedDiffusionInputError(ValueError):
    """A prompt manifest cannot be turned into the Predicated Diffusion input file."""


def _normalise(text: str) -> str:
    return _NORMALISE_RE.sub("", text).lower()


# --- surface-token anchoring ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceTokenAnchor:
    """A parser token anchored to the final CLIP BPE piece of its surface form.

    ``token_index`` is the parser's token position and ``clip_index`` uses
    CLIP's leading-BOS convention, so a recorded index can be traced back to
    the prompt text.
    """

    token_index: int
    text: str
    lemma: str
    pos: str
    char_start: int
    char_end: int
    clip_index: Optional[int]


def tokenize_prompt_pieces(tokenizer: Any, prompt: str) -> list[str]:
    """Tokenize ``prompt`` and return the normalised BPE pieces (entry 0 is BOS, the last is EOS)."""

    ids = tokenizer(prompt).input_ids
    return [_normalise(tokenizer.decode(t)) for t in ids]


def find_word_in_pieces(
    word: str,
    pieces: list[str],
    *,
    occurrence: int = 0,
) -> Optional[int]:
    """Locate ``word`` in normalised ``pieces``; return the index of its final piece, or ``None``.

    ``occurrence`` selects which match to return when the word appears more than once.
    """
    target = _normalise(word)
    if not target:
        return None
    hits = 0
    for start in range(1, len(pieces) - 1):
        buf = ""
        for end in range(start, len(pieces) - 1):
            buf += pieces[end]
            if buf == target:
                if hits == occurrence:
                    return end
                hits += 1
                break
            if not target.startswith(buf):
                break
    return None


def anchor_surface_tokens(
    tokenizer: Any,
    prompt: str,
    tokens: Iterable[Any],
) -> list[SurfaceTokenAnchor]:
    """Anchor parser tokens to CLIP by their occurrence-aware surface form.

    ``tokens`` is a spaCy ``Doc`` or any iterable exposing ``i``, ``text``,
    ``lemma_``, ``pos_`` and ``idx``. The final BPE piece of a word is its
    index. An unalignable token keeps ``clip_index=None``.
    """

    pieces = tokenize_prompt_pieces(tokenizer, prompt)
    seen: Counter[str] = Counter()
    anchors: list[SurfaceTokenAnchor] = []
    for fallback_i, token in enumerate(tokens):
        text = str(getattr(token, "text", token))
        normalised = _normalise(text)
        occurrence = seen[normalised]
        clip_index = find_word_in_pieces(text, pieces, occurrence=occurrence)
        seen[normalised] += 1
        token_index = int(getattr(token, "i", fallback_i))
        char_start = int(getattr(token, "idx", -1))
        anchors.append(
            SurfaceTokenAnchor(
                token_index=token_index,
                text=text,
                lemma=str(getattr(token, "lemma_", text) or text),
                pos=str(getattr(token, "pos_", "")),
                char_start=char_start,
                char_end=char_start + len(text) if char_start >= 0 else -1,
                clip_index=clip_index,
            )
        )
    return anchors


# --- extraction ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """One fail-closed prompt extraction, ready to be written as a guidance-target row."""

    status: str
    pd_indices: PDIndices = field(default_factory=PDIndices)
    alignment: tuple[Mapping[str, Any], ...] = ()
    parser_provenance: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def target_row(self, *, prompt_id: str, prompt: str) -> dict[str, Any]:
        """The guidance-target row of one base prompt (the recorded key set)."""

        provider = PROVIDER if self.parser_provenance.get("sidecar_provider") == PROVIDER else "unknown"
        row: dict[str, Any] = {
            "prompt_id": prompt_id,
            "provider": provider,
            "prompt_sha256": sha256_text(prompt),
            "status": self.status,
            "pd_indices": pd_indices_payload(self.pd_indices),
            "program_sha256": program_sha256(self.pd_indices) if self.ok else None,
            "alignment": [dict(item) for item in self.alignment],
            "parser_provenance": dict(self.parser_provenance),
        }
        if self.error is not None:
            row["error"] = self.error
        return row


def _anchor_payload(
    anchor: SurfaceTokenAnchor,
    *,
    role: str,
    source: str,
    status: str = "anchored",
    reason: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "role": role,
        "source": source,
        "token_index": anchor.token_index,
        "text": anchor.text,
        "lemma": anchor.lemma,
        "pos": anchor.pos,
        "char_start": anchor.char_start,
        "char_end": anchor.char_end,
        "clip_index": anchor.clip_index,
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def _token_anchor_map(
    tokenizer: Any, prompt: str, doc: Any
) -> tuple[dict[int, SurfaceTokenAnchor], list[dict[str, Any]]]:
    anchors = anchor_surface_tokens(tokenizer, prompt, doc)
    by_i = {anchor.token_index: anchor for anchor in anchors}
    ignored = [
        _anchor_payload(
            anchor,
            role="non_pd_token",
            source="spacy_token_offset",
            status="ignored",
            reason="token_not_selected_by_b3_rules",
        )
        for anchor in anchors
        if anchor.pos not in {"NOUN", "PROPN", "ADJ", "VERB", "AUX"}
    ]
    return by_i, ignored


def _spacy_subject(verb: Any, seen: Optional[set[int]] = None) -> Optional[Any]:
    """Resolve the direct or inherited subject of a possession verb."""

    seen = set() if seen is None else seen
    if verb.i in seen:
        return None
    seen.add(verb.i)
    direct = [
        child
        for child in verb.children
        if child.dep_.lower() in {"nsubj", "nsubjpass", "csubj"}
        and child.pos_ in {"NOUN", "PROPN"}
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        return None
    if verb.dep_.lower() == "conj" and verb.head.pos_ in {"VERB", "AUX"}:
        return _spacy_subject(verb.head, seen)
    if verb.dep_.lower() in {"acl", "relcl"} and verb.head.pos_ in {"NOUN", "PROPN"}:
        # spaCy may attach a serial participle to the preceding object (e.g.
        # "woman wearing a coat holding a phone").  If that noun is itself an
        # object of a possession verb, inherit that verb's subject.
        noun = verb.head
        parent = noun.head
        if (
            noun.dep_.lower() in {"dobj", "obj"}
            and parent.pos_ in {"VERB", "AUX"}
            and parent.lemma_.lower() in POSSESSION_LEMMAS
        ):
            inherited = _spacy_subject(parent, seen)
            if inherited is not None:
                return inherited
        return noun
    if verb.head.pos_ in {"VERB", "AUX"} and verb.dep_.lower() in {
        "xcomp",
        "ccomp",
        "advcl",
    }:
        return _spacy_subject(verb.head, seen)
    return None


def _acomp_entity(modifier: Any) -> Optional[Any]:
    head = modifier.head
    if head.pos_ not in {"VERB", "AUX"}:
        return None
    return _spacy_subject(head)


class SpacyProvider:
    """Predicated Diffusion guidance targets from a spaCy dependency parse (provider ``spacy_sd14``)."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        nlp: Any = None,
        model_name: str = SPACY_MODEL,
    ) -> None:
        self.tokenizer_identity = require_clip_tokenizer_identity(
            tokenizer,
            expected=SD14_GENERATION.tokenizer_identity(),
        )
        self.tokenizer = tokenizer
        self.model_name = model_name
        if nlp is None:
            import spacy

            nlp = spacy.load(model_name)
        self.nlp = nlp

    def provenance(self) -> dict[str, Any]:
        """The ``parser_provenance`` of every row (the recorded key set)."""

        import spacy

        meta = getattr(self.nlp, "meta", {})
        return {
            "provider": "spacy",
            "sidecar_provider": PROVIDER,
            "spacy_version": spacy.__version__,
            "spacy_model": meta.get("name", self.model_name),
            "spacy_model_version": meta.get("version"),
            "entity_pos": ["NOUN", "PROPN"],
            "modifier_dependencies": sorted(MODIFIER_DEPENDENCIES),
            "possession_lemmas": sorted(POSSESSION_LEMMAS),
            "preposition_possession": False,
            "leak_policy": "modifier_to_every_non_head_non_self_entity",
            "tokenizer_model": SD14_GENERATION.model_id,
            "tokenizer_revision": SD14_GENERATION.model_revision,
            "backbone": "sd14",
            "canonical_arm_id": ARM,
            "tokenizer": dict(self.tokenizer_identity),
        }

    def extract(self, prompt: str) -> ExtractionResult:
        provenance = self.provenance()
        try:
            doc = self.nlp(prompt)
            anchors, alignment = _token_anchor_map(self.tokenizer, prompt, doc)
        except Exception as exc:
            return ExtractionResult(
                status="spacy_error", parser_provenance=provenance, error=str(exc)
            )
        failures: list[str] = []
        entities = [token for token in doc if token.pos_ in {"NOUN", "PROPN"}]
        entity_ids = {token.i for token in entities}

        for token in entities:
            anchor = anchors[token.i]
            alignment.append(
                _anchor_payload(anchor, role="entity", source="spacy_token_offset")
            )
            if anchor.clip_index is None:
                failures.append(
                    f"CLIP alignment failed for entity token {token.i}:{token.text}"
                )

        modifier_heads: list[tuple[Any, Any]] = []
        for modifier in doc:
            dep = modifier.dep_.upper()
            if dep not in MODIFIER_DEPENDENCIES:
                continue
            head = modifier.head if modifier.head.i in entity_ids else None
            if dep == "ACOMP" and head is None:
                head = _acomp_entity(modifier)
            if head is None or head.i not in entity_ids:
                failures.append(
                    f"B.3 modifier has no entity head {modifier.i}:{modifier.text}/{dep}"
                )
                continue
            modifier_heads.append((modifier, head))
            anchor = anchors[modifier.i]
            alignment.append(
                _anchor_payload(anchor, role="modifier", source=f"spacy_{dep.lower()}")
            )
            if anchor.clip_index is None:
                failures.append(
                    f"CLIP alignment failed for modifier token {modifier.i}:{modifier.text}"
                )

        exist = sorted(
            {
                anchors[token.i].clip_index
                for token in entities
                if anchors[token.i].clip_index is not None
            }
        )
        corr_set: set[tuple[int, int]] = set()
        leak_set: set[tuple[int, int]] = set()
        for modifier, head in modifier_heads:
            modifier_idx = anchors[modifier.i].clip_index
            head_idx = anchors[head.i].clip_index
            if modifier_idx is None or head_idx is None:
                continue
            corr_set.add((modifier_idx, head_idx))
            for other in entities:
                # COMPOUND modifiers are nouns and therefore entities too;
                # without excluding the modifier itself the loop would emit a
                # degenerate self-leak pair (token vs itself).
                if other.i in (head.i, modifier.i):
                    continue
                other_idx = anchors[other.i].clip_index
                if other_idx is not None:
                    leak_set.add((modifier_idx, other_idx))

        possession_set: set[tuple[int, int]] = set()
        for verb in doc:
            if (
                verb.pos_ not in {"VERB", "AUX"}
                or verb.lemma_.lower() not in POSSESSION_LEMMAS
            ):
                continue
            subject = _spacy_subject(verb)
            objects = [
                child
                for child in verb.children
                if child.dep_.lower() in {"dobj", "obj"}
                and child.pos_ in {"NOUN", "PROPN"}
            ]
            if subject is None or len(objects) != 1:
                failures.append(
                    f"unresolved B.3 subject-object for {verb.i}:{verb.text}"
                )
                continue
            subject_idx = anchors[subject.i].clip_index
            object_idx = anchors[objects[0].i].clip_index
            if subject_idx is None or object_idx is None:
                failures.append(
                    f"CLIP alignment failed for possession verb {verb.i}:{verb.text}"
                )
                continue
            possession_set.add((subject_idx, object_idx))
            alignment.append(
                _anchor_payload(
                    anchors[verb.i], role="possession_verb", source="spacy_verb"
                )
            )

        indices = PDIndices(
            exist=[int(index) for index in exist],
            corr=[
                list(pair)
                for pair in sorted(corr_set, key=lambda pair: (pair[1], pair[0]))
            ],
            leak=[list(pair) for pair in sorted(leak_set)],
            possession=[list(pair) for pair in sorted(possession_set)],
        )
        if not indices.exist:
            failures.append("no entity nouns selected by B.3")
        if failures:
            return ExtractionResult(
                status="alignment_error",
                alignment=tuple(alignment),
                parser_provenance=provenance,
                error="; ".join(sorted(set(failures))),
            )
        return ExtractionResult(
            status="ok",
            pd_indices=indices,
            alignment=tuple(alignment),
            parser_provenance=provenance,
        )


# --- the input file of the sd14_pd_spacy arm -----------------------------------------------------------


def _prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(path, reject_duplicate_keys=True)]
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if set(row) != PROMPT_FIELDS:
            raise PredicatedDiffusionInputError(
                f"{path}:{row_number}: expected fields {sorted(PROMPT_FIELDS)}"
            )
        for name in ("experiment", "prompt_id", "prompt"):
            if not isinstance(row[name], str) or not row[name]:
                raise PredicatedDiffusionInputError(
                    f"{path}:{row_number}: {name} must be non-empty text"
                )
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise PredicatedDiffusionInputError(
                f"{path}:{row_number}: seed must be a non-negative integer"
            )
        if row["prompt_id"] in seen:
            raise PredicatedDiffusionInputError(
                f"{path}:{row_number}: duplicate prompt_id {row['prompt_id']!r}"
            )
        seen.add(row["prompt_id"])
    if not rows:
        raise PredicatedDiffusionInputError(f"{path}: prompt manifest is empty")
    return rows


def base_prompt_rows(blind_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per base prompt (``prompt_id`` without its ``_s<seed>`` suffix), in manifest order, seed 0."""

    rows: dict[str, dict[str, Any]] = {}
    for row in blind_rows:
        base = str(row["prompt_id"]).rsplit("_s", 1)[0]
        if base not in rows:
            rows[base] = {"experiment": row["experiment"], "prompt_id": base, "prompt": row["prompt"], "seed": 0}
    return list(rows.values())


def extract_rows(
    prompt_rows: Sequence[Mapping[str, Any]], provider: SpacyProvider
) -> tuple[list[dict[str, Any]], int]:
    """One guidance-target row per prompt row, and the number of rows that are not ``ok``."""

    output: list[dict[str, Any]] = []
    failures = 0
    for row in prompt_rows:
        try:
            result: ExtractionResult = provider.extract(row["prompt"])
        except Exception as exc:  # preserve a per-prompt audit on unexpected bugs
            result = ExtractionResult(
                status="provider_error",
                parser_provenance={"provider": type(provider).__name__},
                error=str(exc),
            )
        if not result.ok:
            failures += 1
        output.append(
            result.target_row(prompt_id=row["prompt_id"], prompt=row["prompt"])
        )
    return output, failures


def build_input_file(
    blind_prompt_manifest: Path,
    output: Path,
    *,
    tokenizer: Any,
    nlp: Any = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the seed-expanded ``sd14_pd_spacy`` guidance-target file and its ``.meta.json``.

    Every base prompt of the prompt-only manifest is parsed once; the rows are
    expanded to the image rows of the manifest (``source_sidecar_prompt_id``
    names the base prompt). Returns the metadata record.
    """

    blind_rows = _prompt_rows(blind_prompt_manifest)
    base_rows = base_prompt_rows(blind_rows)
    provider = SpacyProvider(tokenizer, nlp=nlp)
    rows, _failures = extract_rows(base_rows, provider)
    expanded, used = expand_to_image_rows(base_rows, rows, blind_rows)
    return write_image_rows_file(
        output,
        expanded,
        provider=PROVIDER,
        tokenizer_identity=provider.tokenizer_identity,
        parser_provenance=provider.provenance(),
        target_prompt_manifest=blind_prompt_manifest,
        used_source_prompt_count=len(used),
        overwrite=overwrite,
    )
