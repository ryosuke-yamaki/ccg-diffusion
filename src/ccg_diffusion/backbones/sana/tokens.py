"""Sana's Gemma token axis and the projection of CLIP-indexed guidance targets onto it.

Guidance targets I = (E, C, K, P) are token indices on the SD1.4 CLIP axis
(leading BOS, trailing EOS). Sana encodes a prompt with Gemma after
prepending its instruction prompt, and ``SanaPipeline.encode_prompt`` keeps a
300-token window: the BOS token and the last 299 tokens. This module recreates
that "selected" axis (BOS first, no trailing EOS) and maps every CLIP index to
the Gemma tokens of the same prompt word through character offsets, refusing
any mapping that would need a guess.

Imports neither torch nor diffusers.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ccg_diffusion.targets.pd_indices import PD_FIELDS, PDIndices
from ccg_diffusion.targets.pd_indices import pd_indices_payload as pd_payload

SANA_TARGET_ROW_SCHEMA = "holsd.sana_pd_projection.v1"
SANA_SELECTED_INDEX_SPACE = "sana_selected_bos_inclusive_v1"
_NORMALISE_RE = re.compile(r"[\s_▁\W]+", re.UNICODE)
_WORD_RE = re.compile(r"\S+")


class SanaPDProjectionError(ValueError):
    """Raised when guidance targets cannot be mapped onto the Gemma axis without guessing."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalise_token(text: str) -> str:
    """Lower-case text without whitespace, SentencePiece markers or punctuation."""
    return _NORMALISE_RE.sub("", text).lower()


def parse_pd_payload(raw: Any, *, location: str) -> PDIndices:
    """Read a recorded guidance-target payload, rejecting any malformed or non-positive index."""
    if not isinstance(raw, Mapping) or set(raw) != set(PD_FIELDS):
        raise SanaPDProjectionError(
            f"{location}: PD payload fields must be exactly {list(PD_FIELDS)}"
        )

    def indexes(field: str) -> list[int]:
        value = raw[field]
        if not isinstance(value, list):
            raise SanaPDProjectionError(f"{location}.{field} must be a list")
        result: list[int] = []
        for item in value:
            if type(item) is not int or item < 1:
                raise SanaPDProjectionError(
                    f"{location}.{field} contains invalid index {item!r}"
                )
            result.append(item)
        return result

    def pairs(field: str) -> list[list[int]]:
        value = raw[field]
        if not isinstance(value, list):
            raise SanaPDProjectionError(f"{location}.{field} must be a list")
        result: list[list[int]] = []
        for item in value:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(type(index) is not int or index < 1 for index in item)
            ):
                raise SanaPDProjectionError(
                    f"{location}.{field} contains invalid pair {item!r}"
                )
            result.append([int(item[0]), int(item[1])])
        return result

    return PDIndices(
        exist=indexes("exist"),
        corr=pairs("corr"),
        leak=pairs("leak"),
        possession=pairs("possession"),
    )


def selected_token_offsets(
    tokenizer: Any,
    prompt: str,
    *,
    instruction: Sequence[str],
    max_sequence_length: int = 300,
) -> tuple[dict[int, str], dict[int, tuple[int, int]]]:
    """Decoded pieces of the selected Gemma axis and their character offsets relative to ``prompt``.

    Only pieces that overlap ``prompt`` (not the instruction prompt) receive an offset.
    """
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        instruction_text = ""
        prompt_text = prompt
        if instruction:
            instruction_text = "\n".join(instruction)
            prompt_text = instruction_text + prompt
            instruction_tokens = len(tokenizer.encode(instruction_text))
            max_length_all = instruction_tokens + max_sequence_length - 2
        else:
            max_length_all = max_sequence_length
        encoded = tokenizer(
            prompt_text,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        select_index = [0] + list(range(-max_sequence_length + 1, 0))
        token_ids = encoded.input_ids[0][select_index]
        offsets = encoded.offset_mapping[0][select_index]
        decoded = {
            index: tokenizer.decode([int(token_ids[index])])
            for index in range(int(token_ids.shape[0]))
        }
        relative: dict[int, tuple[int, int]] = {}
        instruction_length = len(instruction_text)
        for index, raw_offset in enumerate(offsets):
            start, end = (int(raw_offset[0]), int(raw_offset[1]))
            start -= instruction_length
            end -= instruction_length
            if end <= 0 or start >= len(prompt) or end <= start:
                continue
            relative[index] = (max(0, start), min(len(prompt), end))
        return decoded, relative
    finally:
        tokenizer.padding_side = saved_padding_side


def _unique_source_indices(indices: PDIndices) -> list[int]:
    return sorted(
        set(indices.exist)
        | {index for pair in indices.corr for index in pair}
        | {index for pair in indices.leak for index in pair}
        | {index for pair in indices.possession for index in pair}
    )


def _word_for_char_span(prompt: str, char_start: int, char_end: int) -> tuple[int, str]:
    candidates = []
    for index, match in enumerate(_WORD_RE.finditer(prompt)):
        overlap = max(0, min(match.end(), char_end) - max(match.start(), char_start))
        if overlap:
            candidates.append((overlap, index, match.group(0)))
    if not candidates:
        raise SanaPDProjectionError(
            f"anchor span [{char_start}, {char_end}) does not overlap a prompt word"
        )
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0]:
        raise SanaPDProjectionError(
            f"anchor span [{char_start}, {char_end}) maps ambiguously to prompt words"
        )
    return best[1], best[2]


def _anchor_for_clip_index(
    alignment: Any,
    source_index: int,
    *,
    location: str,
) -> dict[str, Any]:
    if not isinstance(alignment, list):
        raise SanaPDProjectionError(f"{location}.alignment must be a list")
    candidates: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in alignment:
        if not isinstance(raw, Mapping) or raw.get("clip_index") != source_index:
            continue
        start, end, text = raw.get("char_start"), raw.get("char_end"), raw.get("text")
        if type(start) is not int or type(end) is not int or not isinstance(text, str):
            continue
        candidates[(start, end, text)] = dict(raw)
    if len(candidates) != 1:
        raise SanaPDProjectionError(
            f"{location}: CLIP index {source_index} has {len(candidates)} unique anchors"
        )
    return next(iter(candidates.values()))


def project_target_row(
    *,
    prompt_id: str,
    prompt: str,
    source_row: Mapping[str, Any],
    tokenizer: Any,
    instruction: Sequence[str],
    max_sequence_length: int = 300,
    fallback_statuses: Sequence[str] = (),
) -> dict[str, Any]:
    """Project one CLIP-indexed guidance-target row onto Sana's selected Gemma axis.

    Each CLIP index is mapped to the last Gemma token of the prompt word that
    its alignment anchor covers; the decoded Gemma span must spell the anchor
    text. A row whose status is in ``fallback_statuses`` becomes a
    ``"stock_vanilla_fallback"`` record with empty targets.
    """
    expected_prompt_hash = prompt_sha256(prompt)
    if source_row.get("prompt_id") != prompt_id:
        raise SanaPDProjectionError(f"{prompt_id}: guidance-target row prompt_id mismatch")
    if source_row.get("prompt_sha256") != expected_prompt_hash:
        raise SanaPDProjectionError(f"{prompt_id}: guidance-target row prompt hash mismatch")
    status = source_row.get("status")
    if status != "ok":
        if status not in set(fallback_statuses):
            raise SanaPDProjectionError(
                f"{prompt_id}: guidance-target row status {status!r} is not an allowed fallback"
            )
        return {
            "schema": SANA_TARGET_ROW_SCHEMA,
            "prompt_id": prompt_id,
            "prompt_sha256": expected_prompt_hash,
            "provider": source_row.get("provider"),
            "source_status": status,
            "guidance_mode": "stock_vanilla_fallback",
            "fallback_reason": status,
            "source_program_sha256": source_row.get("program_sha256"),
            "source_row_sha256": _canonical_sha256(dict(source_row)),
            "source_pd_indices": {field: [] for field in PD_FIELDS},
            "sana_pd_indices": {field: [] for field in PD_FIELDS},
            "sana_program_sha256": _canonical_sha256({field: [] for field in PD_FIELDS}),
            "index_space": SANA_SELECTED_INDEX_SPACE,
            "token_index_map": {},
            "token_layout_verified": False,
        }

    source_indices = parse_pd_payload(
        source_row.get("pd_indices"), location=f"{prompt_id}.pd_indices"
    )
    decoded, sana_offsets = selected_token_offsets(
        tokenizer,
        prompt,
        instruction=instruction,
        max_sequence_length=max_sequence_length,
    )
    word_matches = list(_WORD_RE.finditer(prompt))
    provider = str(source_row.get("provider"))
    token_map: dict[int, int] = {}
    audit_map: dict[str, dict[str, Any]] = {}
    for source_index in _unique_source_indices(source_indices):
        anchor = _anchor_for_clip_index(
            source_row.get("alignment"),
            source_index,
            location=prompt_id,
        )
        char_start, char_end = int(anchor["char_start"]), int(anchor["char_end"])
        word_index, word_text = _word_for_char_span(prompt, char_start, char_end)
        anchor_text = str(anchor["text"])
        if normalise_token(anchor_text) not in normalise_token(word_text):
            raise SanaPDProjectionError(
                f"{prompt_id}: anchor {anchor_text!r} does not match word {word_text!r}"
            )
        target_span = [
            int(index)
            for index, (token_start, token_end) in sorted(sana_offsets.items())
            if max(0, min(token_end, char_end) - max(token_start, char_start)) > 0
        ]
        if not target_span:
            raise SanaPDProjectionError(
                f"{prompt_id}: source index {source_index} mapped to an empty Sana span"
            )
        target_index = target_span[-1]
        decoded_text = "".join(decoded[index] for index in target_span)
        expected_normalized = normalise_token(anchor_text)
        decoded_normalized = normalise_token(decoded_text)
        if decoded_normalized != expected_normalized:
            raise SanaPDProjectionError(
                f"{prompt_id}: Sana token mismatch for source index {source_index}: "
                f"{decoded_text!r} != {word_matches[word_index].group(0)!r}"
            )
        token_map[source_index] = target_index
        audit_map[str(source_index)] = {
            "source_index": source_index,
            "source_index_space": "clip_bos_inclusive",
            "target_index": target_index,
            "target_span": target_span,
            "word_index": word_index,
            "char_start": char_start,
            "char_end": char_end,
            "anchor_text": anchor_text,
            "expected_text": word_matches[word_index].group(0),
            "expected_normalized": expected_normalized,
            "decoded_span": [decoded[index] for index in target_span],
            "decoded_normalized": decoded_normalized,
            "verified": True,
        }

    def mapped(index: int) -> int:
        return token_map[int(index)]

    projected = PDIndices(
        exist=sorted({mapped(index) for index in source_indices.exist}),
        corr=[[mapped(a), mapped(b)] for a, b in source_indices.corr],
        leak=[[mapped(a), mapped(b)] for a, b in source_indices.leak],
        possession=[[mapped(a), mapped(b)] for a, b in source_indices.possession],
    )
    for field_name in ("corr", "leak", "possession"):
        source_pairs = getattr(source_indices, field_name)
        projected_pairs = getattr(projected, field_name)
        for source_pair, projected_pair in zip(source_pairs, projected_pairs):
            if source_pair[0] != source_pair[1] and projected_pair[0] == projected_pair[1]:
                raise SanaPDProjectionError(
                    f"{prompt_id}: Sana projection collapsed {field_name} pair "
                    f"{source_pair!r} onto one token"
                )
    source_payload = pd_payload(source_indices)
    sana_payload = pd_payload(projected)
    return {
        "schema": SANA_TARGET_ROW_SCHEMA,
        "prompt_id": prompt_id,
        "prompt_sha256": expected_prompt_hash,
        "provider": provider,
        "source_status": "ok",
        "guidance_mode": "pd_guided",
        "fallback_reason": None,
        "source_program_sha256": source_row.get("program_sha256"),
        "source_row_sha256": _canonical_sha256(dict(source_row)),
        "source_pd_indices": source_payload,
        "sana_pd_indices": sana_payload,
        "sana_program_sha256": _canonical_sha256(sana_payload),
        "index_space": SANA_SELECTED_INDEX_SPACE,
        "token_index_map": audit_map,
        "token_layout_verified": True,
    }


def preprocess_prompt(pipe: Any, prompt: str) -> str:
    """Apply ``SanaPipeline``'s caption preprocessing (``clean_caption=False``: lower-case and strip)."""
    if hasattr(pipe, "_text_preprocessing"):
        return pipe._text_preprocessing([prompt], clean_caption=False)[0]
    return prompt


def _selected_token_ids(
    pipe: Any,
    prompt: str,
    *,
    instruction: Sequence[str],
    max_sequence_length: int = 300,
) -> Any:
    """Recreate the selected token-id axis of ``SanaPipeline.encode_prompt``."""
    tokenizer = pipe.tokenizer
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        prompt_text = preprocess_prompt(pipe, prompt)
        if instruction:
            chi_prompt = "\n".join(instruction)
            prompt_text = chi_prompt + prompt_text
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_sequence_length - 2
        else:
            max_length_all = max_sequence_length
        text_inputs = tokenizer(
            prompt_text,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        select_index = [0] + list(range(-max_sequence_length + 1, 0))
        return text_inputs.input_ids[0][select_index]
    finally:
        tokenizer.padding_side = saved_padding_side


def decoded_prompt_tokens(
    pipe: Any,
    prompt: str,
    prompt_attention_mask: Any,
    *,
    instruction: Sequence[str],
    max_sequence_length: int = 300,
) -> dict[int, str]:
    """Decode the real (unmasked) tokens of the selected Gemma axis; used for diagnostics only."""
    ids = _selected_token_ids(
        pipe,
        prompt,
        instruction=instruction,
        max_sequence_length=max_sequence_length,
    )
    real_count = int(prompt_attention_mask.reshape(-1).sum().item())
    return {
        i: pipe.tokenizer.decode([int(ids[i])])
        for i in range(min(real_count, ids.shape[0]))
    }
