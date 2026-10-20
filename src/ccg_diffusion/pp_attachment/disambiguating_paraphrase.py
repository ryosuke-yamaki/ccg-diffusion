"""Disambiguating Paraphrase (paper Sec. IV-C2).

The ambiguous prompt is rewritten so that it states only the selected
interpretation, using the constrained derivation of that interpretation:

* NP attachment, object cleft: "The girl leaves the boy with a yellow shirt" ->
  "It is the boy with a yellow shirt that the girl leaves.";
* VP attachment, PP fronting: -> "With a yellow shirt, the girl leaves the boy.".

Every original word keeps its CLIP BPE pieces exactly once; only sentence
case, the added function words and punctuation change. Every guidance index is
remapped to the rewritten prompt. Prompts outside the rule (no finite
declarative root clause with a listed present-tense predicate, a site other
than ``with``/``in``, ...) keep the source prompt and record the reason.
No gold setup text or VQA output is read.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

from ccg_diffusion.config import TAB_PARAPHRASE_RULE
from ccg_diffusion.parsing.ccg import (
    AutoNode,
    AutoSentence,
    ClipSourceAnchor,
    align_whitespace_tokens_to_clip,
    parse_ccg_category,
    tokenizer_ids,
)
from ccg_diffusion.pp_attachment.objective import TAB_BASE_LOSS_CONFIG
from ccg_diffusion.provenance import sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices, program_sha256
from ccg_diffusion.targets.tokenizer import require_sd14_tokenizer

PARAPHRASE_RULE = TAB_PARAPHRASE_RULE
# The present-tense matrix predicates of the fixed TAB population (audited from
# its input text only). Other forms are explicit no-ops, not suffix heuristics.
_PRESENT_MATRIX_PREDICATES = frozenset({
    "stands", "enters", "takes", "kills", "threatens", "approaches", "leaves", "looks", "invites",
    "kisses", "hits", "sits", "waves", "plays", "chases",
})


@dataclass(frozen=True)
class Paraphrase:
    """The rewritten prompt (or the source prompt when unsupported), its remapped targets and the trace."""

    prompt: str
    indices: PDIndices
    trace: Mapping[str, Any]
    program_sha256: str


class _UnsupportedConditioning(ValueError):
    """Valid source syntax outside the single predeclared realization."""


def _nodes(node: AutoNode) -> Iterator[AutoNode]:
    yield node
    for child in node.children:
        yield from _nodes(child)


def _is_np(node: AutoNode) -> bool:
    category = parse_ccg_category(node.category)
    return category.is_atomic and category.base_atom == "NP"


def _is_finite_vp(node: AutoNode) -> bool:
    category = parse_ccg_category(node.category)
    return bool(
        category.slash == "\\" and category.result is not None and category.argument is not None
        and category.result.atom == "S[dcl]" and category.argument.atom == "NP"
    )


def _check_auto_source(sentence: AutoSentence, target_row: Mapping[str, Any], trace: Mapping[str, Any]) -> None:
    if sentence.id_line != trace.get("auto_id"):
        raise ValueError("conditioning AUTO id does not match the source conversion trace")
    alignment = target_row["alignment"]
    if len(sentence.leaves) != len(alignment):
        raise ValueError("conditioning AUTO token count does not match the source alignment")
    by_source = {row["source_index"]: row for row in alignment}
    for leaf in sentence.leaves:
        anchor = by_source.get(leaf.source_index)
        if anchor is None or (
            leaf.word != anchor["text"] or leaf.category != anchor["auto_category"]
            or list(leaf.projected_categories) != anchor.get("projected_categories")
        ):
            raise ValueError("conditioning AUTO leaf/projection identity does not match the source alignment")


def _source_anchors(
    prompt: str, target_row: Mapping[str, Any], tokenizer: Any,
) -> dict[int, ClipSourceAnchor]:
    anchors = {(anchor.char_start, anchor.char_end): anchor
               for anchor in align_whitespace_tokens_to_clip(prompt, tokenizer)}
    result: dict[int, ClipSourceAnchor] = {}
    for row in target_row["alignment"]:
        anchor = anchors.get((row["char_start"], row["char_end"]))
        if anchor is None or anchor.anchor_index != row["clip_index"]:
            raise ValueError("conditioning source CLIP anchor does not match the source guidance targets")
        result[row["source_index"]] = anchor
    return result


def _realization_order(
    sentence: AutoSentence, trace: Mapping[str, Any], site: int,
) -> tuple[list[int | str], dict[str, Any]]:
    root = sentence.root
    if (
        root.category != "S[dcl]" or len(root.children) != 2
        or not _is_np(root.children[0]) or not _is_finite_vp(root.children[1])
    ):
        raise _UnsupportedConditioning("requires_finite_declarative_clause")
    subject_np, matrix_vp = root.children
    matrix_predicate = root.head_index
    if sentence.leaves[root.head_index - 1].word.casefold() not in _PRESENT_MATRIX_PREDICATES:
        raise _UnsupportedConditioning("unsupported_matrix_predicate_tense")
    frames = [frame for frame in trace["attachment_possession_frames"] if frame["head_source"] == site]
    if len(frames) != 1 or len(frames[0]["pairs"]) != 1 or not frames[0]["fired"]:
        raise _UnsupportedConditioning("requires_one_explicit_site_holder_object_pair")
    frame = frames[0]
    holder, object_ = frame["pairs"][0]
    all_nodes = list(_nodes(root))
    holder_nps = [node for node in all_nodes if _is_np(node) and node.head_index == holder and node.end < site]
    if not holder_nps:
        raise _UnsupportedConditioning("requires_complete_holder_np_before_site")
    holder_np = max(holder_nps, key=lambda node: node.end - node.start)
    attachment_parents = [
        node for node in all_nodes if len(node.children) == 2
        and node.children[1].start == site and node.children[1].end == root.end
        and node.children[1].head_index == site and node.children[0].end == site - 1
    ]
    if len(attachment_parents) != 1:
        raise _UnsupportedConditioning("requires_one_complete_site_suffix_attachment")
    attachment_parent = attachment_parents[0]
    modified_node, suffix_node = attachment_parent.children
    if not suffix_node.start < object_ <= suffix_node.end:
        raise _UnsupportedConditioning("site_object_is_outside_suffix")
    role = frame["modified_core_role"]
    if role == "atomic_np":
        if modified_node is not holder_np or holder_np.start <= subject_np.end:
            raise _UnsupportedConditioning("requires_matrix_object_np_attachment")
        reading = "np_attach"
    elif role == "verb_phrase":
        if (
            attachment_parent is not matrix_vp or holder_np is not subject_np
            or modified_node.head_index != matrix_predicate
            or frame["raw_holder_sources"] != [matrix_predicate]
        ):
            raise _UnsupportedConditioning("requires_matrix_subject_controlled_event_attachment")
        reading = "vp_attach"
    else:
        raise _UnsupportedConditioning("unsupported_site_modified_core")

    prefix = list(range(1, site))
    suffix = list(range(site, root.end + 1))
    holder_tokens = list(range(holder_np.start, holder_np.end + 1))
    site_word = sentence.leaves[site - 1].word
    if site_word.casefold() in {"with", "in"}:
        if reading == "np_attach":
            order = ["It", "is", *holder_tokens, *suffix, "that", *prefix[:holder_np.start - 1], "."]
            realization = "pp_object_np_cleft"
        else:
            order = [*suffix, ",", *prefix, "."]
            realization = "pp_event_fronting"
    else:
        raise _UnsupportedConditioning("unsupported_site_lexical_form")
    source_order = [part for part in order if isinstance(part, int)]
    if sorted(source_order) != list(range(1, root.end + 1)):
        raise _UnsupportedConditioning("realization_would_drop_or_duplicate_source_tokens")
    return order, {
        "realization": realization, "reading": reading, "holder_source_index": holder,
        "object_source_index": object_, "holder_np_source_span": [holder_np.start, holder_np.end],
        "site_suffix_source_span": [suffix_node.start, suffix_node.end],
        "matrix_subject_source_index": subject_np.head_index, "matrix_predicate_source_index": matrix_predicate,
        "site_modified_core_role": role, "source_token_order": source_order,
    }


def _render_order(
    order: Sequence[int | str], sentence: AutoSentence,
) -> tuple[str, dict[int, tuple[int, int]], list[dict[str, Any]]]:
    text = ""
    spans: dict[int, tuple[int, int]] = {}
    additions: list[dict[str, Any]] = []
    for part in order:
        word = sentence.leaves[part - 1].word if isinstance(part, int) else part
        if not text:
            word = word[0].upper() + word[1:]
        elif part == 1 and word in {"The", "A", "An"}:
            word = word.lower()
        if text and word not in {".", ","}:
            text += " "
        start = len(text)
        text += word
        if isinstance(part, int):
            spans[part] = (start, len(text))
        else:
            additions.append({"text": word, "char_start": start, "char_end": len(text)})
    return text, spans, additions


def _remap_guidance(indices: PDIndices, mapping: Mapping[int, int]) -> PDIndices:
    def anchor(index: int) -> int:
        if type(index) is not int or index not in mapping:
            raise ValueError("guidance index is not a verified original lexical anchor")
        return mapping[index]

    return PDIndices(
        exist=sorted(anchor(index) for index in indices.exist),
        corr=sorted(
            [[anchor(left), anchor(right)] for left, right in indices.corr], key=lambda pair: (pair[1], pair[0])
        ),
        leak=sorted([[anchor(left), anchor(right)] for left, right in indices.leak]),
        possession=sorted([[anchor(left), anchor(right)] for left, right in indices.possession]),
    )


def paraphrase_interpretation(
    *,
    prompt: str,
    target_row: Mapping[str, Any],
    conversion_trace_row: Mapping[str, Any],
    site_source_index: int,
    sentence: AutoSentence,
    tokenizer: Any,
) -> Paraphrase:
    """Rewrite ``prompt`` for the interpretation of ``sentence`` and remap its guidance targets.

    ``target_row`` holds the interpretation's guidance targets on the source
    prompt (``prompt_id``, ``prompt_sha256``, ``program_sha256``,
    ``alignment``, ``pd_indices``); ``conversion_trace_row`` holds their
    conversion trace and its digest. Unsupported syntax returns the source
    prompt; identity, tokenizer and coordinate mismatches raise.
    """

    base = PDIndices(**deepcopy(target_row["pd_indices"]))
    tokenizer_identity = require_sd14_tokenizer(tokenizer)
    source_trace = conversion_trace_row["conversion_trace"]
    _check_auto_source(sentence, target_row, source_trace)
    old_anchors = _source_anchors(prompt, target_row, tokenizer)
    identity_map = {anchor.anchor_index: anchor.anchor_index for anchor in old_anchors.values()}
    _remap_guidance(base, identity_map)
    evidence: dict[str, Any] = {
        "rule": PARAPHRASE_RULE, "applicable": False,
        "source_prompt_id": target_row["prompt_id"], "source_prompt_sha256": target_row["prompt_sha256"],
        "source_program_sha256": target_row["program_sha256"], "input_program_sha256": program_sha256(base),
        "source_conversion_trace_sha256": conversion_trace_row["conversion_trace_sha256"],
        "source_auto_derivation_sha256": sha256_text(sentence.derivation), "tokenizer": tokenizer_identity,
        "site_source_index": site_source_index,
        "input_relation_constraints": [],
    }
    try:
        order, structure = _realization_order(sentence, source_trace, site_source_index)
        rewritten, source_spans, additions = _render_order(order, sentence)
        raw_new_ids = tokenizer_ids(tokenizer(rewritten))
        if len(raw_new_ids) > tokenizer_identity["model_max_length"]:
            raise _UnsupportedConditioning("conditioning_exceeds_token_limit")
        new_anchors = align_whitespace_tokens_to_clip(rewritten, tokenizer)
    except _UnsupportedConditioning as exc:
        digest = program_sha256(base)
        evidence.update(reason=str(exc), conditioning_prompt_sha256=sha256_text(prompt),
                        program_sha256=digest, indices_changed=False, source_to_conditioning_clip={
                            str(source): anchor.anchor_index for source, anchor in old_anchors.items()
                        })
        return Paraphrase(prompt, base, evidence, digest)

    old_ids = tokenizer_ids(tokenizer(prompt))
    mapping: dict[int, int] = {}
    source_mapping: list[dict[str, Any]] = []
    for source, old_anchor in old_anchors.items():
        start, end = source_spans[source]
        matches = [anchor for anchor in new_anchors if anchor.char_start <= start and end <= anchor.char_end]
        if len(matches) != 1:
            raise ValueError("conditioning source word does not have one exact new token span")
        new_anchor = matches[0]
        if [old_ids[index] for index in old_anchor.lexical_piece_indices] != [
            raw_new_ids[index] for index in new_anchor.lexical_piece_indices
        ]:
            raise ValueError("conditioning changed an original word's lexical BPE identity")
        previous = mapping.setdefault(old_anchor.anchor_index, new_anchor.anchor_index)
        if previous != new_anchor.anchor_index:
            raise ValueError("conditioning cannot split an aliased source CLIP anchor")
        source_mapping.append({
            "source_index": source, "source_text": sentence.leaves[source - 1].word,
            "conditioning_text": rewritten[start:end],
            "source_char_span": [old_anchor.char_start, old_anchor.char_end],
            "conditioning_char_span": [start, end],
            "source_clip_index": old_anchor.anchor_index, "conditioning_clip_index": new_anchor.anchor_index,
            "lexical_token_ids": [old_ids[index] for index in old_anchor.lexical_piece_indices],
        })
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("conditioning merged distinct original CLIP anchors")
    indices = _remap_guidance(base, mapping)
    digest = program_sha256(indices)
    evidence.update(
        **structure, applicable=True, reason=None, source_mapping=source_mapping, added_tokens=additions,
        source_to_conditioning_clip={
            str(row["source_index"]): row["conditioning_clip_index"] for row in source_mapping
        },
        conditioning_prompt_sha256=sha256_text(rewritten), program_sha256=digest,
        conditioning_token_count=len(raw_new_ids), source_tokens_preserved_once=True,
        original_lexical_token_ids_preserved=True, indices_changed=digest != program_sha256(base),
        conditioning_relation_constraints=[],
    )
    return Paraphrase(rewritten, indices, evidence, digest)


def source_job(
    *, tab_id: str, reading: str, paraphrase: Paraphrase, source_program_sha256: str, tokenizer: Any,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """The interpretation's job template before CCG Diffusion's TAB components are applied.

    ``evaluation`` holds the evaluation identity (``tab_category``,
    ``tab_requires_reading``, ``tab_evaluation_mode``, ``target_setup_index``,
    ``evaluation_setup_indices``, ``target_setup_mapping_evidence_sha256``).
    """
    return {
        "tab_id": tab_id, "reading": reading, "prompt": paraphrase.prompt,
        "token_ids": tokenizer(paraphrase.prompt, truncation=False)["input_ids"],
        "pd_indices": asdict(paraphrase.indices), "program_sha256": paraphrase.program_sha256,
        "source_program_sha256": source_program_sha256,
        "loss_config": {**TAB_BASE_LOSS_CONFIG, "scale_factor": 10.0, "max_update_ratio": 0.02},
        "relations": [], **evaluation,
    }
