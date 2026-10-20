"""Holder-Object Region Assignment (paper Sec. IV-C2).

The two interpretations of a TAB PP prompt attach the same PP object to
different holders. Their paraphrase traces identify the selected holder, the
alternative holder and the PP object. The region layout puts the selected
holder, the object and their modifiers in one group and the other holder with
its modifiers in the other; the two holders keep their source order (left
group = the holder that comes first). The layout is a ``PrefixLayoutPlan``
without a prefix (the Disambiguating Paraphrase is never combined with the
Positional Prefix); the Region Assignment Loss reads its groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ccg_diffusion.guidance.positional_prefix import PrefixLayoutPlan, lexical_mapping
from ccg_diffusion.provenance import canonical_json, digest, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices, program_sha256


@dataclass(frozen=True)
class HolderObjectPair:
    """Three distinct lexical endpoints (CLIP indices of the current prompt) and their source identity."""

    holder: int
    alternative: int
    object: int
    prompt_sha256: str
    program_sha256: str
    source_indices: tuple[int, int, int]
    source_trace_sha256: tuple[str, str]

    def __post_init__(self) -> None:
        for indices in ((self.holder, self.alternative, self.object), self.source_indices):
            if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or i < 1 for i in indices):
                raise ValueError("attachment endpoints must be three distinct positive lexical indices")
        digests = (self.prompt_sha256, self.program_sha256, *self.source_trace_sha256)
        if len(self.source_trace_sha256) != 2 or any(
            not isinstance(d, str) or len(d) != 64 or set(d) - set("0123456789abcdef") for d in digests
        ):
            raise ValueError("attachment plan identities must be SHA-256 digests")


def holder_object_pair(
    *,
    prompt: str,
    indices: PDIndices,
    selected: Mapping[str, Any],
    alternative: Mapping[str, Any],
) -> HolderObjectPair | None:
    """The holder/object pair of the selected interpretation, from the two paraphrase traces alone.

    ``None`` when a trace has no supported realization or the holders and the
    object are not three distinct words; inconsistent traces raise.
    """
    allowed = {"pp_object_np_cleft", "pp_event_fronting"}
    if selected.get("realization") not in allowed or alternative.get("realization") not in allowed:
        return None
    if selected["realization"] == alternative["realization"]:
        raise ValueError("attachment plan needs opposite forced readings")
    for field in ("source_prompt_sha256", "site_source_index", "object_source_index"):
        if selected[field] != alternative[field]:
            raise ValueError("forced readings disagree on source prompt, PP site, or object")
    if selected["conditioning_prompt_sha256"] != sha256_text(prompt):
        raise ValueError("selected trace does not identify the current prompt")
    if selected["program_sha256"] != program_sha256(indices):
        raise ValueError("selected trace does not identify the current PD program")
    source_indices = (
        selected["holder_source_index"],
        alternative["holder_source_index"],
        selected["object_source_index"],
    )
    if len(set(source_indices)) != 3:
        return None
    mapping = {r["source_index"]: r for r in selected["source_mapping"]}
    other = {r["source_index"]: r for r in alternative["source_mapping"]}
    if len(mapping) != len(selected["source_mapping"]) or mapping.keys() != other.keys():
        raise ValueError("forced readings lack the same unique source tokens")
    for i, row in mapping.items():
        if any(row[f] != other[i][f] for f in ("source_text", "source_char_span", "lexical_token_ids")):
            raise ValueError("forced readings disagree on source token identity")
    holder, other_holder, object_ = (mapping[i]["conditioning_clip_index"] for i in source_indices)
    if [holder, object_] not in indices.possession:
        raise ValueError("selected PP holder/object pair is absent from the current PD program")
    return HolderObjectPair(
        holder,
        other_holder,
        object_,
        sha256_text(prompt),
        program_sha256(indices),
        source_indices,
        (sha256_text(canonical_json(selected)), sha256_text(canonical_json(alternative))),
    )


def region_layout(
    row: Mapping[str, Any], selected: Mapping[str, Any], opposite: Mapping[str, Any], targets: Any,
    tokenizer: Any,
) -> tuple[PrefixLayoutPlan | None, dict[int, int], str]:
    """Region layout of one job row: ``(plan, token mapping, reason)``.

    ``row`` is the job (prompt, token IDs, guidance targets with the attachment
    possession), ``selected``/``opposite`` are the paraphrase traces of the two
    interpretations and ``targets`` are the guidance targets on the source
    prompt (``compile.InterpretationTargets``). Each group holds a holder and
    its source-bound correspondence modifiers; the selected holder's group also
    holds the PP object. The PP suffix stays verbatim.
    """
    indices = PDIndices(**row["pd_indices"])
    attachment = holder_object_pair(
        prompt=row["prompt"], indices=indices, selected=selected, alternative=opposite)
    if attachment is None:
        return None, {}, "unsupported_attachment_pair"
    source_prompt = targets.prompt
    if selected["source_prompt_sha256"] != sha256_text(source_prompt):
        raise ValueError("Layout audit and conditioning refer to different source prompts")
    current = {r["source_index"]: r for r in selected["source_mapping"]}
    source_alignment = {r["source_index"]: r for r in targets.alignment}
    if current.keys() != source_alignment.keys():
        raise ValueError("Layout source token coverage differs")
    for source, entry in current.items():
        original = source_alignment[source]
        if entry["source_text"] != original["text"] or entry["source_clip_index"] != original["clip_index"]:
            raise ValueError("Layout source word/anchor differs from the CCG audit")
    holder = selected["holder_source_index"]
    alternate = opposite["holder_source_index"]
    object_source = selected["object_source_index"]
    ordered = sorted((holder, alternate))
    spans = {
        holder: tuple(range(selected["holder_np_source_span"][0], selected["holder_np_source_span"][1] + 1)),
        alternate: tuple(range(opposite["holder_np_source_span"][0], opposite["holder_np_source_span"][1] + 1)),
    }
    suffix = tuple(range(selected["site_suffix_source_span"][0], selected["site_suffix_source_span"][1] + 1))
    if set(spans[holder]) & set(spans[alternate]) or any(set(suffix) & set(s) for s in spans.values()):
        raise ValueError("Layout phrases overlap")

    # lexical_mapping uses positions in the existing reading-conditioned text.
    alignment = [{
        "source_index": source, "char_start": entry["conditioning_char_span"][0],
        "char_end": entry["conditioning_char_span"][1],
        "clip_lexical_piece_indices": list(range(
            entry["conditioning_clip_index"] - len(entry["lexical_token_ids"]) + 1,
            entry["conditioning_clip_index"] + 1)),
    } for source, entry in current.items()]
    prefix_text, copies = "", []
    new_prompt = prefix_text + row["prompt"]
    if len(tokenizer(new_prompt, truncation=False)["input_ids"]) > 77:
        return None, {}, "conditioning_would_truncate"
    old_ids, new_ids, mapping = lexical_mapping(row["prompt"], prefix_text, alignment, copies, tokenizer)
    if old_ids != row["token_ids"]:
        raise ValueError("Layout source token IDs changed")
    source_to_new = {s: mapping[r["conditioning_clip_index"]] for s, r in current.items()}
    source_anchor = {r["clip_index"]: s for s, r in source_alignment.items()}
    groups = []
    for noun in ordered:
        entities = [noun] + ([object_source] if noun == holder else [])
        words = set(entities)
        for modifier, entity in targets.pd_indices["corr"]:
            if source_anchor[entity] in entities:
                words.add(source_anchor[modifier])
        group = []
        for source in sorted(words):
            entry = current[source]
            anchor = entry["conditioning_clip_index"]
            pieces = range(anchor - len(entry["lexical_token_ids"]) + 1, anchor + 1)
            group.append(tuple(mapping[piece] for piece in pieces))
        groups.append(tuple(group))
    mapped = {"exist": [mapping[i] for i in indices.exist]}
    for field in ("corr", "leak", "possession"):
        mapped[field] = [[mapping[i] for i in pair] for pair in row["pd_indices"][field]]
    if [source_to_new[holder], source_to_new[object_source]] not in mapped["possession"]:
        raise ValueError("Layout lost its selected possession pair")
    plan = PrefixLayoutPlan(
        sha256_text(row["prompt"]), digest(row["pd_indices"]), digest(alignment),
        new_prompt, len(old_ids), len(new_ids), "tab_attachment", "left",
        tuple(sorted(mapping.items())), canonical_json(mapped), tuple(groups), None,
        digest({"selected": selected, "opposite": opposite, "alignment": list(targets.alignment)}),
        "source_order_holders_with_selected_PP_object",
    )
    return plan, mapping, "active"
