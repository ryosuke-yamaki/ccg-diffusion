"""Spatial Relation Loss on TAB: ``next_to`` constraints from the constrained derivation.

A TAB prompt may say that the holder is "next to" something. The relation is
taken only when the marked dependencies prove the frame: ``next`` heads a
fired verb-phrase attachment frame with a ``PP`` complement and slots 1 and 2,
``to`` follows it, and each slot resolves through marked slot-1 edges to one
atomic nominal entity. The constraint is projected onto the CLIP columns of
the source prompt. Only the clean path calls this; the frozen bundle stores
the relations.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ccg_diffusion.guidance.spatial_relations import RelationTriple, project_relation_triple
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency, is_atomic_nominal, parse_ccg_category

NEXT_TO_RULE = "tab_next_to_vp_attachment_slot1_resolution_v1"


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("TAB next_to frame is not proven by the derivation and its marked dependencies")


def validate_resolution_paths(
    frame: Mapping[str, Any], role: str, dependencies: Sequence[MarkedDependency],
) -> None:
    """Prove every recorded endpoint path of ``role`` with actual marked slot-1 edges."""
    raw = set(frame[f"raw_{role}_sources"])
    resolved = set(frame[f"{role}_sources"])
    paths = frame[f"{role}_resolution_paths"]
    _require(bool(paths and resolved))
    observed: set[int] = set()
    for path in paths:
        positions = path["source_path"]
        indices = path["dependency_indices"]
        _require(positions[0] in raw and positions[-1] in resolved)
        _require(positions[-1] == path["resolved_entity"])
        _require(len(indices) == len(positions) - 1)
        for start, stop, edge_index in zip(positions, positions[1:], indices):
            edge = dependencies[edge_index]
            _require(edge.head_index == start and edge.argument_index == stop and edge.slot == 1)
        observed.add(positions[-1])
    _require(observed == resolved)


def next_to_relations(
    *, prompt: str, site_source_index: int, alignment: Sequence[Mapping[str, Any]],
    pd_indices: Mapping[str, Any], conversion_trace: Mapping[str, Any], sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(constraints, evidence)``: at most one ``next_to`` constraint on the source prompt's CLIP columns.

    The derivation and its marked dependencies determine the endpoints; the
    words next/to only validate the closed relation label.
    """
    leaves = {leaf.source_index: leaf for leaf in sentence.leaves}
    constraints: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for frame in conversion_trace["attachment_possession_frames"]:
        if frame["head_word"].lower() != "next":
            continue
        _require(frame["fired"] and frame["modified_core_role"] == "verb_phrase")
        _require(frame["complement_atom"] == "PP" and frame["slots_present"] == [1, 2])
        _require(len(frame["holder_sources"]) == len(frame["object_sources"]) == 1)
        head = int(frame["head_source"])
        _require(leaves[head].word.lower() == "next")
        to_source = head + 1
        _require(leaves[to_source].word.lower() == "to")
        _require(frame["raw_object_sources"] == [to_source])
        for role, slot in (("holder", 1), ("object", 2)):
            arguments = {d.argument_index for d in dependencies if d.head_index == head and d.slot == slot}
            _require(arguments == set(frame[f"raw_{role}_sources"]))
            validate_resolution_paths(frame, role, dependencies)
        _require(all(d.marked_category == frame["marked_category"] for d in dependencies if d.head_index == head))
        subject, obj = int(frame["holder_sources"][0]), int(frame["object_sources"][0])
        _require(subject != obj and [subject, obj] in frame["pairs"])
        for endpoint in (subject, obj):
            _require(is_atomic_nominal(parse_ccg_category(leaves[endpoint].category)))
            _require(endpoint in conversion_trace["source_indices"]["entity"])
        triple = RelationTriple(
            subject_source_index=subject, object_source_index=obj, label="next_to",
            subject_word=leaves[subject].word, object_word=leaves[obj].word,
            relation_surface="next to", relation_source_indices=(head, to_source), extraction="derivation",
        )
        constraint = project_relation_triple(triple, alignment)
        _require(constraint.subject_clip_index in pd_indices["exist"])
        _require(constraint.object_clip_index in pd_indices["exist"])
        constraints.append(asdict(constraint))
        evidence.append({
            "rule": NEXT_TO_RULE, "triple": asdict(triple),
            "head_source": head, "head_category": leaves[head].category,
            "is_ambiguity_site": head == site_source_index,
            "raw_holder_sources": frame["raw_holder_sources"], "raw_object_sources": frame["raw_object_sources"],
            "holder_resolution_paths": frame["holder_resolution_paths"],
            "object_resolution_paths": frame["object_resolution_paths"],
            "original_possession_pair": [constraint.subject_clip_index, constraint.object_clip_index],
        })
    _require(len(constraints) <= 1)
    return constraints, evidence
