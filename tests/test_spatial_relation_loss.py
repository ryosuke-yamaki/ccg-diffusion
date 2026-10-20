"""Closed-set 2D relation phrases from the derivation and the Spatial Relation Loss."""
from __future__ import annotations

import math

import pytest
import torch

from conftest import marked_dependencies, synthetic_sentence
from ccg_diffusion.guidance.spatial_relation_loss import (
    MARGIN,
    PROXIMITY_MAX,
    SPATIAL_RELATION_LOSS_PARAMETERS,
    SPATIAL_RELATION_LOSS_WEIGHT,
    compute_spatial_relation_loss,
)
from ccg_diffusion.guidance.spatial_relations import (
    RELATION_SURFACE_TO_LABEL,
    RelationConstraint,
    RelationExtractionError,
    extract_relation_triples_from_derivation,
    project_relation_triple,
)

NEXT_TO = ("a giraffe next to a lamp", ["NP[nb]/N", "N", "(NP\\NP)/PP", "PP/NP", "NP[nb]/N", "N"], (
    "next_3 ((NP{Y}\\NP{Y}<1>){_}/PP{Z}<2>){_} 1 giraffe_2 0", "a_1 (NP[nb]{Y}/N{Y}<1>){_} 1 giraffe_2 0",
    "next_3 ((NP{Y}\\NP{Y}<1>){_}/PP{Z}<2>){_} 2 to_4 0", "to_4 (PP{_}/NP{Y}<1>){_} 1 lamp_6 0",
    "a_5 (NP[nb]{Y}/N{Y}<1>){_} 1 lamp_6 0"))
TOP = ("a girl on the top of a frog",
       ["NP[nb]/N", "N", "(NP\\NP)/NP", "NP[nb]/N", "N", "(NP\\NP)/NP", "NP[nb]/N", "N"], (
           "on_3 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 1 girl_2 0", "on_3 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 2 top_5 0",
           "of_6 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 1 top_5 0", "the_4 (NP[nb]{Y}/N{Y}<1>){_} 1 top_5 0",
           "of_6 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 2 frog_8 0", "a_7 (NP[nb]{Y}/N{Y}<1>){_} 1 frog_8 0"))
WITH = ("a man with a hat", ["NP[nb]/N", "N", "(NP\\NP)/NP", "NP[nb]/N", "N"], (
    "with_3 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 1 man_2 0", "with_3 ((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_} 2 hat_5 0"))


def _triples(example: tuple) -> tuple:
    prompt, categories, deps = example
    return extract_relation_triples_from_derivation(synthetic_sentence(prompt, categories), marked_dependencies(*deps))


def _blob(placements: dict[int, tuple[int, int]], tokens: int = 10) -> torch.Tensor:
    maps = torch.zeros(16, 16, tokens)
    for token, (row, col) in placements.items():
        maps[row, col, token] = 1.0
    return maps


def _loss(maps: torch.Tensor, *relations: tuple[str, int, int], **kwargs) -> float:
    return float(compute_spatial_relation_loss(maps, [RelationConstraint(*r) for r in relations], **kwargs))


def test_closed_relation_set_and_constants() -> None:
    assert set(RELATION_SURFACE_TO_LABEL) == {"on the left of", "on the right of", "on the top of",
                                              "on the bottom of", "next to", "near", "on side of"}
    assert (SPATIAL_RELATION_LOSS_WEIGHT, MARGIN, PROXIMITY_MAX) == (6.0, 0.25, 150.0 / 512.0)
    assert dict(SPATIAL_RELATION_LOSS_PARAMETERS) == {
        "weight": 6.0, "margin": 0.25, "proximity_max": 150.0 / 512.0, "separation_weight": 1.0,
        "geometry_variant": "metric_aligned_v1", "off_axis_weight": 1.0}


def test_derivation_walk_reads_the_preposition_and_of_chains() -> None:
    (next_to,) = _triples(NEXT_TO)
    assert (next_to.label, next_to.subject_source_index, next_to.object_source_index) == ("next_to", 2, 6)
    assert next_to.relation_surface == "next to" and next_to.extraction == "derivation"
    assert next_to.relational_nominal_source_index is None
    (top,) = _triples(TOP)
    assert (top.label, top.subject_word, top.object_word) == ("top", "girl", "frog")
    assert top.relation_surface == "on the top of"
    assert (top.relational_nominal_source_index, top.relational_nominal_word) == (5, "top")


def test_unsupported_surface_raises_the_recorded_text() -> None:
    with pytest.raises(RelationExtractionError) as error:
        _triples(WITH)
    assert str(error.value) == "no closed-set relation triple: surface 'with' is outside the closed relation set"
    with pytest.raises(RelationExtractionError, match="no nominal attachment functor"):
        extract_relation_triples_from_derivation(synthetic_sentence("a red box", ["NP[nb]/N", "N/N", "N"]), [])


def test_projection_uses_the_alignment_rows() -> None:
    (top,) = _triples(TOP)
    alignment = [{"source_index": i, "prompt_word_index": i, "clip_index": i + 1} for i in range(1, 9)]
    assert project_relation_triple(top, alignment) == RelationConstraint("top", 3, 9)
    alignment[1]["prompt_word_index"] = 99
    with pytest.raises(RelationExtractionError, match="source_index/prompt_word_index mismatch"):
        project_relation_triple(top, alignment)


def test_directional_hinges() -> None:
    assert _loss(_blob({2: (8, 2), 8: (8, 14)}), ("left", 2, 8), separation_weight=0.0) == pytest.approx(0.0, abs=1e-6)
    # reversed layout: cx_object - cx_subject = -0.75, relu(0.25 + 0.75) = 1
    assert _loss(_blob({2: (8, 14), 8: (8, 2)}), ("left", 2, 8), separation_weight=0.0) == pytest.approx(1.0, abs=1e-6)
    vertical = _blob({2: (2, 8), 8: (14, 8)})
    assert _loss(vertical, ("top", 2, 8), separation_weight=0.0) == pytest.approx(0.0, abs=1e-6)
    assert _loss(vertical, ("bottom", 2, 8), separation_weight=0.0) == pytest.approx(1.0, abs=1e-6)
    # the requested axis must dominate: horizontal gap 6/16, vertical gap 12/16
    assert _loss(_blob({2: (2, 2), 8: (14, 8)}), ("left", 2, 8), separation_weight=0.0) == pytest.approx(6 / 16)


def test_proximity_uses_the_smaller_axis_distance() -> None:
    assert _loss(_blob({2: (2, 2), 8: (14, 3)}), ("near", 2, 8), separation_weight=0.0) == pytest.approx(0, abs=1e-6)
    far = _loss(_blob({2: (0, 0), 8: (15, 15)}), ("next_to", 2, 8), separation_weight=0.0)
    assert far == pytest.approx(15 / 16 - 150 / 512, abs=1e-5)


def test_separation_averaging_and_gradient() -> None:
    overlapping = torch.zeros(16, 16, 10)
    overlapping[8, 2, [2, 8]] = 1.0
    overlapping[8, 14, 8] = 1.0
    assert _loss(overlapping, ("left", 2, 8)) > _loss(overlapping, ("left", 2, 8), separation_weight=0.0)
    maps = _blob({2: (8, 14), 4: (2, 8), 6: (14, 8), 8: (8, 2)})
    left, top = ("left", 2, 8), ("top", 4, 6)
    assert _loss(maps, left, top) == pytest.approx((_loss(maps, left) + _loss(maps, top)) / 2)
    attention = torch.rand(16, 16, 10, requires_grad=True)
    compute_spatial_relation_loss(attention, [RelationConstraint("right", 2, 8)]).backward()
    assert torch.isfinite(attention.grad).all() and attention.grad.abs().sum() > 0
    empty = compute_spatial_relation_loss(torch.rand(16, 16, 10, requires_grad=True), [])
    assert float(empty.detach()) == 0.0 and empty.requires_grad
    with pytest.raises(IndexError):
        compute_spatial_relation_loss(torch.rand(16, 16, 5), [RelationConstraint("left", 2, 9)])
    with pytest.raises(ValueError, match="must be finite"):
        compute_spatial_relation_loss(torch.rand(16, 16, 10), [], margin=math.nan)
