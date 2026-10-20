"""The frozen TAB PP bundle reproduces the recorded SD1.4 jobs and Sana plans of the four TAB PP cells."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from conftest import import_closure
from ccg_diffusion.methods import PAPER_TAB_CELLS, TabAblation
from ccg_diffusion.pp_attachment import jobs
from ccg_diffusion.provenance import canonical_json, repository_root, sha256_bytes

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_POSITIONS = (0, 5, 210, 215, 370, 375)


@lru_cache(maxsize=1)
def _bundle() -> jobs.FrozenBundle:
    return jobs.load_frozen_bundle(verify=True)


@lru_cache(maxsize=1)
def _clip() -> object:
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    return load_sd14_tokenizer()


@lru_cache(maxsize=1)
def _gemma() -> object:
    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer

    return load_sana_tokenizer()


def _compile(cell: str) -> list[dict]:
    bundle = _bundle()
    return jobs.place_jobs(interpretations=bundle.interpretations, source_jobs=bundle.source_jobs,
                           ablation=TabAblation.from_cell_id(cell), run_id="20261006_test", tokenizer=_clip())


def test_bundle_files_match_the_manifest() -> None:
    bundle = _bundle()
    assert len(bundle.source_jobs) == jobs.POPULATION_JOBS and len(bundle.interpretations) == 148
    assert [(r["tab_id"], r["reading"]) for r in bundle.source_jobs[:10]] == [("tab-pp-000", "np_attach")] * 5 + [
        ("tab-pp-000", "vp_attach")] * 5
    assert all(r["tab_id"].startswith("tab-pp-") == (i < jobs.PP_JOBS) for i, r in enumerate(bundle.source_jobs))
    assert not set(jobs.IDENTITY) & {k for row in bundle.source_jobs for k in row}
    tiers = [bundle.interpretation_rows[i]["preposition"]["tier"] for i in range(0, 148, 2)]
    counts = {tier: tiers.count(tier) for tier in set(tiers)}
    assert counts == {"single_site": 69, "frame_repair": 2, "nominal_core_variant": 3}
    assert bundle.reference["schema"] == "ccg_diffusion.tab_pp_reference.v1"


@pytest.mark.requires_clip_tokenizer
def test_fixture_jobs_equal_the_compiled_paper_jobs() -> None:
    rows = _compile("c1r1p0m0")
    fixture = json.loads((FIXTURES / "tab_pp_frozen_jobs.json").read_text())
    for position, expected in zip(FIXTURE_POSITIONS, fixture, strict=True):
        drop = set(jobs.IDENTITY)
        assert {k: v for k, v in rows[position].items() if k not in drop} == {
            k: v for k, v in expected.items() if k not in drop}


@pytest.mark.requires_clip_tokenizer
@pytest.mark.parametrize("cell", PAPER_TAB_CELLS)
def test_compiled_cells_equal_the_evaluated_sd14_jobs(cell: str) -> None:
    rows = _compile(cell)
    reference = _bundle().reference["sd14"]["cells"][cell]
    assert jobs.compare_with_reference(rows, reference)["matched"]
    ablation = TabAblation.from_cell_id(cell)
    assert jobs.check_population(rows, ablation=ablation, interpretations=_bundle().interpretations,
                                 source_jobs=_bundle().source_jobs) == reference["census"]
    assert {row["arm"] for row in rows} == {cell} and rows[0]["condition"] == f"20261006_test:{cell}"


@pytest.mark.requires_clip_tokenizer
@pytest.mark.requires_gemma_tokenizer
@pytest.mark.parametrize("cell", PAPER_TAB_CELLS)
def test_compiled_sana_plans_equal_the_evaluated_plans(cell: str) -> None:
    from ccg_diffusion.evaluation.tab import load_pp_items
    from ccg_diffusion.pp_attachment import sana

    plans, sana_jobs = sana.guided_plans_and_jobs(
        specs=list(load_pp_items().values()), interpretations=_bundle().interpretations,
        ablation=TabAblation.from_cell_id(cell), clip_tokenizer=_clip(), gemma_tokenizer=_gemma())
    reference = _bundle().reference["sana"]["cells"][cell]
    assert sha256_bytes("".join(canonical_json(p) + "\n" for p in plans).encode()) == reference["plans_sha256"]
    assert sha256_bytes("".join(canonical_json(j) + "\n" for j in sana_jobs).encode()) == reference["jobs_sha256"]
    assert sana.census(plans) == reference["census"]


@pytest.mark.requires_clip_tokenizer
@pytest.mark.requires_gemma_tokenizer
def test_gemma_projection_of_a_region_interpretation() -> None:
    from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION
    from ccg_diffusion.pp_attachment.compile import compile_interpretation
    from ccg_diffusion.pp_attachment.sana import project_interpretation

    interpretations = _bundle().interpretations
    row = compile_interpretation(interpretations["tab-pp-037", "np_attach"],
                                 interpretations["tab-pp-037", "vp_attach"], _clip(), TabAblation())
    gemma = _gemma()
    plan = project_interpretation(row, _clip(), gemma, COMPLEX_HUMAN_INSTRUCTION)
    assert plan["phrase"] is None and plan["conditioning"] == row["prompt"]
    assert len(plan["region_groups"]) == 2 and plan["loss_config"] == row["loss_config"]
    assert plan["source_program"] == row["pd_indices"] and plan["transfer_trace"] == row["transfer_trace"]
    spans = {s["source_clip_anchor"]: s["target_span"] for s in plan["lexical_projection"]}
    assert all(plan["pd_indices"]["exist"][i] == spans[e][-1] for i, e in enumerate(row["pd_indices"]["exist"]))
    with pytest.raises(ValueError, match="token IDs changed"):
        project_interpretation({**row, "token_ids": row["token_ids"][:-1]}, _clip(), gemma, COMPLEX_HUMAN_INSTRUCTION)
    with pytest.raises(ValueError, match="unsupported TAB plan"):
        project_interpretation({**row, "phrase_conditioning": {"weight": .75}}, _clip(), gemma,
                               COMPLEX_HUMAN_INSTRUCTION)


def test_sealed_sources_are_the_import_closure_of_the_sd14_runner() -> None:
    assert jobs.SD14_SOURCE_FILES == tuple(sorted(jobs.SD14_SOURCE_FILES))
    closure = import_closure(repository_root() / "src/ccg_diffusion/pp_attachment/sd14.py")
    assert set(jobs.SD14_SOURCE_FILES) == closure | {"scripts/tab_pp/run_ccg_diffusion.py", "data/tab_pp/manifest.json"}
