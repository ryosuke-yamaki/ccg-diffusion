"""Pinned contracts, paper configurations and provenance helpers.

Every reference value here is either a paper value or read from a tracked record.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from dataclasses import asdict
from pathlib import Path

import pytest

import ccg_diffusion
from ccg_diffusion import config, methods, provenance
from ccg_diffusion.backbones.jobs import MAX_IMAGE_DIR_COMPONENT_LENGTH, Job, image_path, metadata_path
from ccg_diffusion.methods import T2IAblation, TabAblation
from ccg_diffusion.targets.pd_indices import PDIndices, pd_indices_payload

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# --- config -------------------------------------------------------------------------------------------


def test_sana_generation_contract_is_its_fields() -> None:
    assert config.SANA_GENERATION.to_dict() == asdict(config.SANA_GENERATION)


def test_parse_contracts() -> None:
    assert config.HOLCCG_PARSE.parsing_kwargs() == {
        "stag_threshold": 0.01, "phrase_threshold": 0.001, "span_threshold": 0.001, "apply_skimmer": True}
    constrained = config.HOLCCG_SUPERTAG_CONSTRAINED_PARSE
    assert (constrained.stag_threshold, constrained.phrase_threshold, constrained.span_threshold) == (0.01, 1e-6, 1e-6)
    assert constrained.apply_skimmer is False and constrained.min_rule_freq is None
    site = config.HOLCCG_TAB_SITE_PARSE
    assert site.inference_settings() == {**config.HOLCCG_PARSE.inference_settings(),
                                         "apply_skimmer": False, "min_rule_freq": None}
    for contract in (config.HOLCCG_PARSE, site, constrained):
        assert (contract.checkpoint, contract.revision) == (config.HOLCCG_CHECKPOINT, config.HOLCCG_REVISION)
        assert contract.to_dict() == {"checkpoint": contract.checkpoint, "revision": contract.revision,
                                      **contract.inference_settings()}


@pytest.mark.parametrize("change", [
    {"stag_threshold": 0.0}, {"phrase_threshold": 1.5}, {"span_threshold": float("nan")}, {"stag_threshold": True},
    {"grammar": "ccgbank"}, {"chunk_size": 0}, {"chunk_size": True},
])
def test_parse_contract_validation(change: dict) -> None:
    values = {**asdict(config.HOLCCG_PARSE), **change}
    with pytest.raises(ValueError):
        config.HolCCGParseContract(**values)


def test_sd14_generation_contract_validation() -> None:
    values = asdict(config.SD14_GENERATION)
    assert config.SD14GenerationContract(**values) == config.SD14_GENERATION
    for change, message in (
        ({"precision_profile": "fp32_stable"}, "SD1.4 requires precision_profile='bf16_stable'"),
        ({"compile_mode": "max-autotune"}, "SD1.4 requires compile_mode='off'"),
        ({"scheduler_class": "DDIMScheduler"}, "SD1.4 requires scheduler_class='PNDMScheduler'"),
        ({"tokenizer_pad_token_id": 0}, "SD1.4 requires tokenizer_pad_token_id=49407"),
    ):
        with pytest.raises(ValueError, match=message):
            config.SD14GenerationContract(**{**values, **change})


def test_dataset_constants_equal_the_tracked_contracts() -> None:
    t2i = _json(ROOT / config.T2I_COMPBENCH_DATASET_DIR / "dataset_contract.json")
    assert (t2i["dataset_id"], t2i["version"], t2i["root"]) == (
        config.T2I_COMPBENCH_DATASET_ID, config.T2I_COMPBENCH_DATASET_VERSION, config.T2I_COMPBENCH_DATASET_DIR)
    assert tuple(t2i["population"]["categories"]) == config.T2I_COMPBENCH_CATEGORIES
    assert tuple(t2i["population"]["image_seeds"]) == config.T2I_COMPBENCH_SEEDS
    tab = _json(ROOT / config.TAB_DATASET_DIR / "dataset_contract.json")
    assert (tab["dataset_id"], tab["version"], tab["root"]) == (
        config.TAB_DATASET_ID, config.TAB_DATASET_VERSION, config.TAB_DATASET_DIR)
    assert config.TAB_PP_SEEDS == (0, 1, 2, 3, 4)


def test_record_identifiers() -> None:
    assert dict(config.SD14_ARM_PROVIDERS) == {
        "sd14_pd_holccg": "deps2index_sd14", "sd14_pd_spacy": "spacy_sd14",
        "sd14_vanilla": None, "sd14_tied_oracle": None}
    assert config.TAB_PP_RECIPE == f"{config.TAB_PP_REGION_RECIPE}_{config.TAB_PP_CELL}"
    assert TabAblation().cell_id == config.TAB_PP_CELL and TabAblation().recipe == config.TAB_PP_REGION_RECIPE
    assert config.SD14_GUIDANCE_SCHEDULE == config.GuidanceSchedule(25, (0,), 5, (1.0, 0.5))
    assert config.SANA_GUIDANCE_SCHEDULE.guided_steps == config.SANA_GENERATION.guided_steps
    assert config.SANA_GUIDANCE_SCHEDULE.refinement_steps == config.SANA_GENERATION.refinement_steps


def test_package_version_equals_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == ccg_diffusion.__version__
    assert project["name"] == "ccg-diffusion"


# --- methods ------------------------------------------------------------------------------------------


PAPER_TABLE8 = [
    ("Resolution", "512x512", "512x512"),
    ("Denoising steps", "50 (PNDM)", "20"),
    ("Classifier-free guidance", 7.5, 4.5),
    ("Guided steps", "first 25", "all 20"),
    ("Extra updates at step 0", 5, 5),
    ("Attention maps", "5 layers (16x16)", "blocks [0, 8) (16x16)"),
    ("(s, delta), default, compositional", (20.0, 0.02), (20.0, 0.02)),
    ("(s, delta), default, PP ambiguity", (10.0, 0.02), (10.0, 0.02)),
    ("(s, delta), boosted, both tasks", (80.0, 0.06), (80.0, 0.06)),
    ("gamma_exist, gamma_corr, gamma_leak, gamma_poss", (0.1, 1.0, 0.3, 1.0), (0.1, 1.0, 0.3, 1.0)),
    ("lambda_region, x_b, kappa", (0.25, 0.5, 0.08), (0.25, 0.5, 0.08)),
    ("lambda_spatial, m, r", (6.0, 0.25, 150 / 512), (6.0, 0.25, 150 / 512)),
    ("lambda_count (steps), theta", (0.1, tuple(range(10)), 0.3), (0.1, tuple(range(10)), 0.3)),
]
# Pinned digests of the four configuration records; a changed configuration needs a new configuration ID.
CONFIG_SHA256 = {
    "ccg_diffusion_t2i_sd14": "51169c8b324fb56913e0b016875a634c64c17a8a0b3978e98b7ff81407e33103",
    "ccg_diffusion_t2i_sana": "0898f10109b78c1ea051b1dacb0657bea7fae63907b8deab1e79259e95881933",
    "ccg_diffusion_tab_sd14": "272a8b010658ff56267f967e35a783ab10a3a1affa80fee06775180c3094634d",
    "ccg_diffusion_tab_sana": "71c3eb4bb77d84144925d6ea271dd1d8918f83e256e0deef56ef16e0a7e5f615",
}


def test_table8_equals_the_paper() -> None:
    assert [(row["row"], row["sd14"], row["sana"]) for row in methods.table8()] == PAPER_TABLE8


def test_four_configurations() -> None:
    assert [c.config_id for c in methods.list_configs()] == list(CONFIG_SHA256)
    assert {c.config_id: c.sha256 for c in methods.list_configs()} == CONFIG_SHA256
    for method_config in methods.list_configs():
        record = method_config.as_record()
        assert record["schema"] == methods.SCHEMA and record["family"] == "CCG Diffusion"
        assert record == json.loads(json.dumps(record))
        assert methods.config_for(method_config.task, method_config.backbone) is method_config
        assert methods.get_config(method_config.config_id) is method_config
    t2i, tab = methods.get_config(methods.T2I_SD14), methods.get_config(methods.TAB_SD14)
    assert t2i.parser == config.HOLCCG_PARSE and tab.parser == config.HOLCCG_SUPERTAG_CONSTRAINED_PARSE
    assert t2i.generation is config.SD14_GENERATION
    assert methods.get_config(methods.T2I_SANA).generation is config.SANA_GENERATION
    assert tab.record_keys.recorded_profile_id == "sd14_ccg_pp_attachment_v3"
    assert tab.record_keys.recorded_arm_id is None
    assert tab.disambiguating_paraphrase.rule == config.TAB_PARAPHRASE_RULE and tab.positional_prefix is None
    assert methods.catalog()["configs"] == [c.as_record() for c in methods.list_configs()]


def test_historical_profile_ids_are_not_configurations(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(KeyError, match="ccg_diffusion_t2i_sd14"):
        methods.get_config("sd14_ccg_compositional_v2")
    with pytest.raises(SystemExit) as raised:
        methods.main(["inspect", "sd14_ccg_compositional_v2"])
    assert raised.value.code != 0
    error = capsys.readouterr().err
    assert all(config_id in error for config_id in CONFIG_SHA256)


def test_t2i_ablation_conditions() -> None:
    for condition_id in methods.EVALUATED_T2I_CONDITIONS:
        ablation = T2IAblation.from_condition_id(condition_id)
        assert ablation.condition_id == condition_id
        removed = [name for name in methods.T2I_REMOVABLE if not getattr(ablation, name)]
        assert T2IAblation.from_removed(removed) == ablation
    assert methods.FULL_T2I.is_full and methods.FULL_T2I.scored_categories == config.T2I_COMPBENCH_CATEGORIES
    assert T2IAblation.from_removed(["positional_prefix", "region_assignment_loss"]).condition_id == "gs1_p0_r0"
    no_2d = T2IAblation.from_condition_id("no_2d")
    assert no_2d.components() == {"guidance_strength": True, "prefix": True, "region": True, "spatial_loss": False}
    assert no_2d.scored_categories == ("spatial",)
    assert T2IAblation.from_condition_id("no_count").scored_categories == ("numeracy",)
    assert T2IAblation.from_condition_id("gs0_p1_r0").components() == {
        "guidance_strength": False, "prefix": True, "region": False}
    assert T2IAblation.from_condition_id("gs0_p1_r0").scored_categories == ("color", "shape", "texture")
    assert [a.condition_id for a in methods.T2I_TABLE3_ABLATIONS] == ["gs0_p1_r1", "gs1_p0_r1", "gs1_p1_r0"]
    assert [a.condition_id for a in methods.T2I_TABLE7_ABLATIONS] == [
        "gs0_p0_r0", "gs0_p0_r1", "gs0_p1_r0", "gs1_p0_r0"]
    assert [a.condition_id for a in methods.T2I_TABLE4_ABLATIONS] == ["no_2d", "no_count"]
    for invalid in (
        lambda: T2IAblation(spatial_relation_loss=False, count_loss=False),
        lambda: T2IAblation(positional_prefix=False, count_loss=False),
        lambda: T2IAblation(selective_guidance_boost=0),
        lambda: T2IAblation.from_condition_id("gs2_p1_r1"),
        lambda: T2IAblation.from_removed(["spatial_relation_loss", "count_loss"]),
        lambda: T2IAblation.from_removed(["prefix"]),
        lambda: T2IAblation.from_removed(["positional_prefix", "positional_prefix"]),
    ):
        with pytest.raises(ValueError):
            invalid()


def test_tab_ablation_cells() -> None:
    expected = {
        "c1r1p0m0": ({"cond": True, "reg": True, "poss": False, "mix": False}, "combined_region_only", []),
        "c1r0p0m0": ({"cond": True, "reg": False, "poss": False, "mix": False}, "combined_no_layout",
                     ["holder_object_region_assignment"]),
        "c0r1p0m0": ({"cond": False, "reg": True, "poss": False, "mix": False}, "combined_region_only",
                     ["disambiguating_paraphrase"]),
        "c0r0p0m0": ({"cond": False, "reg": False, "poss": False, "mix": False}, "combined_no_layout",
                     ["disambiguating_paraphrase", "holder_object_region_assignment"]),
    }
    assert tuple(expected) == methods.PAPER_TAB_CELLS
    for cell, (switches, recipe, removed) in expected.items():
        ablation = TabAblation.from_cell_id(cell)
        assert (ablation.cell_id, ablation.switches, ablation.recipe) == (cell, switches, recipe)
        assert list(ablation.switches) == ["cond", "reg", "poss", "mix"]
        assert TabAblation.from_removed(removed) == ablation
    assert [a.cell_id for a in methods.TAB_TABLE6_ABLATIONS] == ["c0r1p0m0", "c1r0p0m0", "c0r0p0m0"]
    with pytest.raises(ValueError, match="docs/provenance.md"):
        TabAblation.from_cell_id("c1r1p1m1")
    with pytest.raises(ValueError):
        TabAblation.from_cell_id("c1r1")
    with pytest.raises(ValueError):
        TabAblation.from_removed(["region_assignment"])


def test_ablation_listing_covers_the_evaluated_conditions() -> None:
    rows = methods.ablation_rows()
    for backbone in methods.BACKBONES:
        assert [r["id"] for r in rows if r["task"] == "t2i" and r["backbone"] == backbone] == list(
            methods.EVALUATED_T2I_CONDITIONS)
        assert [r["id"] for r in rows if r["task"] == "tab" and r["backbone"] == backbone] == list(
            methods.PAPER_TAB_CELLS)


# --- guidance targets and jobs ---------------------------------------------------------------------------


def test_pd_indices_payload() -> None:
    indices = PDIndices(exist=[1], corr=[[2, 1]], leak=[], possession=[])
    assert pd_indices_payload(indices) == {"exist": [1], "corr": [[2, 1]], "leak": [], "possession": []}
    assert not PDIndices().has_any() and indices.has_any()


def test_image_path_layout() -> None:
    job = Job("color", "t2icb_color_0001", "a green bench and a blue bowl", 42, [3], [[2, 3]], [], [], "test")
    assert image_path(Path("out"), "sd14_pd_holccg", job) == Path(
        "out/sd14_pd_holccg/color/t2icb_color_0001_a_green_bench_and_a_blue_bowl/seed_00042.png")
    assert metadata_path(Path("out"), "m", "color") == Path("out/m/color/metadata.jsonl")
    assert job.pd_indices() == PDIndices(exist=[3], corr=[[2, 3]], leak=[], possession=[])
    # A recorded long prompt: the directory component keeps 220 bytes with sha256(prompt)[:16].
    prompt = ("a vibrant plant with star-shaped orange flowers blooming amidst lush green leaves. The plant is "
              "potted in a terracotta container that sits on a wooden shelf. Around it, other houseplants "
              "contribute to a small indoor garden atmosphere.")
    long_job = Job("dpg_native_pilot", "dpg_partiprompts302", prompt, 42, [], [], [], [], "test")
    path = image_path(Path("out"), "m", long_job)
    assert path.parent.name == (
        "dpg_partiprompts302_a_vibrant_plant_with_star_shaped_orange_flowers_blooming_amidst_lush_green_leaves_"
        "the_plant_is_potted_in_a_terracotta_container_that_sits_on_a_wooden_shelf_around_it_other_houseplants_"
        "39f51de00ec1186c")
    assert len(path.parent.name.encode()) <= MAX_IMAGE_DIR_COMPONENT_LENGTH
    assert path.parent.name.endswith(hashlib.sha256(prompt.encode()).hexdigest()[:16])


# --- provenance ---------------------------------------------------------------------------------------


def test_canonical_json_and_atomic_writes(tmp_path: Path) -> None:
    value = {"b": [1, 2], "a": "é"}
    assert provenance.canonical_json(value) == '{"a":"é","b":[1,2]}'
    assert provenance.canonical_json(value, pretty=True).endswith("}\n")
    with pytest.raises(ValueError):
        provenance.canonical_json(float("nan"))
    record = provenance.atomic_write_jsonl(tmp_path / "rows.jsonl", [value, {"c": None}])
    assert (tmp_path / "rows.jsonl").read_bytes() == '{"a":"é","b":[1,2]}\n{"c":null}\n'.encode()
    assert record.sha256 == provenance.sha256_file(tmp_path / "rows.jsonl")
    assert provenance.read_jsonl(tmp_path / "rows.jsonl") == [value, {"c": None}]
    (tmp_path / "dup.jsonl").write_text('{"a": 1, "a": 2}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        provenance.read_jsonl(tmp_path / "dup.jsonl", reject_duplicate_keys=True)
    with pytest.raises(FileExistsError):
        provenance.atomic_write_json(tmp_path / "rows.jsonl", value, overwrite=False)
    assert provenance.digest(value) == provenance.sha256_text(provenance.canonical_json(value))


def test_immutable_artifacts(tmp_path: Path) -> None:
    provenance.immutable_json(tmp_path / "plan.json", {"a": 1})
    provenance.immutable_json(tmp_path / "plan.json", {"a": 1})
    with pytest.raises(ValueError, match="Immutable artifact differs"):
        provenance.immutable_json(tmp_path / "plan.json", {"a": 2})
    provenance.immutable_rows(tmp_path / "rows.jsonl", [{"a": 1}])
    with pytest.raises(ValueError, match="Immutable rows differ"):
        provenance.immutable_rows(tmp_path / "rows.jsonl", [{"a": 2}])


def test_repository_paths(tmp_path: Path) -> None:
    root = provenance.repository_root()
    assert root == ROOT.resolve()
    assert provenance.repo_relative(root / "src/ccg_diffusion/config.py") == "src/ccg_diffusion/config.py"
    assert provenance.repo_relative("configs/parsing") == "configs/parsing"
    historical = f"{provenance.HISTORICAL_REPOSITORY_ROOT}/pyproject.toml"
    assert provenance.resolve_recorded_path(historical, repository_root=root) in {
        root / "pyproject.toml", Path(historical)}
    artifact = tmp_path / "a.txt"
    artifact.write_text("x")
    record = provenance.artifact_record(artifact).to_dict()
    assert provenance.verify_artifact_record(record) == artifact.resolve()
    with pytest.raises(ValueError, match="artifact bytes do not match"):
        provenance.verify_artifact_record({**record, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact size differs"):
        provenance.verify_artifact_record({**record, "size_bytes": 2})


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def sealed_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/a.py").write_text("A = 1\n")
    (root / "pkg/b.py").write_text("B = 2\n")
    for args in (("init", "-q"), ("config", "user.email", "test@example.invalid"), ("config", "user.name", "Test"),
                 ("add", "."), ("commit", "-qm", "initial")):
        _git(root, *args)
    return root


def test_source_sealing(sealed_repo: Path) -> None:
    files = ("pkg/a.py", "pkg/b.py")
    commit, components = provenance.seal_components(files, repository_root=sealed_repo)
    assert commit == _git(sealed_repo, "rev-parse", "HEAD")
    assert components["pkg/a.py"] == {"sha256": hashlib.sha256(b"A = 1\n").hexdigest(), "size_bytes": 6}
    record = provenance.implementation_record(files, schema="test.v1", repository_root=sealed_repo)
    provenance.verify_implementation_record(record, files, schema="test.v1", at_head=True, repository_root=sealed_repo)
    with pytest.raises(provenance.SourceSealError, match="schema"):
        provenance.verify_implementation_record(record, files, schema="other.v1", repository_root=sealed_repo)
    head = provenance.git_head_components_record(files, repository_root=sealed_repo)
    assert head["schema"] == "ccg_diffusion.git_head_components.v1"
    assert head["component_sha256"] == {path: value["sha256"] for path, value in components.items()}
    snapshot = provenance.source_tree_snapshot(("pkg",), repository_root=sealed_repo)
    assert snapshot == {"commit": commit, "dirty": False,
                        "files": {path: value["sha256"] for path, value in components.items()}}

    (sealed_repo / "pkg/a.py").write_text("A = 3\n")
    with pytest.raises(provenance.SourceSealError, match="uncommitted changes"):
        provenance.require_committed(files, repository_root=sealed_repo)
    assert provenance.source_tree_snapshot(("pkg",), repository_root=sealed_repo)["dirty"] is True
    _git(sealed_repo, "commit", "-qam", "change")
    with pytest.raises(provenance.SourceSealError, match="differ from the implementation record"):
        provenance.verify_implementation_record(record, files, schema="test.v1", repository_root=sealed_repo)
    with pytest.raises(provenance.SourceSealError, match="paths are invalid"):
        provenance.seal_components(("../pkg/a.py",), repository_root=sealed_repo)


def test_verify_implementation_record_at_head(sealed_repo: Path) -> None:
    files = ("pkg/a.py",)
    record = provenance.implementation_record(files, schema="test.v1", repository_root=sealed_repo)
    (sealed_repo / "pkg/b.py").write_text("B = 5\n")
    _git(sealed_repo, "commit", "-qam", "unrelated change")
    provenance.verify_implementation_record(record, files, schema="test.v1", repository_root=sealed_repo)
    with pytest.raises(provenance.SourceSealError, match="differs from HEAD"):
        provenance.verify_implementation_record(record, files, schema="test.v1", at_head=True,
                                                repository_root=sealed_repo)


# --- pinned tokenizer and lexicon (optional resources) ---------------------------------------------------


@pytest.mark.requires_clip_tokenizer
def test_pinned_sd14_tokenizer() -> None:
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer, require_sd14_tokenizer

    expected = config.SD14_GENERATION.tokenizer_identity()
    assert require_sd14_tokenizer(load_sd14_tokenizer()) == expected
    assert require_sd14_tokenizer(load_sd14_tokenizer(fast=True)) == expected


@pytest.mark.requires_spacy_model
def test_pinned_lexicon() -> None:
    from ccg_diffusion.targets.lexicon import load_nominal_modifier_lexicon

    lexicon = load_nominal_modifier_lexicon()
    assert len(lexicon.nouns) == 57528 and len(lexicon.adjectives) == 20993


# --- guidance kernel constants and frozen policy records -------------------------------------------------


def test_configurations_equal_the_guidance_kernel_constants() -> None:
    from ccg_diffusion.guidance import count_loss, positional_prefix, predicated_diffusion_loss as pd_loss
    from ccg_diffusion.guidance import region_assignment_loss, selective_guidance_boost as boost
    from ccg_diffusion.guidance.spatial_relation_loss import SPATIAL_RELATION_LOSS_PARAMETERS

    for method_config in methods.list_configs():
        weights = method_config.predicated_diffusion_weights
        assert (weights.exist, weights.leak) == (pd_loss.EXISTENCE_WEIGHT, pd_loss.PD_LEAK_WEIGHT)
        sgb = method_config.selective_guidance_boost
        assert (sgb.boosted.scale, sgb.boosted.relative_l2_cap) == (
            boost.BASE_SCALE * boost.BOOST_SCALE_MULTIPLIER, boost.BOOSTED_UPDATE_CAP)
        assert sgb.eligibility_rule == boost.ROUTING_STRATEGY
        assert sgb.word_pooling_scope == boost.WHOLE_WORD_POOLING_SCOPE
        spatial = method_config.spatial_relation_loss
        assert {"weight": spatial.weight, "margin": spatial.margin, "proximity_max": spatial.proximity_max,
                "separation_weight": spatial.separation_weight, "geometry_variant": spatial.geometry,
                "off_axis_weight": spatial.off_axis_weight} == dict(SPATIAL_RELATION_LOSS_PARAMETERS)
        region = method_config.region_assignment_loss
        assert (region.weight, region.boundary, region.softness) == (
            region_assignment_loss.REGION_WEIGHT, region_assignment_loss.BOUNDARY,
            region_assignment_loss.BOUNDARY_WIDTH)
    for method_config in (methods.get_config(methods.T2I_SD14), methods.get_config(methods.T2I_SANA)):
        assert (method_config.default_strength.scale, method_config.default_strength.relative_l2_cap) == (
            boost.BASE_SCALE, boost.DEFAULT_UPDATE_CAP)
        assert method_config.positional_prefix.recipe == tuple(positional_prefix.APPLIED_SELECTION.values())
        count = method_config.count_loss
        definition = count_loss.numeric_definition()
        assert (count.weight, count.steps, count.peak_threshold, count.epsilon) == (
            count_loss.COUNT_WEIGHT, count_loss.ACTIVE_OUTER_STEPS, count_loss.TAU, count_loss.EPSILON)
        assert count.count_range == (1, count_loss.MAX_COUNT) and list(count.count_range) == definition["count_range"]
        assert (count.attention_resolution, count.mass_scale) == (count_loss.RESOLUTION, count_loss.TOKEN_AXIS_SIZE)
        assert count.smoothing_sigma == definition["smoothing"]["sigma"]


def test_frozen_guidance_definition_shas() -> None:
    from ccg_diffusion.guidance.count_loss import numeric_definition_sha256
    from ccg_diffusion.guidance.selective_guidance_boost import base_policy_sha256, boost_policy_sha256

    assert base_policy_sha256() == "b8413ccd1688550bf17ca5b3b716848d8ca8b7ce360603d95ada75cc56de14cd"
    assert boost_policy_sha256() == "7038c5ac8e27c25b992375794b510620988a917b561826531967844508cdf0ca"
    assert numeric_definition_sha256() == "2088ca14ac30c9f44d1d065f7ea84d8aec46a0011f34bbaaf9094552faddf46f"


def test_t2i_ablation_transforms_of_the_executed_guidance() -> None:
    from ccg_diffusion.compositional import ablation
    from ccg_diffusion.guidance.count_loss import CountConstraint, CountDecision, CountGuidance
    from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans
    from ccg_diffusion.guidance.selective_guidance_boost import GuidanceDecision, SelectiveBoostGuidance
    from ccg_diffusion.guidance.spatial_relations import RelationConstraint

    def route(profile: str, *, whole_word: bool = True) -> SelectiveBoostGuidance:
        return SelectiveBoostGuidance(
            policy="routing_whole_word_v1" if whole_word else "routing_v1",
            decision=GuidanceDecision(profile, ("synthetic",), "{}"), source_prompt_sha256="a" * 64,
            source_program_sha256="b" * 64,
            pair_spans=WholeWordSpans.from_mapping({2: (1, 2), 3: (3,)}, text_token_count=5),
            derived_relations=(RelationConstraint("left", 2, 3),), whole_word=whole_word)

    boosted, spatial, default = route("P2"), route("P1"), route("B0")
    assert boosted.active_pair_spans is not None and boosted.scale_multiplier == 4
    low = ablation.without_selective_guidance_boost(boosted)
    assert (low.effective_profile, low.decision.reasons, low.whole_word, low.active_pair_spans) == (
        "B0", (ablation.BOOST_OFF_REASON,), False, None)
    assert (low.scale_multiplier, low.max_update_ratio) == (1.0, 0.02)
    assert json.loads(low.decision.evidence_json) == {"guidance_strength_ablation": boosted.decision.as_record()}
    for kept in (spatial, default):
        assert ablation.without_selective_guidance_boost(kept).decision == kept.decision
        assert ablation.without_selective_guidance_boost(kept).whole_word is False
    removed = ablation.without_spatial_relation_loss(spatial)
    assert (removed.effective_profile, removed.relations, removed.decision.reasons) == (
        "B0", (), (ablation.SPATIAL_OFF_REASON,))
    assert (removed.scale_multiplier, removed.max_update_ratio, removed.whole_word) == (1.0, 0.02, True)
    assert ablation.without_spatial_relation_loss(boosted) is boosted
    count = CountGuidance(True, CountDecision("ok", ("two",), (CountConstraint(2, 1, 2, "cats", 2, (2,), 5),)),
                          "a" * 64, "b" * 64, "c" * 64)
    disabled = ablation.without_count_loss(count)
    assert disabled.decision == count.decision and not disabled.enabled
    routes = {"p2": boosted, "p1": spatial}
    for condition_id, expected in (("gs0_p1_r1", {"p2": "B0", "p1": "P1"}), ("no_2d", {"p2": "P2", "p1": "B0"}),
                                   ("no_count", {"p2": "P2", "p1": "P1"}), ("gs1_p0_r0", {"p2": "P2", "p1": "P1"})):
        executed, counted = ablation.apply_ablation(T2IAblation.from_condition_id(condition_id), routes,
                                                    {"p2": count, "p1": count})
        assert {key: spec.effective_profile for key, spec in executed.items()} == expected
        assert all(spec.enabled is (condition_id != "no_count") for spec in counted.values())
