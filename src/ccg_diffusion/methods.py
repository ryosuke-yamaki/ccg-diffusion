"""The four paper configurations of CCG Diffusion, the ablation switches and the display names.

``MethodConfig`` is a descriptive record of a paper configuration (Table 8):
one per (task, backbone). Kernels own their constants;
``tests/test_config_and_provenance.py`` asserts that every value here equals
the kernel constant. ``T2IAblation`` and ``TabAblation`` are the switches of
the ablation studies (Tables 3, 4 and 7 for T2I-CompBench++, Table 6 for
TAB PP). ``RecordKeys`` names the identifiers that evaluated runs already wrote
into their records (KEEP-keys).

The historical method profiles of the development repository are not live
configurations; their frozen records are in ``docs/paper/historical_profiles.json``
and their history is described in ``docs/provenance.md``.

Command line (no model is loaded)::

    python -m ccg_diffusion.methods {list,inspect ID,catalog,table8,ablations}

The module uses only the standard library and ``ccg_diffusion.config``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any, Iterable, Sequence

from ccg_diffusion import config

FAMILY_DISPLAY_NAME = "CCG Diffusion"
SCHEMA = "ccg_diffusion.method_config.v1"
CATALOG_SCHEMA = "ccg_diffusion.method_catalog.v1"

T2I_SD14 = "ccg_diffusion_t2i_sd14"
T2I_SANA = "ccg_diffusion_t2i_sana"
TAB_SD14 = "ccg_diffusion_tab_sd14"
TAB_SANA = "ccg_diffusion_tab_sana"
BACKBONES = ("sd14", "sana")


@dataclass(frozen=True)
class GuidanceStrength:
    """Guidance strength (s, delta): base scale and relative-L2 cap of one latent update."""

    scale: float
    relative_l2_cap: float


@dataclass(frozen=True)
class PredicatedDiffusionWeights:
    """Weights gamma of the Predicated Diffusion loss terms (Eq. 3)."""

    exist: float = 0.1
    corr: float = 1.0
    leak: float = 0.3
    possession: float = 1.0


@dataclass(frozen=True)
class SelectiveGuidanceBoost:
    """Selective Guidance Boost (Sec. IV-B1): the boosted strength and its eligibility rule."""

    boosted: GuidanceStrength = GuidanceStrength(80.0, 0.06)
    eligibility_rule: str = "attribute_v2"  # KEEP: recorded routing strategy
    word_pooling_scope: str = "mean_raw_lexical_attention_for_corr_leak_endpoints_on_P2_rows_only"  # KEEP


@dataclass(frozen=True)
class PositionalPrefix:
    """Positional Prefix (Sec. IV-B2, Eq. 4) for coordinated noun phrases."""

    realization: str = "coordinated_np_prefix_color_texture_shape_size"  # KEEP
    prune_phantom_anchors: bool = True
    recipe: tuple[str, str, str] = ("phrase", "prefix_layout", "source_prefix_regions")  # KEEP


@dataclass(frozen=True)
class RegionAssignmentLoss:
    """Region Assignment Loss (Eq. 5-6); TAB uses it as Holder-Object Region Assignment."""

    weight: float = 0.25
    boundary: float = 0.5  # x_b
    softness: float = 0.08  # kappa


@dataclass(frozen=True)
class SpatialRelationLoss:
    """Spatial Relation Loss (Eq. 7)."""

    weight: float = 6.0
    margin: float = 0.25
    proximity_max: float = 150.0 / 512.0
    separation_weight: float = 1.0
    off_axis_weight: float = 1.0
    geometry: str = "metric_aligned_v1"  # KEEP


@dataclass(frozen=True)
class CountLoss:
    """Count Loss (Eq. 8)."""

    weight: float = 0.1
    steps: tuple[int, ...] = tuple(range(10))
    peak_threshold: float = 0.3  # theta
    epsilon: float = 1e-6
    count_range: tuple[int, int] = (1, 10)
    smoothing_sigma: float = 0.5
    attention_resolution: int = 16
    mass_scale: float = 77.0  # g_n = min(1, (77 / I) sum_i A_n[i])


@dataclass(frozen=True)
class DisambiguatingParaphrase:
    """Disambiguating Paraphrase (Sec. IV-C2): rule and the two realization labels."""

    rule: str = config.TAB_PARAPHRASE_RULE  # KEEP
    np_attachment_form: str = "pp_object_np_cleft"  # KEEP
    vp_attachment_form: str = "pp_event_fronting"  # KEEP


@dataclass(frozen=True)
class RecordKeys:
    """Identifiers already written into evaluated records (KEEP-keys).

    ``recorded_profile_id`` and ``recorded_arm_id`` are the profile and arm
    labels of the evaluated SD1.4 runs; ``transfer_recipe`` is the value of
    ``transfer_trace.recipe`` in TAB jobs.
    """

    recorded_profile_id: str | None
    recorded_arm_id: str | None
    guidance_target_transform: str
    run_arm: str
    condition: str | None
    transfer_recipe: str | None


@dataclass(frozen=True)
class MethodConfig:
    """Descriptive record of a paper configuration (Table 8).

    Kernels own their constants; ``tests/test_config_and_provenance.py``
    asserts that every value here equals the kernel constant.
    """

    config_id: str
    task: str
    backbone: str
    display_name: str
    default_strength: GuidanceStrength
    selective_guidance_boost: SelectiveGuidanceBoost
    predicated_diffusion_weights: PredicatedDiffusionWeights
    spatial_relation_loss: SpatialRelationLoss
    region_assignment_loss: RegionAssignmentLoss
    positional_prefix: PositionalPrefix | None
    count_loss: CountLoss | None
    disambiguating_paraphrase: DisambiguatingParaphrase | None
    parser: config.HolCCGParseContract
    execution_shards: int | None
    record_keys: RecordKeys
    evaluation_scope: str

    @property
    def generation(self) -> config.SD14GenerationContract | config.SanaGenerationContract:
        return config.SD14_GENERATION if self.backbone == "sd14" else config.SANA_GENERATION

    @property
    def schedule(self) -> config.GuidanceSchedule:
        return config.SD14_GUIDANCE_SCHEDULE if self.backbone == "sd14" else config.SANA_GUIDANCE_SCHEDULE

    def as_record(self) -> dict[str, Any]:
        record = {
            "schema": SCHEMA,
            "family": FAMILY_DISPLAY_NAME,
            **asdict(self),
            "generation": self.generation.to_dict(),
            "schedule": asdict(self.schedule),
        }
        # JSON conversion makes nested tuples portable and returns an independent copy.
        return json.loads(json.dumps(record, allow_nan=False))

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.as_record(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


# --- ablation switches -------------------------------------------------------------------------------

T2I_REMOVABLE = (
    "selective_guidance_boost", "positional_prefix", "region_assignment_loss", "spatial_relation_loss", "count_loss",
)
TAB_REMOVABLE = ("disambiguating_paraphrase", "holder_object_region_assignment")
EVALUATED_T2I_CONDITIONS = (
    "gs1_p1_r1", "gs1_p1_r0", "gs1_p0_r1", "gs1_p0_r0", "gs0_p1_r1", "gs0_p1_r0", "gs0_p0_r1", "gs0_p0_r0",
    "no_2d", "no_count",
)
PAPER_TAB_CELLS = ("c1r1p0m0", "c1r0p0m0", "c0r1p0m0", "c0r0p0m0")
_COMPONENT_CONDITION = re.compile(r"gs([01])_p([01])_r([01])")
_TAB_CELL = re.compile(r"c([01])r([01])p([01])m([01])")
_ATTRIBUTE_BINDING_CATEGORIES = ("color", "shape", "texture")


def _require_bools(switches: object) -> None:
    if any(type(getattr(switches, field.name)) is not bool for field in fields(switches)):
        raise ValueError("ablation switches must be booleans")


def _removed(cls: type, names: Iterable[str], removable: Sequence[str]) -> Any:
    names = list(names)
    unknown = sorted(set(names) - set(removable))
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"--remove takes distinct names from {removable}, got {names}")
    return cls(**{name: False for name in names})


@dataclass(frozen=True)
class T2IAblation:
    """T2I-CompBench++ ablation (Tables 3, 4 and 7). Each switch says whether the component stays enabled.

    The evaluated combinations are any subset of the three components
    (Selective Guidance Boost, Positional Prefix, Region Assignment Loss) with
    both objectives on, or exactly one objective (Spatial Relation Loss or
    Count Loss) removed with all three components on.
    """

    selective_guidance_boost: bool = True
    positional_prefix: bool = True
    region_assignment_loss: bool = True
    spatial_relation_loss: bool = True
    count_loss: bool = True

    def __post_init__(self) -> None:
        _require_bools(self)
        components = self.selective_guidance_boost and self.positional_prefix and self.region_assignment_loss
        if not (self.spatial_relation_loss and self.count_loss) and not (
            components and (self.spatial_relation_loss or self.count_loss)
        ):
            raise ValueError(
                "an objective may be removed only alone and with all components enabled; "
                f"evaluated conditions: {EVALUATED_T2I_CONDITIONS}"
            )

    @property
    def condition_id(self) -> str:
        if not self.spatial_relation_loss:
            return "no_2d"
        if not self.count_loss:
            return "no_count"
        return (
            f"gs{self.selective_guidance_boost:d}_p{self.positional_prefix:d}"
            f"_r{self.region_assignment_loss:d}"
        )

    @property
    def is_full(self) -> bool:
        return all(getattr(self, name) for name in T2I_REMOVABLE)

    @property
    def scored_categories(self) -> tuple[str, ...]:
        if self.is_full:
            return config.T2I_COMPBENCH_CATEGORIES
        if not self.spatial_relation_loss:
            return ("spatial",)
        if not self.count_loss:
            return ("numeracy",)
        return _ATTRIBUTE_BINDING_CATEGORIES

    def components(self) -> dict[str, bool]:
        """The recorded switch dict (KEEP keys); a removed objective is added as ``False``."""

        record = {
            "guidance_strength": self.selective_guidance_boost,
            "prefix": self.positional_prefix,
            "region": self.region_assignment_loss,
        }
        if not self.spatial_relation_loss:
            record["spatial_loss"] = False
        if not self.count_loss:
            record["count_loss"] = False
        return record

    @classmethod
    def from_condition_id(cls, condition_id: str) -> T2IAblation:
        if condition_id == "no_2d":
            return cls(spatial_relation_loss=False)
        if condition_id == "no_count":
            return cls(count_loss=False)
        match = _COMPONENT_CONDITION.fullmatch(condition_id)
        if match is None:
            raise ValueError(f"unknown T2I condition {condition_id!r}; evaluated: {EVALUATED_T2I_CONDITIONS}")
        boost, prefix, region = (value == "1" for value in match.groups())
        return cls(selective_guidance_boost=boost, positional_prefix=prefix, region_assignment_loss=region)

    @classmethod
    def from_removed(cls, names: Iterable[str]) -> T2IAblation:
        """The ablation that removes ``names`` (``--remove`` on the T2I CLI)."""

        return _removed(cls, names, T2I_REMOVABLE)


@dataclass(frozen=True)
class TabAblation:
    """TAB PP ablation (Table 6): Disambiguating Paraphrase and Holder-Object Region Assignment.

    The cell ID and switch dict are the recorded ones; attachment possession
    (``p``) and phrase value mixing (``m``) are always off.
    """

    disambiguating_paraphrase: bool = True
    holder_object_region_assignment: bool = True

    def __post_init__(self) -> None:
        _require_bools(self)

    @property
    def cell_id(self) -> str:
        return f"c{self.disambiguating_paraphrase:d}r{self.holder_object_region_assignment:d}p0m0"

    @property
    def switches(self) -> dict[str, bool]:
        return {
            "cond": self.disambiguating_paraphrase,
            "reg": self.holder_object_region_assignment,
            "poss": False,
            "mix": False,
        }

    @property
    def recipe(self) -> str:
        return config.TAB_PP_REGION_RECIPE if self.holder_object_region_assignment else config.TAB_PP_NO_REGION_RECIPE

    @classmethod
    def from_cell_id(cls, cell: str) -> TabAblation:
        match = _TAB_CELL.fullmatch(cell)
        if match is None:
            raise ValueError(f"unknown TAB cell {cell!r}; paper cells: {PAPER_TAB_CELLS}")
        paraphrase, region, possession, mixing = (value == "1" for value in match.groups())
        if possession or mixing:
            raise ValueError(
                f"TAB cell {cell!r} uses attachment possession or phrase value mixing, which are not part "
                "of CCG Diffusion; see docs/provenance.md for the historical cells"
            )
        return cls(disambiguating_paraphrase=paraphrase, holder_object_region_assignment=region)

    @classmethod
    def from_removed(cls, names: Iterable[str]) -> TabAblation:
        """The ablation that removes ``names`` (``--remove`` on the TAB CLI)."""

        return _removed(cls, names, TAB_REMOVABLE)


FULL_T2I = T2IAblation()
T2I_TABLE3_ABLATIONS = tuple(T2IAblation.from_condition_id(c) for c in ("gs0_p1_r1", "gs1_p0_r1", "gs1_p1_r0"))
T2I_TABLE7_ABLATIONS = tuple(
    T2IAblation.from_condition_id(c) for c in ("gs0_p0_r0", "gs0_p0_r1", "gs0_p1_r0", "gs1_p0_r0")
)
T2I_TABLE4_ABLATIONS = tuple(T2IAblation.from_condition_id(c) for c in ("no_2d", "no_count"))
TAB_TABLE6_ABLATIONS = tuple(TabAblation.from_cell_id(c) for c in ("c0r1p0m0", "c1r0p0m0", "c0r0p0m0"))


# --- the four configurations --------------------------------------------------------------------------

_T2I_SCOPE = "T2I-CompBench++: 8 categories x 300 prompts x seeds 42-51"
_TAB_SCOPE = "TAB PP: 74 items x 2 interpretations x seeds 0-4"


def _config(config_id: str, task: str, backbone: str, **values: Any) -> MethodConfig:
    t2i = task == "t2i"
    return MethodConfig(
        config_id=config_id,
        task=task,
        backbone=backbone,
        display_name="CCG Diffusion (ours)",
        default_strength=GuidanceStrength(20.0, 0.02) if t2i else GuidanceStrength(10.0, 0.02),
        selective_guidance_boost=SelectiveGuidanceBoost(),
        predicated_diffusion_weights=PredicatedDiffusionWeights(),
        spatial_relation_loss=SpatialRelationLoss(),
        region_assignment_loss=RegionAssignmentLoss(),
        positional_prefix=PositionalPrefix() if t2i else None,
        count_loss=CountLoss() if t2i else None,
        disambiguating_paraphrase=None if t2i else DisambiguatingParaphrase(),
        parser=config.HOLCCG_PARSE if t2i else config.HOLCCG_SUPERTAG_CONSTRAINED_PARSE,
        **values,
    )


_CONFIGS = MappingProxyType({c.config_id: c for c in (
    _config(
        T2I_SD14, "t2i", "sd14", execution_shards=4, evaluation_scope=_T2I_SCOPE,
        record_keys=RecordKeys(
            recorded_profile_id="sd14_ccg_compositional_v2", recorded_arm_id=config.SD14_CCG_ARM,
            guidance_target_transform=config.SD14_GUIDANCE_TARGET_PROVIDER, run_arm=config.SD14_CCG_ARM,
            condition=None, transfer_recipe=None),
    ),
    _config(
        T2I_SANA, "t2i", "sana", execution_shards=None, evaluation_scope=_T2I_SCOPE,
        record_keys=RecordKeys(
            recorded_profile_id=None, recorded_arm_id=None, guidance_target_transform="deps2index_sana",
            run_arm="sana_ccg_compositional_shape_v1", condition="final_shape", transfer_recipe=None),
    ),
    _config(
        TAB_SD14, "tab", "sd14", execution_shards=1,
        evaluation_scope=_TAB_SCOPE + "; 3,150-job layout with physical batches of 4 retained",
        record_keys=RecordKeys(
            recorded_profile_id="sd14_ccg_pp_attachment_v3", recorded_arm_id=None,
            guidance_target_transform="deps2index_syntax_ambiguity", run_arm=config.TAB_PP_RECIPE,
            condition=config.TAB_PP_CELL, transfer_recipe=config.TAB_PP_REGION_RECIPE),
    ),
    _config(
        TAB_SANA, "tab", "sana", execution_shards=None, evaluation_scope=_TAB_SCOPE,
        record_keys=RecordKeys(
            recorded_profile_id=None, recorded_arm_id=None,
            guidance_target_transform="deps2index_syntax_ambiguity", run_arm="guided",
            condition=config.TAB_PP_CELL, transfer_recipe=config.TAB_PP_REGION_RECIPE),
    ),
)})


def get_config(config_id: str) -> MethodConfig:
    try:
        return _CONFIGS[config_id]
    except KeyError:
        raise KeyError(f"unknown configuration {config_id!r}; valid IDs: {', '.join(_CONFIGS)}") from None


def config_for(task: str, backbone: str) -> MethodConfig:
    for method_config in _CONFIGS.values():
        if (method_config.task, method_config.backbone) == (task, backbone):
            return method_config
    raise KeyError(f"no configuration for task {task!r} and backbone {backbone!r}")


def list_configs() -> tuple[MethodConfig, ...]:
    return tuple(_CONFIGS.values())


def catalog() -> dict[str, Any]:
    return {"schema": CATALOG_SCHEMA, "family": FAMILY_DISPLAY_NAME,
            "configs": [method_config.as_record() for method_config in list_configs()]}


def _attention_maps(backbone: str) -> str:
    if backbone == "sd14":
        generation = config.SD14_GENERATION
        layers = f"{generation.expected_attention_processor_count} layers"
    else:
        generation = config.SANA_GENERATION
        layers = "blocks [{}, {})".format(*generation.block_range)
    return f"{layers} ({generation.attention_res}x{generation.attention_res})"


def _table8_value(row: str, backbone: str) -> Any:
    t2i, tab = config_for("t2i", backbone), config_for("tab", backbone)
    generation, schedule = t2i.generation, t2i.schedule
    if row == "Resolution":
        return f"{generation.height}x{generation.width}"
    if row == "Denoising steps":
        if backbone == "sd14":
            return f"{generation.steps} ({generation.scheduler_class.removesuffix('Scheduler')})"
        return str(generation.steps)
    if row == "Classifier-free guidance":
        return generation.guidance_scale
    if row == "Guided steps":
        if schedule.guided_steps == generation.steps:
            return f"all {schedule.guided_steps}"
        return f"first {schedule.guided_steps}"
    if row == "Extra updates at step 0":
        return schedule.refinement_updates
    if row == "Attention maps":
        return _attention_maps(backbone)
    if row == "(s, delta), default, compositional":
        return (t2i.default_strength.scale, t2i.default_strength.relative_l2_cap)
    if row == "(s, delta), default, PP ambiguity":
        return (tab.default_strength.scale, tab.default_strength.relative_l2_cap)
    if row == "(s, delta), boosted, both tasks":
        boosted = t2i.selective_guidance_boost.boosted
        assert boosted == tab.selective_guidance_boost.boosted
        return (boosted.scale, boosted.relative_l2_cap)
    if row == "gamma_exist, gamma_corr, gamma_leak, gamma_poss":
        weights = t2i.predicated_diffusion_weights
        return (weights.exist, weights.corr, weights.leak, weights.possession)
    if row == "lambda_region, x_b, kappa":
        region = t2i.region_assignment_loss
        return (region.weight, region.boundary, region.softness)
    if row == "lambda_spatial, m, r":
        spatial = t2i.spatial_relation_loss
        return (spatial.weight, spatial.margin, spatial.proximity_max)
    if row == "lambda_count (steps), theta":
        count = t2i.count_loss
        return (count.weight, count.steps, count.peak_threshold)
    raise KeyError(row)


TABLE8_ROWS = (
    "Resolution",
    "Denoising steps",
    "Classifier-free guidance",
    "Guided steps",
    "Extra updates at step 0",
    "Attention maps",
    "(s, delta), default, compositional",
    "(s, delta), default, PP ambiguity",
    "(s, delta), boosted, both tasks",
    "gamma_exist, gamma_corr, gamma_leak, gamma_poss",
    "lambda_region, x_b, kappa",
    "lambda_spatial, m, r",
    "lambda_count (steps), theta",
)


def table8() -> list[dict[str, Any]]:
    """The rows of paper Table 8, derived from the configurations and contracts."""

    return [{"row": row, "sd14": _table8_value(row, "sd14"), "sana": _table8_value(row, "sana")}
            for row in TABLE8_ROWS]


# --- display names and recorded labels ---------------------------------------------------------------

METHOD_DISPLAY_NAMES = MappingProxyType({
    "vanilla": "Vanilla",
    "composable_diffusion": "Composable Diffusion",
    "structure_diffusion": "StructureDiffusion",
    "attend_and_excite": "Attend-and-Excite",
    "syngen": "SynGen",
    "predicated_diffusion": "Predicated Diffusion",
    "gold_visual_setup": "VS-TIED + Gold-visual setup",
    "gold_visual_setup_predicated_diffusion": "VS-TIED + Gold-visual setup + Predicated Diffusion",
    "ccg_diffusion": "CCG Diffusion (ours)",
})
COMPONENT_DISPLAY_NAMES = MappingProxyType({
    "predicated_diffusion_loss": "Predicated Diffusion loss",
    "guidance_targets": "Guidance Targets from CCG Derivation",
    "selective_guidance_boost": "Selective Guidance Boost",
    "positional_prefix": "Positional Prefix",
    "region_assignment_loss": "Region Assignment Loss",
    "spatial_relation_loss": "Spatial Relation Loss",
    "count_loss": "Count Loss",
    "supertag_constrained_parsing": "Supertag-Constrained Parsing",
    "disambiguating_paraphrase": "Disambiguating Paraphrase",
    "holder_object_region_assignment": "Holder-Object Region Assignment",
})


# --- command line --------------------------------------------------------------------------------------

def _paper_table(ablation: T2IAblation | TabAblation) -> str:
    if isinstance(ablation, TabAblation):
        return "Table 6" if ablation in TAB_TABLE6_ABLATIONS else "Table 5"
    for table, rows in (("Table 3", T2I_TABLE3_ABLATIONS), ("Table 4", T2I_TABLE4_ABLATIONS),
                        ("Table 7", T2I_TABLE7_ABLATIONS)):
        if ablation in rows:
            return table
    return "Table 2" if ablation.is_full else "-"


def ablation_rows() -> list[dict[str, Any]]:
    """One row per evaluated ablation condition and backbone (the full method included)."""

    rows = []
    for backbone in BACKBONES:
        for condition_id in EVALUATED_T2I_CONDITIONS:
            ablation = T2IAblation.from_condition_id(condition_id)
            rows.append({
                "task": "t2i", "backbone": backbone, "id": condition_id,
                "remove": [name for name in T2I_REMOVABLE if not getattr(ablation, name)],
                "scored_categories": list(ablation.scored_categories), "table": _paper_table(ablation),
            })
    for backbone in BACKBONES:
        for cell in PAPER_TAB_CELLS:
            ablation = TabAblation.from_cell_id(cell)
            rows.append({
                "task": "tab", "backbone": backbone, "id": cell,
                "remove": [name for name in TAB_REMOVABLE if not getattr(ablation, name)],
                "scored_categories": [], "table": _paper_table(ablation),
            })
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ccg_diffusion.methods",
        description="List or inspect the CCG Diffusion paper configurations without loading any model.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="one line per configuration: ID, task, backbone, display name")
    inspect = sub.add_parser("inspect", help="print one configuration record as JSON")
    inspect.add_argument("config_id", help=f"one of: {', '.join(_CONFIGS)}")
    sub.add_parser("catalog", help="print all configuration records as JSON")
    sub.add_parser("table8", help="print the rows of paper Table 8 as JSON")
    sub.add_parser("ablations", help="list the evaluated ablations and their --remove names")
    args = parser.parse_args(argv)
    if args.command == "list":
        for method_config in list_configs():
            print(f"{method_config.config_id}\t{method_config.task}\t{method_config.backbone}"
                  f"\t{method_config.display_name}")
    elif args.command == "inspect":
        try:
            record = get_config(args.config_id).as_record()
        except KeyError as exc:
            parser.error(exc.args[0])
        print(json.dumps(record, ensure_ascii=False, indent=2))
    elif args.command == "catalog":
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
    elif args.command == "table8":
        print(json.dumps(table8(), ensure_ascii=False, indent=2))
    else:
        print("task\tbackbone\tid\tremove\tscored_categories\ttable")
        for row in ablation_rows():
            print("\t".join((row["task"], row["backbone"], row["id"], " ".join(row["remove"]) or "-",
                             ",".join(row["scored_categories"]) or "-", row["table"])))


if __name__ == "__main__":
    main()
