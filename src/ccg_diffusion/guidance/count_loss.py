"""Count Loss (paper Eq. 8): count constraints from the derivation and the density loss.

Extraction (``extract_count_constraints``): an explicit cardinal 1-10 that
takes an atomic noun ``N`` as its slot-1 argument in a complete parse gives one
``CountConstraint`` (count, noun, the noun's CLIP lexical pieces). Every
ambiguous case (vague quantifiers, coordinated or modified cardinals, repeated
noun referents, incomplete parses) abstains, and the image keeps plain
guidance.

Loss (``CountState.augment``): at the first loss evaluation within outer steps
0-9 at which every counted noun has a numerically usable map, one target per
noun is frozen: ``count`` peaks of the smoothed attention map with salience at
least theta = 0.3, an equal-mass Gaussian mixture restricted to their Voronoi
cells. Afterwards the loss adds
``0.1 * mean_nouns(mass_factor * symmetric_KL(target, observed))`` to the
image's base loss (``mass_factor = min(1, 77 * mean(smoothed map))``, detached).

``numeric_definition()`` is the frozen record of this contract; its sha256 is
recorded in every count guidance record. Its ``"status"`` text is a historical
label kept byte-identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence

import torch
import torch.nn.functional as F

from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency, parse_ccg_category
from ccg_diffusion.provenance import canonical_json, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices

VERSION = "holsd.sd14_count_gaussian_mixture.proposed_v3"
RESOLUTION = 16
EPSILON = 1e-6
TAU = .3
COUNT_WEIGHT = 0.1
MIN_MEAN_ATTENTION = 100 * EPSILON
MIN_SPATIAL_RANGE = 100 * EPSILON
TOKEN_AXIS_SIZE = 77
ACTIVE_OUTER_STEPS = tuple(range(10))
MAX_NOUNS = 37  # At most floor(75 lexical CLIP positions / (one numeral + one noun)).
MAX_COUNT = 10
MAX_RADIUS = 2.0
CARDINALS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand".split(),
    (*range(21), 30, 40, 50, 60, 70, 80, 90, 100, 1000),
))
VAGUE = frozenset(
    "both several many few multiple pair couple dozen half halves quarter quarters no none neither every each all some "
    "any more most fewer less least enough double twice once".split()
)


# --- count constraints ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CountConstraint:
    """One numeral/noun binding on the CLIP token axis."""

    max_tokens: ClassVar[int] = 77
    trailing_special_tokens: ClassVar[int] = 1

    count: int
    numeral_source: int
    noun_source: int
    noun_word: str
    noun_anchor: int
    lexical_piece_indices: tuple[int, ...]
    text_token_count: int

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or not 1 <= self.count <= MAX_COUNT:
            raise ValueError("supported explicit count is an integer from 1 through 10")
        if not isinstance(self.noun_word, str) or not self.noun_word:
            raise ValueError("noun word must be nonempty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.numeral_source, self.noun_source, self.noun_anchor)):
            raise ValueError("source and anchor positions must be positive integers")
        if (isinstance(self.text_token_count, bool) or not isinstance(self.text_token_count, int)
                or not 2 + self.trailing_special_tokens <= self.text_token_count <= self.max_tokens):
            raise ValueError("actual text token count must fit CLIP BOS/lexical/EOS")
        if (not self.lexical_piece_indices
                or tuple(sorted(set(self.lexical_piece_indices))) != self.lexical_piece_indices):
            raise ValueError("noun lexical span must be ordered, unique, and nonempty")
        if self.noun_anchor not in self.lexical_piece_indices:
            raise ValueError("noun anchor must be in its full lexical span")
        if any(isinstance(index, bool) or not isinstance(index, int)
               or not 1 <= index < self.text_token_count - self.trailing_special_tokens
               for index in self.lexical_piece_indices):
            raise ValueError("noun span may not include BOS/EOS/padding")


@dataclass(frozen=True)
class SanaCountConstraint(CountConstraint):
    """The same numeral/noun binding on Sana's selected Gemma token axis."""

    max_tokens: ClassVar[int] = 300
    trailing_special_tokens: ClassVar[int] = 0


@dataclass(frozen=True)
class CountDecision:
    """Count-only source eligibility with immutable noun bindings."""

    status: str
    reasons: tuple[str, ...]
    constraints: tuple[CountConstraint, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "abstain_B0", "no_explicit_cardinal"}:
            raise ValueError("unsupported COUNT source decision")
        if (not isinstance(self.reasons, tuple) or not self.reasons
                or any(not isinstance(reason, str) or not reason for reason in self.reasons)):
            raise ValueError("COUNT decision reasons must be immutable nonempty strings")
        if (not isinstance(self.constraints, tuple)
                or any(not isinstance(value, CountConstraint) for value in self.constraints)
                or bool(self.constraints) != (self.status == "ok")):
            raise ValueError("only an eligible COUNT decision may contain typed noun constraints")
        validate_constraint_population(self.constraints)

    def as_record(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons),
                "constraints": [{**asdict(constraint), "lexical_piece_indices": list(constraint.lexical_piece_indices)}
                                for constraint in self.constraints]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CountDecision:
        if not isinstance(record, Mapping) or set(record) != {"status", "reasons", "constraints"}:
            raise ValueError("COUNT source decision fields differ from the typed contract")
        if not isinstance(record["reasons"], list) or not isinstance(record["constraints"], list):
            raise ValueError("serialized COUNT decisions require ordered lists")
        constraints = []
        fields = {"count", "numeral_source", "noun_source", "noun_word", "noun_anchor", "lexical_piece_indices",
                  "text_token_count"}
        for row in record["constraints"]:
            if not isinstance(row, Mapping) or set(row) != fields or not isinstance(row["lexical_piece_indices"], list):
                raise ValueError("serialized COUNT constraint has unsupported fields")
            constraints.append(CountConstraint(**{**row, "lexical_piece_indices": tuple(row["lexical_piece_indices"])}))
        return cls(record["status"], tuple(record["reasons"]), tuple(constraints))


def validate_constraint_population(constraints: tuple[CountConstraint, ...]) -> None:
    """Reject shared/overlapping noun slots or inconsistent physical text axes."""

    if not isinstance(constraints, tuple) or len(constraints) > MAX_NOUNS:
        raise ValueError("COUNT constraints must be an immutable tuple within CLIP lexical capacity")
    if any(not isinstance(value, CountConstraint) for value in constraints):
        raise ValueError("COUNT constraints must retain their typed source bindings")
    for attribute in ("numeral_source", "noun_source", "noun_anchor"):
        values = [getattr(constraint, attribute) for constraint in constraints]
        if len(values) != len(set(values)):
            raise ValueError("COUNT noun constraints may not share numeral, noun or anchor positions")
    pieces = [index for constraint in constraints for index in constraint.lexical_piece_indices]
    if len(pieces) != len(set(pieces)) or len({value.text_token_count for value in constraints}) > 1:
        raise ValueError("COUNT noun spans must be disjoint and use one text token layout")


def numeric_definition_sha256() -> str:
    return sha256_text(canonical_json(numeric_definition()))


@dataclass(frozen=True)
class CountGuidance:
    """One image's count decision and whether the Count Loss uses it (``enabled``)."""

    enabled: bool
    decision: CountDecision
    source_prompt_sha256: str
    source_program_sha256: str
    source_alignment_sha256: str

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or not isinstance(self.decision, CountDecision):
            raise ValueError("COUNT guidance needs an explicit enabled flag and typed source decision")
        for value in (self.source_prompt_sha256, self.source_program_sha256, self.source_alignment_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("COUNT guidance source identities must be full lowercase SHA256 values")

    def as_record(self) -> dict[str, Any]:
        return {"schema": "holsd.sd14_count_job_guidance.v1", "enabled": self.enabled,
                "candidate": VERSION, "numeric_definition_sha256": numeric_definition_sha256(),
                "decision": self.decision.as_record(), "source_prompt_sha256": self.source_prompt_sha256,
                "source_program_sha256": self.source_program_sha256,
                "source_alignment_sha256": self.source_alignment_sha256}


def numeric_definition() -> dict[str, Any]:
    """Frozen record of the Count Loss contract (recorded by sha256; kept byte-identical)."""
    return {
        "schema": VERSION, "status": "proposed_not_frozen_for_generation",
        "attention_resolution": RESOLUTION, "token_axis_size": TOKEN_AXIS_SIZE,
        "count_range": [1, MAX_COUNT], "max_nouns": MAX_NOUNS,
        "epsilon": EPSILON, "weight": COUNT_WEIGHT, "threshold": TAU,
        "min_mean_attention": MIN_MEAN_ATTENTION, "min_spatial_range": MIN_SPATIAL_RANGE,
        "mass_factor": "stop_gradient(min(1,77*mean(gaussian_smooth(raw_whole_noun_map))))",
        "loss": "0.1*mean_nouns(mass_factor*0.5*(KL(target||observed)+KL(observed||target)))",
        "observation": "unit_sum(gaussian_smooth(mean_raw_lexical_BPE_map)+epsilon)",
        "target": "equal_mass_Voronoi_truncated_Gaussian_mixture_plus_epsilon_then_unit_sum",
        "smoothing": {"kernel_size": 3, "sigma": .5, "padding": "replicate"},
        "min_center_distance": "min(16/requested_count,2*maximum_radius=4)",
        "min_distance_rationale": (
            'tau-support disks only need two maximum radii of separation; capping avoids needless low-count '
            'edge completion'
        ),
        "selected_centers": "descending_minmax_salience>=0.3; min_distance; image_center_then_row_major_ties",
        "completion_centers": (
            'descending_salience_below_threshold; min_distance; '
            'least_extra_separation_then_image_center_then_row_major_ties'
        ),
        "radius": "min(2,nearest_center_distance/2); count1_radius=2",
        "sigma": "radius/sqrt(-2*log(0.3))",
        "active_outer_steps": list(ACTIVE_OUTER_STEPS),
        "initialization": "all_nouns_together_at_first_qualifying_loss_evaluation_within_steps_0_through_9",
        "freeze": "centers_voronoi_sigma_density_once_per_image_never_retarget",
        "weak_later": "base_loss_object_returned_unchanged_for_entire_image_evaluation; frozen_targets_retained",
        "refinement": (
            'include_original_fixed_refinement_evaluations_at_active_outer_steps; '
            'no_extra_forwards_or_updates'
        ),
        "schedule": "original_PD_step_scale_and_per_image_cap_unchanged",
        "cross_noun_repulsion": "none; noun targets may overlap; mean all nouns",
        "resume": "whole_image_only; reject_mid_image_restore; restart_unfinished_image_with_fresh_state",
        "weight_rationale": (
            'fixed heuristic 0.1 anchored to existing PD existence coefficient; not an empirical '
            'latent-gradient bound'
        ),
        "gate_rationale": (
            'epsilon pseudocount <=~1% of mean/range; absolute scale gates weak but high-relative-contrast '
            'maps'
        ),
        "count_source": (
            'pure_HolCCG_complete_parse; explicit_cardinal_1_to_10_slot1_N_or_NP_over_N_to_atomic_N; '
            'no_a_or_an'
        ),
        "noun_coreference_guard": (
            'any overlapping generic pinned noun lemma candidates across count nouns abstains whole prompt'
        ),
        "paper": "https://arxiv.org/html/2508.10710v1",
        "paper_deviations": [
            "SD1.4 existing 16x16 attention and PD update",
            "global equal-mass symmetric KL instead of per-cluster KL/sqrt(k)",
            "absolute numerical gate and detached mass factor",
            "explicit missing-center completion and min-distance capped at 2*maximum radius",
            "one target freeze per image",
        ],
    }


# --- extraction from the derivation --------------------------------------------------------------------


def _cardinal(word: str) -> int | None:
    lowered = word.lower()
    if lowered in CARDINALS:
        return CARDINALS[lowered]
    if lowered.isascii() and lowered.isdecimal():
        return int(lowered)
    return None


@dataclass(frozen=True)
class NounForms:
    """Pinned spaCy noun lemma tables; ``candidates`` gives the possible lemmas of a word."""

    nouns: frozenset[str]
    rules: tuple[tuple[str, str], ...]
    exceptions: Mapping[str, tuple[str, ...]]

    def candidates(self, word: str) -> frozenset[str]:
        lowered = word.lower()
        values = set(self.exceptions.get(lowered, ()))
        if lowered in self.nouns:
            values.add(lowered)
        for suffix, replacement in self.rules:
            if lowered.endswith(suffix) and len(lowered) > len(suffix):
                candidate = lowered[:-len(suffix)] + replacement
                if candidate in self.nouns:
                    values.add(candidate)
        return frozenset(values or (lowered,))


@lru_cache(maxsize=1)
def load_noun_forms() -> NounForms:
    """Read generic, already-pinned noun morphology tables; no model inference."""
    from spacy.lookups import Lookups

    from ccg_diffusion.targets.lexicon import lexical_resource_path
    tables = Lookups().from_disk(lexical_resource_path().parent)
    return NounForms(
        frozenset(tables.get_table("lemma_index")["noun"]),
        tuple(tuple(rule) for rule in tables.get_table("lemma_rules")["noun"]),
        MappingProxyType({word: tuple(lemmas) for word, lemmas in tables.get_table("lemma_exc")["noun"].items()}),
    )


def extract_count_constraints(
    sentence: AutoSentence, dependencies: Sequence[MarkedDependency], alignment: Sequence[Mapping[str, Any]],
    indices: PDIndices, *, parse_status: str, unresolved_modifiers: Sequence[Any], text_token_count: int,
    noun_forms: NounForms,
) -> CountDecision:
    """Read explicit numeral-slot-1/atomic-N bindings; all ambiguous cases abstain."""
    leaves = {leaf.source_index: leaf for leaf in sentence.leaves}
    aligned = {row["source_index"]: row for row in alignment}
    if len(aligned) != len(alignment) or set(aligned) != set(leaves):
        raise ValueError("source alignment integrity mismatch")
    candidates = {index: _cardinal(leaf.word) for index, leaf in leaves.items() if _cardinal(leaf.word) is not None}
    vague = [leaf.word for leaf in leaves.values() if leaf.word.lower() in VAGUE]
    digit_composites = [leaf.word for leaf in leaves.values()
                        if any(char.isdigit() for char in leaf.word) and _cardinal(leaf.word) is None]
    reasons = []
    if parse_status != "complete":
        reasons.append("incomplete_parse")
    if unresolved_modifiers:
        reasons.append("unresolved_modifier")
    if vague or digit_composites:
        reasons.append("unsupported_quantifier")
    if any(not 1 <= count <= MAX_COUNT for count in candidates.values()):
        reasons.append("unsupported_cardinality")
    if reasons:
        return CountDecision("abstain_B0", tuple(reasons), ())
    if not candidates:
        return CountDecision("no_explicit_cardinal", ("no_explicit_cardinal",), ())
    if any(dep.argument_index in candidates for dep in dependencies):
        return CountDecision("abstain_B0", ("modified_or_coordinated_cardinal_scope",), ())
    edges = []
    for dep in dependencies:
        if dep.head_index not in candidates:
            continue
        if dep.argument_index not in leaves or dep.head_index not in leaves:
            raise ValueError("dependency references absent source")
        if dep.head_word != leaves[dep.head_index].word or dep.argument_word != leaves[dep.argument_index].word:
            raise ValueError("dependency source token identity mismatch")
        category = parse_ccg_category(dep.marked_category)
        noun_category = parse_ccg_category(leaves[dep.argument_index].category)
        shape_ok = (
            category.slash == "/" and category.result is not None and category.argument is not None
            and category.result.is_atomic and category.result.base_atom in {"N", "NP"}
            and category.argument.is_atomic and category.argument.base_atom == "N"
        )
        if dep.slot == 1 and shape_ok and noun_category.is_atomic and noun_category.base_atom == "N":
            edges.append((dep.head_index, dep.argument_index))
    if len(edges) != len(set(edges)):
        return CountDecision("abstain_B0", ("duplicate_count_edge",), ())
    if any(sum(head == source for head, _noun in edges) != 1 for source in candidates):
        return CountDecision("abstain_B0", ("missing_or_distributed_cardinal_binding",), ())
    nouns = [noun for _head, noun in edges]
    if len(nouns) != len(set(nouns)):
        return CountDecision("abstain_B0", ("conflicting_counts_on_one_noun",), ())
    if len(nouns) > MAX_NOUNS:
        return CountDecision("abstain_B0", ("count_nouns_exceed_CLIP_lexical_capacity",), ())
    forms = [noun_forms.candidates(leaves[noun].word) for noun in nouns]
    if any(left.intersection(right) for index, left in enumerate(forms) for right in forms[index + 1:]):
        return CountDecision("abstain_B0", ("repeated_noun_referent_ambiguous",), ())
    constraints = []
    seen_pieces: set[int] = set()
    for numeral, noun in sorted(edges):
        row = aligned[noun]
        anchor = row["clip_index"]
        if row["text"] != leaves[noun].word or anchor not in indices.exist:
            raise ValueError("count noun differs from verified PD existence anchor")
        span = tuple(row["clip_lexical_piece_indices"])
        if seen_pieces.intersection(span):
            raise ValueError("different count nouns have overlapping lexical spans")
        seen_pieces.update(span)
        constraints.append(CountConstraint(
            candidates[numeral], numeral, noun, leaves[noun].word, anchor, span, text_token_count,
        ))
    return CountDecision("ok", ("explicit_cardinal_noun_bindings",), tuple(constraints))


# --- density loss ----------------------------------------------------------------------------------------


class CountNumericalError(ValueError):
    """Nonfinite COUNT computation must fail rather than become an inactive subset."""


def smooth_map(attention: torch.Tensor) -> torch.Tensor:
    """Differentiable, nonnegative 3x3 Gaussian smoothing, sigma .5, replicate padding."""
    if attention.shape[-2:] != (RESOLUTION, RESOLUTION):
        raise ValueError("count maps must be 16x16")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise CountNumericalError("count attention must be finite and nonnegative")
    offsets = torch.arange(-1, 2, device=attention.device, dtype=torch.float32)
    weights = torch.exp(-offsets.square() / (2 * .5 ** 2))
    weights = weights[:, None] * weights[None, :]
    weights = weights / weights.sum()
    result = F.conv2d(
        F.pad(attention.float().reshape(-1, 1, RESOLUTION, RESOLUTION), (1, 1, 1, 1), mode="replicate"),
        weights[None, None],
    ).reshape(attention.shape)
    if not torch.isfinite(result).all():
        raise CountNumericalError("smoothed COUNT attention is nonfinite")
    return result


@dataclass(frozen=True)
class MapEvidence:
    mean_attention: float
    spatial_range: float
    ready: bool
    reasons: tuple[str, ...]
    mass_factor: float


def map_evidence(raw_map: torch.Tensor, *, mass_scale: float = TOKEN_AXIS_SIZE) -> MapEvidence:
    """A numerical evidence gate, not an object detector or a relative-contrast gate."""
    if raw_map.shape != (RESOLUTION, RESOLUTION):
        raise ValueError("each noun map must be exactly 16x16")
    work = smooth_map(raw_map).detach()
    mean = float(work.mean())
    spatial_range = float(work.max() - work.min())
    reasons = []
    if mean < MIN_MEAN_ATTENTION:
        reasons.append("mean_below_100epsilon")
    if spatial_range < MIN_SPATIAL_RANGE:
        reasons.append("range_below_100epsilon")
    return MapEvidence(mean, spatial_range, not reasons, tuple(reasons), min(1., mass_scale * mean))


@dataclass(frozen=True)
class FrozenTarget:
    centers: tuple[tuple[int, int], ...]
    sigma: tuple[float, ...]
    completion_count: int
    density: torch.Tensor
    region_assignment: torch.Tensor
    sha256: str


def make_target(raw_map: torch.Tensor, count: int) -> FrozenTarget:
    """Choose centers once from detached current attention and freeze a unit-sum mixture."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
        raise ValueError("count must be an integer from 1 through 10")
    evidence = map_evidence(raw_map)
    if not evidence.ready:
        raise ValueError("count target needs finite numerical noun evidence")
    attention = smooth_map(raw_map).detach().cpu()
    low, high = float(attention.min()), float(attention.max())
    normalized = (attention - low) / max(high - low, EPSILON)
    center_distance = min(RESOLUTION / count, 2 * MAX_RADIUS)
    candidates = [(row, col) for row in range(RESOLUTION) for col in range(RESOLUTION)]
    # Ties favor the image center and then row-major order, avoiding an implicit top-left first center.

    def tie(point: tuple[int, int]) -> tuple[float, int, int]:
        return ((point[0] - 7.5) ** 2 + (point[1] - 7.5) ** 2, point[0], point[1])
    ordered = sorted(candidates, key=lambda point: (-float(normalized[point]), *tie(point)))
    centers: list[tuple[int, int]] = []

    def separation(point: tuple[int, int]) -> float:
        return min((math.dist(point, center) for center in centers), default=float("inf"))
    for point in ordered:
        if float(normalized[point]) >= TAU and separation(point) >= center_distance:
            centers.append(point)
            if len(centers) == count:
                break
    initial_count = len(centers)
    if not centers:
        centers.append(ordered[0])
    # Complete missing instances using the highest remaining salience that meets
    # the same minimum distance. Equal salience favors the least extra separation,
    # then the image center and row-major order. No farthest-point corner prior.
    while len(centers) < count:
        remaining = [point for point in candidates if separation(point) >= center_distance]
        if not remaining:
            raise RuntimeError("deterministic center completion failed; never silently disable count guidance")
        centers.append(min(remaining, key=lambda point: (
            -float(normalized[point]), separation(point), *tie(point),
        )))
    coords = torch.stack(torch.meshgrid(torch.arange(RESOLUTION), torch.arange(RESOLUTION), indexing="ij"), -1).float()
    center_tensor = torch.tensor(centers, dtype=torch.float32)
    squared_distance = (coords[None] - center_tensor[:, None, None]).square().sum(-1)
    assignment = squared_distance.argmin(0)  # Equal distances use center selection order.
    sigma = []
    components = []
    for index, center in enumerate(centers):
        nearest = min((math.dist(center, other) for j, other in enumerate(centers) if j != index), default=4.)
        radius = min(MAX_RADIUS, nearest / 2)
        width = radius / math.sqrt(-2 * math.log(TAU))
        sigma.append(width)
        gaussian = torch.exp(-squared_distance[index] / (2 * width ** 2)) * (assignment == index)
        components.append(gaussian / gaussian.sum())
    target = torch.stack(components).mean(0)
    target = (target + EPSILON) / (target.sum() + EPSILON * target.numel())
    digest = hashlib.sha256(target.numpy().tobytes()).hexdigest()
    return FrozenTarget(tuple(centers), tuple(sigma), count - initial_count, target, assignment, digest)


def distribution(raw_map: torch.Tensor) -> torch.Tensor:
    smoothed = smooth_map(raw_map)
    return (smoothed + EPSILON) / (smoothed.sum(dim=(-2, -1), keepdim=True) + EPSILON * RESOLUTION ** 2)


def count_divergence(raw_map: torch.Tensor, target: FrozenTarget) -> torch.Tensor:
    """Positive symmetric KL of normalized full-map densities; gradient descent follows -grad."""
    observed = distribution(raw_map)
    expected = target.density.to(device=observed.device, dtype=torch.float32)
    loss = (.5 * (expected - observed) * (expected.log() - observed.log())).sum().clamp_min(0.)
    if not torch.isfinite(loss):
        raise CountNumericalError("COUNT density loss is nonfinite")
    return loss


@dataclass
class CountState:
    """Fresh per physical image row; targets cannot be restored through construction."""

    identity: str
    constraints: tuple[CountConstraint, ...]
    enabled: bool = True
    token_axis_size: int = field(default=TOKEN_AXIS_SIZE, kw_only=True)
    mass_scale: float = field(default=TOKEN_AXIS_SIZE, kw_only=True)
    weight: float = field(default=COUNT_WEIGHT, kw_only=True)
    active_outer_steps: tuple[int, ...] = field(default=ACTIVE_OUTER_STEPS, kw_only=True)
    targets: tuple[FrozenTarget, ...] = field(default=(), init=False)
    first_active_step: int | None = field(default=None, init=False)
    first_active_evaluation: int | None = field(default=None, init=False)
    last_step: int = field(default=-1, init=False)
    calls: int = field(default=0, init=False)
    evaluations: int = field(default=0, init=False)
    active_evaluations: int = field(default=0, init=False)
    skipped_evaluations: int = field(default=0, init=False)
    nonfinite_failures: int = field(default=0, init=False)
    expired: bool = field(default=False, init=False)
    last_evidence: tuple[MapEvidence, ...] = field(default=(), init=False)
    loss_trajectory: list[dict[str, Any]] = field(default_factory=list, init=False)
    inactive_calls: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity or type(self.enabled) is not bool:
            raise ValueError("COUNT state needs an image identity and explicit enabled flag")
        validate_constraint_population(self.constraints)
        if type(self.token_axis_size) is not int or self.token_axis_size < 2:
            raise ValueError("COUNT requires a declared token axis")
        if any(c.text_token_count > self.token_axis_size for c in self.constraints):
            raise ValueError("COUNT spans exceed its declared token axis")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in (self.mass_scale, self.weight)):
            raise ValueError("COUNT weights must be finite and nonnegative")
        if (not isinstance(self.active_outer_steps, tuple)
                or any(type(v) is not int or v < 0 for v in self.active_outer_steps)
                or self.active_outer_steps != tuple(sorted(set(self.active_outer_steps)))):
            raise ValueError("COUNT active steps must be ordered unique nonnegative integers")

    def _inactive(self, reason: str, base_loss: torch.Tensor) -> torch.Tensor:
        self.inactive_calls[reason] = self.inactive_calls.get(reason, 0) + 1
        return base_loss

    def augment(self, base_loss: torch.Tensor, attention: torch.Tensor, *, outer_step_index: int) -> torch.Tensor:
        """Return the identical base object unless every counted noun qualifies."""

        if (type(outer_step_index) is not int or outer_step_index < 0 or outer_step_index < self.last_step):
            raise ValueError("outer steps must be nonnegative monotonic integers")
        self.last_step = outer_step_index
        self.calls += 1
        if not self.enabled:
            return self._inactive("disabled", base_loss)
        if not self.constraints:
            return self._inactive("source_ineligible", base_loss)
        if outer_step_index not in self.active_outer_steps:
            self.expired = True
            return self._inactive("outside_active_outer_steps", base_loss)
        if attention.shape != (RESOLUTION, RESOLUTION, self.token_axis_size):
            raise ValueError("count attention must use 16x16 and its declared token axis")
        if base_loss.numel() != 1 or base_loss.ndim != 0:
            raise ValueError("COUNT augments one scalar base loss for one physical image")
        self.evaluations += 1
        try:
            if not torch.isfinite(base_loss):
                raise CountNumericalError("COUNT received a nonfinite base loss")
            maps = tuple(attention[:, :, list(spec.lexical_piece_indices)].float().mean(-1)
                         for spec in self.constraints)
            evidence = tuple(map_evidence(values, mass_scale=self.mass_scale) for values in maps)
            self.last_evidence = evidence
            # Keep every noun in the gate. A weak/flat noun never selects an easier subset.
            if not all(item.ready for item in evidence):
                self.skipped_evaluations += 1
                self.loss_trajectory.append({
                    "outer_step_index": outer_step_index, "evaluation": self.evaluations,
                    "status": "weak_or_flat", "base_loss": float(base_loss.detach()),
                    "count_loss": 0.0, "combined_loss": float(base_loss.detach()),
                    "evidence": [asdict(item) for item in evidence],
                })
                return base_loss
            if not self.targets:
                self.targets = tuple(make_target(values, spec.count) for values, spec in zip(maps, self.constraints))
                self.first_active_step = outer_step_index
                self.first_active_evaluation = self.evaluations
            values = [item.mass_factor * count_divergence(values, target)
                      for values, target, item in zip(maps, self.targets, evidence)]
            # Detached float factors and arithmetic order exactly match reviewed v3.
            added = self.weight * torch.stack(values).mean()
            result = base_loss + added
            if not torch.isfinite(result):
                raise CountNumericalError("combined COUNT and base loss is nonfinite")
        except CountNumericalError:
            self.nonfinite_failures += 1
            raise
        self.active_evaluations += 1
        self.loss_trajectory.append({
            "outer_step_index": outer_step_index, "evaluation": self.evaluations, "status": "active",
            "base_loss": float(base_loss.detach()), "count_loss": float(added.detach()),
            "combined_loss": float(result.detach()), "evidence": [asdict(item) for item in evidence],
        })
        return result

    def audit_record(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "enabled": self.enabled,
            "constraints": [asdict(spec) for spec in self.constraints],
            "first_active_step": self.first_active_step, "first_active_evaluation": self.first_active_evaluation,
            "last_step": self.last_step, "calls": self.calls, "evaluations": self.evaluations,
            "active_evaluations": self.active_evaluations, "skipped_evaluations": self.skipped_evaluations,
            "nonfinite_failures": self.nonfinite_failures, "expired": self.expired,
            "inactive_calls": dict(self.inactive_calls), "evidence": [asdict(item) for item in self.last_evidence],
            "loss_trajectory": list(self.loss_trajectory),
            "targets": [{"centers": target.centers, "sigma": target.sigma,
                         "completion_count": target.completion_count, "sha256": target.sha256,
                         "density": target.density.tolist(), "region_assignment": target.region_assignment.tolist(),
                         "region_assignment_sha256": hashlib.sha256(
                             target.region_assignment.numpy().tobytes()).hexdigest()} for target in self.targets],
            "resume": "whole_image_only; unfinished images restart from initial seed with fresh state",
        }


def fresh_count_states(
    count_guidance: Sequence[CountGuidance], identities: Sequence[str],
) -> tuple[CountState, ...]:
    """Keep padding rows independent even when their source job/spec is repeated."""

    if len(count_guidance) != len(identities) or not identities:
        raise ValueError("COUNT needs one typed specification per physical image row")
    if any(not isinstance(spec, CountGuidance) for spec in count_guidance):
        raise ValueError("COUNT physical rows require typed guidance specifications")
    return tuple(CountState(identity, spec.decision.constraints, enabled=spec.enabled)
                 for identity, spec in zip(identities, count_guidance))


def augment_count_losses(
    base_losses: list[torch.Tensor], attention_maps: torch.Tensor, states: Sequence[CountState] | None, *,
    outer_step_index: int,
) -> list[torch.Tensor]:
    """An absent COUNT channel returns the original loss list without tensor work."""

    if states is None:
        return base_losses
    if len(states) != len(base_losses) or len(states) != len(attention_maps):
        raise ValueError("COUNT states, base losses, and attention rows must align")
    return [state.augment(base_loss, attention, outer_step_index=outer_step_index)
            for state, base_loss, attention in zip(states, base_losses, attention_maps)]
