"""TAB guidance strength and the combined TAB objective (paper Sec. IV-C, Table 8).

Every TAB interpretation is guided with the default strength (scale 10,
relative-L2 cap 0.02); an interpretation whose guidance targets receive the
Selective Guidance Boost uses scale 80 and cap 0.06. The SD1.4 kernel applies
a route's scale as ``scale_factor * scale_multiplier``: a physical batch that
holds a boosted row runs at kernel scale 20 with multipliers 0.5 (default) and
4.0 (boosted); a batch without one runs at kernel scale 10 without routes.

The objective of one image is ``L_PD + 6 * L_spatial + 0.25 * L_region``: the
Predicated Diffusion loss, the Spatial Relation Loss of the ``next_to``
relations and the Region Assignment Loss of the Holder-Object Region
Assignment, shared by SD1.4 and Sana. The weights are the constants written in
``tab_objective``; the per-row ``loss_config.relation_loss_weight`` and
``transfer_region.weight`` are records of the same values.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ccg_diffusion.guidance.positional_prefix import PrefixLayoutPlan
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans, compute_pd_loss
from ccg_diffusion.guidance.region_assignment_loss import region_assignment_loss
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceDecision, GuidanceRoute, ImageGuidance
from ccg_diffusion.guidance.spatial_relation_loss import compute_spatial_relation_loss
from ccg_diffusion.guidance.spatial_relations import RelationConstraint
from ccg_diffusion.provenance import digest, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices, program_sha256

# The loss configuration recorded in every TAB job (verbatim). Only scale_factor,
# max_update_ratio and relation_loss_weight vary per job; holder_aware_weight is
# an inert recorded key.
TAB_BASE_LOSS_CONFIG = MappingProxyType({
    "scale_factor": 20.0,
    "max_update_ratio": 0.02,
    "binding_loss": "pd",
    "leak_weight": 0.3,
    "holder_aware_weight": 0.0,
    "relation_loss_weight": 0.0,
    "relation_geometry": "metric_aligned_v1",
    "relation_margin": 0.25,
    "relation_proximity_max": 0.29296875,
    "relation_separation_weight": 1.0,
    "relation_off_axis_weight": 1.0,
})


@dataclass(frozen=True)
class TabGuidanceStrength(ImageGuidance):
    """The TAB strength on the kernel's route interface: kernel scale 20 x 0.5 = 10, boosted 20 x 4 = 80."""

    @property
    def scale_multiplier(self) -> float:
        return 4.0 if self.effective_profile == GuidanceRoute.BOOSTED else 0.5


def guidance_strength(row: Mapping[str, Any]) -> TabGuidanceStrength:
    """The route of one TAB job row (no whole-word pooling: policy ``routing_v1``)."""
    active = row["loss_config"]["scale_factor"] == 80.0
    if row["loss_config"]["scale_factor"] not in (10.0, 80.0):
        raise ValueError("Transfer only supports TAB 10 and selected P2 80")
    if row["loss_config"]["max_update_ratio"] != (0.06 if active else 0.02):
        raise ValueError("Transfer dose and cap differ")
    return TabGuidanceStrength(
        "routing_v1",
        GuidanceDecision(GuidanceRoute.BOOSTED if active else GuidanceRoute.DEFAULT, ("TAB_transfer_dose",), "{}"),
        sha256_text(row["prompt"]), digest(row["pd_indices"]),
        WholeWordSpans(len(row["token_ids"]), ()), (),
    )


def tab_objective(
    maps: torch.Tensor, indices: PDIndices, relations: tuple[RelationConstraint, ...],
    region: Any, **pd_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(total, L_spatial, L_region)`` of one image; only the token layout and capture are backbone-specific."""
    value = compute_pd_loss(maps, indices, **pd_kwargs)
    spatial = region_value = maps.new_zeros(())
    if relations:
        spatial = compute_spatial_relation_loss(
            maps, relations, attention_res=16, margin=.25, proximity_max=150/512,
            separation_weight=1., off_axis_weight=1.)
        value = value + 6.0 * spatial
    if region is not None:
        region_value = region_assignment_loss(maps, region)
        value = value + .25 * region_value
    return value, spatial, region_value


class TabObjective:
    """SD1.4 per-image loss override of a TAB job row with relations or a region layout."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.program_hash = digest(row["pd_indices"])
        self.relations = tuple(RelationConstraint(**r) for r in row.get("relations", ()))
        self.region = None
        if row.get("transfer_region"):
            self.region = PrefixLayoutPlan.from_record(row["transfer_region"]["plan"])
        self.evaluations = 0
        self.spatial_sum = self.region_sum = 0.0

    def __call__(self, maps: torch.Tensor, indices: PDIndices, **kwargs: Any) -> torch.Tensor:
        if program_sha256(indices) != self.program_hash:
            raise ValueError("Transfer loss program changed")
        value, spatial, region = tab_objective(maps, indices, self.relations, self.region, **kwargs)
        self.spatial_sum += float(spatial.detach())
        self.region_sum += float(region.detach())
        self.evaluations += 1
        return value

    def diagnostics(self) -> dict[str, Any]:
        return {"evaluations": self.evaluations, "mean_spatial_loss": self.spatial_sum / max(1, self.evaluations),
                "mean_region_loss": self.region_sum / max(1, self.evaluations)}
