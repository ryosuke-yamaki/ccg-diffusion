"""Arm identifiers of the SD1.4 and Sana T2I-CompBench++ baselines (KEEP-keys).

An arm ID names one generated image population in run locks, metadata rows,
input-file provenance and evaluation directories. The four arms of the SD1.4
batch driver (CCG Diffusion, Predicated Diffusion, Vanilla and VS-TIED +
Gold-visual setup) and their guidance-target providers are defined in
``ccg_diffusion.config``; this module adds the four external SD1.4 baselines
(run by ``baselines.sd14_generation``), the upstream source commits of the
compared methods and the paper-method -> arm map used by the CLIs.

The module imports only ``ccg_diffusion.config`` (it is shared with the Python
3.11 adapter environment of the baseline preprocessing).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ccg_diffusion.config import SD14_CCG_ARM, SD14_PREDICATED_DIFFUSION_ARM

# The order is part of the six-arm experiment identity (KEEP).
EXTERNAL_ARMS = ("sd14_composable", "sd14_structure", "sd14_attend_excite", "sd14_syngen")
# Identity of the six-arm SD1.4 T2I run (``sd14_pd_holccg`` is CCG Diffusion's
# earlier configuration there, not a paper row; KEEP).
SD14_SIX_ARM_METHODS = (*EXTERNAL_ARMS, SD14_PREDICATED_DIFFUSION_ARM, SD14_CCG_ARM)
# Recorded in every external input file and run lock (KEEP: exactly these six keys).
SOURCE_COMMIT_BY_ARM: Mapping[str, str] = MappingProxyType({
    # energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
    "sd14_composable": "77af82bbc73bbc987ca238bb9a3b0902ea0e12d0",
    # weixi-feng/Structured-Diffusion-Guidance
    "sd14_structure": "773b2ba068c6cc60443d5d747a725413c8c9d46f",
    # yuval-alaluf/Attend-and-Excite
    "sd14_attend_excite": "163efdfd341bf3590df3c0c2b582935fbc8e8343",
    # RoyiRa/Linguistic-Binding-in-Diffusion-Models
    "sd14_syngen": "50b8eab175dc9b31ce9c7c4b2eba28c817fc440e",
    # tksmatsubara/PredicatedDiffusion (both arms run its loss)
    SD14_PREDICATED_DIFFUSION_ARM: "af06f1be19dc6f0be468b271714c4904e623928e",
    SD14_CCG_ARM: "af06f1be19dc6f0be468b271714c4904e623928e",
})
# Paper method -> arm ID per backbone (Table 2 baseline rows).
T2I_BASELINE_ARMS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "sd14": MappingProxyType({
        "composable_diffusion": "sd14_composable",
        "structure_diffusion": "sd14_structure",
        "attend_and_excite": "sd14_attend_excite",
        "syngen": "sd14_syngen",
        "predicated_diffusion": SD14_PREDICATED_DIFFUSION_ARM,
    }),
    "sana": MappingProxyType({
        "composable_diffusion": "sana_composable",
        "structure_diffusion": "sana_structure",
        "attend_and_excite": "sana_attend_excite",
        "syngen": "sana_syngen",
        "predicated_diffusion": "sana_pd_spacy",
    }),
})
