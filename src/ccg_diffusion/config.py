"""Pinned contracts and record identifiers of CCG Diffusion.

This module holds the values that every part of the pipeline must agree on:

* the SD1.4 and Sana 0.6B generation contracts (Table 8, backbone rows);
* the Hol-CCG parse contracts (T2I parse, TAB preposition-site parse and
  Supertag-Constrained Parsing);
* the guidance schedules and the SD1.4 determinism record;
* the dataset identifiers of T2I-CompBench++ and TAB;
* the arm, provider, recipe, cell, paraphrase-rule and judge identifiers that
  evaluated runs wrote into their records.

The identifier values are KEEP-keys: they are spelled exactly as the evaluated
records spell them and must never change, even where the paper uses a
different name (the constant names use the paper terms). Method
hyperparameters (the Table 8 method rows) are described in
``ccg_diffusion.methods``.

The module uses only the standard library.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict


@dataclass(frozen=True)
class SanaGenerationContract:
    """Sana 0.6B generation settings (512px checkpoint, 20 steps, all steps guided)."""

    model_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    model_revision: str = "83d7a190bfd1fd070570a793d2dab5c7a3231b9d"
    model_size: str = "0.6b"
    checkpoint_resolution: int = 512
    height: int = 512
    width: int = 512
    steps: int = 20
    guided_steps: int = 20
    guidance_scale: float = 4.5
    scale_factor: float = 20.0
    model_dtype: str = "float16"
    master_dtype: str = "float32"
    batch_size: int = 4
    attention_res: int = 16
    block_range: tuple[int, int] = (0, 8)
    refinement_steps: tuple[int, ...] = (0,)
    execution_mode: str = "optimized"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SANA_GENERATION = SanaGenerationContract()


HOLCCG_CHECKPOINT = "ryosuke-yamaki/hol-ccg-roberta-large"
# The only snapshot of the checkpoint on both hosts of the paper runs; the T2I
# Stage 1 metadata never recorded it, so for T2I runs it is an inference.
HOLCCG_REVISION = "1888cd6343d8df52cf871e1a86924f500a6d7b6b"
HOLCCG_PACKAGE_VERSION = "1.0.0"
_GRAMMARS = ("treebank", "candc")


@dataclass(frozen=True)
class HolCCGParseContract:
    """Hol-CCG checkpoint and CKY settings for one parsing use."""

    checkpoint: str
    revision: str
    stag_threshold: float
    phrase_threshold: float
    span_threshold: float
    apply_skimmer: bool = True
    grammar: str = "treebank"
    chunk_size: int = 1
    min_rule_freq: int | None = 1  # None = checkpoint default (the TAB parses did not record it)

    def __post_init__(self) -> None:
        for name in ("stag_threshold", "phrase_threshold", "span_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.grammar not in _GRAMMARS:
            raise ValueError(f"grammar must be one of {_GRAMMARS}")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

    def parsing_kwargs(self) -> Dict[str, Any]:
        """Return a fresh mapping suitable for ``HolCCG.parse``."""

        return {
            "stag_threshold": self.stag_threshold,
            "phrase_threshold": self.phrase_threshold,
            "span_threshold": self.span_threshold,
            "apply_skimmer": self.apply_skimmer,
        }

    def inference_settings(self) -> Dict[str, Any]:
        """Return the seven keys of the ``parsing`` block of the Stage 1 metadata."""

        return {
            "stag_threshold": self.stag_threshold,
            "phrase_threshold": self.phrase_threshold,
            "span_threshold": self.span_threshold,
            "apply_skimmer": self.apply_skimmer,
            "min_rule_freq": self.min_rule_freq,
            "grammar": self.grammar,
            "chunk_size": self.chunk_size,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible provenance representation."""

        return {"checkpoint": self.checkpoint, "revision": self.revision, **self.inference_settings()}


# Stage 1 of the T2I pipeline. The skimmer is on: a sentence that fails to
# parse completely still yields dependencies from its PARTIAL fragments.
HOLCCG_PARSE = HolCCGParseContract(
    HOLCCG_CHECKPOINT, HOLCCG_REVISION, 0.01, 0.001, 0.001,
    apply_skimmer=True, grammar="treebank", chunk_size=1, min_rule_freq=1,
)
# Unconstrained TAB parse that locates the preposition before Supertag-Constrained Parsing.
HOLCCG_TAB_SITE_PARSE = HolCCGParseContract(
    HOLCCG_CHECKPOINT, HOLCCG_REVISION, 0.01, 0.001, 0.001,
    apply_skimmer=False, grammar="treebank", chunk_size=1, min_rule_freq=None,
)
# Supertag-Constrained Parsing of one TAB interpretation (paper Sec. IV-C1).
HOLCCG_SUPERTAG_CONSTRAINED_PARSE = HolCCGParseContract(
    HOLCCG_CHECKPOINT, HOLCCG_REVISION, 0.01, 1e-6, 1e-6,
    apply_skimmer=False, grammar="treebank", chunk_size=1, min_rule_freq=None,
)


@dataclass(frozen=True)
class SD14GenerationContract:
    """Pinned SD1.4 generation settings and CLIP tokenizer identity."""

    model_id: str
    model_revision: str
    tokenizer_subfolder: str
    tokenizer_vocab_sha256: str
    tokenizer_merges_sha256: str
    tokenizer_config_sha256: str
    tokenizer_bos_token: str
    tokenizer_eos_token: str
    tokenizer_pad_token: str
    tokenizer_bos_token_id: int
    tokenizer_eos_token_id: int
    tokenizer_pad_token_id: int
    tokenizer_model_max_length: int
    scheduler_class: str
    prediction_type: str
    precision_profile: str
    execution_mode: str
    batch_size: int
    compile_mode: str
    memory_format: str
    finite_guard: str
    update_clip_gate: str
    safety_checker_enabled: bool
    negative_prompt: str | None
    height: int
    width: int
    steps: int
    guidance_scale: float
    attention_res: int
    attention_block_scope: str
    expected_attention_processor_count: int

    def __post_init__(self) -> None:
        # The SD1.4 code implements only these settings (the driver resolves the
        # bf16_stable precision profile by name); the tokenizer files are checked
        # against tokenizer_identity() when they are loaded.
        for name, expected in (
            ("scheduler_class", "PNDMScheduler"),
            ("prediction_type", "epsilon"),
            ("precision_profile", "bf16_stable"),
            ("execution_mode", "throughput"),
            ("compile_mode", "off"),
            ("memory_format", "contiguous"),
            ("finite_guard", "strict"),
            ("update_clip_gate", "none"),
            ("attention_block_scope", "all_at_attention_res"),
            ("safety_checker_enabled", False),
            ("negative_prompt", None),
            ("tokenizer_bos_token", "<|startoftext|>"),
            ("tokenizer_eos_token", "<|endoftext|>"),
            ("tokenizer_pad_token", "<|endoftext|>"),
            ("tokenizer_bos_token_id", 49406),
            ("tokenizer_eos_token_id", 49407),
            ("tokenizer_pad_token_id", 49407),
            ("tokenizer_model_max_length", 77),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"SD1.4 requires {name}={expected!r}")

    def tokenizer_identity(self) -> Dict[str, Any]:
        """Return the exact SD1.4 tokenizer identity required by Stage 3."""

        return {
            "model_id": self.model_id,
            "revision": self.model_revision,
            "subfolder": self.tokenizer_subfolder,
            "vocab_sha256": self.tokenizer_vocab_sha256,
            "merges_sha256": self.tokenizer_merges_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "bos_token": self.tokenizer_bos_token,
            "eos_token": self.tokenizer_eos_token,
            "pad_token": self.tokenizer_pad_token,
            "bos_token_id": self.tokenizer_bos_token_id,
            "eos_token_id": self.tokenizer_eos_token_id,
            "pad_token_id": self.tokenizer_pad_token_id,
            "model_max_length": self.tokenizer_model_max_length,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible provenance representation."""

        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_subfolder": self.tokenizer_subfolder,
            "tokenizer": self.tokenizer_identity(),
            "scheduler_class": self.scheduler_class,
            "prediction_type": self.prediction_type,
            "precision_profile": self.precision_profile,
            "execution_mode": self.execution_mode,
            "batch_size": self.batch_size,
            "compile_mode": self.compile_mode,
            "memory_format": self.memory_format,
            "finite_guard": self.finite_guard,
            "update_clip_gate": self.update_clip_gate,
            "safety_checker_enabled": self.safety_checker_enabled,
            "negative_prompt": self.negative_prompt,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "attention_res": self.attention_res,
            "attention_block_scope": self.attention_block_scope,
            "expected_attention_processor_count": (
                self.expected_attention_processor_count
            ),
        }


# Model modules run in BF16 while latent, scheduler and guidance arithmetic
# stays FP32 (the ``bf16_stable`` precision profile). The three tokenizer
# artifacts are pinned by their sha256.
SD14_GENERATION = SD14GenerationContract(
    model_id="CompVis/stable-diffusion-v1-4",
    model_revision="133a221b8aa7292a167afc5127cb63fb5005638b",
    tokenizer_subfolder="tokenizer",
    tokenizer_vocab_sha256=(
        "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
    ),
    tokenizer_merges_sha256=(
        "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
    ),
    tokenizer_config_sha256=(
        "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882"
    ),
    tokenizer_bos_token="<|startoftext|>",
    tokenizer_eos_token="<|endoftext|>",
    tokenizer_pad_token="<|endoftext|>",
    tokenizer_bos_token_id=49406,
    tokenizer_eos_token_id=49407,
    tokenizer_pad_token_id=49407,
    tokenizer_model_max_length=77,
    scheduler_class="PNDMScheduler",
    prediction_type="epsilon",
    precision_profile="bf16_stable",
    execution_mode="throughput",
    batch_size=4,
    compile_mode="off",
    memory_format="contiguous",
    finite_guard="strict",
    update_clip_gate="none",
    safety_checker_enabled=False,
    negative_prompt=None,
    height=512,
    width=512,
    steps=50,
    guidance_scale=7.5,
    attention_res=16,
    attention_block_scope="all_at_attention_res",
    expected_attention_processor_count=5,
)


@dataclass(frozen=True)
class GuidanceSchedule:
    """When the latent update runs: Table 8 "Guided steps" and "Extra updates at step 0".

    ``scale_range`` gives the endpoints of the step-size decay
    eta_t = s * sqrt(1 - (1 - scale_range[1]) * j / (T - 1)).
    """

    guided_steps: int
    refinement_steps: tuple[int, ...]
    refinement_updates: int
    scale_range: tuple[float, float]


SD14_GUIDANCE_SCHEDULE = GuidanceSchedule(25, (0,), 5, (1.0, 0.5))
SANA_GUIDANCE_SCHEDULE = GuidanceSchedule(20, (0,), 5, (1.0, 0.5))
SD14_SCHEDULER_CONFIG_SHA256 = "c6589022d9d8432e019b2750354e061eec62aed48d1a7e82c036dc40f33a534f"
SD14_MAX_LATENT_ABS = 20.0
# Process-global determinism state of SD1.4 generation (``bf16_stable``).
SD14_DETERMINISM = MappingProxyType({
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
    "cublas_workspace_config": ":4096:8",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "float32_matmul_precision": "highest",
    "global_torch_seed": 0,
    "torch_deterministic_algorithms": True,
})

# Datasets. These constants are the only place that spells the dataset IDs,
# versions and directories (directories are repository-relative paths, not IDs).
T2I_COMPBENCH_DATASET_ID = "t2i-compbench-full-v1"
T2I_COMPBENCH_DATASET_VERSION = 1
T2I_COMPBENCH_DATASET_DIR = "dataset/derived/t2i-compbench/t2i-compbench-full-v1"
T2I_COMPBENCH_CATEGORIES = ("color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex")
T2I_COMPBENCH_SEEDS = tuple(range(42, 52))
TAB_DATASET_ID = "tab-ppvp-disambiguation"
TAB_DATASET_VERSION = 1
TAB_DATASET_DIR = "dataset/derived/tab/tab-ppvp-disambiguation-v1"
TAB_PP_SEEDS = tuple(range(5))

# Arms and guidance-target providers of the SD1.4 driver (KEEP-keys).
SD14_VANILLA_ARM = "sd14_vanilla"
SD14_GOLD_VISUAL_SETUP_ARM = "sd14_tied_oracle"  # VS-TIED + Gold-visual setup
SD14_PREDICATED_DIFFUSION_ARM = "sd14_pd_spacy"
SD14_PREDICATED_DIFFUSION_PROVIDER = "spacy_sd14"
SD14_CCG_ARM = "sd14_pd_holccg"
SD14_GUIDANCE_TARGET_PROVIDER = "deps2index_sd14"
SD14_ARM_PROVIDERS = MappingProxyType({
    SD14_CCG_ARM: SD14_GUIDANCE_TARGET_PROVIDER,
    SD14_PREDICATED_DIFFUSION_ARM: SD14_PREDICATED_DIFFUSION_PROVIDER,
    SD14_VANILLA_ARM: None,
    SD14_GOLD_VISUAL_SETUP_ARM: None,
})

# TAB PP identifiers (KEEP-keys). The recipe names the guidance written into
# ``transfer_trace.recipe``: with or without Holder-Object Region Assignment.
TAB_PP_REGION_RECIPE = "combined_region_only"
TAB_PP_NO_REGION_RECIPE = "combined_no_layout"
TAB_PP_CELL = "c1r1p0m0"
TAB_PP_RECIPE = "combined_region_only_c1r1p0m0"
TAB_PARAPHRASE_RULE = "tab_ppvp_holder_relative_pp_cleft_fronting_v1"  # Disambiguating Paraphrase
TAB_JUDGE = "gpt-6-astra-high"
