"""Precision and determinism contract of SD1.4 guided generation.

``bf16_stable`` (Table 8) runs the U-Net, text encoder and VAE in BF16 while
the master latent, the scheduler, the attention-probability reduction, every
loss and the latent update stay in FP32. ``fp32_stable`` is the same contract
in FP32 and is used by CPU tests only.

``configure_determinism`` sets process-global CUDA and cuDNN state
(``CUBLAS_WORKSPACE_CONFIG``, deterministic algorithms, the global seed). It is
called by SD1.4 code only, once per process, before ``load_pipe`` and before
any CUDA matmul (including the Hol-CCG parse of a single prompt). Sana code
never calls it: the evaluated Sana runs set only the three TF32/benchmark
flags, and a process that generates with Sana must not have deterministic
algorithms enabled. One backbone runs per process.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

PrecisionProfileName = Literal["fp32_stable", "bf16_stable"]
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class PrecisionProfile:
    """Closed precision/numerics contract for one generation process."""

    name: PrecisionProfileName
    model_dtype: torch.dtype
    stable_numerics: bool
    allow_tf32: bool
    pair_boundary_margin: float = 1e-6
    max_update_ratio: float | None = None
    max_latent_abs: float | None = None
    allow_fp16_reduced_precision_reduction: bool = False
    allow_bf16_reduced_precision_reduction: bool = False

    @property
    def model_dtype_name(self) -> str:
        return str(self.model_dtype).removeprefix("torch.")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["model_dtype"] = self.model_dtype_name
        record["master_latent_dtype"] = "float32" if self.stable_numerics else self.model_dtype_name
        record["pd_loss_dtype"] = "float32" if self.stable_numerics else self.model_dtype_name
        record["scheduler_dtype"] = "float32" if self.stable_numerics else self.model_dtype_name
        return record


PROFILES: dict[str, PrecisionProfile] = {
    "fp32_stable": PrecisionProfile(
        name="fp32_stable",
        model_dtype=torch.float32,
        stable_numerics=True,
        allow_tf32=False,
        max_update_ratio=0.02,
        max_latent_abs=20.0,
    ),
    "bf16_stable": PrecisionProfile(
        name="bf16_stable",
        model_dtype=torch.bfloat16,
        stable_numerics=True,
        allow_tf32=False,
        max_update_ratio=0.02,
        max_latent_abs=20.0,
    ),
}


def resolve_precision_profile(name: str, *, device: torch.device) -> PrecisionProfile:
    """Return the named profile; BF16 on CUDA requires native BF16 support."""

    try:
        profile = PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported precision profile: {name}") from exc
    if profile.model_dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("bf16_stable requires native CUDA BF16 support")
    return profile


def configure_precision_backend(profile: PrecisionProfile) -> dict[str, Any]:
    """Apply and return the CUDA math settings fixed by ``profile``."""

    torch.backends.cuda.matmul.allow_tf32 = profile.allow_tf32
    torch.backends.cudnn.allow_tf32 = profile.allow_tf32
    torch.set_float32_matmul_precision("high" if profile.allow_tf32 else "highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        profile.allow_fp16_reduced_precision_reduction
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        profile.allow_bf16_reduced_precision_reduction
    )
    return {
        "cuda_matmul_allow_tf32": profile.allow_tf32,
        "cudnn_allow_tf32": profile.allow_tf32,
        "float32_matmul_precision": "high" if profile.allow_tf32 else "highest",
        "allow_fp16_reduced_precision_reduction": (
            profile.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": (
            profile.allow_bf16_reduced_precision_reduction
        ),
    }


def configure_determinism(
    precision_profile: PrecisionProfile | None = None,
) -> dict[str, Any]:
    """Lock CUDA math choices before the pipeline creates its first context.

    Process-global and SD1.4 only (see the module docstring). The returned
    record equals ``config.SD14_DETERMINISM`` for ``PROFILES["bf16_stable"]``.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    record = {
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "torch_deterministic_algorithms": True,
        "global_torch_seed": 0,
    }
    if precision_profile is not None:
        record.update(configure_precision_backend(precision_profile))
    return record


def freeze_pipeline_for_latent_grad(pipe: Any) -> None:
    """Freeze SD modules while preserving gradients with respect to inputs."""

    for component_name in ("unet", "text_encoder", "vae"):
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        component.eval()
        component.requires_grad_(False)
