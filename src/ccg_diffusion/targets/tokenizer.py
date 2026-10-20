"""Pinned SD1.4 CLIP tokenizer: identity checks and loader.

Guidance targets are CLIP token indices, so every producer and consumer of
those indices checks the tokenizer files and its BPE behaviour against the
pinned SD1.4 identity. ``transformers`` is imported only by the loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ccg_diffusion.config import SD14_GENERATION
from ccg_diffusion.provenance import sha256_file


def _tokenizer_file(tokenizer: Any, name: str) -> Path:
    value = getattr(tokenizer, name, None)
    if value is None:
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        if isinstance(init_kwargs, Mapping):
            value = init_kwargs.get(name)
    if not isinstance(value, (str, Path)) or not value:
        raise ValueError(f"CLIP tokenizer does not expose {name}")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"CLIP tokenizer {name} does not exist: {path}")
    return path


def clip_tokenizer_identity(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    subfolder: str,
    include_tokenizer_config: bool = False,
) -> dict[str, Any]:
    """Fingerprint tokenizer files and special-token semantics."""

    vocab_path = _tokenizer_file(tokenizer, "vocab_file")
    merges_path = _tokenizer_file(tokenizer, "merges_file")
    identity = {
        "model_id": model_id,
        "revision": revision,
        "subfolder": subfolder,
        "vocab_sha256": sha256_file(vocab_path),
        "merges_sha256": sha256_file(merges_path),
        "bos_token": str(tokenizer.bos_token),
        "eos_token": str(tokenizer.eos_token),
        "pad_token": str(tokenizer.pad_token),
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "model_max_length": int(tokenizer.model_max_length),
    }
    if include_tokenizer_config:
        tokenizer_config_path = vocab_path.parent / "tokenizer_config.json"
        if not tokenizer_config_path.is_file():
            raise ValueError(
                "CLIP tokenizer does not expose tokenizer_config.json beside vocab.json"
            )
        identity["tokenizer_config_sha256"] = sha256_file(tokenizer_config_path)
    return identity


# File-hash identity cannot detect implementation drift: transformers'
# CLIPTokenizer silently falls back to a BasicTokenizer pre-split when ftfy is
# missing, which breaks contractions ("rubik's" -> "rubik ' s") relative to the
# original openai BPE that SD1.x CLIP checkpoints were trained with.  These
# probes pin the openai behavior so every producer/consumer of CLIP token
# indices fails closed under a drifted tokenizer implementation.
CLIP_BPE_BEHAVIOR_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "a cubic rubik's cube and a cylindrical pencil holder",
        (
            "a</w>",
            "cubic</w>",
            "rubi",
            "k</w>",
            "'s</w>",
            "cube</w>",
            "and</w>",
            "a</w>",
            "cy",
            "lind",
            "rical</w>",
            "pencil</w>",
            "holder</w>",
        ),
    ),
    (
        "The farmer's market isn't closed.",
        (
            "the</w>",
            "farmer</w>",
            "'s</w>",
            "market</w>",
            "isn</w>",
            "'t</w>",
            "closed</w>",
            ".</w>",
        ),
    ),
)


def require_clip_bpe_behavior(
    tokenizer: Any,
    *,
    probes: tuple[tuple[str, tuple[str, ...]], ...] = CLIP_BPE_BEHAVIOR_PROBES,
) -> None:
    """Fail closed unless ``tokenize`` reproduces the pinned openai CLIP BPE."""

    mismatches = []
    for text, expected_tokens in probes:
        actual = tuple(tokenizer.tokenize(text))
        if actual != expected_tokens:
            mismatches.append(
                {
                    "text": text,
                    "expected": list(expected_tokens),
                    "actual": list(actual),
                }
            )
    if mismatches:
        raise ValueError(
            "CLIP tokenizer behavior mismatch (BasicTokenizer fallback without "
            f"ftfy, or an incompatible transformers build): {mismatches}"
        )


def require_clip_tokenizer_identity(
    tokenizer: Any,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the runtime identity or fail closed on any pinned-field drift."""

    required = {"model_id", "revision", "subfolder"}
    missing = sorted(required - set(expected))
    if missing:
        raise ValueError(f"expected tokenizer identity is missing {missing}")
    actual = clip_tokenizer_identity(
        tokenizer,
        model_id=str(expected["model_id"]),
        revision=str(expected["revision"]),
        subfolder=str(expected["subfolder"]),
        include_tokenizer_config="tokenizer_config_sha256" in expected,
    )
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CLIP tokenizer identity mismatch: {mismatches}")
    return actual


def require_sd14_tokenizer(tokenizer: Any) -> dict[str, Any]:
    """Fingerprint and require the pinned SD1.4 tokenizer assets and behavior."""

    identity = require_clip_tokenizer_identity(
        tokenizer,
        expected=SD14_GENERATION.tokenizer_identity(),
    )
    require_clip_bpe_behavior(tokenizer)
    return identity


def load_sd14_tokenizer(*, fast: bool = False, local_files_only: bool = True) -> Any:
    """Load the pinned SD1.4 CLIP tokenizer (``CLIPTokenizerFast`` when ``fast``) without a diffusion model."""

    if fast:
        from transformers import CLIPTokenizerFast as tokenizer_class
    else:
        from transformers import CLIPTokenizer as tokenizer_class
    return tokenizer_class.from_pretrained(
        SD14_GENERATION.model_id,
        revision=SD14_GENERATION.model_revision,
        subfolder=SD14_GENERATION.tokenizer_subfolder,
        local_files_only=local_files_only,
    )
