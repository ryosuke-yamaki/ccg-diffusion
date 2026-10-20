"""StructureDiffusion (Feng et al., 2023): noun-phrase conditioning and structured cross-attention.

The method-specific parts of the official implementation:

* prompt normalisation and Stanza 1.4.2 constituency noun-phrase extraction;
* CLIP-BPE span alignment and phrase-conditioned embedding replacement;
* the structured cross-attention average over phrase-specific keys and values
  (official ``multi_qkv``).

The official code assumes one prompt repeated ``n_samples`` times. Here the
same operations are kept per image and padded only at the batch boundary, so
the four images of a physical batch never share phrases or masks.

Reference source: ``weixi-feng/Structured-Diffusion-Guidance`` at commit
``773b2ba068c6cc60443d5d747a725413c8c9d46f``.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from nltk.tree import Tree
import torch

if TYPE_CHECKING:
    from diffusers.models.attention_processor import Attention


SOURCE_REPOSITORY = "weixi-feng/Structured-Diffusion-Guidance"
SOURCE_COMMIT = "773b2ba068c6cc60443d5d747a725413c8c9d46f"
STANZA_VERSION = "1.4.2"
STANZA_PROCESSORS = ("tokenize", "pos", "constituency")


class _TokenizerProtocol(Protocol):
    def tokenize(self, text: str) -> list[str]:
        """Return CLIP token strings without BOS/EOS."""


@dataclass(frozen=True)
class StructurePlan:
    """Constituency-derived phrase plan for one normalised prompt."""

    original_prompt: str
    method_prompt: str
    first_sentence: str
    conditioning_texts: tuple[str, ...]
    token_spans: tuple[tuple[int, int], ...]
    np_texts: tuple[str, ...]
    np_token_spans: tuple[tuple[int, int], ...]
    conjunction: bool
    stanza_version: str = STANZA_VERSION
    source_commit: str = SOURCE_COMMIT

    def __post_init__(self) -> None:
        if not self.original_prompt or not self.method_prompt:
            raise ValueError("original_prompt and method_prompt must be non-empty")
        if not self.first_sentence:
            raise ValueError("first_sentence must be non-empty")
        if not self.conditioning_texts or self.conditioning_texts[0] != self.method_prompt:
            raise ValueError("conditioning_texts must begin with method_prompt")
        if len(self.conditioning_texts) != len(self.token_spans):
            raise ValueError("conditioning texts and spans must have equal lengths")
        if len(self.np_texts) != len(self.np_token_spans):
            raise ValueError("NP texts and spans must have equal lengths")
        if any(not text for text in self.conditioning_texts):
            raise ValueError("conditioning texts must be non-empty")
        for start, end in self.token_spans:
            if start < 0 or end <= start:
                raise ValueError("token spans must be non-empty half-open intervals")
        if self.stanza_version != STANZA_VERSION:
            raise ValueError("Structure Diffusion requires Stanza 1.4.2")
        if self.source_commit != SOURCE_COMMIT:
            raise ValueError("unexpected Structure Diffusion source commit")

    @property
    def guidance_applied(self) -> bool:
        """Whether the parser supplied at least one multi-leaf NP."""

        return bool(self.np_texts)


@dataclass(frozen=True)
class StructureConditioning:
    """Unpadded structured K/V text contexts for one image sample."""

    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        _validate_context_tensor(self.keys, name="keys")
        _validate_context_tensor(self.values, name="values")
        if tuple(self.keys.shape[1:]) != tuple(self.values.shape[1:]):
            raise ValueError("key and value sequence/embedding dimensions must match")
        if self.keys.device != self.values.device or self.keys.dtype != self.values.dtype:
            raise ValueError("keys and values must share device and dtype")
        if self.keys.shape[0] != self.values.shape[0] and min(
            self.keys.shape[0], self.values.shape[0]
        ) != 1:
            raise ValueError("official structured attention requires K == V, K == 1, or V == 1")


@dataclass(frozen=True)
class StructureConditioningBatch:
    """Padded, sample-local structured K/V contexts for a physical batch."""

    keys: torch.Tensor
    values: torch.Tensor
    key_mask: torch.Tensor
    value_mask: torch.Tensor
    key_counts: tuple[int, ...] | None = None
    value_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise ValueError("batched keys and values must have shape [batch, variants, tokens, dim]")
        if self.keys.shape[0] != self.values.shape[0]:
            raise ValueError("key and value batch sizes differ")
        if tuple(self.keys.shape[2:]) != tuple(self.values.shape[2:]):
            raise ValueError("key and value sequence/embedding dimensions differ")
        if self.key_mask.shape != self.keys.shape[:2]:
            raise ValueError("key_mask must match [batch, key variants]")
        if self.value_mask.shape != self.values.shape[:2]:
            raise ValueError("value_mask must match [batch, value variants]")
        if self.key_mask.dtype != torch.bool or self.value_mask.dtype != torch.bool:
            raise TypeError("key_mask and value_mask must be boolean")
        tensors = (self.values, self.key_mask, self.value_mask)
        if any(tensor.device != self.keys.device for tensor in tensors):
            raise ValueError("all conditioning tensors must share one device")
        if self.values.dtype != self.keys.dtype:
            raise ValueError("batched keys and values must share dtype")
        if not bool(self.key_mask.any(dim=1).all().item()):
            raise ValueError("every sample needs at least one key variant")
        if not bool(self.value_mask.any(dim=1).all().item()):
            raise ValueError("every sample needs at least one value variant")
        if not bool(torch.isfinite(self.keys).all().item()):
            raise FloatingPointError("structured keys contain NaN or Inf")
        if not bool(torch.isfinite(self.values).all().item()):
            raise FloatingPointError("structured values contain NaN or Inf")
        mask_key_counts = tuple(int(value) for value in self.key_mask.sum(dim=1).cpu().tolist())
        mask_value_counts = tuple(int(value) for value in self.value_mask.sum(dim=1).cpu().tolist())
        if self.key_counts is None:
            object.__setattr__(self, "key_counts", mask_key_counts)
        elif self.key_counts != mask_key_counts:
            raise ValueError("key_counts do not match key_mask")
        if self.value_counts is None:
            object.__setattr__(self, "value_counts", mask_value_counts)
        elif self.value_counts != mask_value_counts:
            raise ValueError("value_counts do not match value_mask")
        for key_count, value_count in zip(mask_key_counts, mask_value_counts):
            if key_count != value_count and min(key_count, value_count) != 1:
                raise ValueError("each sample requires K == V, K == 1, or V == 1")

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])


def normalize_prompt(prompt: str) -> str:
    """Apply the official ``lower().strip().strip('.').strip()`` sequence."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = prompt.lower().strip().strip(".").strip()
    if not normalized:
        raise ValueError("normalisation produced an empty prompt")
    return normalized


def get_clip_token_alignment_map(
    tree: Tree,
    clip_tokens: Sequence[str] | None,
) -> dict[int, tuple[int, ...]]:
    """Port the official Stanza-leaf to CLIP-token alignment algorithm.

    Returned indices exclude BOS/EOS.  The extra entry at ``len(leaves)``
    contains the exclusive token boundary used by the official NP span code.
    """

    leaves = tree.leaves()
    if not leaves:
        raise ValueError("constituency tree has no leaves")
    if clip_tokens is None:
        return {index: (index,) for index in range(len(leaves) + 1)}
    if not clip_tokens:
        raise ValueError("clip_tokens must not be empty")
    if any(not isinstance(token, str) or not token for token in clip_tokens):
        raise ValueError("clip_tokens must be non-empty strings")

    def token_surface(token: str) -> str:
        return token[:-4] if token.endswith("</w>") else token

    index_map: dict[int, tuple[int, ...]] = {}
    token_index = 0
    previous_tree_leaf = ""
    try:
        for leaf_index, leaf_value in enumerate(leaves):
            leaf = str(leaf_value)
            token = token_surface(clip_tokens[token_index])
            aligned = [token_index]
            if token == previous_tree_leaf + leaf:
                previous_tree_leaf = ""
                token_index += 1
            elif len(token) < len(leaf):
                prefix = ""
                while prefix + token != leaf:
                    prefix += token
                    token_index += 1
                    token = token_surface(clip_tokens[token_index])
                    aligned.append(token_index)
                token_index += 1
            else:
                previous_tree_leaf += leaf
                token_index -= 1
                token_index += 1
            index_map[leaf_index] = tuple(aligned)
    except IndexError as exc:
        raise ValueError("CLIP tokens ended before Stanza leaves could be aligned") from exc

    if token_index < 0 or token_index > len(clip_tokens):
        raise ValueError("Stanza/CLIP alignment produced an invalid terminal boundary")
    index_map[len(leaves)] = (token_index,)
    return index_map


def extract_noun_phrases(
    tree: Tree,
    full_prompt: str,
    clip_tokens: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...], tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Extract all multi-leaf NP subtrees in the official traversal order.

    The first two outputs include the full prompt exactly as the official
    runner does.  The final two outputs retain only parser-produced NP rows so
    a caller can distinguish a real structured intervention from stock fallback.
    """

    if not isinstance(tree, Tree):
        raise TypeError("tree must be an nltk.tree.Tree")
    if not full_prompt:
        raise ValueError("full_prompt must be non-empty")
    index_map = get_clip_token_alignment_map(tree, clip_tokens)
    leaves = tree.leaves()
    extracted: list[tuple[str, tuple[int, int]]] = []

    def visit(subtree: Tree | str, left: int, right: int) -> None:
        if isinstance(subtree, str) or len(subtree.leaves()) == 1:
            return
        subtree_leaf_count = len(subtree.leaves())
        if right - left != subtree_leaf_count:
            raise ValueError("invalid constituency subtree offsets")
        child_leaf_counts = [len(child.leaves()) if isinstance(child, Tree) else 1 for child in subtree]
        offsets: list[int] = []
        running = 0
        for count in child_leaf_counts:
            offsets.append(running)
            running += count

        if subtree.label() == "NP" and subtree_leaf_count > 1:
            start = min(index_map[left])
            end = min(index_map[right])
            if end <= start:
                raise ValueError("NP alignment produced an empty token span")
            extracted.append((" ".join(str(leaf) for leaf in subtree.leaves()), (start, end)))
        for child_index, child in enumerate(subtree):
            child_left = left + offsets[child_index]
            child_right = child_left + child_leaf_counts[child_index]
            visit(child, child_left, child_right)

    visit(tree, 0, len(leaves))
    np_texts = tuple(text for text, _ in extracted)
    np_spans = tuple(span for _, span in extracted)

    conditioning = list(extracted)
    if full_prompt not in np_texts:
        full_span = (min(index_map[0]), min(index_map[len(leaves)]))
        if full_span[1] <= full_span[0]:
            raise ValueError("full prompt alignment produced an empty token span")
        conditioning.insert(0, (full_prompt, full_span))
    elif conditioning[0][0] != full_prompt:
        raise ValueError("official NP traversal did not place the full prompt first")

    return (
        tuple(text for text, _ in conditioning),
        tuple(span for _, span in conditioning),
        np_texts,
        np_spans,
    )


def has_full_np_coordination(tree: Tree) -> bool:
    """Detect the frozen automatic conjunction mode from the Stanza tree.

    A qualifying NP covers every leaf in the first sentence and directly
    contains at least two NP children plus a coordinating ``and`` child.
    """

    if not isinstance(tree, Tree):
        raise TypeError("tree must be an nltk.tree.Tree")
    full_leaves = tuple(str(leaf) for leaf in tree.leaves())
    for subtree in tree.subtrees(lambda item: item.label() == "NP"):
        if tuple(str(leaf) for leaf in subtree.leaves()) != full_leaves:
            continue
        np_children = sum(
            1 for child in subtree if isinstance(child, Tree) and child.label() == "NP"
        )
        has_and = any(
            isinstance(child, Tree)
            and child.label() in {"CC", "CONJP"}
            and any(str(leaf).lower() == "and" for leaf in child.leaves())
            for child in subtree
        )
        if np_children >= 2 and has_and:
            return True
    return False


def plan_from_tree(
    original_prompt: str,
    tree: Tree,
    clip_tokens: Sequence[str],
    *,
    first_sentence: str | None = None,
    stanza_version: str = STANZA_VERSION,
) -> StructurePlan:
    """Build a pure prompt plan from a pinned Stanza constituency tree."""

    method_prompt = normalize_prompt(original_prompt)
    conditioning_texts, spans, np_texts, np_spans = extract_noun_phrases(
        tree,
        method_prompt,
        clip_tokens,
    )
    sentence = first_sentence.strip() if first_sentence is not None else " ".join(tree.leaves())
    return StructurePlan(
        original_prompt=original_prompt,
        method_prompt=method_prompt,
        first_sentence=sentence,
        conditioning_texts=conditioning_texts,
        token_spans=spans,
        np_texts=np_texts,
        np_token_spans=np_spans,
        conjunction=has_full_np_coordination(tree),
        stanza_version=stanza_version,
    )


class StanzaConstituencyPreprocessor:
    """Pinned Stanza 1.4.2 wrapper around the pure plan builder."""

    def __init__(
        self,
        pipeline: Any,
        tokenizer: _TokenizerProtocol,
        *,
        stanza_version: str | None = None,
    ) -> None:
        resolved_version = stanza_version
        if resolved_version is None:
            try:
                resolved_version = metadata.version("stanza")
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError("Stanza is required for Structure Diffusion preprocessing") from exc
        if resolved_version != STANZA_VERSION:
            raise RuntimeError(
                f"Structure Diffusion requires stanza=={STANZA_VERSION}, got {resolved_version}"
            )
        processors = getattr(pipeline, "processors", None)
        if not isinstance(processors, Mapping):
            raise TypeError("Stanza pipeline must expose a processors mapping")
        processor_names = tuple(processors.keys())
        if set(processor_names) != set(STANZA_PROCESSORS):
            raise RuntimeError(
                "Stanza pipeline processors must be exactly tokenize,pos,constituency"
            )
        if not callable(getattr(tokenizer, "tokenize", None)):
            raise TypeError("tokenizer must expose tokenize(text)")
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.stanza_version = resolved_version

    def __call__(self, prompt: str) -> StructurePlan:
        method_prompt = normalize_prompt(prompt)
        document = self.pipeline(method_prompt)
        sentences = getattr(document, "sentences", None)
        if not sentences:
            raise ValueError("Stanza returned no sentences")
        first = sentences[0]
        constituency = getattr(first, "constituency", None)
        if constituency is None:
            raise ValueError("Stanza first sentence has no constituency tree")
        tree = Tree.fromstring(str(constituency))
        tokens = self.tokenizer.tokenize(method_prompt)
        first_text = str(getattr(first, "text", "")).strip() or " ".join(tree.leaves())
        return plan_from_tree(
            prompt,
            tree,
            tokens,
            first_sentence=first_text,
            stanza_version=self.stanza_version,
        )


def align_embedding(
    full_embedding: torch.Tensor,
    phrase_embedding: torch.Tensor,
    span: tuple[int, int],
) -> torch.Tensor:
    """Port official ``multi_align`` for one BOS-exclusive token span."""

    _validate_embedding(full_embedding, name="full_embedding")
    _validate_embedding(phrase_embedding, name="phrase_embedding")
    if full_embedding.shape != phrase_embedding.shape:
        raise ValueError("full and phrase embeddings must have equal shapes")
    if full_embedding.device != phrase_embedding.device or full_embedding.dtype != phrase_embedding.dtype:
        raise ValueError("full and phrase embeddings must share device and dtype")
    start, end = span
    if start < 0 or end <= start:
        raise ValueError("span must be a non-empty half-open interval")
    bos_start, bos_end = start + 1, end + 1
    segment_length = bos_end - bos_start
    if bos_end > full_embedding.shape[0] or 1 + segment_length > phrase_embedding.shape[0]:
        raise ValueError("aligned span exceeds the CLIP embedding sequence")
    aligned = full_embedding.clone()
    aligned[bos_start:bos_end] = phrase_embedding[1:1 + segment_length]
    return aligned


def build_conditioning(
    text_embeddings: torch.Tensor,
    token_spans: Sequence[tuple[int, int]],
    *,
    conjunction: bool,
) -> StructureConditioning:
    """Build official structured K/V contexts for one sample.

    ``text_embeddings[0]`` is the full-prompt embedding; later rows are
    phrase embeddings with spans at matching positions.
    """

    if text_embeddings.ndim != 3:
        raise ValueError("text_embeddings must have shape [texts, tokens, dim]")
    if text_embeddings.shape[0] == 0:
        raise ValueError("at least the full-prompt embedding is required")
    if len(token_spans) != text_embeddings.shape[0]:
        raise ValueError("token_spans must match the text embedding count")
    if not bool(torch.isfinite(text_embeddings).all().item()):
        raise FloatingPointError("text embeddings contain NaN or Inf")

    full = text_embeddings[0]
    aligned = [
        align_embedding(full, text_embeddings[index], token_spans[index])
        for index in range(1, text_embeddings.shape[0])
    ]
    if conjunction:
        keys = torch.stack([full, *aligned], dim=0)
        combined_value = full.clone()
        for index in range(1, text_embeddings.shape[0]):
            combined_value = align_embedding(
                combined_value,
                text_embeddings[index],
                token_spans[index],
            )
        values = combined_value.unsqueeze(0)
    else:
        keys = full.unsqueeze(0)
        values = torch.stack([full, *aligned], dim=0)
    return StructureConditioning(keys=keys, values=values)


def pack_conditioning(
    samples: Sequence[StructureConditioning],
) -> StructureConditioningBatch:
    """Pad structured contexts while keeping each sample's mask independent."""

    if not samples:
        raise ValueError("samples must not be empty")
    sequence_shape = tuple(samples[0].keys.shape[1:])
    dtype = samples[0].keys.dtype
    device = samples[0].keys.device
    for sample in samples:
        if tuple(sample.keys.shape[1:]) != sequence_shape:
            raise ValueError("all samples must share token and embedding dimensions")
        if sample.keys.dtype != dtype or sample.keys.device != device:
            raise ValueError("all samples must share dtype and device")

    batch_size = len(samples)
    max_keys = max(sample.keys.shape[0] for sample in samples)
    max_values = max(sample.values.shape[0] for sample in samples)
    keys = torch.zeros((batch_size, max_keys, *sequence_shape), dtype=dtype, device=device)
    values = torch.zeros((batch_size, max_values, *sequence_shape), dtype=dtype, device=device)
    key_mask = torch.zeros((batch_size, max_keys), dtype=torch.bool, device=device)
    value_mask = torch.zeros((batch_size, max_values), dtype=torch.bool, device=device)
    for batch_index, sample in enumerate(samples):
        key_count = sample.keys.shape[0]
        value_count = sample.values.shape[0]
        keys[batch_index, :key_count] = sample.keys
        values[batch_index, :value_count] = sample.values
        key_mask[batch_index, :key_count] = True
        value_mask[batch_index, :value_count] = True
    return StructureConditioningBatch(
        keys=keys,
        values=values,
        key_mask=key_mask,
        value_mask=value_mask,
        key_counts=tuple(int(sample.keys.shape[0]) for sample in samples),
        value_counts=tuple(int(sample.values.shape[0]) for sample in samples),
    )


def compute_structured_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor,
    value_mask: torch.Tensor,
    *,
    scale: float,
    stable_fp32: bool = True,
    key_counts: Sequence[int] | None = None,
    value_counts: Sequence[int] | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Apply the official ``multi_qkv`` averaging rule per batch sample.

    Inputs are projected and head-separated:
    ``query [B,H,Q,D]``, ``keys [B,K,H,T,D]``, and
    ``values [B,V,H,T,D]``.  Variable K/V counts are selected by masks;
    no phrase from another image can enter a sample's attention result.
    """

    if validate:
        _validate_projected_attention_inputs(query, keys, values, key_mask, value_mask)
    else:
        _validate_projected_attention_shapes(query, keys, values, key_mask, value_mask)
    if not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(float(scale))):
        raise ValueError("scale must be finite")
    work_dtype = torch.float32 if stable_fp32 else query.dtype
    query_work = query.to(dtype=work_dtype)
    keys_work = keys.to(dtype=work_dtype)
    values_work = values.to(dtype=work_dtype)
    if key_counts is None:
        resolved_key_counts = tuple(int(value) for value in key_mask.sum(dim=1).cpu().tolist())
    else:
        resolved_key_counts = tuple(int(value) for value in key_counts)
    if value_counts is None:
        resolved_value_counts = tuple(int(value) for value in value_mask.sum(dim=1).cpu().tolist())
    else:
        resolved_value_counts = tuple(int(value) for value in value_counts)
    if len(resolved_key_counts) != query.shape[0] or len(resolved_value_counts) != query.shape[0]:
        raise ValueError("key_counts and value_counts must match the batch size")

    if stable_fp32:
        # Experiment mode: one batched attention launch.  Actual Structure
        # Diffusion layouts have either one key or one value per sample; the
        # diagonal branch also preserves the official K == V pairing rule.
        logits = torch.einsum("bhqd,bkhtd->bkhqt", query_work, keys_work) * float(scale)
        attentions = torch.softmax(logits, dim=-1)
        key_weights = key_mask.to(dtype=work_dtype)
        value_weights = value_mask.to(dtype=work_dtype)
        key_denominator = key_weights.sum(dim=1).clamp_min(1.0)
        value_denominator = value_weights.sum(dim=1).clamp_min(1.0)

        # K == V: pair variants by index, exactly as official ``zip``.
        paired_count = min(keys.shape[1], values.shape[1])
        paired_terms = torch.einsum(
            "bkhqt,bkhtd->bkhqd",
            attentions[:, :paired_count],
            values_work[:, :paired_count],
        )
        paired_weights = (
            key_weights[:, :paired_count] * value_weights[:, :paired_count]
        )
        paired_output = (
            paired_terms * paired_weights[:, :, None, None, None]
        ).sum(dim=1) / key_denominator[:, None, None, None]

        # K == 1: one attention map is applied to the mean aligned value.
        mean_values = (
            values_work * value_weights[:, :, None, None, None]
        ).sum(dim=1) / value_denominator[:, None, None, None]
        single_key_output = torch.einsum(
            "bhqt,bhtd->bhqd",
            attentions[:, 0],
            mean_values,
        )

        # V == 1: average key-specific attention outputs against one value.
        mean_attentions = (
            attentions * key_weights[:, :, None, None, None]
        ).sum(dim=1) / key_denominator[:, None, None, None]
        single_value_output = torch.einsum(
            "bhqt,bhtd->bhqd",
            mean_attentions,
            values_work[:, 0],
        )

        equal_counts = torch.tensor(
            [key == value for key, value in zip(resolved_key_counts, resolved_value_counts)],
            dtype=torch.bool,
            device=query.device,
        )
        single_key = torch.tensor(
            [key == 1 for key in resolved_key_counts],
            dtype=torch.bool,
            device=query.device,
        )
        output = torch.where(
            equal_counts[:, None, None, None],
            paired_output,
            torch.where(
                single_key[:, None, None, None],
                single_key_output,
                single_value_output,
            ),
        )
        if validate and not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError("structured attention output contains NaN or Inf")
        return output.to(dtype=query.dtype)

    outputs: list[torch.Tensor] = []

    for batch_index in range(query.shape[0]):
        key_count = resolved_key_counts[batch_index]
        value_count = resolved_value_counts[batch_index]
        sample_query = query_work[batch_index]
        sample_keys = keys_work[batch_index, :key_count]
        sample_values = values_work[batch_index, :value_count]
        attentions = [
            torch.softmax(
                torch.einsum("hqd,htd->hqt", sample_query, sample_keys[key_index]) * float(scale),
                dim=-1,
            )
            for key_index in range(key_count)
        ]
        terms: list[torch.Tensor]
        if key_count == value_count:
            terms = [
                torch.einsum("hqt,htd->hqd", attention, sample_values[index])
                for index, attention in enumerate(attentions)
            ]
        else:
            terms = [
                torch.einsum("hqt,htd->hqd", attention, sample_value)
                for attention in attentions
                for sample_value in sample_values
            ]
        sample_output = terms[0]
        for term in terms[1:]:
            sample_output = sample_output + term
        outputs.append(sample_output / len(terms))

    output = torch.stack(outputs, dim=0)
    if validate and not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError("structured attention output contains NaN or Inf")
    return output.to(dtype=query.dtype)


class StructureAttentionController:
    """Batch-scoped conditioning shared by all structured cross-attention layers."""

    def __init__(self, *, stable_fp32: bool = True) -> None:
        self.stable_fp32 = stable_fp32
        self.conditioning: StructureConditioningBatch | None = None
        self.do_classifier_free_guidance = True

    def set_conditioning(
        self,
        conditioning: StructureConditioningBatch,
        *,
        do_classifier_free_guidance: bool,
    ) -> None:
        self.conditioning = conditioning
        self.do_classifier_free_guidance = do_classifier_free_guidance

    def clear(self) -> None:
        self.conditioning = None


class StructureDiffusionCrossAttnProcessor:
    """Diffusers processor implementing official structured cross-attention."""

    def __init__(self, controller: StructureAttentionController) -> None:
        self.controller = controller

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("Structure Diffusion processor is only valid for cross-attention")
        conditioning = self.controller.conditioning
        if conditioning is None:
            raise RuntimeError("structured conditioning must be set before the U-Net forward")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch, channels, height * width).transpose(1, 2)
        elif input_ndim == 3:
            total_batch = hidden_states.shape[0]
            channels = height = width = -1
        else:
            raise ValueError("hidden_states must have three or four dimensions")

        expected_batch = conditioning.batch_size * (
            2 if self.controller.do_classifier_free_guidance else 1
        )
        if total_batch != expected_batch or encoder_hidden_states.shape[0] != expected_batch:
            raise ValueError("U-Net batch does not match structured conditioning/CFG layout")
        if hidden_states.device != conditioning.keys.device:
            raise ValueError("hidden states and structured conditioning must share a device")
        if hidden_states.shape[-1] != attn.to_q.in_features:
            raise ValueError("hidden state dimension does not match attention projection")
        if conditioning.keys.shape[-1] != attn.to_k.in_features:
            raise ValueError("conditioning dimension does not match attention projection")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query_flat = attn.head_to_batch_dim(attn.to_q(hidden_states))
        heads = int(attn.heads)
        query = query_flat.reshape(total_batch, heads, query_flat.shape[1], query_flat.shape[2])
        conditional_query = query[-conditioning.batch_size:]

        keys_raw = conditioning.keys
        values_raw = conditioning.values
        if attn.norm_cross:
            keys_shape = keys_raw.shape
            values_shape = values_raw.shape
            keys_raw = attn.norm_encoder_hidden_states(keys_raw.flatten(0, 1)).reshape(keys_shape)
            values_raw = attn.norm_encoder_hidden_states(values_raw.flatten(0, 1)).reshape(values_shape)
        projected_keys = _project_context_variants(attn, keys_raw, projection="key")
        projected_values = _project_context_variants(attn, values_raw, projection="value")
        conditional_output = compute_structured_attention(
            conditional_query,
            projected_keys,
            projected_values,
            conditioning.key_mask,
            conditioning.value_mask,
            scale=float(attn.scale),
            stable_fp32=self.controller.stable_fp32,
            key_counts=conditioning.key_counts,
            value_counts=conditioning.value_counts,
            validate=False,
        )

        if self.controller.do_classifier_free_guidance:
            batch_size = conditioning.batch_size
            unconditional_context = encoder_hidden_states[:batch_size]
            if attn.norm_cross:
                unconditional_context = attn.norm_encoder_hidden_states(unconditional_context)
            key_uc = attn.head_to_batch_dim(attn.to_k(unconditional_context))
            value_uc = attn.head_to_batch_dim(attn.to_v(unconditional_context))
            query_uc = query_flat[:batch_size * heads]
            sequence_length = unconditional_context.shape[1]
            prepared_mask = attn.prepare_attention_mask(attention_mask, sequence_length, total_batch)
            if prepared_mask is not None:
                prepared_mask = prepared_mask[:batch_size * heads]
            attention_probs = attn.get_attention_scores(query_uc, key_uc, prepared_mask)
            unconditional_output_flat = torch.bmm(attention_probs, value_uc)
            unconditional_output = unconditional_output_flat.reshape(
                batch_size,
                heads,
                unconditional_output_flat.shape[1],
                unconditional_output_flat.shape[2],
            )
            output = torch.cat([unconditional_output, conditional_output], dim=0)
        else:
            output = conditional_output

        output_flat = output.reshape(total_batch * heads, output.shape[2], output.shape[3])
        hidden_states = attn.batch_to_head_dim(output_flat)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch,
                channels,
                height,
                width,
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@dataclass(frozen=True)
class StructureAttentionRegistration:
    """Registered controller and exact processor mapping needed for restoration."""

    controller: StructureAttentionController
    original_processors: Mapping[str, Any]

    def restore(self, unet: Any) -> None:
        if not callable(getattr(unet, "set_attn_processor", None)):
            raise TypeError("unet must expose set_attn_processor")
        unet.set_attn_processor(dict(self.original_processors))


def register_attention(
    unet: Any,
    *,
    stable_fp32: bool = True,
) -> StructureAttentionRegistration:
    """Replace only U-Net cross-attention (``attn2``) processors."""

    current = getattr(unet, "attn_processors", None)
    if not isinstance(current, Mapping) or not current:
        raise TypeError("unet must expose a non-empty attn_processors mapping")
    if not callable(getattr(unet, "set_attn_processor", None)):
        raise TypeError("unet must expose set_attn_processor")
    original = dict(current)
    controller = StructureAttentionController(stable_fp32=stable_fp32)
    replacements: dict[str, Any] = {}
    cross_count = 0
    for name, processor in original.items():
        if name.endswith("attn2.processor"):
            replacements[name] = StructureDiffusionCrossAttnProcessor(controller)
            cross_count += 1
        else:
            replacements[name] = processor
    if cross_count == 0:
        raise ValueError("no cross-attention processors were found on the U-Net")
    unet.set_attn_processor(replacements)
    return StructureAttentionRegistration(
        controller=controller,
        original_processors=original,
    )


def _validate_embedding(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [tokens, dim]")
    if tensor.shape[0] < 2 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must contain a token sequence and embedding dimension")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _validate_context_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [variants, tokens, dim]")
    if 0 in tensor.shape:
        raise ValueError(f"{name} dimensions must be non-zero")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _validate_projected_attention_inputs(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor,
    value_mask: torch.Tensor,
) -> None:
    _validate_projected_attention_shapes(query, keys, values, key_mask, value_mask)
    if not bool(key_mask.any(dim=1).all().item()) or not bool(value_mask.any(dim=1).all().item()):
        raise ValueError("every sample needs an active key and value")
    if not bool(torch.isfinite(query).all().item()):
        raise FloatingPointError("query contains NaN or Inf")
    if not bool(torch.isfinite(keys).all().item()) or not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("keys or values contain NaN or Inf")
    for batch_index in range(query.shape[0]):
        key_count = int(key_mask[batch_index].sum().item())
        value_count = int(value_mask[batch_index].sum().item())
        if not bool(key_mask[batch_index, :key_count].all().item()):
            raise ValueError("key_mask active entries must form a prefix")
        if not bool(value_mask[batch_index, :value_count].all().item()):
            raise ValueError("value_mask active entries must form a prefix")
        if key_count != value_count and min(key_count, value_count) != 1:
            raise ValueError("official structured attention requires K == V, K == 1, or V == 1")


def _validate_projected_attention_shapes(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor,
    value_mask: torch.Tensor,
) -> None:
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, queries, dim]")
    if keys.ndim != 5 or values.ndim != 5:
        raise ValueError("keys and values must have shape [batch, variants, heads, tokens, dim]")
    if query.shape[0] != keys.shape[0] or query.shape[0] != values.shape[0]:
        raise ValueError("query/key/value batch sizes differ")
    if query.shape[1] != keys.shape[2] or query.shape[1] != values.shape[2]:
        raise ValueError("query/key/value head counts differ")
    if query.shape[-1] != keys.shape[-1] or query.shape[-1] != values.shape[-1]:
        raise ValueError("query/key/value head dimensions differ")
    if keys.shape[3] != values.shape[3]:
        raise ValueError("key/value token sequence lengths differ")
    if key_mask.shape != keys.shape[:2] or value_mask.shape != values.shape[:2]:
        raise ValueError("variant masks do not match projected K/V tensors")
    if key_mask.dtype != torch.bool or value_mask.dtype != torch.bool:
        raise TypeError("variant masks must be boolean")
    tensors = (keys, values, key_mask, value_mask)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("projected attention inputs must share one device")
    if not query.is_floating_point() or not keys.is_floating_point() or not values.is_floating_point():
        raise TypeError("query, keys, and values must be floating point")


def _project_context_variants(
    attn: Attention,
    contexts: torch.Tensor,
    *,
    projection: str,
) -> torch.Tensor:
    batch_size, variants, tokens, _ = contexts.shape
    flattened = contexts.flatten(0, 1)
    if projection == "key":
        projected = attn.to_k(flattened)
    elif projection == "value":
        projected = attn.to_v(flattened)
    else:
        raise ValueError("projection must be 'key' or 'value'")
    head_folded = attn.head_to_batch_dim(projected)
    return head_folded.reshape(
        batch_size,
        variants,
        int(attn.heads),
        tokens,
        head_folded.shape[-1],
    )
