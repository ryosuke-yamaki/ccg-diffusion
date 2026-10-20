"""The Predicated Diffusion loss L_PD over guidance targets I = (E, C, K, P) (paper Eq. 2-3, Table 1).

For attention maps ``a_i`` of the guidance-target tokens::

    L_exist(i)        = -log(1 - exp(mean_x log(1 - a_i(x))))
    L_corr(i, j)      = -log(1 - a_j(1-a_i)) - log(1 - a_i(1-a_j))   (mean over pixels)
    L_leak(i, j)      = -log(1 - a_i * a_j)                          (mean over pixels)
    L_possession(h,p) = -log(1 - a_p * (1 - a_h))                    (mean over pixels)

with the weights gamma = (0.1, 1, 0.3, 1) of Predicated Diffusion. The
existence term uses a softmax over the lexical token columns (x100); the pair
terms use per-token min-max normalized maps. All arithmetic is FP32. The pair
terms use the bounded barrier ``-log1p(-(1 - 1e-6) * product)``; the existence
term has its own guard: ``mean_x log1p(1e-4 - a_i(x))`` is clamped to at most
``-1e-6`` before ``-log(-expm1(.))``.

``WholeWordSpans`` describe, for Selective Guidance Boost rows, the lexical
pieces of each correspondence/leak endpoint word; with spans the pair terms
read the mean of the raw piece maps before normalization (whole-word pooling).
``SanaWholeWordSpans`` is the same contract on Sana's Gemma token axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Literal, Mapping, MutableMapping, Optional, Sequence

import torch

from ccg_diffusion.targets.pd_indices import PDIndices

EXISTENCE_WEIGHT = 0.1  # gamma_exist (the literal 0.1 is also written in the loss expression)
PD_LEAK_WEIGHT = 0.3  # gamma_leak
PAIR_BOUNDARY_MARGIN = 1e-6  # barrier margin of the pair terms

# Accepted ``special_layout`` values — see :func:`compute_pd_loss`.
_SPECIAL_LAYOUTS = ("clip", "leading_bos")


@dataclass(frozen=True)
class WholeWordSpans:
    """Immutable CLIP spans; indices include BOS and exclude EOS/padding.

    A span is keyed by the historical final lexical-piece anchor. The
    text_token_count includes the first BOS/EOS pair, independently of
    the unchanged existence-softmax window used by a particular backbone.
    """

    max_tokens: ClassVar[int] = 77
    trailing_special_tokens: ClassVar[int] = 1
    special_layout: ClassVar[str] = "clip"

    text_token_count: int
    spans: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.text_token_count, bool)
            or not isinstance(self.text_token_count, int)
            or not 2 + self.trailing_special_tokens <= self.text_token_count <= self.max_tokens
        ):
            raise ValueError("pair span token count must include CLIP BOS/EOS and fit 77 tokens")
        if not isinstance(self.spans, tuple):
            raise ValueError("pair spans must be immutable tuples")
        anchors: set[int] = set()
        owners: dict[int, int] = {}
        for entry in self.spans:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("pair span entries must be (anchor, lexical pieces)")
            anchor, pieces = entry
            if isinstance(anchor, bool) or not isinstance(anchor, int):
                raise ValueError("pair span anchor must be an integer")
            if anchor in anchors:
                raise ValueError("pair span anchors must be unique")
            anchors.add(anchor)
            if not isinstance(pieces, tuple) or not pieces:
                raise ValueError("pair span lexical pieces must be a nonempty tuple")
            if any(
                isinstance(piece, bool)
                or not isinstance(piece, int)
                or not 0 < piece < self.text_token_count - self.trailing_special_tokens
                for piece in pieces
            ):
                raise ValueError("pair span includes a special, padding, or out-of-range token")
            if pieces != tuple(sorted(set(pieces))) or anchor != pieces[-1]:
                raise ValueError("pair span must be ordered, unique, and end at its anchor")
            for piece in pieces:
                if piece in owners and owners[piece] != anchor:
                    raise ValueError("different lexical pair spans may not overlap")
                owners[piece] = anchor
        if tuple(anchor for anchor, _pieces in self.spans) != tuple(sorted(anchors)):
            raise ValueError("pair spans must be ordered by anchor")

    @classmethod
    def from_mapping(
        cls, value: Mapping[int, Sequence[int]], *, text_token_count: int
    ) -> "WholeWordSpans":
        """Freeze a validated anchor mapping without coercing malformed types."""

        if any(isinstance(anchor, bool) or not isinstance(anchor, int) for anchor in value):
            raise ValueError("pair span anchor must be an integer")
        return cls(
            text_token_count=text_token_count,
            spans=tuple((anchor, tuple(value[anchor])) for anchor in sorted(value)),
        )

    def require_coverage(self, anchors: set[int], *, token_axis_size: int) -> None:
        """Require the exact corr/leak endpoint set and the runtime token axis."""

        if self.text_token_count > token_axis_size:
            raise ValueError("pair spans exceed the runtime attention token axis")
        if {anchor for anchor, _pieces in self.spans} != anchors:
            raise ValueError("pair spans must cover exactly the corr/leak endpoint anchors")

    def as_mapping(self) -> dict[int, tuple[int, ...]]:
        """Return a fresh mapping; mutating it cannot change this contract."""

        return dict(self.spans)

    def as_record(self) -> dict[str, object]:
        """Return a JSON-safe record without stringifying token-index keys."""

        return {
            "text_token_count": self.text_token_count,
            "spans": [
                {"anchor": anchor, "lexical_piece_indices": list(pieces)}
                for anchor, pieces in self.spans
            ],
        }


@dataclass(frozen=True)
class SanaWholeWordSpans(WholeWordSpans):
    """Gemma selected-axis spans: leading BOS, no trailing EOS, at most 300 tokens."""

    max_tokens: ClassVar[int] = 300
    trailing_special_tokens: ClassVar[int] = 0
    special_layout: ClassVar[str] = "leading_bos"


def whole_word_spans(alignment: Any, indices: Any, *, text_token_count: int) -> WholeWordSpans:
    """Whole-word spans of every correspondence/leak endpoint, read from the CLIP alignment rows."""

    anchors = {index for pair in (*indices.corr, *indices.leak) for index in pair}
    spans: dict[int, tuple[int, ...]] = {}
    for row in alignment:
        anchor = row.get("clip_index")
        if anchor not in anchors:
            continue
        pieces = tuple(row["clip_lexical_piece_indices"])
        if anchor in spans and spans[anchor] != pieces:
            raise ValueError("one anchor has inconsistent lexical spans")
        spans[anchor] = pieces
    result = WholeWordSpans.from_mapping(spans, text_token_count=text_token_count)
    result.require_coverage(anchors, token_axis_size=77)
    return result


def _stable_barrier(product: torch.Tensor, *, margin: float) -> torch.Tensor:
    """Evaluate ``-log(1-product)`` with a finite endpoint derivative."""

    bounded = torch.clamp(product, min=0.0, max=1.0)
    return -torch.log1p(-(1.0 - margin) * bounded)


def compute_pd_loss(
    attention_maps: torch.Tensor,
    pd_indices: PDIndices,
    *,
    attention_res: int = 16,
    special_layout: Literal["clip", "leading_bos"] = "clip",
    guard_counts: Optional[MutableMapping[str, int]] = None,
    pair_token_spans: WholeWordSpans | None = None,
) -> torch.Tensor:
    """Return the Predicated Diffusion loss L_PD as an FP32 scalar tensor.

    Parameters
    ----------
    attention_maps:
        ``[res, res, num_tokens]`` cross-attention maps of one image.
    pd_indices:
        Guidance targets I = (E, C, K, P) as token indices that include the
        leading BOS (1-based lexical positions).
    attention_res:
        Edge length of the attention grid; per-pixel terms are divided by
        ``attention_res ** 2``.
    special_layout:
        ``"clip"`` (SD1.4: ``[bos] ... [eos]``, existence softmax over
        ``[1:-1]``, i.e. the fixed 75 lexical columns) or ``"leading_bos"``
        (Sana/Gemma: ``<bos> ...`` without EOS, softmax over ``[1:]``).
    guard_counts:
        Optional counters of the numerical guards (clamped existence terms,
        near-constant maps, whole-word reductions).
    pair_token_spans:
        Whole-word spans of the correspondence/leak endpoints (Selective
        Guidance Boost rows). Existence and possession terms are unchanged.
    """
    if special_layout not in _SPECIAL_LAYOUTS:
        raise ValueError(
            f"special_layout must be one of {_SPECIAL_LAYOUTS}, got {special_layout!r}"
        )

    size = attention_res ** 2
    work_maps = attention_maps.float()
    loss = torch.zeros((), device=work_maps.device, dtype=work_maps.dtype)

    # Cache 0-1 normalisations: the same token attention map can show up
    # in corr, leak, AND possession.
    normalised: Dict[int, torch.Tensor] = {}
    pair_normalised: Dict[int, torch.Tensor] = {}
    pair_pieces: dict[int, tuple[int, ...]] = {}
    if pair_token_spans is not None:
        if not isinstance(pair_token_spans, WholeWordSpans):
            raise ValueError("pair_token_spans must be a validated WholeWordSpans")
        if special_layout != pair_token_spans.special_layout:
            raise ValueError("whole-word pair spans require their declared CLIP layout or Gemma layout")
        pair_token_spans.require_coverage(
            {index for pair in (*pd_indices.corr, *pd_indices.leak) for index in pair},
            token_axis_size=attention_maps.shape[-1],
        )
        pair_pieces = pair_token_spans.as_mapping()
        if guard_counts is not None:
            guard_counts["whole_word_pair_loss_calls"] = guard_counts.get("whole_word_pair_loss_calls", 0) + 1

    def _norm(idx: int) -> torch.Tensor:
        cached = normalised.get(idx)
        if cached is None:
            cached = _minmax_normalize(
                work_maps[:, :, idx],
                guard_counts=guard_counts,
            )
            normalised[idx] = cached
        return cached

    def _pair_norm(idx: int) -> torch.Tensor:
        pieces = pair_pieces.get(idx)
        # Singleton spans retain the original operations. Multi-piece maps
        # never enter the cache shared with the unchanged possession term.
        if pieces is None or len(pieces) == 1:
            return _norm(idx)
        cached = pair_normalised.get(idx)
        if cached is None:
            if guard_counts is not None:
                key = "whole_word_multi_piece_map_reductions"
                guard_counts[key] = guard_counts.get(key, 0) + 1
            cached = _minmax_normalize(
                work_maps[:, :, pieces].mean(dim=-1),
                guard_counts=guard_counts,
            )
            pair_normalised[idx] = cached
        return cached

    if pd_indices.exist:
        # exist softmax trim depends on the layout:
        #   clip         -> drop <bos> and <eos>            (1, -1)
        #   leading_bos  -> drop <bos> only                 (1, None)
        # Both layouts use <bos>-inclusive 1-indexed targets (offset 1).
        if special_layout == "clip":
            exist_slice = slice(1, -1)
        else:  # leading_bos
            exist_slice = slice(1, None)
        offset = 1
        attention_normalize = work_maps[:, :, exist_slice] * 100
        if attention_normalize.shape[-1] == 0:
            raise ValueError("existence loss requires at least one lexical token")
        attention_normalize = torch.nn.functional.softmax(attention_normalize, dim=-1)
        n_exist = len(pd_indices.exist)
        for i in pd_indices.exist:
            atten = attention_normalize[:, :, i - offset]
            log_term = torch.sum(torch.log1p(-atten + 1e-4)) / size
            # -log(1 - exp(log_term)) is defined only for log_term < 0; the clamp
            # keeps it finite and is a no-op whenever log_term is already negative.
            if guard_counts is not None and bool((log_term > -1e-6).item()):
                guard_counts["exist_log_term_clamps"] = (
                    int(guard_counts.get("exist_log_term_clamps", 0)) + 1
                )
            log_term = torch.clamp(log_term, max=-1e-6)
            exist_loss = -torch.log(-torch.expm1(log_term))
            loss = loss + 0.1 * exist_loss / n_exist

    if pd_indices.corr:
        n_corr = len(pd_indices.corr)
        for adj_idx, obj_idx in pd_indices.corr:
            atten_adj = _pair_norm(adj_idx)
            atten_obj = _pair_norm(obj_idx)
            loss_corr = _stable_barrier(
                atten_obj * (1 - atten_adj), margin=PAIR_BOUNDARY_MARGIN
            ) + _stable_barrier(
                atten_adj * (1 - atten_obj), margin=PAIR_BOUNDARY_MARGIN
            )
            loss = loss + torch.sum(loss_corr) / size / n_corr

    if pd_indices.leak:
        n_leak = len(pd_indices.leak)
        for a_idx, b_idx in pd_indices.leak:
            atten_a = _pair_norm(a_idx)
            atten_b = _pair_norm(b_idx)
            loss_leak = _stable_barrier(
                atten_a * atten_b, margin=PAIR_BOUNDARY_MARGIN
            )
            loss = loss + PD_LEAK_WEIGHT * torch.sum(loss_leak) / size / n_leak

    if pd_indices.possession:
        n_poss = len(pd_indices.possession)
        for holder_idx, possession_idx in pd_indices.possession:
            atten_holder = _norm(holder_idx)
            atten_possession = _norm(possession_idx)
            loss_poss = _stable_barrier(
                atten_possession * (1 - atten_holder),
                margin=PAIR_BOUNDARY_MARGIN,
            )
            loss = loss + torch.sum(loss_poss) / size / n_poss

    return loss


def _minmax_normalize(
    attention: torch.Tensor,
    *,
    guard_counts: Optional[MutableMapping[str, int]] = None,
) -> torch.Tensor:
    """Rescale a 2-D attention map to ``[0, 1]`` via min-max (PD convention).

    A (near-)constant map carries no localization signal and its min-max
    rescale is undefined; it maps to all-zeros so the pair terms stay finite.
    """
    maximum = attention.max()
    minimum = attention.min()
    span = maximum - minimum
    finfo = torch.finfo(attention.dtype)
    scale = torch.maximum(maximum.abs(), minimum.abs())
    span_floor = torch.maximum(
        scale * (32.0 * finfo.eps),
        torch.tensor(finfo.tiny, device=attention.device, dtype=attention.dtype),
    )
    is_constant = span <= span_floor
    guard_key = "near_constant_attention_maps"
    if is_constant:
        if guard_counts is not None:
            guard_counts[guard_key] = (
                int(guard_counts.get(guard_key, 0)) + 1
            )
        return torch.zeros_like(attention)
    return (attention - minimum) / span
