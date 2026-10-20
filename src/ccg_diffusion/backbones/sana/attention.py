"""Cross-attention capture on Sana's ``attn2`` layers (paper Table 8: blocks [0, 8) at 16x16).

``SanaTransformer2DModel`` is a flat list of transformer blocks; each block's
``attn2`` attends from the image tokens to the text tokens. Diffusers runs it
with ``scaled_dot_product_attention``, which never exposes the attention
probabilities, so ``SanaStoreAttnProcessor`` replaces it with an explicit
FP32 ``softmax(QK^T) V`` and hands the probabilities of the selected blocks to
``SanaAttentionStore``. After the last selected block the processor raises
``SanaCaptureComplete``: the guidance forward needs no later block. The text
keys and values are computed once per phase (guidance forward or CFG forward)
and cached, because the text input does not change within a generation.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import torch
from diffusers.models.attention_processor import Attention, SanaLinearAttnProcessor2_0

# Block index of an ``attn_processors`` key such as ``transformer_blocks.7.attn2.processor``.
_BLOCK_INDEX_RE = re.compile(r"transformer_blocks\.(\d+)\.attn2")


class SanaCaptureComplete(Exception):
    """Internal signal: every requested guidance map is available."""


class SanaAttentionStore:
    """``attn2`` probabilities keyed by transformer block (``block_<index>``).

    ``step_store`` collects the maps of the running forward; once
    ``num_att_layers`` maps have arrived they become ``attention_store`` and
    ``step_store`` is emptied. Each map has shape ``[B, heads, Q, K]``
    (``Q`` image tokens, ``K`` text tokens) and keeps its autograd graph, so the
    guidance loss can be differentiated with respect to the latent.
    """

    def __init__(self) -> None:
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.enabled = True
        self.selected_blocks: Optional[set[int]] = None
        self.stop_after: Optional[int] = None
        self.cache_text_kv = False
        self.num_blocks = 0
        self.step_store: dict[str, torch.Tensor] = {}
        self.attention_store: dict[str, torch.Tensor] = {}

    def __call__(self, attn: torch.Tensor, block_index: int) -> None:
        self.step_store[f"block_{block_index}"] = attn
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.attention_store = self.step_store
            self.step_store = {}

    def reset(self) -> None:
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = {}
        self.attention_store = {}


class SanaStoreAttnProcessor:
    """``attn2`` processor that records the cross-attention probabilities.

    Same projections, head split, ``to_out`` and ``rescale_output_factor`` as
    diffusers' Sana attention processor; only the attention product is an
    explicit FP32 softmax whose probabilities are cast back to the value dtype.
    """

    def __init__(self, store: SanaAttentionStore, block_index: int) -> None:
        self.store = store
        self.block_index = block_index
        # One processor set belongs to one generation invocation. Conditional
        # guidance and CFG denoising have independent, immutable text inputs.
        self._text_kv: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # Shape to (batch, heads, source_length, target_length).
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        cache_key = self.store.enabled
        cached = self._text_kv.get(cache_key) if self.store.cache_text_kv else None
        if cached is None:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            if self.store.cache_text_kv:
                if key.requires_grad or value.requires_grad:
                    raise ValueError("Sana text KV caching requires frozen text projections")
                self._text_kv[cache_key] = (key, value)
        else:
            key, value = cached
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scores = torch.matmul(query.float(), key.transpose(-1, -2).float()) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        attention_probs = scores.softmax(dim=-1)
        if self.store.enabled and (self.store.selected_blocks is None
                                   or self.block_index in self.store.selected_blocks):
            self.store(attention_probs, block_index=self.block_index)
            if self.block_index == self.store.stop_after:
                raise SanaCaptureComplete
        hidden_states = torch.matmul(attention_probs.to(value.dtype), value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_sana_attention_store(transformer: Any, store: SanaAttentionStore) -> None:
    """Install a capture processor on every ``attn2`` of ``transformer``.

    ``set_attn_processor`` needs a processor for every key, so the ``attn1``
    (linear self-attention) keys receive a fresh ``SanaLinearAttnProcessor2_0``,
    which leaves them unchanged. Sets ``store.num_blocks`` and
    ``store.num_att_layers`` (the number of maps that complete one forward).
    """
    procs: dict[str, Any] = {}
    num_blocks = 0
    for name in transformer.attn_processors:
        match = _BLOCK_INDEX_RE.search(name)
        if name.endswith("attn2.processor") and match is not None:
            procs[name] = SanaStoreAttnProcessor(store, block_index=int(match.group(1)))
            num_blocks += 1
        else:
            procs[name] = SanaLinearAttnProcessor2_0()
    transformer.set_attn_processor(procs)
    store.num_blocks = num_blocks
    store.num_att_layers = num_blocks if store.selected_blocks is None else len(store.selected_blocks)
    if store.selected_blocks is not None and not store.selected_blocks <= set(range(num_blocks)):
        raise ValueError("Sana guidance requested an unavailable attention block")


def aggregate_sana_attention(store: SanaAttentionStore, block_range: Sequence[int]) -> torch.Tensor:
    """Average the stored maps over ``block_range`` and the heads into ``[B, grid, grid, K]``."""
    maps: list[torch.Tensor] = []
    for i in block_range:
        key = f"block_{i}"
        if key not in store.attention_store:
            raise KeyError(f"block index {i} not present in attention_store (have {sorted(store.attention_store)})")
        maps.append(store.attention_store[key])
    if not maps:
        raise RuntimeError(f"block_range {list(block_range)} selected no blocks")

    stacked = torch.stack(maps, dim=0)  # [n_block, B, heads, Q, K]
    m = stacked.mean(dim=0)             # [B, heads, Q, K]
    m = m.mean(dim=1)
    batch_size, q, k = m.shape
    grid = int(round(q ** .5))
    if grid * grid != q:
        raise ValueError("non-square latent unsupported")
    return m.reshape(batch_size, grid, grid, k)
