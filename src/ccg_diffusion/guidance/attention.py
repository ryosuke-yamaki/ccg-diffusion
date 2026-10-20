"""Cross-attention capture of the SD1.4 guided forward pass.

Every loss of CCG Diffusion on SD1.4 (the Predicated Diffusion loss L_PD and
the Spatial Relation, Region Assignment and Count losses) reads the 16x16
cross-attention probabilities of the five ``attn2`` processors of the U-Net.
``register_attention_control`` installs a ``SelectiveCrossAttnProcessor`` on
every cross-attention processor; during a guided forward it computes the
attention logits and softmax in FP32 and hands the probabilities to an
``AttentionStore``. ``aggregate_attention_batched`` then averages the stored
maps over heads and layers, one ``[16, 16, 77]`` map per image.

The external SD1.4 baselines (Attend-and-Excite, SynGen) use the same capture.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

# Order in which stored layers are summed; it fixes the floating-point summation order (KEEP).
ATTENTION_PLACES = ("up", "down", "mid")


class AttentionStore:
    """Buffers the attention probabilities of one guided forward, keyed by U-Net block location.

    ``step_store`` collects the maps of the current forward; ``finish_selective_forward``
    publishes them as ``attention_store``. Maps whose token grid is larger than
    ``max_size x max_size`` are not kept.
    """

    @staticmethod
    def get_empty_store() -> Dict[str, List[torch.Tensor]]:
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> None:
        """Append ``attn`` to ``step_store[f"{place_in_unet}_{cross|self}"]``.

        Skips layers whose token grid exceeds ``max_size ** 2`` to bound
        memory.
        """
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= self.max_size ** 2:
            self.step_store[key].append(attn)

    def between_steps(self) -> None:
        """Fold ``step_store`` into ``attention_store``, then reset ``step_store``."""
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def reset(self) -> None:
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.step_capture_processor_names = []

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable map retention for the next selective guided forward."""

        self.capture_enabled = enabled

    def finish_selective_forward(self) -> None:
        """Publish the maps collected during the guided forward."""

        self.last_capture_processor_names = tuple(self.step_capture_processor_names)
        self.between_steps()

    def record_capture_processor(self, name: str) -> None:
        """Record the exact processor that retained a map in this forward."""

        if name not in self.step_capture_processor_names:
            self.step_capture_processor_names.append(name)

    def __init__(self, max_size: int = 32) -> None:
        self.max_size = max_size
        self.step_store = self.get_empty_store()
        self.attention_store: Dict[str, List[torch.Tensor]] = {}
        self.capture_enabled = True
        self.step_capture_processor_names: List[str] = []
        self.last_capture_processor_names: tuple[str, ...] = ()


class SelectiveCrossAttnProcessor:
    """Capture one cross-attention resolution and use SDPA otherwise.

    The guided target path evaluates QK logits and softmax in FP32 so the loss
    never receives a half-precision probability map.  Normal denoising and
    non-target resolutions delegate to diffusers' fused ``AttnProcessor2_0``.
    """

    def __init__(
        self,
        attnstore: AttentionStore,
        place_in_unet: str,
        *,
        attention_res: int,
        processor_name: str = "<unnamed>",
    ) -> None:
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.num_pixels = attention_res ** 2
        self.processor_name = processor_name
        self.fast_processor = AttnProcessor2_0()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        sequence_pixels = hidden_states.shape[-2]
        should_capture = (
            self.attnstore.capture_enabled
            and is_cross
            and sequence_pixels == self.num_pixels
        )
        if not should_capture:
            return self.fast_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )

        batch_size, sequence_length, _ = hidden_states.shape
        prepared_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        scale = getattr(attn, "scale", query.shape[-1] ** -0.5)
        logits = torch.bmm(
            query.float(), key.float().transpose(-1, -2)
        ) * float(scale)
        if prepared_mask is not None:
            logits = logits + prepared_mask.float()
        attention_probs = torch.softmax(logits, dim=-1)
        self.attnstore.record_capture_processor(self.processor_name)
        self.attnstore.forward(attention_probs, True, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs.to(value.dtype), value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def register_attention_control(unet: Any, controller: AttentionStore, *, attention_res: int = 16) -> None:
    """Install a ``SelectiveCrossAttnProcessor`` on every cross-attention processor of ``unet``.

    Self-attention processors keep their original processor. Processors outside
    the down, mid and up blocks are not passed to ``set_attn_processor``.
    """

    attn_procs: Dict[str, Any] = {}
    for name, original_processor in unet.attn_processors.items():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        if ".attn2.processor" not in name:
            attn_procs[name] = original_processor
            continue

        attn_procs[name] = SelectiveCrossAttnProcessor(
            attnstore=controller,
            place_in_unet=place_in_unet,
            attention_res=attention_res,
            processor_name=name,
        )

    unet.set_attn_processor(attn_procs)


def aggregate_attention_batched(store: AttentionStore, *, res: int = 16, batch_size: int) -> List[torch.Tensor]:
    """Average the stored cross-attention maps of resolution ``res`` per image.

    Each stored tensor has shape ``(batch_size * heads, res * res, num_tokens)``.
    Heads are averaged first, then layers are summed in ``ATTENTION_PLACES``
    order and divided by the number of layers. Returns ``batch_size`` tensors of
    shape ``(res, res, num_tokens)``.
    """
    num_pixels = res ** 2
    accum: Optional[torch.Tensor] = None
    num_layers = 0
    for location in ATTENTION_PLACES:
        for item in store.attention_store.get(f"{location}_cross", []):
            if item.shape[1] != num_pixels:
                continue
            if item.shape[0] % batch_size != 0:
                raise ValueError(
                    f"attention tensor leading dim {item.shape[0]} is not divisible by batch_size={batch_size}"
                )
            heads = item.shape[0] // batch_size
            last_dim = item.shape[-1]
            reshaped = item.reshape(batch_size, heads, res, res, last_dim)
            per_track = reshaped.mean(dim=1)
            accum = per_track if accum is None else accum + per_track
            num_layers += 1
    if num_layers == 0:
        raise RuntimeError(f"No attention maps at res={res} in from_where={ATTENTION_PLACES}")
    averaged = accum / num_layers
    return [averaged[b] for b in range(batch_size)]
