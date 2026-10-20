"""Sana backbone: CLIP vs Gemma token layouts, attention capture, the pinned contract and the guided loop."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from diffusers.models.attention_processor import Attention

from conftest import QuadraticSanaObjective, sana_prompt_inputs, tiny_sana_pipe
from ccg_diffusion.backbones.sana.attention import (
    SanaAttentionStore,
    SanaCaptureComplete,
    SanaStoreAttnProcessor,
    aggregate_sana_attention,
)
from ccg_diffusion.backbones.sana.guidance import DEFAULT_BLOCK_RANGE, generate_guided
from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION, tensor_sha256
from ccg_diffusion.config import SANA_GENERATION
from ccg_diffusion.guidance.count_loss import CountConstraint, CountState, SanaCountConstraint
from ccg_diffusion.guidance.predicated_diffusion_loss import SanaWholeWordSpans, WholeWordSpans, compute_pd_loss
from ccg_diffusion.provenance import canonical_json, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices

# sha256 of the canonical JSON of the instruction list recorded in the Sana T2I protocols and TAB run manifests.
INSTRUCTION_SHA256 = "0c2a85f2ced0eb140197686156669a7b37a6b4c3987df943940f218eac443adf"


@pytest.fixture
def default_algorithms():
    """Run with deterministic algorithms off, as every Sana paper run did, and restore the previous state."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(previous)


# --- the same guidance on the CLIP and the Gemma token layouts ---------------------------------------------------


def test_whole_word_pair_loss_is_equal_on_the_clip_and_gemma_layouts() -> None:
    generator = torch.Generator().manual_seed(421)
    raw = torch.rand(16, 16, 8, generator=generator)
    clip = raw.clone().requires_grad_()
    sana = raw[..., :-1].clone().requires_grad_()  # Gemma: no trailing EOS column
    spans = {3: (2, 3), 6: (5, 6)}
    indices = PDIndices(corr=[[3, 6]], leak=[[3, 6]])
    a = compute_pd_loss(clip, indices, pair_token_spans=WholeWordSpans.from_mapping(spans, text_token_count=8))
    b = compute_pd_loss(sana, indices, special_layout="leading_bos",
                        pair_token_spans=SanaWholeWordSpans.from_mapping(spans, text_token_count=7))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga, = torch.autograd.grad(a, clip)
    gb, = torch.autograd.grad(b, sana)
    torch.testing.assert_close(ga[..., :-1], gb, rtol=0, atol=0)
    with pytest.raises(ValueError, match="out-of-range"):
        SanaWholeWordSpans.from_mapping({7: (7,)}, text_token_count=7)


def test_count_loss_is_equal_on_the_clip_and_gemma_layouts() -> None:
    generator = torch.Generator().manual_seed(423)
    raw = torch.rand(16, 16, 80, generator=generator) * .05
    clip = raw[..., :77].clone().requires_grad_()
    sana = raw.clone().requires_grad_()
    a = CountState("clip", (CountConstraint(3, 1, 2, "cats", 3, (2, 3), 6),))
    b = CountState("sana", (SanaCountConstraint(3, 1, 2, "cats", 3, (2, 3), 80),), token_axis_size=80)
    la = a.augment(clip.sum() * 0, clip, outer_step_index=0)
    lb = b.augment(sana.sum() * 0, sana, outer_step_index=0)
    assert a.targets[0].sha256 == b.targets[0].sha256
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    ga, = torch.autograd.grad(la, clip)
    gb, = torch.autograd.grad(lb, sana)
    torch.testing.assert_close(ga, gb[..., :77], rtol=0, atol=0)
    assert torch.count_nonzero(gb[..., 77:]) == 0


# --- cross-attention capture -------------------------------------------------------------------------------


def _attention_layer() -> Attention:
    return Attention(query_dim=16, cross_attention_dim=16, heads=2, dim_head=8).eval().requires_grad_(False)


def test_early_capture_keeps_the_selected_maps_and_latent_gradients() -> None:
    torch.manual_seed(4)
    layers = [_attention_layer() for _ in range(3)]
    text = torch.randn(2, 5, 16)
    latent = torch.randn(2, 4, 16)
    observed = []
    for early in [False, True]:
        store = SanaAttentionStore()
        store.num_att_layers = 2 if early else 3
        if early:
            store.selected_blocks = {0, 1}
            store.stop_after = 1
        x = latent.clone().requires_grad_()
        h = x
        visited = []
        try:
            for index, layer in enumerate(layers):
                visited.append(index)
                h = h + SanaStoreAttnProcessor(store, index)(layer, h, text)
        except SanaCaptureComplete:
            assert early
        maps = aggregate_sana_attention(store, [0, 1])
        grad = torch.autograd.grad(maps.square().sum(), x)[0]
        observed.append((maps.detach(), grad))
        assert visited == ([0, 1] if early else [0, 1, 2])
    torch.testing.assert_close(observed[0][0], observed[1][0], rtol=0, atol=0)
    torch.testing.assert_close(observed[0][1], observed[1][1], rtol=0, atol=0)


def test_text_kv_cache_separates_guidance_and_cfg_inputs() -> None:
    torch.manual_seed(7)
    layer = _attention_layer()
    store = SanaAttentionStore()
    store.cache_text_kv = True
    processor = SanaStoreAttnProcessor(store, 0)
    for enabled, batch in [(True, 2), (False, 4)]:
        store.enabled = enabled
        x, text = torch.randn(batch, 4, 16), torch.randn(batch, 5, 16)
        plain_store = SanaAttentionStore()
        plain_store.enabled = enabled
        expected = SanaStoreAttnProcessor(plain_store, 0)(layer, x, text)
        for _ in range(2):
            torch.testing.assert_close(processor(layer, x, text), expected, rtol=0, atol=0)
        if not enabled:
            store.reset()
            processor(layer, x, text)
            assert not store.step_store and not store.attention_store
    assert set(processor._text_kv) == {False, True}


# --- pinned contract ---------------------------------------------------------------------------------------


def test_sana_contract_instruction_prompt_and_latent_hash_layout() -> None:
    c = SANA_GENERATION
    assert (c.model_id, c.model_revision) == (
        "Efficient-Large-Model/Sana_600M_512px_diffusers", "83d7a190bfd1fd070570a793d2dab5c7a3231b9d")
    assert (c.steps, c.guided_steps, c.guidance_scale, c.height, c.width, c.batch_size, c.block_range,
            c.refinement_steps, c.scale_factor, c.model_dtype, c.master_dtype) == (
        20, 20, 4.5, 512, 512, 4, (0, 8), (0,), 20.0, "float16", "float32")
    assert DEFAULT_BLOCK_RANGE == tuple(range(*c.block_range))
    assert sha256_text(canonical_json(COMPLEX_HUMAN_INSTRUCTION)) == INSTRUCTION_SHA256
    # Sana layout: dtype string, then the shape tuple, then the raw bytes.
    value = torch.zeros(2, dtype=torch.float16)
    assert tensor_sha256(value) == hashlib.sha256(b"torch.float16(2,)" + bytes(4)).hexdigest()


# --- the guided denoising loop on CPU fakes ----------------------------------------------------------------


def _guided_run(pipe, *, weight: float = 1e3) -> tuple[list[QuadraticSanaObjective], list, object, list[int]]:
    plan, encoding = sana_prompt_inputs()
    objectives = [QuadraticSanaObjective(plan, PDIndices(**plan["source_program"]), scale_multiplier=multiplier,
                                         max_update_ratio=cap, weight=weight)
                  for multiplier, cap in ((1.0, 0.02), (4.0, 0.06))]
    forwards: list[int] = []
    hook = pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: forwards.append(args[0].shape[0]), with_kwargs=True)
    try:
        images, diagnostics = generate_guided(
            pipe, plan["conditioning"], objectives, [torch.Generator().manual_seed(seed) for seed in (42, 43)],
            prompt_encoding=encoding, instruction=COMPLEX_HUMAN_INSTRUCTION, steps=3, guided_steps=2)
    finally:
        hook.remove()
    return objectives, images, diagnostics, forwards


def test_generate_guided_cpu_smoke(default_algorithms: None) -> None:
    pipe = tiny_sana_pipe()
    objectives, images, diagnostics, forwards = _guided_run(pipe)
    # Guided steps 0 and 1, plus five extra updates at step 0: 7 capture forwards (batch 2), 3 CFG forwards (batch 4).
    assert forwards == [2] * 6 + [4, 2, 4, 4]
    assert diagnostics.loss_trace and len(diagnostics.loss_trace) == 7
    for objective in objectives:
        assert objective.steps == [0] * 6 + [1]
        record = objective.update_diagnostics
        assert record["guidance_updates_attempted"] == record["guidance_updates_applied"] == 7
        assert record["latent_update_clips"] == 7
        assert record["max_proposed_update_ratio"] > objective.max_update_ratio
        assert record["max_applied_update_ratio"] == pytest.approx(objective.max_update_ratio, rel=1e-5)
    assert diagnostics.actual_attention_res == 16 and diagnostics.token_layout_verified
    assert [image.size for image in images] == [(512, 512)] * 2
    assert not torch.are_deterministic_algorithms_enabled()

    repeated, repeated_images, repeated_diagnostics, _ = _guided_run(pipe)
    assert repeated_diagnostics.final_latent_hash == diagnostics.final_latent_hash
    assert [o.update_diagnostics for o in repeated] == [o.update_diagnostics for o in objectives]
    assert [np.asarray(image).tobytes() for image in repeated_images] == [
        np.asarray(image).tobytes() for image in images]
    assert not torch.are_deterministic_algorithms_enabled()


def test_generate_guided_refuses_deterministic_algorithms(default_algorithms: None) -> None:
    torch.use_deterministic_algorithms(True)
    with pytest.raises(AssertionError, match="configure_determinism"):
        _guided_run(tiny_sana_pipe())


# --- CompositionalLoss: the T2I-CompBench++ objective on the Gemma axis ---------------------------------------


def _compositional_plan(**changes: object) -> dict:
    from ccg_diffusion.compositional.sana import SCHEMA

    plan = {"schema": SCHEMA, "condition": "ablation", "pd_indices": {"exist": [2, 4]},
            "source_guidance": {"effective_profile": "P2"},
            "token_spans": SanaWholeWordSpans.from_mapping({}, text_token_count=6).as_record(),
            "text_token_count": 6, "relations": [], "counts": [],
            "components": {"guidance_strength": True, "prefix": False, "region": True},
            "prefix_active": False, "region_active": True, "region_relation": "left",
            "region_groups": [[[2]], [[4]]]}
    return {**plan, **changes}


def test_compositional_loss_adds_the_region_loss_without_the_prefix() -> None:
    from ccg_diffusion.compositional.sana import CompositionalLoss
    from ccg_diffusion.guidance.region_assignment_loss import region_assignment_loss

    guided = CompositionalLoss(_compositional_plan(), identity="test")
    plain = CompositionalLoss(_compositional_plan(region_active=False), identity="test")
    attention = torch.rand(16, 16, 6, generator=torch.Generator().manual_seed(42)).requires_grad_()
    actual = guided(attention, 0)
    expected = plain(attention, 0) + .25 * region_assignment_loss(attention, guided.region)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ga, = torch.autograd.grad(actual, attention, retain_graph=True)
    gb, = torch.autograd.grad(expected, attention)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert (guided.scale_multiplier, guided.max_update_ratio) == (4, .06)
    record = guided.audit_record()
    assert record["region_active"] and not record["prefix_active"]
    assert set(record) == {"route", "prefix_active", "components", "relations_applied", "region_active",
                           "scale_multiplier", "max_update_ratio", "loss_trace", "count", "update_diagnostics"}
    with pytest.raises(ValueError, match="16x16"):
        guided(torch.rand(8, 8, 6), 1)


@pytest.mark.parametrize("boost", [True, False])
def test_compositional_loss_selective_guidance_boost_switch(boost: bool) -> None:
    from ccg_diffusion.compositional.sana import CompositionalLoss

    components = {"guidance_strength": boost, "prefix": False, "region": True}
    loss = CompositionalLoss(_compositional_plan(components=components), identity="test")
    assert loss.route == ("P2" if boost else "B0") and (loss.pair_spans is None) != boost
    assert (loss.scale_multiplier, loss.max_update_ratio) == ((4, .06) if boost else (1, .02))
    assert loss.region.active and loss.audit_record()["components"] == components
    final = _compositional_plan(condition="final_shape")
    for key in ("components", "region_active"):
        del final[key]
    full = CompositionalLoss(final, identity="test")
    assert full.route == "P2" and not full.region.active and set(full.audit_record()) == {
        "route", "prefix_active", "scale_multiplier", "max_update_ratio", "loss_trace", "count", "update_diagnostics"}
    with pytest.raises(ValueError, match="component switches"):
        CompositionalLoss({**final, "condition": "ablation"}, identity="t")


@pytest.mark.parametrize("objective", ["spatial_loss", "count_loss"])
def test_compositional_loss_switches_drop_only_their_objective(objective: str) -> None:
    from ccg_diffusion.compositional.sana import CompositionalLoss

    plan = _compositional_plan(
        source_guidance={"effective_profile": "P1"},
        relations=[{"label": "left", "subject_clip_index": 2, "object_clip_index": 4}],
        counts=[{"count": 2, "numeral_source": 1, "noun_source": 2, "noun_word": "cats", "noun_anchor": 2,
                 "lexical_piece_indices": [2], "text_token_count": 6}],
        components={"guidance_strength": True, "prefix": True, "region": True, objective: False},
        region_active=False, region_relation=None, region_groups=[])
    loss = CompositionalLoss(plan, identity="test")
    assert loss.route == "P1" and (loss.scale_multiplier, loss.max_update_ratio) == (1, .02)
    assert bool(loss.relations) == (objective != "spatial_loss")
    assert bool(loss.count_state.audit_record()["constraints"]) == (objective != "count_loss")
    assert loss.audit_record()["relations_applied"] == len(loss.relations)
