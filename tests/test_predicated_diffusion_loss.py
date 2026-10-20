"""Predicated Diffusion loss L_PD (existence, correspondence, leakage and possession terms), whole-word pooling,
attention capture, the precision/determinism contract and the capped latent update."""
from __future__ import annotations

import os

import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from ccg_diffusion import config
from ccg_diffusion.guidance.attention import (
    ATTENTION_PLACES,
    AttentionStore,
    SelectiveCrossAttnProcessor,
    aggregate_attention_batched,
    register_attention_control,
)
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics, update_latents_batched
from ccg_diffusion.guidance.precision import PROFILES, configure_determinism
from ccg_diffusion.guidance.predicated_diffusion_loss import (
    PAIR_BOUNDARY_MARGIN,
    PD_LEAK_WEIGHT,
    SanaWholeWordSpans,
    WholeWordSpans,
    compute_pd_loss,
)
from ccg_diffusion.targets.pd_indices import PDIndices


def _maps(seed: int = 0, tokens: int = 10) -> torch.Tensor:
    return torch.rand(4, 4, tokens, generator=torch.Generator().manual_seed(seed)).requires_grad_()


def _minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min())


def _barrier(product: torch.Tensor) -> torch.Tensor:
    return -torch.log(1 - (1 - PAIR_BOUNDARY_MARGIN) * product.clamp(0, 1))


# --- the four terms in closed form (gamma = 0.1, 1, 0.3, 1) -------------------------------------------


def test_existence_term_closed_form() -> None:
    maps = _maps()
    loss = compute_pd_loss(maps, PDIndices(exist=[2, 5]), attention_res=4)
    softmax = torch.softmax(maps[:, :, 1:-1].detach() * 100, dim=-1)
    expected = 0.0
    for index in (2, 5):
        log_term = torch.log(1 - softmax[:, :, index - 1] + 1e-4).sum() / 16
        expected += 0.1 * -torch.log(1 - torch.exp(log_term)) / 2
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-5, atol=1e-7)


def test_pair_terms_closed_form_and_additivity() -> None:
    maps = _maps()
    a, b = _minmax(maps[:, :, 3].detach()), _minmax(maps[:, :, 6].detach())
    corr = (_barrier(b * (1 - a)) + _barrier(a * (1 - b))).sum() / 16
    leak = PD_LEAK_WEIGHT * _barrier(a * b).sum() / 16
    possession = _barrier(b * (1 - a)).sum() / 16
    for field, expected in (("corr", corr), ("leak", leak), ("possession", possession)):
        loss = compute_pd_loss(maps, PDIndices(**{field: [[3, 6]]}), attention_res=4)
        torch.testing.assert_close(loss.detach(), expected, rtol=1e-6, atol=1e-7)
    full = PDIndices(exist=[3], corr=[[3, 6]], leak=[[3, 6]], possession=[[3, 6]])
    parts = sum(compute_pd_loss(maps, PDIndices(**part), attention_res=4) for part in (
        {"exist": [3]}, {"corr": [[3, 6]]}, {"leak": [[3, 6]]}, {"possession": [[3, 6]]}))
    torch.testing.assert_close(compute_pd_loss(maps, full, attention_res=4), parts)


def test_barrier_is_finite_at_zero_one_maps_in_low_precision() -> None:
    master = torch.zeros(2, 2, 4, requires_grad=True)
    with torch.no_grad():
        master[:, :, 1] = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        master[:, :, 2] = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    loss = compute_pd_loss(master.to(torch.float16), PDIndices(exist=[1], leak=[[1, 2]], corr=[[1, 2]]),
                           attention_res=2)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    (gradient,) = torch.autograd.grad(loss, master)
    assert torch.isfinite(gradient).all()


def test_empty_targets_and_guards() -> None:
    assert compute_pd_loss(_maps(), PDIndices(), attention_res=4).item() == 0.0
    flat = torch.zeros(2, 2, 4)
    flat[:, :, 1] = 0.25
    flat[:, :, 2] = torch.tensor([[0.25, 0.25], [0.25, 0.25 + 1e-9]])
    guards: dict[str, int] = {}
    loss = compute_pd_loss(flat, PDIndices(leak=[[1, 2]]), attention_res=2, guard_counts=guards)
    assert torch.isfinite(loss) and guards == {"near_constant_attention_maps": 2}
    with pytest.raises(ValueError, match="special_layout"):
        compute_pd_loss(flat, PDIndices(exist=[1]), special_layout="t5")


def test_existence_window_clip_and_gemma_layouts() -> None:
    maps = _maps()
    indices = PDIndices(exist=[2, 5])
    changed_eos = maps.detach().clone()
    changed_eos[:, :, -1] += 5.0
    changed_bos = maps.detach().clone()
    changed_bos[:, :, 0] += 5.0
    clip = compute_pd_loss(maps, indices, attention_res=4)
    gemma = compute_pd_loss(maps, indices, attention_res=4, special_layout="leading_bos")
    assert torch.equal(clip, compute_pd_loss(changed_eos, indices, attention_res=4))
    assert torch.equal(clip, compute_pd_loss(changed_bos, indices, attention_res=4))
    assert not torch.equal(gemma, compute_pd_loss(changed_eos, indices, attention_res=4, special_layout="leading_bos"))
    assert torch.equal(gemma, compute_pd_loss(changed_bos, indices, attention_res=4, special_layout="leading_bos"))


# --- whole-word pooling -----------------------------------------------------------------------------------


def _spans() -> WholeWordSpans:
    return WholeWordSpans.from_mapping({3: (1, 2, 3), 6: (5, 6), 8: (8,)}, text_token_count=10)


def test_singleton_spans_equal_the_default_loss_and_gradient() -> None:
    maps = _maps()
    indices = PDIndices(exist=[3, 6], corr=[[3, 6]], leak=[[3, 8]], possession=[[3, 6]])
    default = compute_pd_loss(maps, indices, attention_res=4)
    singleton = compute_pd_loss(maps, indices, attention_res=4, pair_token_spans=WholeWordSpans.from_mapping(
        {3: (3,), 6: (6,), 8: (8,)}, text_token_count=10))
    assert torch.equal(default, singleton)
    assert torch.equal(torch.autograd.grad(default, maps, retain_graph=True)[0],
                       torch.autograd.grad(singleton, maps)[0])


def test_pooling_changes_only_the_correspondence_and_leak_terms() -> None:
    maps = _maps()
    pairs = PDIndices(corr=[[3, 6]], leak=[[3, 8]])
    others = PDIndices(exist=[3, 6], possession=[[3, 6]])
    everything = PDIndices(exist=others.exist, corr=pairs.corr, leak=pairs.leak, possession=others.possession)
    residual = (compute_pd_loss(maps, everything, attention_res=4, pair_token_spans=_spans())
                - compute_pd_loss(maps, pairs, attention_res=4, pair_token_spans=_spans()))
    torch.testing.assert_close(residual, compute_pd_loss(maps, others, attention_res=4), rtol=1e-6, atol=1e-6)
    pooled = compute_pd_loss(maps, PDIndices(corr=[[3, 6]]), attention_res=4, pair_token_spans=(
        WholeWordSpans.from_mapping({3: (1, 2, 3), 6: (5, 6)}, text_token_count=10)))
    gradient = torch.autograd.grad(pooled, maps)[0]
    assert all(gradient[:, :, index].abs().sum() > 0 for index in (1, 2, 3, 5, 6))
    assert gradient[:, :, [0, 4, 7, 8, 9]].count_nonzero() == 0
    guards: dict[str, int] = {}
    compute_pd_loss(maps, pairs, attention_res=4, pair_token_spans=_spans(), guard_counts=guards)
    assert guards == {"whole_word_pair_loss_calls": 1, "whole_word_multi_piece_map_reductions": 2}


@pytest.mark.parametrize(("spans", "count"), [
    (((3, (0, 3)),), 10), (((3, (3, 9)),), 10), (((3, (3, 2)),), 10), (((3, (2,)),), 10),
    (((3, (2, 3)), (4, (3, 4))), 10), (((6, (6,)), (3, (3,))), 10), (((3, (3,)),), 78), (((3, (True, 3)),), 10),
])
def test_invalid_spans_fail_closed(spans: object, count: object) -> None:
    with pytest.raises(ValueError):
        WholeWordSpans(text_token_count=count, spans=spans)


def test_span_coverage_and_layout_are_required() -> None:
    maps = _maps()
    with pytest.raises(ValueError, match="cover exactly"):
        compute_pd_loss(maps, PDIndices(corr=[[3, 6]]), attention_res=4,
                        pair_token_spans=WholeWordSpans.from_mapping({3: (2, 3)}, text_token_count=10))
    with pytest.raises(ValueError, match="CLIP layout or Gemma layout"):
        compute_pd_loss(maps, PDIndices(corr=[[3, 6]]), attention_res=4, special_layout="leading_bos",
                        pair_token_spans=WholeWordSpans.from_mapping({3: (3,), 6: (6,)}, text_token_count=10))
    gemma = SanaWholeWordSpans.from_mapping({3: (2, 3), 6: (6,)}, text_token_count=10)
    loss = compute_pd_loss(maps, PDIndices(corr=[[3, 6]]), attention_res=4, special_layout="leading_bos",
                           pair_token_spans=gemma)
    assert torch.isfinite(loss) and gemma.as_record()["spans"][0] == {"anchor": 3, "lexical_piece_indices": [2, 3]}


# --- attention capture and aggregation ---------------------------------------------------------------------


def _attention_module() -> Attention:
    torch.manual_seed(0)
    return Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4)


def test_selective_processor_captures_fp32_probabilities_and_matches_sdpa() -> None:
    store = AttentionStore()
    processor = SelectiveCrossAttnProcessor(store, "up", attention_res=2, processor_name="up.attn2")
    attn = _attention_module()
    hidden, context = torch.randn(1, 4, 8), torch.randn(1, 6, 12)
    output = processor(attn, hidden, encoder_hidden_states=context)
    store.finish_selective_forward()
    torch.testing.assert_close(output, AttnProcessor2_0()(attn, hidden, encoder_hidden_states=context))
    captured = store.attention_store["up_cross"][0]
    assert captured.dtype == torch.float32 and captured.shape == (2, 4, 6)
    assert store.last_capture_processor_names == ("up.attn2",)


def test_selective_processor_delegates_when_capture_is_off_or_resolution_differs() -> None:
    attn = _attention_module()
    context = torch.randn(1, 6, 12)
    for enabled, pixels in ((False, 4), (True, 9)):
        store = AttentionStore()
        store.set_capture_enabled(enabled)
        hidden = torch.randn(1, pixels, 8)
        output = SelectiveCrossAttnProcessor(store, "up", attention_res=2)(attn, hidden, encoder_hidden_states=context)
        assert torch.equal(output, AttnProcessor2_0()(attn, hidden, encoder_hidden_states=context))
        assert store.step_store["up_cross"] == []


def test_registration_keeps_self_attention_processors() -> None:
    self_processor, cross_processor, other = object(), object(), object()

    class FakeUNet:
        attn_processors = {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": self_processor,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": cross_processor,
            "conv_in.processor": other,
        }

        def set_attn_processor(self, processors: dict) -> None:
            self.attn_processors = dict(processors)

    unet = FakeUNet()
    register_attention_control(unet, AttentionStore())
    assert unet.attn_processors["down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"] is self_processor
    selected = unet.attn_processors["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"]
    assert isinstance(selected, SelectiveCrossAttnProcessor) and selected.place_in_unet == "down"
    assert "conv_in.processor" not in unet.attn_processors


def test_aggregation_averages_heads_then_layers_in_place_order() -> None:
    assert ATTENTION_PLACES == ("up", "down", "mid")
    generator = torch.Generator().manual_seed(1)
    layers = {place: torch.rand(2 * 4, 16, 5, generator=generator) for place in ("mid", "down", "up")}
    store = AttentionStore()
    for place in ("mid", "down", "up"):  # stored out of aggregation order
        store.forward(layers[place], True, place)
    store.forward(torch.rand(2 * 4, 4, 5), True, "up")  # another resolution: skipped
    store.finish_selective_forward()
    maps = aggregate_attention_batched(store, res=4, batch_size=2)
    per_track = {place: layers[place].reshape(2, 4, 4, 4, 5).mean(dim=1) for place in layers}
    expected = (per_track["up"] + per_track["down"] + per_track["mid"]) / 3
    assert len(maps) == 2 and all(torch.equal(maps[b], expected[b]) for b in range(2))
    with pytest.raises(ValueError, match="not divisible"):
        aggregate_attention_batched(store, res=4, batch_size=3)
    with pytest.raises(RuntimeError):
        aggregate_attention_batched(store, res=8, batch_size=2)


# --- precision and determinism -----------------------------------------------------------------------------


def test_bf16_stable_profile_uses_the_loss_boundary_margin() -> None:
    assert PROFILES["bf16_stable"].pair_boundary_margin == PAIR_BOUNDARY_MARGIN


def test_determinism_record_equals_the_recorded_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda flag: calls.append(("deterministic", flag)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: calls.append(value))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unset")
    for name in ("benchmark", "deterministic", "allow_tf32"):
        monkeypatch.setattr(torch.backends.cudnn, name, getattr(torch.backends.cudnn, name))
    for name in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        monkeypatch.setattr(torch.backends.cuda.matmul, name, getattr(torch.backends.cuda.matmul, name))
    record = configure_determinism(PROFILES["bf16_stable"])
    assert record == dict(config.SD14_DETERMINISM)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == ["highest", ("deterministic", True), ("seed", 0), "highest"]


# --- the capped latent update -------------------------------------------------------------------------------


def _ratio(updated: torch.Tensor, latents: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(updated - latents.detach()) / torch.linalg.vector_norm(latents.detach()))


def test_update_caps_the_relative_l2_step_per_image() -> None:
    latents = torch.tensor([[2.0, -1.0], [30.0, -40.0]], requires_grad=True)
    diagnostics = [GenerationDiagnostics(), GenerationDiagnostics()]
    updated = update_latents_batched(latents, [1e6 * latents[0].sum(), 1e6 * latents[1].sum()], 20.0,
                                     diagnostics=diagnostics, strict_finite=True, max_update_ratio=0.02,
                                     max_latent_abs=None)
    assert [record.max_pre_update_latent_abs for record in diagnostics] == [pytest.approx(2.0), pytest.approx(40.0)]
    for index, record in enumerate(diagnostics):
        assert _ratio(updated[index], latents[index]) == pytest.approx(0.02, rel=1e-5)
        assert record.latent_update_clips == 1 and record.guidance_updates_applied == 1
        assert record.min_update_clip_factor < 1e-6


def test_update_uses_per_image_multipliers_caps_and_skips_zero_losses() -> None:
    latents = torch.ones(3, 4, requires_grad=True)
    diagnostics = [GenerationDiagnostics() for _ in range(3)]
    diagnostics[0].semantic_guidance = {"update_step_size_trajectory": [], "update_cap_trajectory": []}
    losses = [1e3 * latents[0].sum(), 1e3 * latents[1].sum(), latents[2].sum() * 0.0]
    updated = update_latents_batched(latents, losses, 20.0, diagnostics=diagnostics, strict_finite=True,
                                     max_update_ratio=0.02, max_latent_abs=20.0, step_size_multipliers=[1.0, 4.0, 1.0],
                                     per_job_max_update_ratios=[0.02, 0.06, 0.02])
    assert _ratio(updated[0], latents[0]) == pytest.approx(0.02, rel=1e-5)
    assert _ratio(updated[1], latents[1]) == pytest.approx(0.06, rel=1e-5)
    assert torch.equal(updated[2], latents[2].detach()) and diagnostics[2].zero_loss_skips == 1
    assert diagnostics[0].semantic_guidance == {"update_step_size_trajectory": [20.0], "update_cap_trajectory": [0.02]}
    with pytest.raises(ValueError, match="step_size_multipliers"):
        update_latents_batched(latents, losses, 20.0, diagnostics=diagnostics, strict_finite=True,
                               max_update_ratio=0.02, max_latent_abs=20.0, step_size_multipliers=[1.0, 0.0, 1.0])


def test_update_backtracks_to_the_absolute_latent_bound() -> None:
    latents = torch.tensor([[19.9]], requires_grad=True)
    diagnostics = [GenerationDiagnostics()]
    updated = update_latents_batched(latents, [-latents[0].sum()], 1.0, diagnostics=diagnostics, strict_finite=True,
                                     max_update_ratio=None, max_latent_abs=20.0)
    assert updated.item() == pytest.approx(19.9 + 0.0625)
    assert diagnostics[0].backtracking_updates == 1 and diagnostics[0].latent_bound_violations == 4
