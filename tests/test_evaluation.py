"""Evaluation: statistics, T2I-CompBench++ evaluator driver and results, TAB trial plans and the two judges."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ccg_diffusion.evaluation import statistics as st
from ccg_diffusion.evaluation import tab
from ccg_diffusion.evaluation.judges import astra, qwen38
from ccg_diffusion.evaluation.t2i_compbench import dataset, export, official_evaluator as ev, results
from ccg_diffusion.provenance import digest, repository_root, sha256_file

ROOT = repository_root()


# --- statistics ------------------------------------------------------------------------------------------


def test_sign_flip_p_value_of_an_all_positive_triple() -> None:
    # All 8 sign patterns are equally likely; only the two constant ones reach |mean| = 1.
    result = st.paired_permutation_test(np.array([1.0, 1.0, 1.0]), np.random.default_rng(0))
    assert result["p"] == pytest.approx(0.25, abs=0.005) and result["wtl"] == [3, 0, 0]


def test_holm_ties_keep_input_order_and_marks() -> None:
    assert st.holm([0.01, 0.04, 0.03, 0.01]) == pytest.approx([0.04, 0.06, 0.06, 0.04])
    assert st.holm([0.5, 0.9]) == [1.0, 1.0]
    assert [st.significance_mark(p) for p in (0.0009, 0.001, 0.0099, 0.049, 0.05)] == ["***", "**", "**", "*", "n.s."]


def test_cohen_kappa_on_a_synthetic_table() -> None:
    assert st.cohen_kappa([True, True, False, False], [True, False, False, False]) == (0.75, 0.5)
    assert st.bootstrap_quantiles(list(range(1000))) == [25, 974]


def test_prompt_unit_bootstraps_use_prompt_means() -> None:
    answers = [float(i % 10 == 0) for i in range(30)]
    assert st.prompt_means(answers, expected_prompts=3) == [0.1, 0.1, 0.1]
    with pytest.raises(ValueError):
        st.prompt_means(answers, expected_prompts=4)
    summary = st.paired_prompt_bootstrap([0.1, -0.1, 0.0, 0.2], iterations=200, seed=1)
    assert summary["mean_delta"] == pytest.approx(0.05) and summary["tied_prompt_count"] == 1
    assert st.prompt_mean_bootstrap([0.5, 0.5], iterations=10, seed=1)["ci95"] == [0.5, 0.5]


# --- T2I-CompBench++ results and evaluator driver ---------------------------------------------------------


def test_recorded_work_dir_reads_the_container_path() -> None:
    assert results.recorded_work_dir("/workspace/holsd/output/vqa/R/t2icb_work/gs1_p1_r1/color") == (
        "R", "gs1_p1_r1", "color")


def test_numeric_answers_require_contiguous_question_ids(tmp_path: Path) -> None:
    path = tmp_path / "vqa_result.json"
    path.write_text(json.dumps([{"question_id": 0, "answer": "0.5"}, {"question_id": 2, "answer": "1"}]))
    with pytest.raises(dataset.T2ICompBenchError, match="contiguous"):
        results.numeric_answers(path)


def test_routing_uses_official_script_and_cwd(tmp_path: Path) -> None:
    clone, venv = tmp_path / "clone", tmp_path / "python"
    spatial = ev.build_commands("spatial", clone_dir=clone, venv_python=venv, work_dir=tmp_path / "w")
    assert [c["name"] for c in spatial] == ["spatial_2d"]
    assert spatial[0]["argv"][1].endswith("UniDet_eval/2D_spatial_eval.py")
    assert spatial[0]["cwd"].endswith("UniDet_eval") and "--complex" not in spatial[0]["argv"]
    complex_steps = ev.build_commands("complex", clone_dir=clone, venv_python=venv, work_dir=tmp_path / "w",
                                      dataset_dir=tmp_path / "prompts")
    assert [c["name"] for c in complex_steps] == ["blip_vqa", "spatial_2d", "clip_similarity", "three_in_one"]
    assert [("--complex" in c["argv"]) for c in complex_steps] == [False, True, True, False]
    assert f"--data_path={(tmp_path / 'prompts').resolve()}" in complex_steps[3]["argv"]
    assert "False" not in sum((c["argv"] for c in complex_steps), [])


def _samples(directory: Path, prompts: list[str], names: list[str] | None = None) -> Path:
    directory.mkdir(parents=True)
    for name in names or [f"{p}_{i * 10 + j:06d}.png" for i, p in enumerate(prompts) for j in range(10)]:
        (directory / name).write_bytes(b"png")
    return directory


def test_validate_samples_accepts_the_official_layout_and_rejects_gaps(tmp_path: Path) -> None:
    prompts = ["a red apple", "a blue cup"]
    assert ev.validate_samples(_samples(tmp_path / "ok", prompts), prompts)["n_images"] == 20
    names = [f"{p}_{i * 10 + j:06d}.png" for i, p in enumerate(prompts) for j in range(10)]
    with pytest.raises(dataset.T2ICompBenchError, match="contiguous"):
        ev.validate_samples(_samples(tmp_path / "gap", prompts, names[:-1] + ["a blue cup_000020.png"]), prompts)
    with pytest.raises(dataset.T2ICompBenchError, match="underscores"):
        ev.validate_samples(_samples(tmp_path / "under", prompts, ["a_red apple_000000.png"]), prompts)


def test_read_score_cross_checks_the_official_score(tmp_path: Path) -> None:
    source = ev.METRICS["color"]
    (tmp_path / Path(source.score_path).parent).mkdir(parents=True)
    (tmp_path / source.result_path).write_text(json.dumps([{"question_id": 0, "answer": "0.25"},
                                                           {"question_id": 1, "answer": "0.75"}]))
    (tmp_path / source.score_path).write_text("BLIP-VQA score: 0.5\n")
    assert ev.read_score(tmp_path, "color")["score"] == 0.5
    (tmp_path / source.score_path).write_text("BLIP-VQA score: 0.6\n")
    with pytest.raises(dataset.T2ICompBenchError, match="re-averaging"):
        ev.read_score(tmp_path, "color")


def test_curated_seal_rejects_a_mutated_result(tmp_path: Path) -> None:
    work, curated = tmp_path / "work", tmp_path / "curated"
    source = ev.METRICS["color"]
    (work / Path(source.score_path).parent).mkdir(parents=True)
    (work / source.result_path).write_text(json.dumps([{"question_id": 0, "answer": "1"}]))
    (work / source.score_path).write_text("BLIP-VQA score: 1.0\n")
    manifest = {"category": "color", "curated_artifacts": ev.curate(work, curated, "color")}
    assert [a["role"] for a in ev.validate_curated_artifact_seals(manifest, curated)] == ["result", "score"]
    (curated / "vqa_result.json").write_text(json.dumps([{"question_id": 0, "answer": "0"}]))
    with pytest.raises(dataset.T2ICompBenchError, match="sha256"):
        ev.validate_curated_artifact_seals(manifest, curated)


def test_setup_keeps_the_tracked_source_pin_and_installs_the_tracked_freeze(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.setup import setup_t2i_compbench as setup

    tracked_path = ROOT / dataset.SOURCE_PIN_PATH
    tracked = json.loads(tracked_path.read_text())
    venv = tracked["venv"]
    assert sha256_file(setup.TRACKED_FREEZE_PATH) == venv["freeze_sha256"]
    assert setup.EVAL_PYTHON_VERSION == venv["python_version"]
    frozen = setup.TRACKED_FREEZE_PATH.read_text().splitlines()
    assert sorted(set(frozen) - set(setup.freeze_requirements())) == [
        f"detectron2 @ git+{setup.DETECTRON2_REPOSITORY}@{setup.DETECTRON2_COMMIT}", *setup.TORCH_REQUIREMENTS]
    assert setup.freeze_differences("a==1\nb==2\n", "a==1\nc==3\n") == ["-c==3", "+b==2"]
    # The pin of a setup with the recorded environment equals the tracked one apart from retrieved_at.
    versions = {"python": venv["python_version"], **{key: venv[key] for key in (
        "torch", "torchvision", "torch_cuda", "detectron2", "transformers", "timm", "spacy", "numpy", "opencv",
        "diffusers", "accelerate")}}
    monkeypatch.setattr(setup, "collect_val_txt", lambda: tracked["val_txt"])
    assert setup.pin_differences({**setup.pin_payload(versions, venv["freeze_sha256"]), "retrieved_at": "now"},
                                 tracked) == []
    copy = tmp_path / "t2icb_source_pin.json"
    copy.write_bytes(tracked_path.read_bytes())
    monkeypatch.setattr(setup, "PIN_MIRROR_PATH", copy)
    setup.write_pin_mirror(versions, venv["freeze_sha256"])
    assert copy.read_bytes() == tracked_path.read_bytes()
    changed = json.loads(json.dumps(tracked))
    changed["venv"]["torch"] = "2.0.2"
    changed["weights"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(setup, "pin_payload", lambda *_: changed)
    with pytest.raises(RuntimeError, match=r"\['venv\.torch', 'weights\[0\]\.sha256'\]"):
        setup.write_pin_mirror(versions, venv["freeze_sha256"])
    assert copy.read_bytes() == tracked_path.read_bytes()


def test_local_evaluator_components_are_the_evaluation_package() -> None:
    assert all((ROOT / path).is_file() for path in ev.LOCAL_EVALUATOR_COMPONENTS)
    assert ev.resolve_categories(["all"]) == list(dataset.CATEGORIES)


def test_export_renumbers_prompts_and_locks_registered_arms(tmp_path: Path) -> None:
    from ccg_diffusion.backbones.jobs import Job, image_path

    prompts = ["a red apple", "a blue cup"]
    rows = [{"experiment": "t2icb_color", "prompt_id": f"t2icb_color_{i:04d}_s{s}", "prompt": p, "seed": s}
            for i, p in enumerate(prompts) for s in dataset.SEEDS]
    data = tmp_path / "data"
    (data / "prompts").mkdir(parents=True)
    (data / "prompts/color_val.txt").write_text("\n".join(prompts) + "\n")
    (data / "independent_seed_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    generation = tmp_path / "generation"
    for row in rows:
        job = Job(**row, attention_exist_indices=[], attention_corr_indices=[], attention_leak_indices=[],
                  attention_possession_indices=[], source="test")
        path = image_path(generation, "sd14_pd_spacy", job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row["prompt_id"].encode())
    manifest = export.export_samples("color", run_id="R", arm="gs1_p1_r1", method="sd14_pd_spacy",
                                     generation_out_dir=generation, dataset_dir=data, vqa_root=tmp_path / "vqa")
    assert [image["name"] for image in manifest["images"]][:2] == ["a red apple_000000.png", "a red apple_000001.png"]
    assert manifest["images"][10]["name"] == "a blue cup_000010.png" and manifest["dataset_id"] is None
    with pytest.raises(dataset.T2ICompBenchError, match="arm == --method"):
        export.export_samples("color", run_id="R", arm="sd14_syngen", method="sd14_pd_spacy",
                              generation_out_dir=generation, dataset_dir=data, vqa_root=tmp_path / "vqa")


# --- TAB PP trial plans and per-trial judgments -----------------------------------------------------------


def test_trial_plan_pairs_images_with_official_questions(tmp_path: Path) -> None:
    spec = {"tab_id": "tab-pp-000", "tab_category": tab.PP_CATEGORY, "ambiguous_prompt": "p",
            "targets": [{"setup_index": i, "question": f"q{i}", "visual_setup": f"v{i}", "commonsense_label": "CS"}
                        for i in (0, 1)]}
    metadata = []
    for reading, setup in (("np_attach", 0), ("vp_attach", 1)):
        for seed in tab.SEEDS:
            path = tmp_path / f"{reading}_{seed}.png"
            path.write_bytes(f"{reading}{seed}".encode())
            metadata.append({"condition": "guided", "tab_id": "tab-pp-000", "image_path": str(path),
                             "image_id": f"{reading}_s{seed}", "seed": seed, "reading": reading,
                             "target_setup_index": setup, "tab_evaluation_mode": "targeted_setup",
                             "tab_requires_reading": True, "evaluation_setup_indices": [setup]})
    trials, coverage = tab.build_trial_plan(metadata, [spec], benchmark_scope="sample",
                                            include_opposite_diagnostic=True, profile=tab.pp_profile("sana-tab-pp"))
    assert len(trials) == 20 and coverage["conditions"]["guided"]["primary_trials"] == 10
    assert [(t["image_id"], t["evaluation_role"]) for t in trials[:2]] == [
        ("np_attach_s0", "opposite_diagnostic"), ("np_attach_s0", "target")]
    target = trials[1]
    assert target["qwen_prompt"] == "q0\nAnswer with only Yes or No." and target["question_setup_index"] == 0
    assert target["trial_id"] == tab.sha256_bytes("guided\0np_attach_s0\0target\x000\0q0".encode())
    assert tab.PROMPT_TEMPLATE_SHA256.startswith("0ad22cac")
    assert [tab.normalize_yes_no(t) for t in ("Yes.", "no", "Yesterday", " NO!")] == ["yes", "no", None, "no"]


# --- GPT-6 Astra ------------------------------------------------------------------------------------------


def _response(answer: str = "Yes", status: str = "completed", model: str = "gpt-6-astra") -> dict[str, Any]:
    return {"id": "resp_fixture", "model": model, "reasoning": {"effort": "high"},
            "service_tier": "default", "status": status,
            "output": [{"type": "reasoning", "content": [{"type": "output_text", "text": "No"}]},
                       {"type": "message", "content": [{"type": "output_text", "text": answer}]}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200},
                      "output_tokens": 400, "output_tokens_details": {"reasoning_tokens": 300},
                      "total_tokens": 1400}}


def _trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trial_id: str = "fixture") -> dict[str, Any]:
    monkeypatch.setattr(astra, "REPO_ROOT", tmp_path)
    path = tmp_path / "fixture.png"
    if not path.exists():
        path.write_bytes(b"\x89PNG fixture")
    value = {"trial_id": trial_id, "condition": "sd14_gold", "tab_id": "tab-pp-001", "seed": 0,
             "question_setup_index": 0, "official_question": "Is the man holding a bag?",
             "image_path": "fixture.png", "image_sha256": sha256_file(path)}
    value["request_sha256"] = digest(astra.request_descriptor(value))
    return value


def test_frozen_protocol_and_judge_contract() -> None:
    assert digest(astra.PROTOCOL) == astra.PROTOCOL_SHA256 == (
        "4bee59de5ff0ff4725f1a93ba7d12fd7a895b16cef62405b4e9df856f6503165")
    contract = astra.judge_contract()
    assert contract["transport"] == "ccg_diffusion.evaluation.judges.astra" and astra.recorded_judge(contract)
    with pytest.raises(ValueError, match="pinned contract"):
        astra.recorded_judge({**contract, "default_workers": 8})


def test_request_sends_the_image_and_official_question_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trial = _trial(tmp_path, monkeypatch)
    trial.update(visual_setup="SECRET GOLD", prompt="GENERATION PROMPT")
    body = astra.request_body(trial)
    content = body["input"][0]["content"]
    assert content[0]["detail"] == "original" and content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["text"] == trial["official_question"] + "\nAnswer with only Yes or No."
    assert "SECRET" not in json.dumps(body) and "GENERATION PROMPT" not in json.dumps(body)
    (tmp_path / "fixture.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        astra.request_body(trial)


def test_cost_answer_adapter_and_drift() -> None:
    usage = astra.usage_cost(_response()["usage"])
    assert usage["cost_usd"] == pytest.approx(.0282) and usage["other_output_tokens"] == 100
    bad = {**_response()["usage"], "total_tokens": 1700}
    with pytest.raises(ValueError):
        astra.usage_cost(bad)
    assert astra.decode_response(_response("Yes."))["answered_yes"] is True
    assert astra.decode_response(_response("Maybe yes"))["invalid_answer"] is True
    truncated = astra.decode_response(_response("Yes", "incomplete"))
    assert truncated["answer"] is None and truncated["usage"]["cost_usd"] > 0
    with pytest.raises(ValueError, match="returned model"):
        astra.decode_response(_response(model="gpt-5"))
    with pytest.raises(ValueError, match="reasoning effort"):
        astra.decode_response({**_response(), "reasoning": {"effort": "low"}})


def _paid_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, fail_at: int | None = None
              ) -> tuple[Path, dict[str, Any]]:
    first = _trial(tmp_path, monkeypatch)
    rows = [{**first, "trial_id": f"trial_{i:03d}"} for i in range(count)]
    plan = {"protocol_sha256": digest(astra.PROTOCOL)}
    output = tmp_path / "run"
    astra.save(output / "preparation.json", plan)
    monkeypatch.setattr(astra, "preflight", lambda *_a, **_k: (plan, rows))
    stats = {"calls": 0, "active": 0, "max_active": 0}
    lock = threading.Lock()

    def create(**kwargs: Any) -> Any:
        assert kwargs["extra_body"] == {"service_tier": "default"}
        with lock:
            stats["calls"] += 1
            number = stats["calls"]
            stats["active"] += 1
            stats["max_active"] = max(stats["active"], stats["max_active"])
        try:
            time.sleep(.01)
            if number == fail_at:
                raise TimeoutError("never serialized")
            raw = _response()
            parsed = SimpleNamespace(model_dump=lambda **_: raw, _request_id=f"req_{number}")
            return SimpleNamespace(headers={"authorization": "SECRET"}, parse=lambda: parsed)
        finally:
            with lock:
                stats["active"] -= 1

    session = SimpleNamespace(models=SimpleNamespace(retrieve=lambda _: SimpleNamespace(id="gpt-6-astra")),
                              responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)),
                              close=lambda: None)
    monkeypatch.setattr(astra, "client", lambda: session)
    return output, stats


def test_budget_reserves_every_in_flight_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, stats = _paid_run(tmp_path, monkeypatch, count=6)
    # Enough for six measured calls, but only one worst-case request may be in flight.
    result = astra.run(output, workers=8, budget_usd=2, launch_interval=0)
    assert result["full_complete"] and stats["calls"] == 6 and stats["max_active"] == 1
    astra.run(output, workers=8, budget_usd=2, launch_interval=0)
    assert stats["calls"] == 6


def test_budget_stop_happens_before_any_paid_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, stats = _paid_run(tmp_path, monkeypatch, count=3)
    with pytest.raises(RuntimeError, match="Budget reserve"):
        astra.run(output, workers=8, budget_usd=1, launch_interval=0)
    assert stats["calls"] == 0


def test_failed_request_drains_and_is_never_resent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, stats = _paid_run(tmp_path, monkeypatch, count=12, fail_at=4)
    with pytest.raises(RuntimeError, match="Request failed"):
        astra.run(output, workers=4, budget_usd=10, launch_interval=0)
    calls = stats["calls"]
    assert stats["active"] == 0 and len(list((output / "errors").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="Unresolved API attempt"):
        astra.run(output, workers=4, budget_usd=10, launch_interval=0)
    assert stats["calls"] == calls
    assert all("SECRET" not in p.read_text() for p in (output / "responses").glob("*.json"))


def test_replay_rederives_predictions_from_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_trial(tmp_path, monkeypatch, f"t{i}") for i in (2, 0, 1)]
    for row in rows:
        row["seed"] = int(row["trial_id"][1])
        row["request_sha256"] = digest(astra.request_descriptor(row))
    output = tmp_path / "judged"
    astra.save_rows(output / "trial_plan.jsonl", rows)
    astra.save(output / "preparation.json", {"protocol": astra.PROTOCOL, "protocol_sha256": digest(astra.PROTOCOL),
                                             "pricing": astra.PRICING, "trials": 3,
                                             "trial_plan_sha256": sha256_file(output / "trial_plan.jsonl")})
    for row in rows:
        raw = _response("No" if row["trial_id"] == "t1" else "Yes")
        astra.save(output / "responses" / f"{row['trial_id']}.json", {
            "trial": row, "protocol_sha256": digest(astra.PROTOCOL), "raw_response": raw,
            "raw_response_sha256": digest(raw), "result": astra.decode_response(raw)})
    replayed = astra.replay(output, seed_count=2)
    assert [r["trial_id"] for r in replayed] == ["t0", "t1"] and [r["answer"] for r in replayed] == ["yes", "no"]
    record = json.loads((output / "responses/t0.json").read_text())
    record["raw_response"]["output"][1]["content"][0]["text"] = "No"
    (output / "responses/t0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Raw response changed"):
        astra.replay(output)


# --- Qwen3.8-Flash-Next -----------------------------------------------------------------------------------


def test_qwen_decodes_answers() -> None:
    raw = {"choices": [{"message": {"content": "Yes, it is.", "reasoning_content": "thinking"},
                        "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}
    assert qwen38.decode(raw) == {"raw_answer": "Yes, it is.", "answer": "yes", "answered_yes": True,
                                  "invalid_answer": False, "finish_reason": "stop", "reasoning_chars": 8,
                                  "usage": {"completion_tokens": 9}}
    truncated = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    assert qwen38.decode(truncated)["invalid_answer"] is True
