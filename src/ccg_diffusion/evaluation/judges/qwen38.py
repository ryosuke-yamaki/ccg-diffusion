"""Re-judge the ten TAB PP method-comparison conditions with Qwen3.8-Flash-Next (reasoning effort xhigh).

The open judge of TAB PP, beside GPT-6 Astra. The trials, images and question
adapter are taken unchanged from the GPT-6 Astra judgments
(``predictions_5seed.jsonl`` of each Astra run), so the only difference from
the Astra protocol is the judge.
The model is served by vLLM through an OpenAI-compatible
``/v1/chat/completions`` endpoint (``scripts/setup/qwen38_flash_next/serve.sh``).
The paper run used the checkpoint ``nvidia/Qwen3.8-Flash-Next-NVFP4`` at the
revision in ``PROTOCOL``. Standard library only.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ccg_diffusion.evaluation.tab import VQA_PROMPT_TEMPLATE, normalize_yes_no, render_vqa_prompt
from ccg_diffusion.provenance import repository_root

REPO_ROOT = repository_root()
Row = dict[str, Any]
DEFAULT_RUN = "20260927_tab_pp_qwen38fn_xhigh_v1"
SERVED_MODEL = "qwen3.8-flash-next"
# The ten method-comparison conditions in the row order of summary.json and scores.csv: (backbone label,
# method label, Astra run, condition). The labels are recorded values (trial plan, summary.json); display
# names are in ccg_diffusion.methods.
TABLE = (
    ("SD1.4", "Vanilla", "20260923_tab_pp_astra_high_v1", "sd14_vanilla"),
    ("SD1.4", "spaCy + PD", "20260926_tab_pp_pd_baselines_astra_v1", "sd14_pd_spacy"),
    ("SD1.4", "Gold-Setup Concatenation", "20260923_tab_pp_astra_high_v1", "sd14_gold"),
    ("SD1.4", "spaCy + PD + Gold-Setup", "20260926_tab_pp_pd_baselines_astra_v1", "sd14_pd_spacy_gold"),
    ("SD1.4", "PP-Attachment Control (ours)", "20260925_tab_pp_ablation_astra_v1", "sd14_c1r1p0m0"),
    ("Sana 0.6B", "Vanilla", "20260923_tab_pp_astra_high_v1", "sana_vanilla"),
    ("Sana 0.6B", "spaCy + PD", "20260926_tab_pp_pd_baselines_astra_v1", "sana_pd_spacy"),
    ("Sana 0.6B", "Gold-Setup Concatenation", "20260923_tab_pp_astra_high_v1", "sana_gold"),
    ("Sana 0.6B", "spaCy + PD + Gold-Setup", "20260926_tab_pp_pd_baselines_astra_v1", "sana_pd_spacy_gold"),
    ("Sana 0.6B", "PP-Attachment Control (ours)", "20260925_tab_pp_ablation_astra_v1", "sana_c1r1p0m0"),
)
# Recorded literal: its insertion order is part of summary.json.
PROTOCOL = {
    "schema": "holsd.tab_qwen38fn_vqa.v1",
    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "model_revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
    "serving": "vLLM 0.30.0, 2x DGX Spark, TP2+EP, MTP3 (MiaAI-Lab v0.30 lane)",
    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    # generation_config.json of the checkpoint (thinking-mode recommendation).
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 0.0, "repetition_penalty": 1.0},
    "max_tokens": 32768,
    "content_order": "image then text, as in the Astra request",
    "question_adapter": VQA_PROMPT_TEMPLATE,
    "answer_adapter": "leading standalone Yes/No of message content (reasoning excluded); "
                      "invalid or truncated remains in the denominator",
    "trials": "Astra predictions_5seed.jsonl of each condition, unchanged (seeds 0-4)",
}
_lock = threading.Lock()


def read_rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def local_image(value: str) -> Path:
    path = Path(value.removeprefix("/workspace/holsd/"))
    return path if path.is_absolute() else REPO_ROOT / path


def build_plan(vqa_root: Path = REPO_ROOT / "output/vqa",
               astra_runs: Mapping[str, tuple[str, str]] | None = None) -> list[Row]:
    """The 7,400 trials (10 conditions x 740) with the Astra answers, in ``TABLE`` order.

    ``astra_runs`` replaces the GPT-6 Astra judgment of a ``TABLE`` condition:
    condition -> (judgment directory under ``vqa_root``, e.g. ``RUN`` or
    ``RUN/astra``; condition ID in its ``predictions_5seed.jsonl``). An Astra
    trial ID digests the condition ID but not the run, and the raw responses are
    stored by trial ID, so conditions that share trial IDs are rejected.
    """
    overrides = dict(astra_runs or {})
    unknown = sorted(set(overrides) - {condition for *_, condition in TABLE})
    if unknown:
        raise ValueError(f"not method-comparison conditions: {unknown}")
    plan = []
    for backbone, method, run, condition in TABLE:
        run, source = overrides.get(condition, (run, condition))
        path = vqa_root / run / "predictions_5seed.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{condition}: GPT-6 Astra predictions not found: {path}; judge the condition "
                                    f"with scripts.tab_pp.judge_astra and name the judgment with --astra-run "
                                    f"{condition}=RUN[:CONDITION]")
        rows = [r for r in read_rows(path) if r["condition"] == source]
        if len(rows) != 740 or len({r["trial_id"] for r in rows}) != 740:
            raise ValueError(f"{condition}: expected 740 unique trials, got {len(rows)}")
        for r in rows:
            plan.append({
                "trial_id": r["trial_id"], "condition": condition, "backbone": backbone,
                "method": method, "tab_id": r["tab_id"], "seed": r["seed"],
                "question_setup_index": r["question_setup_index"],
                "generation_target_setup_index": r["generation_target_setup_index"],
                "reading": r["reading"], "evaluation_role": r["evaluation_role"],
                "image_path": r["image_path"], "image_sha256": r["image_sha256"],
                "official_question": r["official_question"], "astra_run": run,
                "astra_answer": r["answer"], "astra_answered_yes": r["answered_yes"],
                "astra_invalid": r["invalid_answer"],
            })
    counts = Counter(r["trial_id"] for r in plan)
    shared = sorted({r["condition"] for r in plan if counts[r["trial_id"]] > 1})
    if shared:
        raise ValueError(f"conditions {shared} share GPT-6 Astra trial IDs (the judgments use the same condition "
                         f"ID); judge them under their own condition IDs, e.g. judge_astra prepare --run-id J "
                         f"--condition sd14_pd_spacy=R1:pd_spacy:SD1.4 "
                         f"--condition sana_pd_spacy=R2:pd_spacy:Sana0.6B, then "
                         f"--astra-run sd14_pd_spacy=J --astra-run sana_pd_spacy=J")
    return plan


def request_body(row: Row, served_model: str) -> Row:
    path = local_image(row["image_path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Image changed: {path}")
    url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    return {
        "model": served_model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": render_vqa_prompt(row["official_question"])},
        ]}],
        "max_tokens": PROTOCOL["max_tokens"],
        **PROTOCOL["sampling"],
        "chat_template_kwargs": PROTOCOL["chat_template_kwargs"],
    }


def post(base_url: str, body: Row, timeout: float) -> Row:
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def decode(raw: Row) -> Row:
    """Answer of the message content (reasoning excluded), validity and token usage."""
    choice = raw["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    answer = normalize_yes_no(content)
    return {"raw_answer": content, "answer": answer, "answered_yes": answer == "yes",
            "invalid_answer": answer is None, "finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len(reasoning), "usage": raw.get("usage")}


def judge_one(row: Row, out: Path, *, base_url: str, served_model: str, timeout: float, retries: int) -> Row:
    """Judge one trial, retrying network failures with exponential backoff; write ``raw/<trial_id>.json``."""
    target = out / "raw" / (row["trial_id"] + ".json")
    body = request_body(row, served_model)
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            raw = post(base_url, body, timeout)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            detail = exc.read().decode(errors="replace")[:500] if isinstance(exc, urllib.error.HTTPError) else ""
            with _lock:
                with (out / "errors.log").open("a") as fh:
                    fh.write(f"{datetime.now(timezone.utc).isoformat()} {row['trial_id']} "
                             f"attempt={attempt} {exc!r} {detail}\n")
            if attempt == retries:
                raise
            time.sleep(min(300, 15 * 2 ** attempt))
    record = {"trial_id": row["trial_id"], "elapsed_seconds": time.monotonic() - started,
              "received_at": datetime.now(timezone.utc).isoformat(), "raw_response": raw}
    write_json(target, record)
    return record


def run(run_dir: Path, *, base_url: str, served_model: str = SERVED_MODEL, workers: int = 8,
        limit: int | None = None, timeout: float = 3600, retries: int = 4,
        astra_runs: Mapping[str, tuple[str, str]] | None = None) -> None:
    """Judge every pending trial (resumable), then ``summarize``; ``astra_runs`` as in ``build_plan``."""
    plan = build_plan(astra_runs=astra_runs)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "trial_plan.jsonl"
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in plan)
    if plan_path.exists() and plan_path.read_text() != text:
        raise ValueError("Existing trial_plan.jsonl differs from the rebuilt plan")
    plan_path.write_text(text)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError("Existing protocol.json differs from this script's PROTOCOL")
    write_json(protocol_path, PROTOCOL)
    # Deterministic interleaving so that any prefix covers all ten conditions evenly.
    order = sorted(plan, key=lambda r: hashlib.sha256(r["trial_id"].encode()).hexdigest())
    pending = [r for r in order if not (run_dir / "raw" / (r["trial_id"] + ".json")).exists()]
    if limit is not None:
        pending = pending[:limit]
    total, done, started = len(pending), 0, time.monotonic()
    print(f"{len(plan)} trials, {total} pending, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge_one, r, run_dir, base_url=base_url, served_model=served_model,
                               timeout=timeout, retries=retries): r for r in pending}
        for future in as_completed(futures):
            future.result()
            done += 1
            elapsed = time.monotonic() - started
            write_json(run_dir / "progress.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(), "completed_this_session": done,
                "pending_this_session": total - done,
                "completed_total": len(list((run_dir / "raw").glob("*.json"))), "planned_total": len(plan),
                "elapsed_seconds": elapsed, "eta_seconds": elapsed / done * (total - done)})
            if done % 20 == 0 or done == total:
                print(f"{done}/{total} {elapsed / 60:.1f} min", flush=True)
    summarize(run_dir)


def summarize(run_dir: Path) -> None:
    """Write ``predictions.jsonl``, ``summary.json`` and ``scores.csv`` from the stored raw responses."""
    plan = read_rows(run_dir / "trial_plan.jsonl")
    preds = []
    for row in plan:
        path = run_dir / "raw" / (row["trial_id"] + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            preds.append({**row, **decode(record["raw_response"]),
                          "returned_model": record["raw_response"].get("model")})
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in preds))
    table = []
    for backbone, method, _, condition in TABLE:
        rows = [p for p in preds if p["condition"] == condition]
        n = len(rows)
        yes = sum(p["answered_yes"] for p in rows)
        astra_yes = sum(p["astra_answered_yes"] for p in rows)
        tokens = [p["usage"]["completion_tokens"] for p in rows if p.get("usage")]
        table.append({
            "backbone": backbone, "method": method, "condition": condition, "n": n,
            "yes": yes, "rate_pct": 100 * yes / n if n else None,
            "invalid": sum(p["invalid_answer"] for p in rows),
            "truncated": sum(p["finish_reason"] == "length" for p in rows),
            "astra_yes_same_trials": astra_yes,
            "astra_rate_pct_same_trials": 100 * astra_yes / n if n else None,
            "agreement_with_astra_pct": 100 * sum(p["answered_yes"] == p["astra_answered_yes"]
                                                  for p in rows) / n if n else None,
            "mean_completion_tokens": sum(tokens) / len(tokens) if tokens else None,
            "max_completion_tokens": max(tokens) if tokens else None,
        })
    write_json(run_dir / "summary.json", {"protocol": PROTOCOL, "completed": len(preds),
                                          "planned": len(plan), "table": table})
    with (run_dir / "scores.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"completed {len(preds)}/{len(plan)}")
    for t in table:
        rate = "-" if t["rate_pct"] is None else f"{t['rate_pct']:.1f}"
        astra = "-" if t["astra_rate_pct_same_trials"] is None else f"{t['astra_rate_pct_same_trials']:.1f}"
        print(f"{t['backbone']:9} {t['method']:30} n={t['n']:3} qwen={rate:>5} astra={astra:>5} "
              f"invalid={t['invalid']} trunc={t['truncated']} tok={t['mean_completion_tokens']}")
