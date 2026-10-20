"""GPT-6 Astra judgment of TAB PP generations under the frozen request protocol.

One trial is one generated image plus one official TAB intention question,
sent as an independent Responses API request (``PROTOCOL``; its sha256 is
``PROTOCOL_SHA256``). A judgment run directory holds:

* ``preparation.json`` and ``trial_plan.jsonl``: written by ``prepare`` from
  the trial plans of one or more generation runs (offline);
* ``attempts/``, ``raw/``, ``responses/``, ``errors/``, ``executions/``: the
  receipts of ``run``, the only function that sends paid requests. It needs a
  dollar budget, reserves the worst-case cost of every in-flight request and
  never resends an unresolved request automatically;
* ``predictions.jsonl`` / ``predictions_<n>seed.jsonl`` and
  ``summary_<n>seed.json``: the judgments and Yes rates.

``replay`` re-derives the predictions of a finished run from its stored
receipts without any network access; it is how the paper's judgments are
checked at a later commit.
"""
from __future__ import annotations

import base64
import csv
import fcntl
import importlib.metadata
import json
import math
import platform
import re
import signal
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ccg_diffusion import config
from ccg_diffusion.evaluation import tab
from ccg_diffusion.provenance import (
    atomic_write_json, atomic_write_jsonl, digest, git_head, repository_root, require_committed, sha256_file,
    sha256_text,
)

REPO_ROOT = repository_root()
ASTRA_JUDGE = config.TAB_JUDGE
DEFAULT_WORKERS = 16
# The model every paper judgment returned.
RETURNED_MODEL = "gpt-6-astra"
PROTOCOL = {
    "schema": "holsd.tab_astra_vqa.v1", "model": "gpt-6-astra", "reasoning": {"effort": "high"},
    "service_tier": "default", "image_detail": "original", "image_size": [512, 512],
    "max_output_tokens": 32768, "store": False, "tools": [],
    "question_adapter": "official question + newline + Answer with only Yes or No.",
    "answer_adapter": "existing TAB leading standalone Yes/No; invalid remains in denominator",
    "unit": "one image + one official target question per independent Responses API request",
    "metric": "Yes / all primary image-question trials; no opposite diagnostics",
    "model_version": "official alias; record returned model for every response; no silent substitution",
    "seeds_4": [0, 1, 2, 3], "seeds_5": [0, 1, 2, 3, 4],
}
PROTOCOL_SHA256 = "4bee59de5ff0ff4725f1a93ba7d12fd7a895b16cef62405b4e9df856f6503165"
PRICING = {
    "currency": "USD", "per_tokens": 1000000, "input": 10.0, "cached_input": 1.0,
    "cache_write": 12.5, "output_including_reasoning": 50.0,
    "service": "Standard, short context, OpenAI direct API; no regional uplift or taxes",
    "checked_on": "2026-09-23", "source": "https://developers.openai.com/api/docs/pricing",
}
SOURCE_FILES = (
    "src/ccg_diffusion/evaluation/judges/astra.py", "src/ccg_diffusion/evaluation/tab.py",
    "src/ccg_diffusion/provenance.py", "scripts/tab_pp/judge_astra.py",
)
TRIAL_FIELDS = ("tab_id", "seed", "question_setup_index", "target_setup_index", "reading", "official_question",
                "official_question_sha256", "image_sha256", "generation_target_setup_index", "commonsense_label")
PANEL = 740
# Worst-case cost and token use of one in-flight request, reserved before it is sent.
RESERVE_USD = (4096 * PRICING["cache_write"] + PROTOCOL["max_output_tokens"] * PRICING[
    "output_including_reasoning"]) / 1e6
RATE_TOKEN_RESERVE = 4096 + PROTOCOL["max_output_tokens"]
Row = dict[str, Any]


@dataclass(frozen=True)
class ConditionSource:
    """One judged condition: the rows of ``source_condition`` in a generation run's ``trial_plan``."""

    id: str
    source_run: str
    source_condition: str
    trial_plan: str
    backbone: str | None = None


def judge_contract() -> Row:
    """The judge record of a judgment run's ``preparation.json``."""
    return {"judge_profile": ASTRA_JUDGE, "model": PROTOCOL["model"], "reasoning": PROTOCOL["reasoning"],
            "protocol_sha256": digest(PROTOCOL), "transport": "ccg_diffusion.evaluation.judges.astra",
            "default_workers": DEFAULT_WORKERS}


def recorded_judge(record: Row) -> str:
    """Name of a recorded judge that still equals its pinned contract; fail closed on drift."""
    if not isinstance(record, dict) or record.get("judge_profile") != ASTRA_JUDGE or record != judge_contract():
        raise ValueError("Recorded TAB judge differs from its pinned contract")
    return ASTRA_JUDGE


# --- files of a judgment run -------------------------------------------------------------------------


def read_rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to overwrite a different artifact: {path}")
    else:
        atomic_write_json(path, value, overwrite=False)


def save_rows(path: Path, rows: list[Row]) -> None:
    if path.exists():
        if read_rows(path) != rows:
            raise ValueError(f"Refusing to overwrite a different ledger: {path}")
    else:
        atomic_write_jsonl(path, rows, overwrite=False)


def local_image(value: str) -> Path:
    """Repository path of a recorded image path (historical records use ``/workspace/holsd/...``)."""
    path = Path(value.removeprefix("/workspace/holsd/"))
    result = path if path.is_absolute() else REPO_ROOT / path
    result.resolve().relative_to(REPO_ROOT.resolve())
    return result


def output_path(vqa: Path) -> Path:
    """Judgment directory of one generation run's own trial plan: ``output/vqa/<run>/astra``."""
    return vqa / "astra"


def _run_id(output: Path) -> str:
    return output.parent.name if output.name == "astra" else output.name


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[0-9]{8}_[a-z0-9][a-z0-9_-]*", run_id):
        raise ValueError("Run ID must match YYYYMMDD_lowercase-slug")
    return run_id


@contextmanager
def locked(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


# --- request, cost and answer --------------------------------------------------------------------------


def request_descriptor(row: Row, protocol: Row = PROTOCOL) -> Row:
    """What identifies a request: protocol, image bytes (sha256) and rendered question."""
    return {"protocol": protocol, "image_sha256": row["image_sha256"],
            "text": tab.render_vqa_prompt(row["official_question"])}


def request_body(row: Row, protocol: Row = PROTOCOL) -> Row:
    path = local_image(row["image_path"])
    if sha256_file(path) != row["image_sha256"]:
        raise ValueError("Image changed before submission")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "model": protocol["model"], "reasoning": deepcopy(protocol["reasoning"]),
        "max_output_tokens": protocol["max_output_tokens"], "store": False, "tools": [],
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64," + encoded,
             "detail": protocol["image_detail"]},
            {"type": "input_text", "text": tab.render_vqa_prompt(row["official_question"])},
        ]}],
    }


def usage_cost(usage: Row, pricing: Row = PRICING) -> Row:
    """Reasoning is a subset of output_tokens, never an additional charge."""
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    written = usage.get("input_tokens_details", {}).get("cache_write_tokens", 0)
    reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    values = (inputs, outputs, cached, written, reasoning)
    if any(type(v) is not int or v < 0 for v in values) or cached + written > inputs or reasoning > outputs:
        raise ValueError("Invalid API token usage")
    if usage["total_tokens"] != inputs + outputs:
        raise ValueError("API total_tokens differs from input + output")
    actual = ((inputs - cached - written) * pricing["input"] + cached * pricing["cached_input"]
              + written * pricing["cache_write"] + outputs * pricing["output_including_reasoning"]) / 1e6
    uncached = (inputs * pricing["input"] + outputs * pricing["output_including_reasoning"]) / 1e6
    return {"input_tokens": inputs, "cached_input_tokens": cached, "cache_write_tokens": written,
            "output_tokens": outputs, "reasoning_tokens": reasoning,
            "other_output_tokens": outputs - reasoning, "total_tokens": inputs + outputs,
            "cost_usd": actual, "cost_without_cache_usd": uncached}


def decode_response(raw: Row) -> Row:
    """Answer, validity and cost of one stored Responses API reply."""
    model = raw.get("model", "")
    if model != PROTOCOL["model"] and not model.startswith(PROTOCOL["model"] + "-"):
        raise ValueError(f"Unexpected returned model: {model}")
    if raw.get("reasoning", {}).get("effort") != "high":
        raise ValueError("Response did not confirm high reasoning effort")
    if raw.get("service_tier") not in ("default", "standard"):
        raise ValueError("Response did not confirm Standard service pricing")
    text = "".join(part["text"] for item in raw.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")
    completed = raw.get("status") == "completed"
    answer = tab.normalize_yes_no(text) if completed else None
    return {"response_id": raw["id"], "returned_model": model, "status": raw["status"],
            "raw_answer": text, "answer": answer, "answered_yes": answer == "yes",
            "invalid_answer": answer is None, "incomplete_details": raw.get("incomplete_details"),
            "usage": usage_cost(raw["usage"])}


# --- preparation (offline) -------------------------------------------------------------------------------


def prepare(output: Path, conditions: Sequence[ConditionSource], *, run_id: str | None = None) -> Row:
    """Seal the official target trials of ``conditions`` into ``output``; offline.

    Every item/setup/seed is judged under all conditions back to back, so judge
    drift over time is not confounded with a condition.
    """
    from PIL import Image

    require_committed(SOURCE_FILES)
    verification = tab.verify_dataset()
    items = tab.load_pp_items()
    ids = [c.id for c in conditions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Conditions are missing or duplicated")
    paths = {str(tab.ITEMS.relative_to(REPO_ROOT)): sha256_file(tab.ITEMS)}
    sources: dict[str, list[Row]] = {}
    image_hashes: dict[str, str] = {}
    trials: list[Row] = []
    records: list[Row] = []
    for condition in conditions:
        if condition.trial_plan not in sources:
            paths[condition.trial_plan] = sha256_file(REPO_ROOT / condition.trial_plan)
            sources[condition.trial_plan] = read_rows(REPO_ROOT / condition.trial_plan)
        panel = [r for r in sources[condition.trial_plan] if r["tab_id"] in items
                 and r["condition"] == condition.source_condition and r["paper_primary_metric_member"]]
        tab.validate_panel(panel, sorted(items), list(range(5)))
        vanilla = panel[0]["tab_evaluation_mode"] == "ambiguous_both_setups"
        physical, pairs = set(), {}
        for row in sorted(panel, key=tab.trial_key):
            question = items[row["tab_id"]]["targets"][row["question_setup_index"]]["question"]
            if (row["official_question"] != question or row["evaluation_role"] != "target"
                    or row["official_question_sha256"] != sha256_text(question)):
                raise ValueError("Official question or target role differs from TAB")
            if vanilla:
                if (row["tab_evaluation_mode"] != "ambiguous_both_setups"
                        or row["generation_target_setup_index"] is not None):
                    raise ValueError("Ambiguous generations must use both official questions for one image")
            elif (row["tab_evaluation_mode"] != "targeted_setup"
                  or row["generation_target_setup_index"] != row["question_setup_index"]):
                raise ValueError("Generation target does not match the official question")
            relative = str(local_image(row["image_path"]).relative_to(REPO_ROOT))
            if not relative.startswith(f"output/generation/{condition.source_run}/"):
                raise ValueError(f"Image outside its generation run: {relative}")
            if relative not in image_hashes:
                with Image.open(REPO_ROOT / relative) as image:
                    image.verify()
                with Image.open(REPO_ROOT / relative) as image:
                    if image.format != "PNG" or image.size != (512, 512):
                        raise ValueError("Expected an unchanged 512x512 PNG")
                image_hashes[relative] = sha256_file(REPO_ROOT / relative)
            if image_hashes[relative] != row["image_sha256"]:
                raise ValueError(f"Generated image hash mismatch: {relative}")
            # Bound the text payload used for conservative pre-request budget reservation.
            if len(question.encode()) > 1024:
                raise ValueError("Unexpectedly long official question")
            physical.add(relative)
            pairs.setdefault((row["tab_id"], row["seed"]), set()).add(relative)
            trial = {k: row.get(k) for k in TRIAL_FIELDS}
            trial.update(condition=condition.id, image_path=relative, source_trial_id=row["trial_id"],
                         evaluation_role="target")
            if condition.backbone is not None:
                trial["backbone"] = condition.backbone
            trial["trial_id"] = digest([condition.id, *tab.trial_key(trial)])
            trial["request_sha256"] = digest(request_descriptor(trial))
            trials.append(trial)
        if len(physical) != (PANEL // 2 if vanilla else PANEL):
            raise ValueError("Unexpected physical image count")
        if vanilla and any(len(v) != 1 for v in pairs.values()):
            raise ValueError("Both questions of an ambiguous generation must reference one image")
        records.append({**asdict(condition), "vanilla": vanilla, "physical_images": len(physical)})
    if len(image_hashes) != sum(c["physical_images"] for c in records):
        raise ValueError("Conditions share physical images")
    trials.sort(key=lambda t: (tab.trial_key(t), ids.index(t["condition"])))
    save_rows(output / "trial_plan.jsonl", trials)
    plan = {
        "run_id": run_id if run_id is not None else _run_id(output), "judge": judge_contract(),
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "pricing": PRICING,
        "conditions": records, "trials": len(trials), "physical_images": len(image_hashes),
        "trial_plan_sha256": sha256_file(output / "trial_plan.jsonl"), "input_sha256": paths,
        "images_sha256": image_hashes, "dataset_verification": verification,
        "source_sha256": {p: sha256_file(REPO_ROOT / p) for p in SOURCE_FILES},
        "git_commit": git_head(REPO_ROOT),
        "preparation_contacts_api": False,
    }
    save(output / "preparation.json", plan)
    print(json.dumps({"prepared": str(output), "conditions": ids, "trials": len(trials),
                      "physical_images": len(image_hashes)}), flush=True)
    return plan


def prepare_run(vqa: Path, run: str) -> Row:
    """Judge every condition of one generation run's own trial plan into ``output/vqa/<run>/astra``."""
    trial_plan = str((vqa / "trial_plan.jsonl").relative_to(REPO_ROOT))
    rows = [r for r in read_rows(vqa / "trial_plan.jsonl") if r["paper_primary_metric_member"]]
    conditions = [ConditionSource(id=c, source_run=run, source_condition=c, trial_plan=trial_plan)
                  for c in dict.fromkeys(r["condition"] for r in rows)]
    return prepare(output_path(vqa), conditions, run_id=run)


def conditions_from(preparation: Path) -> list[ConditionSource]:
    """The conditions of a recorded judgment run (``trial_plan`` from the record or its ``sources.json``)."""
    plan = json.loads(preparation.read_text())
    sources: dict[str, Row] = {}
    for name in plan["input_sha256"]:
        if name.endswith("/sources.json"):
            sources = {c["id"]: c for c in json.loads((REPO_ROOT / name).read_text())["conditions"]}
    return [ConditionSource(id=c["id"], source_run=c["source_run"], source_condition=c["source_condition"],
                            trial_plan=c.get("trial_plan") or sources[c["id"]]["trial_plan"],
                            backbone=c.get("backbone"))
            for c in plan["conditions"]]


def preflight(output: Path, *, images: bool = True, verify_sources: bool = True) -> tuple[Row, list[Row]]:
    """Re-verify protocol, pricing, judge, run identity, sealed files and trial identities; offline."""
    plan = json.loads((output / "preparation.json").read_text())
    if (plan["protocol"] != PROTOCOL or plan["protocol_sha256"] != digest(PROTOCOL)
            or plan["pricing"] != PRICING or plan["judge"] != judge_contract()
            or plan["run_id"] != _run_id(output)):
        raise ValueError("Frozen Astra protocol, pricing, judge or run identity changed")
    if verify_sources and set(plan["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Incomplete source provenance")
    fields = (*(("source_sha256", "input_sha256") if verify_sources else ()), *(("images_sha256",) if images else ()))
    for field in fields:
        for name, expected in plan[field].items():
            if sha256_file(REPO_ROOT / name) != expected:
                raise ValueError(f"Frozen source/input changed: {name}")
    if sha256_file(output / "trial_plan.jsonl") != plan["trial_plan_sha256"]:
        raise ValueError("Trial plan changed")
    rows = read_rows(output / "trial_plan.jsonl")
    items = sorted(tab.load_pp_items())
    ids = [c["id"] for c in plan["conditions"]]
    if len(rows) != plan["trials"] or len(rows) != PANEL * len(ids) or len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError(f"Expected exactly {PANEL} distinct primary trials per condition")
    for condition in ids:
        tab.validate_panel([r for r in rows if r["condition"] == condition], items, list(range(5)))
    if any(r["request_sha256"] != digest(request_descriptor(r)) for r in rows):
        raise ValueError("Trial request identity changed")
    return plan, rows


def receipts(output: Path, rows: list[Row]) -> list[Row]:
    """Verified stored responses, in receipt file name (= trial ID) order."""
    results = []
    by_id = {r["trial_id"]: r for r in rows}
    for path in sorted((output / "responses").glob("*.json")):
        saved = json.loads(path.read_text())
        trial = by_id[path.stem]
        if saved["trial"] != trial or saved["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Stored response belongs to a different request/protocol")
        if saved["raw_response_sha256"] != digest(saved["raw_response"]):
            raise ValueError("Raw response changed")
        if saved["result"] != decode_response(saved["raw_response"]):
            raise ValueError("Stored answer or token usage changed")
        results.append(saved)
    models = {r["result"]["returned_model"] for r in results}
    if len(models) > 1:
        raise ValueError("Returned model version changed within the evaluation")
    return results


# --- paid transport --------------------------------------------------------------------------------------


def client() -> Any:
    from openai import OpenAI
    # Keep the pinned SDK. New request fields go through its supported extra_body.
    return OpenAI(base_url="https://api.openai.com/v1", max_retries=0, timeout=300.)


def safe_headers(headers: Any) -> dict[str, str]:
    """Only operational rate metadata; never serialize authorization or cookies."""
    return {key.lower(): str(value) for key, value in headers.items()
            if key.lower().startswith("x-ratelimit-") or key.lower() in ("retry-after", "x-request-id")}


def reset_seconds(value: str) -> float:
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", value)
    if not parts or "".join(n + unit for n, unit in parts) != value:
        return 60.
    return sum(float(n) * {"ms": .001, "s": 1., "m": 60., "h": 3600.}[unit] for n, unit in parts)


def rate_pause(headers: dict[str, str], workers: int) -> float:
    """Respect low remaining quotas before further submissions, with an in-flight margin."""
    pauses = [0.]
    for kind, reserve in (("requests", 1), ("tokens", RATE_TOKEN_RESERVE),
                          ("project-tokens", RATE_TOKEN_RESERVE)):
        remaining = headers.get("x-ratelimit-remaining-" + kind)
        if remaining is not None:
            try:
                low = float(remaining) < reserve * (workers + 1)
            except ValueError:
                continue
            if low:
                pauses.append(reset_seconds(headers.get("x-ratelimit-reset-" + kind, "60s")) + .1)
    return max(pauses)


def concurrency_limit(workers: int, received: int) -> int:
    """Expose quota headers first, then double concurrency after successive response groups."""
    stages = ((0, 1), (1, 2), (5, 4), (13, 8), (29, 16), (61, 32), (125, 64))
    return min(workers, max(limit for threshold, limit in stages if received >= threshold))


def send_one(session: Any, output: Path, row: Row, plan: Row, execution_id: str) -> Row:
    """A worker writes only its own receipt; budget and scheduling belong to the coordinator."""
    started = time.monotonic()
    try:
        body = request_body(row)
        reply = session.responses.with_raw_response.create(**body, extra_body={"service_tier": "default"})
        headers = safe_headers(reply.headers)
        response = reply.parse()
        raw = response.model_dump(mode="json")
        save(output / "raw" / (row["trial_id"] + ".json"), raw)
        result = decode_response(raw)
        record = {"trial": row, "protocol_sha256": plan["protocol_sha256"], "phase": "full",
                  "execution_id": execution_id, "request_id": getattr(response, "_request_id", None),
                  "elapsed_seconds": time.monotonic() - started, "rate_limit_headers": headers,
                  "raw_response": raw, "raw_response_sha256": digest(raw), "result": result}
        save(output / "responses" / (row["trial_id"] + ".json"), record)
        return {"record": record, "headers": headers}
    except Exception as exc:
        error_response = getattr(exc, "response", None)
        error = {"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                 "request_id": getattr(exc, "request_id", None), "execution_id": execution_id,
                 "rate_limit_headers": safe_headers(getattr(error_response, "headers", {})),
                 "billing_status": "unknown; no automatic resend; preserve intent/raw response"}
        save(output / "errors" / (row["trial_id"] + ".json"), error)
        return {"error": error, "headers": error["rate_limit_headers"]}


def schedule(
    output: Path, plan: Row, rows: list[Row], stored: list[Row], session: Any, *,
    workers: int, budget_usd: float, launch_interval: float, max_requests: int | None,
    stop: threading.Event, execution_id: str, required_model: str | None = RETURNED_MODEL,
) -> Row:
    """One coordinator reserves all in-flight costs and drains workers before releasing its lock."""
    completed = {record["trial"]["trial_id"] for record in stored}
    pending = [row for row in rows if row["trial_id"] not in completed]
    if max_requests is not None:
        pending = pending[:max_requests]
    spent = sum(record["result"]["usage"]["cost_usd"] for record in stored)
    known_models = {record["result"]["returned_model"] for record in stored}
    futures: dict[Future[Row], Row] = {}
    submitted, received, index = 0, 0, 0
    errors: list[str] = []
    next_launch, rate_ready = 0., 0.
    started = time.monotonic()
    state_path = output / "executions" / (execution_id + "_progress.json")

    def progress(status: str) -> Row:
        state = {"execution_id": execution_id, "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "completed_primary_trials": len(completed), "total_primary_trials": len(rows),
                 "full_complete": len(completed) == len(rows),
                 "actual_spend_usd": spent, "budget_usd": budget_usd, "workers": workers,
                 "in_flight": len(futures), "reserved_in_flight_usd": len(futures) * RESERVE_USD,
                 "submitted_this_execution": submitted, "received_this_execution": received,
                 "elapsed_seconds": time.monotonic() - started, "errors": errors}
        atomic_write_json(state_path, state)
        atomic_write_json(output / "progress.json", state)
        return state

    progress("running")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tab-astra") as pool:
        while index < len(pending) or futures:
            if (output / "PAUSE").exists():
                stop.set()
            # Keep traffic increases gradual even when the requested maximum is 64.
            active_limit = concurrency_limit(workers, received)
            done = [future for future in futures if future.done()]
            for future in done:
                row = futures.pop(future)
                outcome = future.result()
                received += 1
                rate_ready = max(rate_ready, time.monotonic() + rate_pause(outcome["headers"], workers))
                if "error" in outcome:
                    errors.append(f"Request failed ({outcome['error']['error_type']}): {row['trial_id']}")
                    continue
                record = outcome["record"]
                result = record["result"]
                completed.add(row["trial_id"])
                spent += result["usage"]["cost_usd"]
                known_models.add(result["returned_model"])
                if (len(known_models) > 1 or result["usage"]["input_tokens"] > 4096
                        or required_model is not None and result["returned_model"] != required_model):
                    errors.append("Returned model or reserved input bound changed")
                if result["status"] != "completed":
                    errors.append("Incomplete response preserved; review before continuing")
                print(json.dumps({"execution_id": execution_id, "condition": row["condition"],
                                  "completed": len(completed), "in_flight": len(futures),
                                  "cumulative_usd": spent, "status": result["status"],
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
            if done:
                progress("draining" if errors or stop.is_set() else "running")
            now = time.monotonic()
            can_launch = (not errors and not stop.is_set() and index < len(pending)
                          and len(futures) < active_limit and now >= max(next_launch, rate_ready))
            if can_launch:
                if spent + (len(futures) + 1) * RESERVE_USD <= budget_usd:
                    row = pending[index]
                    # Fail locally before creating intent if the image no longer matches its frozen hash.
                    request_body(row)
                    save(output / "attempts" / (row["trial_id"] + ".json"), {
                        "trial_id": row["trial_id"], "request_sha256": row["request_sha256"],
                        "protocol_sha256": plan["protocol_sha256"], "phase": "full",
                        "started_at": datetime.now(timezone.utc).isoformat(), "budget_usd": budget_usd,
                        "execution_id": execution_id,
                    })
                    future = pool.submit(send_one, session, output, row, plan, execution_id)
                    futures[future] = row
                    submitted += 1
                    index += 1
                    next_launch = now + launch_interval
                    progress("running")
                    continue
                if not futures:
                    errors.append(f"Budget reserve reached: spent ${spent:.6f}; reserve ${RESERVE_USD:.6f}")
            if not futures and (errors or stop.is_set()):
                break
            if futures:
                wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
            elif index < len(pending):
                stop.wait(.05)
    status = "stopped_requires_inspection" if errors else "paused" if stop.is_set() else "completed"
    return progress(status)


def run(
    output: Path, *, workers: int = DEFAULT_WORKERS, budget_usd: float, launch_interval: float = .3,
    max_requests: int | None = None, required_model: str | None = RETURNED_MODEL,
) -> Row:
    """Paid concurrent judgment of the pending trials; SIGINT/SIGTERM drain in-flight requests."""
    if not 1 <= workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("A finite positive dollar budget is required")
    if not math.isfinite(launch_interval) or launch_interval < 0:
        raise ValueError("launch_interval must be finite and nonnegative")
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be positive")
    stop = threading.Event()
    previous = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with locked(output):
            plan, rows = preflight(output)
            stored = receipts(output, rows)
            completed = {record["trial"]["trial_id"] for record in stored}
            for intent in (output / "attempts").glob("*.json"):
                if intent.stem not in completed:
                    raise ValueError(f"Unresolved API attempt; inspect before any resend: {intent}")
            if any(record["result"]["status"] != "completed" for record in stored):
                raise ValueError("Incomplete response in ledger; review before continuing")
            if required_model is not None and any(r["result"]["returned_model"] != required_model for r in stored):
                raise ValueError(f"Returned model differs from {required_model}")
            save(output / "runtime.json", {
                "python": platform.python_version(), "openai": importlib.metadata.version("openai"),
                "endpoint": "https://api.openai.com/v1/responses", "docker": Path("/.dockerenv").exists(),
                "workspace": str(REPO_ROOT)})
            execution_id = datetime.now(timezone.utc).strftime("parallel_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
            save(output / "executions" / (execution_id + ".json"), {
                "execution_id": execution_id, "started_at": datetime.now(timezone.utc).isoformat(),
                "workers": workers, "budget_usd": budget_usd, "launch_interval": launch_interval,
                "max_requests": max_requests, "reused_responses": len(stored), "seed_count": 5,
                "protocol_sha256": plan["protocol_sha256"],
                "preparation_sha256": sha256_file(output / "preparation.json"),
                "transport_sha256": {SOURCE_FILES[0]: sha256_file(Path(__file__))},
                "git_commit": git_head(REPO_ROOT),
                "retry_policy": "No automatic resend; drain all in-flight requests on errors or pause",
            })
            session = client()
            try:
                if session.models.retrieve(PROTOCOL["model"]).id != PROTOCOL["model"]:
                    raise ValueError("Requested Astra model unavailable; no substitution")
                result = schedule(
                    output, plan, rows, stored, session, workers=workers, budget_usd=budget_usd,
                    launch_interval=launch_interval, max_requests=max_requests, stop=stop,
                    execution_id=execution_id, required_model=required_model)
            finally:
                session.close()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if result["errors"]:
        raise RuntimeError("Evaluation stopped after draining outstanding calls: " + "; ".join(result["errors"]))
    return result


# --- predictions, summaries and replay -------------------------------------------------------------------


def predictions(output: Path) -> tuple[Row, list[Row]]:
    """All sealed trials with their judgments, in trial-plan order; fails until every trial is judged."""
    plan, rows = preflight(output, images=False)
    saved = {r["trial"]["trial_id"]: r["result"] for r in receipts(output, rows)}
    if set(saved) != {r["trial_id"] for r in rows}:
        raise ValueError(f"Astra judgment incomplete: {len(saved)}/{len(rows)} trials; "
                         "resume the judge with --budget-usd")
    result = [{**r, **saved[r["trial_id"]]} for r in rows]
    save_rows(output / "predictions.jsonl", result)
    return plan, result


def summarize(output: Path, *, seed_counts: Sequence[int] = (5,), bootstrap_seed: int) -> Row:
    """Write ``predictions_<n>seed.jsonl``, ``summary_<n>seed.json`` and ``scores_<n>seed.csv`` (n = max seeds)."""
    plan, rows = preflight(output, images=False)
    saved = receipts(output, rows)
    count = max(seed_counts)
    expected = {r["trial_id"] for r in rows if r["seed"] < count}
    selected = [r for r in saved if r["trial"]["seed"] < count]
    if {r["trial"]["trial_id"] for r in selected} != expected:
        raise ValueError("Full summary requires every selected trial")
    result = [{**r["trial"], **r["result"]} for r in selected]
    scores = tab.score_conditions(result, [c["id"] for c in plan["conditions"]], seed_counts=sorted(seed_counts),
                                  bootstrap_seed=bootstrap_seed)
    summary = {"run_id": plan["run_id"], "protocol": plan["protocol"], "conditions": plan["conditions"],
               "scores": scores, "returned_models": sorted({r["returned_model"] for r in result}),
               "bootstrap_unit": "TAB item (74); exploratory unadjusted CI",
               "bootstrap": {"draws": 20000, "rng": f"numpy.default_rng({bootstrap_seed})"},
               "spend_usd_selected_trials": sum(r["result"]["usage"]["cost_usd"] for r in selected)}
    save_rows(output / f"predictions_{count}seed.jsonl", result)
    save(output / f"summary_{count}seed.json", summary)
    with (output / f"scores_{count}seed.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "seed_count", "yes", "trials", "percent", "invalid"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores)
    return summary


def replay(output: Path, *, seed_count: int = 5) -> list[Row]:
    """Predictions re-derived from the stored receipts of a finished run; offline and location independent.

    Checks the frozen protocol and pricing, the trial plan seal and every
    request identity, then decodes every stored response again. The rows are in
    receipt (trial ID) order, the order of the recorded
    ``predictions_<n>seed.jsonl``. Recorded runs of earlier source layouts are
    accepted here only (their ``preflight`` fails by design).
    """
    plan = json.loads((output / "preparation.json").read_text())
    if plan["protocol"] != PROTOCOL or plan["protocol_sha256"] != digest(PROTOCOL) or plan["pricing"] != PRICING:
        raise ValueError("Frozen Astra protocol or pricing changed")
    if sha256_file(output / "trial_plan.jsonl") != plan["trial_plan_sha256"]:
        raise ValueError("Trial plan changed")
    rows = read_rows(output / "trial_plan.jsonl")
    if len(rows) != plan["trials"] or len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Trial plan does not match its preparation record")
    if any(r["request_sha256"] != digest(request_descriptor(r)) for r in rows):
        raise ValueError("Trial request identity changed")
    return [{**r["trial"], **r["result"]} for r in receipts(output, rows) if r["trial"]["seed"] < seed_count]
