"""Stage 2: C&C ``generate`` marked dependencies from complete and PARTIAL Hol-CCG derivations.

Complete derivations are handed to the pure-Python C&C ``generate`` port shipped
with the ``holccg`` package (no external binary). A synthetic ``PARTIAL``
derivation (Stage 1 skimmer output) is expanded into its real fragments; each
fragment is generated on its own, its dependency token indices are shifted back
to sentence-global positions, and the blocks are aggregated per original
sentence. No dependency crossing two fragments is ever invented.

``generate_dependencies`` writes the Stage 2 files of one AUTO corpus;
``generate_marked_dependencies`` runs it in a temporary directory for in-process
callers (TAB) and returns the parsed blocks. ``holccg`` is imported lazily.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ccg_diffusion.parsing.ccg import MarkedDependency, MarkedDepsError, parse_marked_dependency
from ccg_diffusion.parsing.derivation import extract_partial_subtrees, parse_auto_tree_with_spans
from ccg_diffusion.provenance import repo_relative, repository_root

# The HOL-enhanced candc-1.00 marked-up category database bundled with holccg,
# used by every recorded run.
MARKEDUP_RESOURCE = "hol_en"
# Tracked overlay with the few categories Hol-CCG emits that the database lacks.
DEFAULT_ADDITIONS_PATH = repository_root() / "configs/parsing/candc_markedup_additions"
# Last header line of every aggregated dependency file (recorded value: part of the file bytes).
AGGREGATION_NOTE = "# PARTIAL fragments aggregated with sentence-global token indices by HolSD"


@dataclass(frozen=True)
class AutoRecord:
    sentence_index: int
    id_line: str
    derivation: str


@dataclass(frozen=True)
class GenerateUnit:
    unit_index: int
    sentence_index: int
    fragment_index: int
    fragment_count: int
    start: int
    end: int
    root_category: str
    parse_status: str
    derivation: str


@dataclass
class SentencePlan:
    record: AutoRecord
    root_category: str
    token_count: int
    parse_status: str
    units: list[GenerateUnit]


# --- CCGbank dependency files -------------------------------------------------------------------------


def split_ccgbank_deps(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Split a CCGbank deps file into header lines and per-sentence blocks.

    ``header_lines`` holds the leading ``#`` comments plus the blank line that
    terminates them; ``sentence_blocks`` holds one list of dependency lines per
    sentence.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] = []
    idx = 0
    while idx < len(lines) and lines[idx].startswith("#"):
        header.append(lines[idx])
        idx += 1
    if idx < len(lines) and lines[idx].strip() == "":
        header.append(lines[idx])
        idx += 1

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines[idx:]:
        if line.strip() == "":
            blocks.append(current)
            current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return header, blocks


def write_ccgbank_deps(
    path: str | Path, header: list[str], blocks: list[list[str]]
) -> None:
    """Write CCGbank deps blocks after ``header`` (written verbatim, without a trailing blank line)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for line in header:
            handle.write(line + "\n")
        if header and header[-1].strip() != "" and blocks:
            handle.write("\n")
        for block in blocks:
            for dep_line in block:
                handle.write(dep_line + "\n")
            handle.write("\n")


def load_aligned_dependency_blocks(path: Path, sentence_count: int) -> list[list[MarkedDependency]]:
    """Read ``generate -e`` blocks keeping empty leading blocks in position.

    ``parsing.ccg.load_marked_dependency_blocks`` skips *every* blank line after
    the ``#`` header, so a corpus whose leading sentences carry no dependency
    loses those empty blocks and every later sentence is read one slot early.
    ``write_ccgbank_deps`` emits exactly one header-terminating blank line and
    then one blank line per sentence, so consuming a single blank line after the
    header restores the 1:1 sentence alignment.
    """

    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        index += 1
    if index < len(lines) and not lines[index].strip():
        index += 1
    raw_blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines[index:]:
        if line.strip():
            current.append(line)
        else:
            raw_blocks.append(current)
            current = []
    if current:
        raw_blocks.append(current)
    if len(raw_blocks) != sentence_count:
        raise MarkedDepsError(
            f"{path}: marked dependency blocks ({len(raw_blocks)}) do not match "
            f"the corpus sentence count ({sentence_count})"
        )
    return [[parse_marked_dependency(line) for line in block] for block in raw_blocks]


# --- pure-Python C&C generate backend -----------------------------------------------------------------


def materialize_markedup_resources(cats_dir: Path) -> dict[str, str]:
    """Unpack the bundled ``hol_en`` marked-up-category resources into ``cats_dir``.

    ``generate`` resolves the type-raising files (``trNP`` etc.) relative to the
    supplied cats directory, so the packaged resources are written out once per
    run. ``bundled_resources`` verifies every file against its manifest SHA-256
    first; the digests are returned for the run's provenance record.
    """
    from holccg.evaluation.candc_generate import bundled_resources
    from holccg.evaluation.candc_generate.resources import RESOURCE_FILES

    resources = bundled_resources(name=MARKEDUP_RESOURCE, validate=True)
    cats_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for filename in RESOURCE_FILES:
        payload = resources.root.joinpath(filename).read_bytes()
        (cats_dir / filename).write_bytes(payload)
        digests[filename] = hashlib.sha256(payload).hexdigest()
    return digests


def prepare_markedup_file(
    base_path: Path,
    additions_path: Optional[Path],
    output_path: Path,
) -> list[str]:
    """Create an effective C&C markedup file and return the added bare categories.

    Keeping the small Hol-CCG-specific additions in a tracked overlay makes the
    setup reproducible without editing the marked-up database bundled with the
    holccg package.
    """
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing markedup file: {base_path}")

    base_text = base_path.read_text(encoding="utf-8")
    effective_text = base_text.rstrip() + "\n"
    added: list[str] = []
    if additions_path is not None:
        if not additions_path.is_file():
            raise FileNotFoundError(f"Missing markedup additions file: {additions_path}")
        additions_text = additions_path.read_text(encoding="utf-8")
        blocks = [
            block.strip("\n")
            for block in re.split(r"\n\s*\n", additions_text)
            if block.strip() and not block.lstrip().startswith("#")
        ]
        for block in blocks:
            category = block.splitlines()[0].strip()
            if not category or category.startswith("#"):
                continue
            if re.search(rf"(?m)^{re.escape(category)}\s*$", effective_text):
                continue
            effective_text = effective_text.rstrip() + "\n\n" + block.rstrip() + "\n"
            added.append(category)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(effective_text, encoding="utf-8")
    return added


def convert_auto_to_pipe(auto_path: Path, pipe_path: Path) -> None:
    """Render an AUTO file in C&C's preorder ``.pipe`` layout.

    The holccg AUTO reader has no record delimiter inside lexical items, so a
    lexical quote such as ``"Light"`` needs no special handling.
    """
    from holccg.evaluation.ccgbank import read_auto_file, write_legacy_pipe_sentence

    pipe_path.parent.mkdir(parents=True, exist_ok=True)
    with pipe_path.open("w", encoding="utf-8", newline="") as handle:
        for sentence in read_auto_file(auto_path):
            write_legacy_pipe_sentence(sentence, handle)


def run_generate(
    cats_dir: Path,
    markedup_path: Path,
    pipe_path: Path,
    output_path: Path,
    mode: str,
    errors_log_path: Path,
) -> None:
    """Run the pure-Python C&C ``generate`` port for one output mode (``-e`` or ``-j``)."""
    from holccg.evaluation.candc_generate import cli

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = ["generate", mode, str(cats_dir), str(markedup_path), str(pipe_path)]
    print(f"+ {' '.join(argv)} > {output_path}")
    with output_path.open("w", encoding="utf-8") as out_file:
        status = cli.run(argv, stdout=out_file, errors_log=str(errors_log_path))
    if status != 0:
        raise RuntimeError(f"generate {mode} failed with exit status {status}")


# --- PARTIAL expansion and aggregation ----------------------------------------------------------------


def load_auto_records(path: Path) -> list[AutoRecord]:
    """Load CCGbank AUTO records, accepting wrapped derivation lines."""
    records: list[AutoRecord] = []
    current_id: Optional[str] = None
    derivation_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, derivation_lines
        if current_id is None:
            return
        derivation = " ".join(derivation_lines).strip()
        if not derivation:
            raise ValueError(f"AUTO record has no derivation: {current_id}")
        records.append(
            AutoRecord(
                sentence_index=len(records) + 1,
                id_line=current_id,
                derivation=derivation,
            )
        )
        current_id = None
        derivation_lines = []

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("ID="):
                flush()
                current_id = line
                continue
            if current_id is None:
                if line.strip() and not line.lstrip().startswith("#"):
                    raise ValueError(
                        f"Unexpected AUTO content before first ID line: {line[:120]!r}"
                    )
                continue
            if line.strip():
                derivation_lines.append(line.strip())
    flush()
    if not records:
        raise ValueError(f"No AUTO records found in {path}")
    return records


def build_sentence_plans(
    records: list[AutoRecord],
    partial_root_cat: str,
    fail_root_cats: set[str],
) -> list[SentencePlan]:
    """Split each AUTO record into its generate units: the whole tree, its PARTIAL fragments, or none."""
    plans: list[SentencePlan] = []
    next_unit_index = 1
    for record in records:
        root = parse_auto_tree_with_spans(record.derivation)
        if root.cat in fail_root_cats:
            status = "fail"
            fragment_nodes = []
        elif root.cat == partial_root_cat:
            status = "partial"
            fragment_nodes = extract_partial_subtrees(root, partial_root_cat)
        else:
            status = "complete"
            fragment_nodes = [root]

        units: list[GenerateUnit] = []
        for fragment_index, node in enumerate(fragment_nodes, start=1):
            if not node.auto:
                raise ValueError(
                    f"Missing serialized subtree for sentence {record.sentence_index}, "
                    f"fragment {fragment_index}"
                )
            units.append(
                GenerateUnit(
                    unit_index=next_unit_index,
                    sentence_index=record.sentence_index,
                    fragment_index=fragment_index,
                    fragment_count=len(fragment_nodes),
                    start=node.start,
                    end=node.end,
                    root_category=node.cat,
                    parse_status=status,
                    derivation=node.auto,
                )
            )
            next_unit_index += 1

        plans.append(
            SentencePlan(
                record=record,
                root_category=root.cat,
                token_count=root.end,
                parse_status=status,
                units=units,
            )
        )
    return plans


def write_expanded_auto(path: Path, plans: list[SentencePlan]) -> None:
    """Write one AUTO record per generate unit (the ``ID=holsd.generate_unit.<n>`` lines are recorded values)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            for unit in plan.units:
                handle.write(
                    "ID=holsd.generate_unit."
                    f"{unit.unit_index} ORIGINAL_SENTENCE={unit.sentence_index} "
                    f"FRAGMENT={unit.fragment_index}/{unit.fragment_count} NUMPARSE=1\n"
                )
                handle.write(unit.derivation.strip() + "\n")


def read_generate_messages(path: Path, expected_count: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing generate log: {path}")
    messages: list[Optional[str]] = [None] * expected_count
    pattern = re.compile(r"^(\d+):(.*)$")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = pattern.match(raw_line.rstrip("\r\n"))
            if match is None:
                continue
            index = int(match.group(1))
            if not 1 <= index <= expected_count:
                raise ValueError(f"Out-of-range sentence ID {index} in {path}")
            if messages[index - 1] is not None:
                raise ValueError(f"Duplicate sentence ID {index} in {path}")
            messages[index - 1] = match.group(2).strip()
    missing = [index + 1 for index, message in enumerate(messages) if message is None]
    if missing:
        raise ValueError(f"Missing generate log records in {path}: {missing[:20]}")
    return [str(message) for message in messages]


def read_generate_blocks(path: Path, expected_count: int) -> tuple[list[str], list[list[str]]]:
    header, blocks = split_ccgbank_deps(str(path))
    if len(blocks) != expected_count:
        raise ValueError(
            f"Generate block count mismatch for {path}: "
            f"expected {expected_count}, got {len(blocks)}"
        )
    return header, blocks


def reindex_dependency_token(token: str, offset: int) -> str:
    base, separator, index_text = token.rpartition("_")
    if not separator or not index_text.isdigit():
        raise ValueError(f"Dependency endpoint has no numeric _N suffix: {token!r}")
    return f"{base}_{int(index_text) + offset}"


def reindex_dependency_line(line: str, offset: int) -> str:
    """Shift predicate and argument word indices in one generate dependency."""
    fields = line.split()
    if len(fields) < 5:
        raise ValueError(f"Malformed generate dependency line: {line!r}")
    fields[0] = reindex_dependency_token(fields[0], offset)
    fields[3] = reindex_dependency_token(fields[3], offset)
    return " ".join(fields)


def validate_aggregated_dependency_blocks(
    plans: list[SentencePlan], blocks: list[list[str]]
) -> int:
    """Verify every endpoint is in-range and remains inside one source subtree."""
    if len(plans) != len(blocks):
        raise ValueError(
            f"Plan/dependency block mismatch: {len(plans)} vs {len(blocks)}"
        )
    dependency_count = 0
    for plan, block in zip(plans, blocks):
        spans = [(unit.start, unit.end) for unit in plan.units]
        for line in block:
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Malformed aggregated dependency line: {line!r}")
            predicate = reindex_dependency_token(fields[0], 0)
            argument = reindex_dependency_token(fields[3], 0)
            predicate_index = int(predicate.rsplit("_", 1)[1])
            argument_index = int(argument.rsplit("_", 1)[1])
            within_sentence = all(
                (
                    1 <= predicate_index <= plan.token_count,
                    1 <= argument_index <= plan.token_count,
                )
            )
            if not within_sentence:
                raise ValueError(
                    f"Dependency endpoint outside sentence {plan.record.sentence_index}: "
                    f"{line!r} (tokens={plan.token_count})"
                )
            within_one_subtree = any(
                all(
                    (
                        start <= predicate_index <= end,
                        start <= argument_index <= end,
                    )
                )
                for start, end in spans
            )
            if not within_one_subtree:
                raise ValueError(
                    "Dependency crosses synthetic PARTIAL subtree boundaries in "
                    f"sentence {plan.record.sentence_index}: {line!r}, spans={spans}"
                )
            dependency_count += 1
    return dependency_count


def _clean_header(header: list[str], aggregation_note: str) -> list[str]:
    return [line for line in header if line.strip()] + [aggregation_note]


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    fraction = numerator / denominator if denominator else None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": fraction,
        "percent": 100.0 * fraction if fraction is not None else None,
    }


def _load_raw(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n").strip() for line in handle if line.strip()]


def _record_path(path: Path) -> str:
    """Repository-relative path inside the checkout, absolute otherwise."""
    try:
        return repo_relative(path.resolve())
    except ValueError:
        return str(path.resolve())


def generate_dependencies(  # noqa: C901
    *,
    auto_path: Path,
    output_dir: Path,
    additions_path: Path | None = DEFAULT_ADDITIONS_PATH,
    output_prefix: str | None = None,
    partial_root_cat: str = "PARTIAL",
    fail_root_cats: Sequence[str] = ("FAIL", "OVERLEN"),
    raw_path: Path | None = None,
    dataset_name: str | None = None,
    checkpoint: str | None = None,
    stag_threshold: float | None = None,
    phrase_threshold: float | None = None,
    span_threshold: float | None = None,
    apply_skimmer: bool = False,
) -> dict[str, Any]:
    """Run Stage 2 on one AUTO corpus and write its files into ``output_dir``.

    Writes ``<prefix>.markedup_deps`` (``generate -e``) and ``<prefix>.ccgbank_deps``
    (``generate -j``) aggregated per sentence, the per-sentence
    ``<prefix>.coverage_rows.jsonl`` (with ``prompt`` = the ``.raw`` line when
    ``raw_path`` is given), the per-unit ``<prefix>.generate_units.jsonl``, the
    ``<prefix>.coverage.json`` summary and the intermediate expanded files. The
    threshold and checkpoint arguments are recorded in the summary only. Paths in
    the summary are repository-relative inside the checkout, absolute otherwise.
    Returns the summary.
    """
    if not auto_path.is_file():
        raise FileNotFoundError(f"Missing AUTO input: {auto_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or auto_path.stem
    records = load_auto_records(auto_path)
    plans = build_sentence_plans(
        records,
        partial_root_cat=partial_root_cat,
        fail_root_cats=set(fail_root_cats),
    )
    units = [unit for plan in plans for unit in plan.units]
    if not units:
        raise ValueError("AUTO input contains no complete or partial subtrees to generate")

    raw_sentences: Optional[list[str]] = None
    if raw_path is not None:
        raw_sentences = _load_raw(raw_path)
        if len(raw_sentences) != len(plans):
            raise ValueError(
                f"Raw/AUTO sentence mismatch: {len(raw_sentences)} vs {len(plans)}"
            )
        for plan, sentence in zip(plans, raw_sentences):
            if len(sentence.split()) != plan.token_count:
                raise ValueError(
                    f"Raw/AUTO token mismatch at sentence {plan.record.sentence_index}: "
                    f"raw={len(sentence.split())}, auto={plan.token_count}"
                )

    expanded_auto = output_dir / f"{prefix}.expanded.auto"
    expanded_pipe = output_dir / f"{prefix}.expanded.pipe"
    effective_markedup = output_dir / f"{prefix}.effective_markedup"
    raw_markedup_deps = output_dir / f"{prefix}.expanded.markedup_deps"
    raw_ccgbank_deps = output_dir / f"{prefix}.expanded.ccgbank_deps"
    markedup_deps = output_dir / f"{prefix}.markedup_deps"
    ccgbank_deps = output_dir / f"{prefix}.ccgbank_deps"
    e_log = output_dir / f"{prefix}.generate_e.errors.log"
    j_log = output_dir / f"{prefix}.generate_j.errors.log"

    write_expanded_auto(expanded_auto, plans)
    convert_auto_to_pipe(expanded_auto, expanded_pipe)

    cats_dir = output_dir / f"{prefix}.cats"
    resource_digests = materialize_markedup_resources(cats_dir)
    appended_categories = prepare_markedup_file(
        cats_dir / "markedup", additions_path, effective_markedup
    )
    run_generate(
        cats_dir,
        effective_markedup,
        expanded_pipe,
        raw_markedup_deps,
        "-e",
        e_log,
    )
    run_generate(
        cats_dir,
        effective_markedup,
        expanded_pipe,
        raw_ccgbank_deps,
        "-j",
        j_log,
    )

    e_messages = read_generate_messages(e_log, len(units))
    j_messages = read_generate_messages(j_log, len(units))
    e_header, e_unit_blocks = read_generate_blocks(raw_markedup_deps, len(units))
    j_header, j_unit_blocks = read_generate_blocks(raw_ccgbank_deps, len(units))

    e_sentence_blocks: list[list[str]] = [[] for _ in plans]
    j_sentence_blocks: list[list[str]] = [[] for _ in plans]
    generated_token_sets: list[set[int]] = [set() for _ in plans]
    unit_rows: list[dict[str, Any]] = []
    for unit, e_message, j_message, e_block, j_block in zip(
        units, e_messages, j_messages, e_unit_blocks, j_unit_blocks
    ):
        e_success = e_message == "parse successful"
        j_success = j_message == "parse successful"
        offset = unit.start - 1
        if e_success:
            e_sentence_blocks[unit.sentence_index - 1].extend(
                reindex_dependency_line(line, offset) for line in e_block
            )
        if j_success:
            j_sentence_blocks[unit.sentence_index - 1].extend(
                reindex_dependency_line(line, offset) for line in j_block
            )
        if e_success and j_success:
            generated_token_sets[unit.sentence_index - 1].update(
                range(unit.start, unit.end + 1)
            )
        unit_row = asdict(unit)
        unit_row.pop("derivation")
        unit_row.update(
            {
                "generate_e_message": e_message,
                "generate_j_message": j_message,
                "generate_e_success": e_success,
                "generate_j_success": j_success,
                "generate_both_success": e_success and j_success,
                "markedup_dependency_count": len(e_block) if e_success else 0,
                "ccgbank_dependency_count": len(j_block) if j_success else 0,
            }
        )
        unit_rows.append(unit_row)

    markedup_dependency_total = validate_aggregated_dependency_blocks(
        plans, e_sentence_blocks
    )
    ccgbank_dependency_total = validate_aggregated_dependency_blocks(
        plans, j_sentence_blocks
    )

    write_ccgbank_deps(str(markedup_deps), _clean_header(e_header, AGGREGATION_NOTE), e_sentence_blocks)
    write_ccgbank_deps(str(ccgbank_deps), _clean_header(j_header, AGGREGATION_NOTE), j_sentence_blocks)

    unit_rows_by_sentence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        unit_rows_by_sentence[int(row["sentence_index"])].append(row)

    coverage_rows: list[dict[str, Any]] = []
    for plan, e_block, j_block, token_set in zip(
        plans, e_sentence_blocks, j_sentence_blocks, generated_token_sets
    ):
        sentence_index = plan.record.sentence_index
        sentence_units = unit_rows_by_sentence[sentence_index]
        all_success = bool(sentence_units) and all(
            bool(row["generate_both_success"]) for row in sentence_units
        )
        any_success = any(
            bool(row["generate_both_success"]) for row in sentence_units
        )
        row: dict[str, Any] = {
            "sentence_index": sentence_index,
            "auto_id": plan.record.id_line,
            "root_category": plan.root_category,
            "parse_status": plan.parse_status,
            "token_count": plan.token_count,
            "fragment_count": len(plan.units),
            "successful_fragment_count": sum(
                bool(unit_row["generate_both_success"])
                for unit_row in sentence_units
            ),
            "generated_token_count": len(token_set),
            "generated_token_fraction": len(token_set) / plan.token_count,
            "all_generate_units_success": all_success,
            "any_generate_unit_success": any_success,
            "markedup_dependency_count": len(e_block),
            "ccgbank_dependency_count": len(j_block),
            "has_markedup_dependency": bool(e_block),
            "has_ccgbank_dependency": bool(j_block),
            "failed_units": [
                {
                    "unit_index": unit_row["unit_index"],
                    "fragment_index": unit_row["fragment_index"],
                    "generate_e_message": unit_row["generate_e_message"],
                    "generate_j_message": unit_row["generate_j_message"],
                }
                for unit_row in sentence_units
                if not bool(unit_row["generate_both_success"])
            ],
        }
        if raw_sentences is not None:
            row["prompt"] = raw_sentences[sentence_index - 1]
        coverage_rows.append(row)

    parse_counts = Counter(str(row["parse_status"]) for row in coverage_rows)
    all_success_count = sum(
        bool(row["all_generate_units_success"]) for row in coverage_rows
    )
    any_success_count = sum(
        bool(row["any_generate_unit_success"]) for row in coverage_rows
    )
    dep_sentence_count = sum(bool(row["has_ccgbank_dependency"]) for row in coverage_rows)
    total_tokens = sum(int(row["token_count"]) for row in coverage_rows)
    generated_tokens = sum(int(row["generated_token_count"]) for row in coverage_rows)
    successful_units = sum(bool(row["generate_both_success"]) for row in unit_rows)
    partial_rows = [row for row in coverage_rows if row["parse_status"] == "partial"]
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "input_auto": _record_path(auto_path),
        "auto_sha256": hashlib.sha256(auto_path.read_bytes()).hexdigest(),
        "raw_path": _record_path(raw_path) if raw_path else None,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path else None,
        "checkpoint": checkpoint,
        "parsing": {
            "stag_threshold": stag_threshold,
            "phrase_threshold": phrase_threshold,
            "span_threshold": span_threshold,
            "apply_skimmer": apply_skimmer,
        },
        "partial_semantics": (
            "Dependencies are generated independently inside each skimmer subtree; "
            "no cross-subtree dependency is invented."
        ),
        "sentence_count": len(plans),
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "complete_parse_coverage": _rate(parse_counts["complete"], len(plans)),
        "complete_or_partial_parse_coverage": _rate(
            parse_counts["complete"] + parse_counts["partial"], len(plans)
        ),
        "generate_unit_count": len(units),
        "generate_unit_coverage": _rate(successful_units, len(units)),
        "sentence_coverage": {
            "all_generate_units_success": _rate(all_success_count, len(plans)),
            "at_least_one_generate_unit_success": _rate(any_success_count, len(plans)),
            "has_ccgbank_dependency": _rate(dep_sentence_count, len(plans)),
            "tokens_in_successful_generate_units": _rate(generated_tokens, total_tokens),
        },
        "partial_sentence_coverage": {
            "all_fragments_generate_success": _rate(
                sum(bool(row["all_generate_units_success"]) for row in partial_rows),
                len(partial_rows),
            ),
            "has_ccgbank_dependency": _rate(
                sum(bool(row["has_ccgbank_dependency"]) for row in partial_rows),
                len(partial_rows),
            ),
        },
        "failed_sentence_ids": [
            int(row["sentence_index"])
            for row in coverage_rows
            if not bool(row["all_generate_units_success"])
        ],
        "sentences_without_ccgbank_dependencies": [
            int(row["sentence_index"])
            for row in coverage_rows
            if not bool(row["has_ccgbank_dependency"])
        ],
        "failed_unit_ids": [
            int(row["unit_index"])
            for row in unit_rows
            if not bool(row["generate_both_success"])
        ],
        "aggregation_audit": {
            "markedup_dependency_count": markedup_dependency_total,
            "ccgbank_dependency_count": ccgbank_dependency_total,
            "all_endpoints_within_sentence": True,
            "no_dependency_crosses_partial_subtrees": True,
        },
        "markedup": {
            "resource": MARKEDUP_RESOURCE,
            "resource_sha256": resource_digests,
            "base": _record_path(cats_dir / "markedup"),
            "additions": _record_path(additions_path) if additions_path else None,
            "categories_appended_to_effective_file": appended_categories,
            "effective": _record_path(effective_markedup),
        },
        "outputs": {
            "markedup_dependencies": _record_path(markedup_deps),
            "ccgbank_dependencies": _record_path(ccgbank_deps),
            "coverage_rows": _record_path(output_dir / f"{prefix}.coverage_rows.jsonl"),
            "unit_rows": _record_path(output_dir / f"{prefix}.generate_units.jsonl"),
        },
    }

    coverage_path = output_dir / f"{prefix}.coverage.json"
    coverage_rows_path = output_dir / f"{prefix}.coverage_rows.jsonl"
    unit_rows_path = output_dir / f"{prefix}.generate_units.jsonl"
    coverage_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with coverage_rows_path.open("w", encoding="utf-8") as handle:
        for row in coverage_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with unit_rows_path.open("w", encoding="utf-8") as handle:
        for row in unit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_md = output_dir / f"{prefix}.summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# HolCCG → C&C generate coverage",
                "",
                f"- Sentences: {len(plans)}",
                f"- Parse status: {dict(sorted(parse_counts.items()))}",
                f"- Generate units successful: {successful_units}/{len(units)}",
                f"- Sentences with all units successful: {all_success_count}/{len(plans)}",
                f"- Sentences with CCGbank dependencies: {dep_sentence_count}/{len(plans)}",
                f"- Tokens in successful units: {generated_tokens}/{total_tokens}",
                f"- PARTIAL sentences: {len(partial_rows)}",
                "",
                "PARTIAL dependencies are the union of dependencies internal to each "
                "skimmer subtree. Cross-subtree dependencies are intentionally absent.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def generate_marked_dependencies(
    auto_lines: Sequence[str],
    raw_lines: Sequence[str],
    *,
    additions_path: Path = DEFAULT_ADDITIONS_PATH,
    apply_skimmer: bool = False,
) -> list[list[MarkedDependency]]:
    """Stage 2 for an in-memory corpus: one marked-dependency block per sentence.

    ``auto_lines`` alternates ``ID=`` lines and derivations; ``raw_lines`` holds
    one tokenized sentence per entry. The corpus is written as
    ``prompts.auto``/``prompts.raw`` into a temporary directory and run through
    :func:`generate_dependencies` exactly as the Stage 2 CLI does, so the blocks
    equal those of the CLI's ``prompts.markedup_deps``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        auto_path = root / "prompts.auto"
        raw_path = root / "prompts.raw"
        auto_path.write_text("\n".join(auto_lines) + "\n", encoding="utf-8")
        raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        generate_dependencies(
            auto_path=auto_path,
            output_dir=root / "generate",
            additions_path=additions_path,
            raw_path=raw_path,
            apply_skimmer=apply_skimmer,
        )
        return load_aligned_dependency_blocks(root / "generate" / "prompts.markedup_deps", len(raw_lines))
