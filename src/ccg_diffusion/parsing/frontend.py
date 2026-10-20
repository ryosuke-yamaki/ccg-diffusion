"""Stage 1-2 for one prompt in process: Hol-CCG parse, then C&C ``generate`` marked dependencies.

:class:`CCGFrontEnd` loads the parser and the marked-up category resources once
and keeps one resident ``generate`` engine per output mode. Each
:meth:`CCGFrontEnd.parse` call does the per-sentence work of the Stage 1 and
Stage 2 corpus tools (``parsing.holccg.parse_corpus`` and
``parsing.dependencies.generate_dependencies``) and returns the AUTO sentence,
its marked dependencies and a coverage row. Stage 3 is left to the caller:
``targets.guidance_targets.build_guidance_target_rows(..., sentence_map_rows=None,
coverage_rows=[result.coverage])``.

Used for arbitrary prompts (``single_prompt``, ``scripts/ccg/show_derivation.py``)
and for the generation-time benchmark (``scripts/paper/compute_cost.py``). On
hardware other than the paper's, a near-tie supertag can flip, so the
derivation of a recorded prompt is not guaranteed to match the recorded AUTO.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holccg.evaluation.candc_generate.engine import Generator
from holccg.parsing.inference import HolCCGParser
from holccg.parsing.render import fallback_right_branching

from ccg_diffusion import config
from ccg_diffusion.config import HolCCGParseContract
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency, auto_sentence, parse_marked_dependency
from ccg_diffusion.parsing.dependencies import (
    DEFAULT_ADDITIONS_PATH,
    AutoRecord,
    build_sentence_plans,
    convert_auto_to_pipe,
    materialize_markedup_resources,
    prepare_markedup_file,
    read_generate_blocks,
    read_generate_messages,
    reindex_dependency_line,
    validate_aggregated_dependency_blocks,
    write_expanded_auto,
)
from ccg_diffusion.parsing.holccg import (
    align_encoder_dtype,
    inference_options,
    render_auto,
    resolve_checkpoint,
    tokenizer_max_length,
    would_truncate_words,
)
from ccg_diffusion.parsing.sentences import split_and_tokenize


@dataclass(frozen=True)
class FrontEndResult:
    """Stage 1-2 output for one prompt."""

    tokens: tuple[str, ...]
    id_line: str
    derivation: str
    sentence: AutoSentence
    dependencies: tuple[MarkedDependency, ...]
    dependency_lines: tuple[str, ...]
    coverage: dict[str, Any]
    parse_record_status: str


class CCGFrontEnd:
    """Resident Hol-CCG parser and C&C ``generate`` engines for single-prompt Stage 1-2."""

    def __init__(
        self,
        device: str,
        *,
        contract: HolCCGParseContract = config.HOLCCG_PARSE,
        additions_path: Path = DEFAULT_ADDITIONS_PATH,
        scratch_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        self._scratch_owner = tempfile.TemporaryDirectory() if scratch_dir is None else None
        self.scratch_dir = Path(self._scratch_owner.name) if scratch_dir is None else scratch_dir
        checkpoint = resolve_checkpoint(contract.checkpoint, revision=contract.revision,
                                        local_files_only=local_files_only)
        self.parser = HolCCGParser.from_pretrained(str(checkpoint), options=inference_options(contract),
                                                   device=device, strict=False)
        align_encoder_dtype(self.parser.model)
        self.max_length = tokenizer_max_length(self.parser.model.tokenizer)
        self.cats, self.markedup = self.scratch_dir / "cats", self.scratch_dir / "effective_markedup"
        self.resource_sha256 = materialize_markedup_resources(self.cats)
        prepare_markedup_file(self.cats / "markedup", additions_path, self.markedup)
        self.generators = {mode: Generator(self.cats, self.markedup) for mode in ("-e", "-j")}

    def _generate(self, mode: str, pipe_path: Path, output: Path, log_path: Path) -> None:
        """The -e/-j branch of holccg.evaluation.candc_generate.cli.run with a resident Generator."""
        from holccg.evaluation.candc_generate.output import (
            format_command_header, format_parse_failure, format_word_dependencies)
        from holccg.evaluation.candc_generate.pipe import iter_pipe_sentences

        generator = self.generators[mode]
        generator.nsentences = 0
        argv = ["generate", mode, str(self.cats), str(self.markedup), str(pipe_path)]
        with (pipe_path.open("r", encoding="utf-8", errors="surrogateescape", newline="") as source,
              output.open("w", encoding="utf-8") as out,
              log_path.open("w", encoding="utf-8", errors="surrogateescape", newline="") as log):
            out.write(format_command_header(argv))
            for sentence in iter_pipe_sentences(source):
                result = generator.parse(sentence)
                log.write(result.log_line + "\n")
                if not result.success or result.top is None:
                    out.write(format_parse_failure(mode)[0])
                    continue
                out.write(format_word_dependencies(
                    result.top, result.sentence, generator.categories.relations, generator.categories,
                    julia_slots=mode == "-j"))

    def parse(self, prompt: str, *, sentence_id: str) -> FrontEndResult:
        """Parse one single-sentence prompt; the AUTO ID line is ``ID=<sentence_id> PARSER=HOLCCG NUMPARSE=1``."""
        sentences = split_and_tokenize(prompt)
        if len(sentences) != 1:
            raise ValueError(f"Stage 1 expects one sentence per prompt, found {len(sentences)}: {prompt!r}")
        tokens = [token.text for token in sentences[0].tokens]
        id_line = f"ID={sentence_id} PARSER=HOLCCG NUMPARSE=1"
        if (self.max_length is not None and tokens
                and would_truncate_words(self.parser.model.tokenizer, tokens, self.max_length)):
            derivation, parse_record_status = fallback_right_branching(tokens, "OVERLEN", "POS"), "overlen"
        else:
            record = next(iter(self.parser.iter_parse_chunks([tokens], id_start=1, sentence_ids=[sentence_id])))
            derivation = render_auto(record, partial_root_cat="PARTIAL", fail_root_cat="FAIL", fail_pos="POS")
            parse_record_status = str(record.status)
        derivation = derivation.strip()

        plans = build_sentence_plans([AutoRecord(sentence_index=1, id_line=id_line, derivation=derivation)],
                                     partial_root_cat="PARTIAL", fail_root_cats={"FAIL", "OVERLEN"})
        units = plans[0].units
        e_block: list[str] = []
        j_block: list[str] = []
        success: list[bool] = []
        if units:
            with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmp:
                tmp_path = Path(tmp)
                write_expanded_auto(tmp_path / "x.expanded.auto", plans)
                convert_auto_to_pipe(tmp_path / "x.expanded.auto", tmp_path / "x.expanded.pipe")
                outputs = {}
                for mode in ("-e", "-j"):
                    out, log = tmp_path / f"x{mode}.deps", tmp_path / f"x{mode}.log"
                    self._generate(mode, tmp_path / "x.expanded.pipe", out, log)
                    outputs[mode] = (read_generate_messages(log, len(units)),
                                     read_generate_blocks(out, len(units))[1])
            (e_messages, e_blocks), (j_messages, j_blocks) = outputs["-e"], outputs["-j"]
            for unit, e_message, j_message, e_lines, j_lines in zip(
                    units, e_messages, j_messages, e_blocks, j_blocks, strict=True):
                e_ok, j_ok = e_message == "parse successful", j_message == "parse successful"
                if e_ok:
                    e_block.extend(reindex_dependency_line(line, unit.start - 1) for line in e_lines)
                if j_ok:
                    j_block.extend(reindex_dependency_line(line, unit.start - 1) for line in j_lines)
                success.append(e_ok and j_ok)
            validate_aggregated_dependency_blocks(plans, [e_block])
            validate_aggregated_dependency_blocks(plans, [j_block])
        coverage = {"sentence_index": 1, "auto_id": id_line, "root_category": plans[0].root_category,
                    "parse_status": plans[0].parse_status,
                    "all_generate_units_success": bool(units) and all(success),
                    "any_generate_unit_success": any(success), "prompt": " ".join(tokens)}
        return FrontEndResult(
            tokens=tuple(tokens),
            id_line=id_line,
            derivation=derivation,
            sentence=auto_sentence(id_line, derivation),
            dependencies=tuple(parse_marked_dependency(line) for line in e_block),
            dependency_lines=tuple(e_block),
            coverage=coverage,
            parse_record_status=parse_record_status,
        )
