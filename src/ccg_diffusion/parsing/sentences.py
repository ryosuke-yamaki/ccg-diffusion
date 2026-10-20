r"""Sentence split, parse tokenization, sentence maps and the Stage 1 parse corpus.

Stage 1 parses one sentence per ``.raw`` line (whitespace pre-tokenized). The
checkpoint's grammar has no ``S[dcl] S[dcl] -> X`` rule, so several sentences on
one line never reach a complete parse. This module maps an arbitrary prompt onto
that contract: it splits the prompt at terminal punctuation and tokenizes each
sentence for parsing.

The parse input and the generation prompt are kept apart, and **the generation
prompt is never rewritten**. Punctuation split off for parsing appears only in
the ``.raw`` line; every token carries its character span in the original prompt.
Every token is an exact substring of the prompt
(``prompt[char_start:char_end] == text``), so no escaping or surface rewrite
happens.

Tokenization rules (CCGbank/PTB style). The starting point is the ``[,.!?;:()]``
set of the ABC-6K preparation script, extended as follows:

- Double quotes ``"`` and their curly forms ``“ ”`` are split off (otherwise a
  leaf such as ``"Merryman"`` reaches the supertagger with its quotes).
- The possessive and contraction clitics ``'s`` / ``’s`` are split off
  (``day's`` -> ``day`` + ``'s``). In CCGbank the clitic is its own leaf of
  category ``(NP[nb]/N)\NP``; gluing it to the noun loses the genitive
  dependency.
- Word-internal hyphens (``high-top``) are kept; CCGbank also treats such a word
  as one token.
- Everything else stays whitespace-separated. In particular em dashes, en
  dashes, slashes and a bare word-final apostrophe (``objects'``) are not split
  (see "known glued tokens" below).

Known glued tokens (measured on the 1,065 DPG-Bench prompts; a change of policy
edits the separator constants in one place):

- Of the 8 em dashes and 2 en dashes, those attached to a word (``sandals—one``)
  stay one token.
- A bare word-final apostrophe, i.e. a plural possessive (56 cases,
  ``the objects'``), stays one token.
- The inch mark in ``39"`` is treated as a quote and gives ``39`` + ``"``.

Sentences are split by a deterministic rule with no external NLP dependency: a
run of whitespace right after ``[.!?]`` is a boundary. Closing quotes and closing
brackets stay with their sentence (``... "Stop." A car`` is two sentences), but a
boundary after a closing mark is taken only when the next sentence starts with an
uppercase letter, a digit or an opening quote. This keeps a mid-sentence
quotation that ends in terminal punctuation (``proclaim "Welcome Friends!" to all
...``) in one sentence; on DPG-Bench 4 of the 8 boundaries after a closing mark
were such quotations. There is no abbreviation list (``Dr.`` and so on); the
number of sentences is audited per corpus instead.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion.provenance import artifact_record, repository_root


# Punctuation marks that always become their own token: the ABC-6K tokenize() set plus double quotes.
SEPARATED_PUNCTUATION = frozenset(",.!?;:()\"“”")

# Apostrophes of the possessive clitic (straight and curly).
APOSTROPHES = frozenset("'’")

# Closing marks at the end of a sentence (they belong to the sentence even after terminal punctuation).
_CLOSING_MARKS = "\"'”’)]"

# A boundary may cross closing marks only when the next text looks like the start of a new sentence.
_SENTENCE_START_MARKS = "\"“'‘([A-Z0-9"

# The whitespace after terminal punctuation (and closing marks) is the sentence boundary.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?]+"
    r"(?:[" + re.escape(_CLOSING_MARKS) + r"]+(?=\s+[" + _SENTENCE_START_MARKS + r"]))?"
    r"(?P<gap>\s+)"
)

_NON_SPACE_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ParseToken:
    """One parse token and its character span in the original prompt."""

    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SentenceSpan:
    """Position of one sentence in the original prompt."""

    sentence_index: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class TokenizedSentence:
    """One sentence span and the parse tokens made from it (with original-prompt spans)."""

    sentence_index: int
    char_start: int
    char_end: int
    text: str
    tokens: tuple[ParseToken, ...]

    @property
    def raw_line(self) -> str:
        """The ``.raw`` line handed to Stage 1 (tokens joined by one space)."""
        return " ".join(token.text for token in self.tokens)


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split ``text`` into sentences at terminal punctuation and return their spans.

    Whitespace-only pieces are not sentences. The returned spans exclude
    surrounding whitespace, so ``text[span.char_start:span.char_end] == span.text``
    always holds.
    """
    cuts: list[tuple[int, int]] = [
        (match.start("gap"), match.end("gap")) for match in _SENTENCE_BOUNDARY_RE.finditer(text)
    ]
    pieces: list[tuple[int, int]] = []
    previous_end = 0
    for gap_start, gap_end in cuts:
        pieces.append((previous_end, gap_start))
        previous_end = gap_end
    pieces.append((previous_end, len(text)))

    spans: list[SentenceSpan] = []
    for start, end in pieces:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        spans.append(
            SentenceSpan(
                sentence_index=len(spans),
                char_start=start,
                char_end=end,
                text=text[start:end],
            )
        )
    return spans


def _is_possessive_clitic(chunk: str, index: int) -> bool:
    """Whether ``chunk[index]`` starts a possessive or contraction clitic (``'s`` / ``’s``)."""
    if chunk[index] not in APOSTROPHES:
        return False
    if index == 0 or not chunk[index - 1].isalnum():
        return False
    if index + 1 >= len(chunk) or chunk[index + 1] not in "sS":
        return False
    return index + 2 == len(chunk) or not chunk[index + 2].isalnum()


def _tokenize_chunk(chunk: str, chunk_start: int, out: list[ParseToken]) -> None:
    """Split a whitespace-free ``chunk`` at punctuation and clitic boundaries into ``out``."""
    position = 0
    while position < len(chunk):
        if chunk[position] in SEPARATED_PUNCTUATION:
            end = position + 1
        elif _is_possessive_clitic(chunk, position):
            end = position + 2
        else:
            end = position + 1
            while (
                end < len(chunk)
                and chunk[end] not in SEPARATED_PUNCTUATION
                and not _is_possessive_clitic(chunk, end)
            ):
                end += 1
        out.append(
            ParseToken(
                text=chunk[position:end],
                char_start=chunk_start + position,
                char_end=chunk_start + end,
            )
        )
        position = end


def tokenize_for_parse(text: str, *, char_offset: int = 0) -> list[ParseToken]:
    """Split ``text`` into parse tokens.

    ``char_offset`` is the start of ``text`` in the prompt it was cut from, so the
    returned spans are prompt coordinates. Each token text is an exact substring
    of the original string.
    """
    tokens: list[ParseToken] = []
    for match in _NON_SPACE_RE.finditer(text):
        _tokenize_chunk(match.group(0), char_offset + match.start(), tokens)
    return tokens


def split_and_tokenize(prompt: str) -> list[TokenizedSentence]:
    """Split ``prompt`` into sentences and tokenize each one for parsing (with original-prompt spans).

    A single-sentence prompt is the degenerate case of the same path.
    """
    sentences: list[TokenizedSentence] = []
    for span in split_sentences(prompt):
        tokens = tokenize_for_parse(span.text, char_offset=span.char_start)
        sentences.append(
            TokenizedSentence(
                sentence_index=span.sentence_index,
                char_start=span.char_start,
                char_end=span.char_end,
                text=span.text,
                tokens=tuple(tokens),
            )
        )
    return sentences


def build_sentence_map_rows(
    prompt_id: str,
    prompt: str,
    *,
    raw_line_offset: int,
) -> list[dict[str, Any]]:
    """Build the rows of the sentence map (``<prefix>.sentence_map.jsonl``) for one prompt.

    ``raw_line_offset`` is the 0-based ``.raw`` line index of the prompt's first
    sentence in the whole corpus.
    """
    rows: list[dict[str, Any]] = []
    for sentence in split_and_tokenize(prompt):
        rows.append(
            {
                "prompt_id": prompt_id,
                "sentence_index": sentence.sentence_index,
                "raw_line_index": raw_line_offset + len(rows),
                "sentence_char_start": sentence.char_start,
                "sentence_char_end": sentence.char_end,
                "tokens": [
                    {
                        "text": token.text,
                        "char_start": token.char_start,
                        "char_end": token.char_end,
                    }
                    for token in sentence.tokens
                ],
            }
        )
    return rows


# --- sentence map -------------------------------------------------------------------------------------

SENTENCE_MAP_ROW_KEYS: frozenset[str] = frozenset(
    {
        "prompt_id",
        "sentence_index",
        "raw_line_index",
        "sentence_char_start",
        "sentence_char_end",
        "tokens",
    }
)
SENTENCE_MAP_TOKEN_KEYS: frozenset[str] = frozenset({"text", "char_start", "char_end"})


class SentenceMapError(ValueError):
    """A sentence map violates its schema or its span invariants."""


@dataclass(frozen=True)
class SentenceMapRow:
    """One sentence-map JSONL row (prompt <-> sentence <-> .raw line <-> spans)."""

    prompt_id: str
    sentence_index: int
    raw_line_index: int
    sentence_char_start: int
    sentence_char_end: int
    tokens: tuple[ParseToken, ...]


def _int_field(payload: Mapping[str, Any], key: str, *, location: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise SentenceMapError(f"{location}.{key} must be an integer")
    return value


def sentence_map_row_from_payload(
    payload: Mapping[str, Any], *, location: str
) -> SentenceMapRow:
    """Parse one sentence-map row, rejecting any schema deviation."""

    if not isinstance(payload, Mapping):
        raise SentenceMapError(f"{location} must be an object")
    if set(payload) != SENTENCE_MAP_ROW_KEYS:
        unknown = sorted(set(payload) - SENTENCE_MAP_ROW_KEYS)
        missing = sorted(SENTENCE_MAP_ROW_KEYS - set(payload))
        raise SentenceMapError(
            f"{location}: sentence map key mismatch; unknown={unknown}, missing={missing}"
        )
    prompt_id = payload["prompt_id"]
    if not isinstance(prompt_id, str) or not prompt_id:
        raise SentenceMapError(f"{location}.prompt_id must be a non-empty string")
    raw_tokens = payload["tokens"]
    if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes)):
        raise SentenceMapError(f"{location}.tokens must be a list")
    if not raw_tokens:
        raise SentenceMapError(f"{location}.tokens must not be empty")
    tokens: list[ParseToken] = []
    for position, raw_token in enumerate(raw_tokens):
        where = f"{location}.tokens[{position}]"
        if not isinstance(raw_token, Mapping) or set(raw_token) != SENTENCE_MAP_TOKEN_KEYS:
            raise SentenceMapError(f"{where} must have text/char_start/char_end")
        text = raw_token["text"]
        if not isinstance(text, str) or not text or text.split() != [text]:
            raise SentenceMapError(f"{where}.text must be one whitespace-free token")
        tokens.append(
            ParseToken(
                text=text,
                char_start=_int_field(raw_token, "char_start", location=where),
                char_end=_int_field(raw_token, "char_end", location=where),
            )
        )
    return SentenceMapRow(
        prompt_id=prompt_id,
        sentence_index=_int_field(payload, "sentence_index", location=location),
        raw_line_index=_int_field(payload, "raw_line_index", location=location),
        sentence_char_start=_int_field(payload, "sentence_char_start", location=location),
        sentence_char_end=_int_field(payload, "sentence_char_end", location=location),
        tokens=tuple(tokens),
    )


def load_sentence_map(path: str | Path) -> tuple[SentenceMapRow, ...]:
    """Load ``<prefix>.sentence_map.jsonl`` in file order."""

    rows: list[SentenceMapRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SentenceMapError(f"{location}: invalid JSON: {exc}") from exc
            rows.append(sentence_map_row_from_payload(payload, location=location))
    if not rows:
        raise SentenceMapError(f"sentence map has no rows: {path}")
    for position, row in enumerate(rows):
        if row.raw_line_index != position:
            raise SentenceMapError(
                f"{path}: raw_line_index {row.raw_line_index} at file position {position}"
            )
    return tuple(rows)


def group_sentence_map(
    rows: Sequence[SentenceMapRow],
) -> tuple[tuple[str, tuple[SentenceMapRow, ...]], ...]:
    """Group sentence-map rows per prompt, keeping file order and contiguity."""

    groups: list[tuple[str, list[SentenceMapRow]]] = []
    seen: set[str] = set()
    for row in rows:
        if groups and groups[-1][0] == row.prompt_id:
            groups[-1][1].append(row)
            continue
        if row.prompt_id in seen:
            raise SentenceMapError(
                f"sentence map rows of prompt {row.prompt_id!r} are not contiguous"
            )
        seen.add(row.prompt_id)
        groups.append((row.prompt_id, [row]))
    for prompt_id, group in groups:
        expected = list(range(len(group)))
        if [row.sentence_index for row in group] != expected:
            raise SentenceMapError(
                f"prompt {prompt_id!r}: sentence_index must be 0..{len(group) - 1} in order"
            )
    return tuple((prompt_id, tuple(group)) for prompt_id, group in groups)


def validate_sentence_map_group(
    prompt: str, group: Sequence[SentenceMapRow]
) -> None:
    """Enforce the span invariants of one prompt's sentence-map rows."""

    if not group:
        raise SentenceMapError("sentence map group is empty")
    previous_end = 0
    for row in group:
        location = f"prompt {row.prompt_id!r} sentence {row.sentence_index}"
        if not 0 <= row.sentence_char_start < row.sentence_char_end <= len(prompt):
            raise SentenceMapError(f"{location}: sentence span outside the prompt")
        if row.sentence_char_start < previous_end:
            raise SentenceMapError(f"{location}: sentence span overlaps the previous one")
        previous_end = row.sentence_char_end
        token_end = row.sentence_char_start
        for position, token in enumerate(row.tokens):
            where = f"{location} token {position}"
            if token.char_start < token_end or token.char_end <= token.char_start:
                raise SentenceMapError(f"{where}: char span is not strictly increasing")
            if token.char_end > row.sentence_char_end:
                raise SentenceMapError(f"{where}: char span leaves the sentence span")
            if prompt[token.char_start:token.char_end] != token.text:
                raise SentenceMapError(
                    f"{where}: text {token.text!r} != prompt span "
                    f"{prompt[token.char_start:token.char_end]!r}"
                )
            token_end = token.char_end


# --- parse corpus (Stage 1 input) ---------------------------------------------------------------------


class CorpusError(ValueError):
    """A prompt manifest violates the parse-corpus input contract."""


def read_prompt_rows(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    """Read a prompt-only JSONL manifest: unique ``prompt_id``s, one stripped physical line per prompt."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CorpusError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"non-object row at {path}:{line_number}")
            prompt_id = row.get("prompt_id")
            prompt = row.get("prompt")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise CorpusError(f"invalid prompt_id at {path}:{line_number}")
            if prompt_id in seen:
                raise CorpusError(f"duplicate prompt_id {prompt_id!r}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise CorpusError(f"invalid prompt at {path}:{line_number}")
            if prompt != prompt.strip() or "\n" in prompt or "\r" in prompt:
                raise CorpusError(
                    f"prompt {prompt_id!r} must be one stripped physical line"
                )
            seen.add(prompt_id)
            rows.append(row)
    if len(rows) != expected_count:
        raise CorpusError(f"expected {expected_count} prompts, got {len(rows)}")
    return rows


def default_sentence_map_path(raw_output: Path) -> Path:
    """``<raw output stem>.sentence_map.jsonl`` next to the ``.raw`` file."""
    return raw_output.with_name(f"{raw_output.stem}.sentence_map.jsonl")


def build_sentence_map(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All sentence-map rows of a corpus, in prompt order."""
    sentence_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_rows = build_sentence_map_rows(
            str(row["prompt_id"]),
            str(row["prompt"]),
            raw_line_offset=len(sentence_rows),
        )
        if not prompt_rows:
            raise CorpusError(f"prompt {row['prompt_id']!r} yielded no sentence")
        sentence_rows.extend(prompt_rows)
    return sentence_rows


def raw_line(sentence_row: Mapping[str, Any]) -> str:
    """The ``.raw`` line of one sentence-map row (parse tokens joined by one space)."""
    return " ".join(str(token["text"]) for token in sentence_row["tokens"])


def _atomic_write_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CorpusError(f"output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, str]:
    """``{"path", "sha256"}`` with a repository-relative path when the file is inside the checkout."""
    record = artifact_record(path).to_dict(relative_to=repository_root())
    return {"path": record["path"], "sha256": record["sha256"]}


def write_parse_corpus(
    *,
    prompt_manifest: Path,
    raw_output: Path,
    source_map_output: Path,
    report_output: Path,
    expected_count: int,
    overwrite: bool,
    sentence_map_output: Path | None = None,
) -> dict[str, Any]:
    """Write the Stage 1 parse corpus of a prompt manifest.

    One ``.raw`` line is one tokenized sentence; the source map keeps the prompt
    rows; the sentence map ties each ``prompt_id`` to its sentences, ``.raw`` line
    indices and exact original-prompt character spans, so that Stage 3 can anchor
    sentence-local indices in the full prompt. Returns the report, which is also
    written to ``report_output``.
    """
    rows = read_prompt_rows(prompt_manifest, expected_count=expected_count)
    sentence_rows = build_sentence_map(rows)
    if sentence_map_output is None:
        sentence_map_output = default_sentence_map_path(raw_output)
    _atomic_write_text(
        raw_output,
        "".join(f"{raw_line(sentence_row)}\n" for sentence_row in sentence_rows),
        overwrite=overwrite,
    )
    _atomic_write_text(
        source_map_output,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        overwrite=overwrite,
    )
    _atomic_write_text(
        sentence_map_output,
        "".join(
            json.dumps(sentence_row, ensure_ascii=False, sort_keys=True) + "\n"
            for sentence_row in sentence_rows
        ),
        overwrite=overwrite,
    )
    sentences_per_prompt = Counter(str(sentence_row["prompt_id"]) for sentence_row in sentence_rows)
    report = {
        "schema": "holsd.native_prompt_corpus.v2",
        "status": "passed",
        "row_count": len(rows),
        "sentence_count": len(sentence_rows),
        "max_sentences_per_prompt": max(sentences_per_prompt.values()),
        "multi_sentence_prompt_count": sum(1 for count in sentences_per_prompt.values() if count > 1),
        "prompt_manifest": _file_record(prompt_manifest),
        "raw": _file_record(raw_output),
        "source_map": _file_record(source_map_output),
        "sentence_map": _file_record(sentence_map_output),
        "first_prompt_id": rows[0]["prompt_id"],
        "last_prompt_id": rows[-1]["prompt_id"],
    }
    _atomic_write_text(
        report_output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    return report
