"""Readers for Hol-CCG AUTO derivations and C&C ``generate -e`` marked dependencies.

It also aligns whitespace source tokens to CLIP token indices. The module
uses only the standard library; tokenizers are passed in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


class MarkedDepsError(ValueError):
    """An AUTO, marked-dependency or tokenizer artifact violates the parsing contract."""


@dataclass(frozen=True)
class CCGCategory:
    """A small CCG category AST after C&C head/slot annotations are removed."""

    atom: Optional[str] = None
    result: Optional["CCGCategory"] = None
    argument: Optional["CCGCategory"] = None
    slash: Optional[str] = None

    @property
    def is_atomic(self) -> bool:
        return self.atom is not None

    @property
    def base_atom(self) -> Optional[str]:
        if self.atom is None:
            return None
        return self.atom.split("[", 1)[0]

    @property
    def feature(self) -> Optional[str]:
        if self.atom is None or "[" not in self.atom:
            return None
        return self.atom.split("[", 1)[1].rstrip("]")


_HEAD_ANNOTATION_RE = re.compile(r"\{[^{}]*\}")
_SLOT_ANNOTATION_RE = re.compile(r"<\d+>")


class _CategoryCursor:
    def __init__(self, text: str):
        self.text = text
        self.index = 0

    def peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def get(self) -> str:
        value = self.peek()
        self.index += bool(value)
        return value


def _clean_category(text: str) -> str:
    value = _HEAD_ANNOTATION_RE.sub("", text)
    value = _SLOT_ANNOTATION_RE.sub("", value)
    return value.replace(" ", "").replace("*", "")


def parse_ccg_category(text: str) -> CCGCategory:
    """Parse a bare or C&C marked-up CCG category.

    Slash application is left associative, matching the explicit CCG category
    notation used by AUTO and C&C's markedup inventory.
    """

    cleaned = _clean_category(text)
    if not cleaned:
        raise MarkedDepsError(f"empty CCG category after annotation removal: {text!r}")
    cursor = _CategoryCursor(cleaned)

    def primary() -> CCGCategory:
        if cursor.peek() == "(":
            cursor.get()
            value = expression()
            if cursor.get() != ")":
                raise MarkedDepsError(f"unclosed CCG category group: {text!r}")
            return value
        chars: list[str] = []
        bracket_depth = 0
        while cursor.peek():
            char = cursor.peek()
            if bracket_depth == 0 and char in "()/\\":
                break
            char = cursor.get()
            chars.append(char)
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise MarkedDepsError(f"malformed CCG feature: {text!r}")
        atom = "".join(chars)
        if not atom or bracket_depth:
            raise MarkedDepsError(f"malformed CCG atom: {text!r}")
        return CCGCategory(atom=atom)

    def expression() -> CCGCategory:
        value = primary()
        while cursor.peek() in {"/", "\\"}:
            slash = cursor.get()
            argument = primary()
            value = CCGCategory(result=value, argument=argument, slash=slash)
        return value

    parsed = expression()
    if cursor.peek():
        raise MarkedDepsError(
            f"unconsumed CCG category suffix {cleaned[cursor.index:]!r}: {text!r}"
        )
    return parsed


def _is_atomic(category: CCGCategory, atom: str) -> bool:
    return category.is_atomic and category.base_atom == atom


def is_atomic_nominal(category: CCGCategory) -> bool:
    return category.is_atomic and category.base_atom in {"N", "NP"}


def is_noun_modifier(category: CCGCategory) -> bool:
    return bool(
        not category.is_atomic
        and category.result is not None
        and category.argument is not None
        and _is_atomic(category.result, "N")
        and _is_atomic(category.argument, "N")
    )


def is_degree_modifier(category: CCGCategory) -> bool:
    return bool(
        not category.is_atomic
        and category.result is not None
        and category.argument is not None
        and is_noun_modifier(category.result)
        and is_noun_modifier(category.argument)
    )


def is_predicative_adjective(category: CCGCategory) -> bool:
    return bool(
        category.slash == "\\"
        and category.result is not None
        and category.argument is not None
        and _is_atomic(category.result, "S")
        and category.result.feature == "adj"
        and _is_atomic(category.argument, "NP")
    )


def is_passive_verb_frame(category: CCGCategory) -> bool:
    r"""Return whether a result spine contains a passive ``S\NP`` frame."""

    current = category
    while not current.is_atomic:
        if (
            current.slash == "\\"
            and current.result is not None
            and current.argument is not None
            and _is_atomic(current.result, "S")
            and current.result.feature in {"pss", "pt"}
            and _is_atomic(current.argument, "NP")
        ):
            return True
        if current.result is None:
            return False
        current = current.result
    return False


@dataclass
class AutoNode:
    category: str
    children: tuple["AutoNode", ...]
    head_child: Optional[int]
    start: int
    end: int
    leaf_category: Optional[str] = None
    word: Optional[str] = None

    @property
    def head_index(self) -> int:
        if not self.children:
            return self.start
        assert self.head_child is not None
        return self.children[self.head_child].head_index


class _AutoCursor:
    def __init__(self, text: str):
        self.text = text
        self.index = 0

    def peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def get(self) -> str:
        value = self.peek()
        self.index += bool(value)
        return value

    def skip_ws(self) -> None:
        while self.peek() and self.peek().isspace():
            self.index += 1


def _read_auto_header(cursor: _AutoCursor) -> list[str]:
    tokens: list[str] = []
    chars: list[str] = []
    while True:
        char = cursor.get()
        if not char:
            raise MarkedDepsError("unclosed AUTO <...> header")
        if char == ">":
            if chars:
                tokens.append("".join(chars))
            return tokens
        if char.isspace():
            if chars:
                tokens.append("".join(chars))
                chars = []
        else:
            chars.append(char)


def parse_auto_derivation(derivation: str) -> AutoNode:
    """Parse one AUTO derivation while retaining head direction and leaves."""

    cursor = _AutoCursor(derivation)
    leaf_index = 0

    def node() -> AutoNode:
        nonlocal leaf_index
        cursor.skip_ws()
        if cursor.get() != "(":
            raise MarkedDepsError("expected '(' at AUTO node start")
        cursor.skip_ws()
        if cursor.get() != "<":
            raise MarkedDepsError("expected '<' at AUTO node header")
        header = _read_auto_header(cursor)
        if not header:
            raise MarkedDepsError("empty AUTO node header")
        kind = header[0]
        payload = header[1:]
        if kind == "L":
            if len(payload) < 4:
                raise MarkedDepsError(f"malformed AUTO leaf header: {header!r}")
            leaf_index += 1
            cursor.skip_ws()
            if cursor.get() != ")":
                raise MarkedDepsError("expected ')' after AUTO leaf")
            return AutoNode(
                category=payload[0],
                children=(),
                head_child=None,
                start=leaf_index,
                end=leaf_index,
                leaf_category=payload[0],
                word=payload[3],
            )
        if kind != "T" or len(payload) < 3:
            raise MarkedDepsError(f"malformed AUTO tree header: {header!r}")
        try:
            head_child = int(payload[1])
            child_count = int(payload[2])
        except ValueError as exc:
            raise MarkedDepsError(f"non-integer AUTO tree header: {header!r}") from exc
        children = tuple(node() for _ in range(child_count))
        cursor.skip_ws()
        if cursor.get() != ")":
            raise MarkedDepsError("expected ')' after AUTO tree node")
        if not children or not 0 <= head_child < len(children):
            raise MarkedDepsError(f"invalid AUTO head child: {header!r}")
        return AutoNode(
            category=payload[0],
            children=children,
            head_child=head_child,
            start=children[0].start,
            end=children[-1].end,
        )

    root = node()
    cursor.skip_ws()
    if cursor.peek():
        raise MarkedDepsError(
            f"unconsumed AUTO derivation suffix: {derivation[cursor.index:cursor.index + 80]!r}"
        )
    return root


@dataclass(frozen=True)
class AutoLeaf:
    source_index: int
    category: str
    word: str
    projected_categories: tuple[str, ...]
    fragment_index: int


@dataclass(frozen=True)
class AutoSentence:
    id_line: str
    derivation: str
    root: AutoNode
    leaves: tuple[AutoLeaf, ...]


def _partial_fragments(root: AutoNode) -> list[AutoNode]:
    if root.category != "PARTIAL":
        return [root]
    fragments: list[AutoNode] = []

    def visit(node: AutoNode) -> None:
        if node.category == "PARTIAL":
            for child in node.children:
                visit(child)
        else:
            fragments.append(node)

    visit(root)
    return fragments


def auto_sentence(id_line: str, derivation: str) -> AutoSentence:
    root = parse_auto_derivation(derivation)
    raw_leaves: dict[int, AutoNode] = {}
    projections: dict[int, list[str]] = {}

    def collect(node: AutoNode) -> None:
        projections.setdefault(node.head_index, []).append(node.category)
        if not node.children:
            raw_leaves[node.start] = node
            return
        for child in node.children:
            collect(child)

    collect(root)
    fragment_by_index: dict[int, int] = {}
    for fragment_index, fragment in enumerate(_partial_fragments(root), start=1):
        for source_index in range(fragment.start, fragment.end + 1):
            fragment_by_index[source_index] = fragment_index
    leaves: list[AutoLeaf] = []
    for source_index in range(1, root.end + 1):
        leaf = raw_leaves[source_index]
        assert leaf.leaf_category is not None and leaf.word is not None
        leaves.append(
            AutoLeaf(
                source_index=source_index,
                category=leaf.leaf_category,
                word=leaf.word,
                projected_categories=tuple(projections[source_index]),
                fragment_index=fragment_by_index[source_index],
            )
        )
    return AutoSentence(id_line, derivation, root, tuple(leaves))


def load_auto_sentences(path: str | Path) -> list[AutoSentence]:
    """Load sentence-level AUTO records, including wrapped derivation lines."""

    records: list[AutoSentence] = []
    current_id: Optional[str] = None
    derivation_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, derivation_lines
        if current_id is None:
            return
        derivation = " ".join(derivation_lines).strip()
        if not derivation:
            raise MarkedDepsError(f"AUTO record has no derivation: {current_id}")
        records.append(auto_sentence(current_id, derivation))
        current_id = None
        derivation_lines = []

    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("ID="):
                flush()
                current_id = line
            elif current_id is None:
                if line.strip() and not line.lstrip().startswith("#"):
                    raise MarkedDepsError(
                        f"unexpected AUTO content before first ID: {line[:100]!r}"
                    )
            elif line.strip():
                derivation_lines.append(line.strip())
    flush()
    if not records:
        raise MarkedDepsError(f"no AUTO records found: {path}")
    return records


@dataclass(frozen=True)
class MarkedDependency:
    head_index: int
    head_word: str
    marked_category: str
    slot: int
    argument_index: int
    argument_word: str
    rule_id: int
    long_range_category: Optional[str] = None


def _dependency_endpoint(value: str) -> tuple[str, int]:
    word, separator, index = value.rpartition("_")
    if not separator or not index.isdigit():
        raise MarkedDepsError(f"dependency endpoint lacks a numeric _N suffix: {value!r}")
    return word, int(index)


def parse_marked_dependency(line: str) -> MarkedDependency:
    fields = line.split()
    if len(fields) not in {5, 6}:
        raise MarkedDepsError(f"malformed marked dependency line: {line!r}")
    head_word, head_index = _dependency_endpoint(fields[0])
    argument_word, argument_index = _dependency_endpoint(fields[3])
    try:
        slot = int(fields[2])
        rule_id = int(fields[4])
    except ValueError as exc:
        raise MarkedDepsError(f"non-integer dependency slot/rule: {line!r}") from exc
    return MarkedDependency(
        head_index=head_index,
        head_word=head_word,
        marked_category=fields[1],
        slot=slot,
        argument_index=argument_index,
        argument_word=argument_word,
        rule_id=rule_id,
        long_range_category=fields[5] if len(fields) == 6 else None,
    )


def load_marked_dependency_blocks(path: str | Path) -> list[list[MarkedDependency]]:
    """Load blank-separated sentence blocks from ``generate -e`` output."""

    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        index += 1
    while index < len(lines) and not lines[index].strip():
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
    return [[parse_marked_dependency(line) for line in block] for block in raw_blocks]


@dataclass(frozen=True)
class ClipSourceAnchor:
    source_index: int
    surface: str
    char_start: int
    char_end: int
    piece_indices: tuple[int, ...]
    lexical_piece_indices: tuple[int, ...]
    anchor_index: int
    anchor_kind: str
    decoded_pieces: tuple[str, ...]


def tokenizer_ids(value: Any) -> list[int]:
    ids = value["input_ids"] if isinstance(value, Mapping) else value.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise MarkedDepsError("batched tokenizer output is not supported")
        ids = ids[0]
    return [int(item) for item in ids]


def align_whitespace_tokens_to_clip(
    prompt: str, tokenizer: Any
) -> tuple[ClipSourceAnchor, ...]:
    """Map C&C whitespace positions to BOS-inclusive CLIP indices.

    The exact surface of each source token is encoded independently and the
    concatenation is required to equal the prompt-level encoding.  This makes
    duplicate words and punctuation deterministic without a parser or lemma
    lookup.  The final alphanumeric BPE is the attention anchor; punctuation-
    only tokens retain a deterministic final-piece fallback for full coverage.
    """

    matches = list(re.finditer(r"\S+", prompt))
    full_content = tokenizer_ids(tokenizer(prompt, add_special_tokens=False))
    full_special = tokenizer_ids(tokenizer(prompt))
    if len(full_special) != len(full_content) + 2 or full_special[1:-1] != full_content:
        raise MarkedDepsError("tokenizer does not expose CLIP's one-BOS/one-EOS layout")
    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and max_length < 1_000_000:
        if len(full_special) > max_length:
            raise MarkedDepsError(
                f"prompt requires {len(full_special)} CLIP tokens, maximum is {max_length}"
            )

    concatenated: list[int] = []
    anchors: list[ClipSourceAnchor] = []
    next_global_index = 1
    for source_index, match in enumerate(matches, start=1):
        surface = match.group(0)
        local_ids = tokenizer_ids(
            tokenizer(surface, add_special_tokens=False)
        )
        if not local_ids:
            raise MarkedDepsError(f"CLIP emitted no pieces for source token {surface!r}")
        decoded = tuple(str(tokenizer.decode(token_id)) for token_id in local_ids)
        piece_indices = tuple(
            range(next_global_index, next_global_index + len(local_ids))
        )
        lexical = tuple(
            piece_index
            for piece_index, piece in zip(piece_indices, decoded)
            if any(char.isalnum() for char in piece)
        )
        anchor_kind = "lexical" if lexical else "punctuation_fallback"
        anchor_index = lexical[-1] if lexical else piece_indices[-1]
        anchors.append(
            ClipSourceAnchor(
                source_index=source_index,
                surface=surface,
                char_start=match.start(),
                char_end=match.end(),
                piece_indices=piece_indices,
                lexical_piece_indices=lexical,
                anchor_index=anchor_index,
                anchor_kind=anchor_kind,
                decoded_pieces=decoded,
            )
        )
        concatenated.extend(local_ids)
        next_global_index += len(local_ids)
    if concatenated != full_content:
        raise MarkedDepsError(
            "concatenated source-token CLIP pieces differ from prompt encoding"
        )
    return tuple(anchors)
