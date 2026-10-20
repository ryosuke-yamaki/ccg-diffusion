r"""Derivation-tree helpers: attachment supertags, chart-tree walks and AUTO span trees.

* The supertags of the two PP interpretations (paper Sec. IV-C1): the
  preposition as a noun modifier ``(NP\NP)/NP`` or as a verb-phrase modifier
  ``((S\NP)\(S\NP))/NP``.
* Walks over Hol-CCG chart derivations (``left_child``/``right_child`` nodes
  with a ``span``), used to find the leaf at a token position.
* An AUTO reader that keeps each node's exact source substring and 1-based leaf
  span, used by Stage 2 to split a ``PARTIAL`` derivation into its skimmer
  fragments and re-emit each fragment byte for byte.

The module uses only the standard library.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, List, Optional, Tuple

NP_ATTACHMENT_CATEGORY: str = r"(NP\NP)/NP"
VP_ATTACHMENT_CATEGORY: str = r"((S\NP)\(S\NP))/NP"
PARTICIPIAL_CATEGORY: str = r"S[ng]\NP"
NOMINAL_MODIFIER_CATEGORY: str = r"NP\NP"
VERBAL_MODIFIER_CATEGORY: str = r"(S\NP)\(S\NP)"

# Interpretation label (KEEP value) -> supertag forced on the preposition.
INTERPRETATION_SUPERTAGS = MappingProxyType(
    {"np_attach": NP_ATTACHMENT_CATEGORY, "vp_attach": VP_ATTACHMENT_CATEGORY}
)


# --- chart derivations --------------------------------------------------------------------------------


def node_category(node: Any) -> str:
    return str(getattr(node, "cat", ""))


def children(node: Any) -> Tuple[Any, ...]:
    result: List[Any] = []
    left = getattr(node, "left_child", None)
    right = getattr(node, "right_child", None)
    if left is not None:
        result.append(left)
    if right is not None:
        result.append(right)
    return tuple(result)


def iter_leaves(root: Any) -> Iterable[Any]:
    """Yield derivation leaves from left to right."""

    if root is None:
        return
    if bool(getattr(root, "is_leaf", False)):
        yield root
        return
    for child in children(root):
        yield from iter_leaves(child)


def leaf_position(leaf: Any) -> Optional[int]:
    """Return a leaf's zero-based token position, or ``None`` if malformed."""

    span = getattr(leaf, "span", None)
    if not isinstance(span, tuple) or len(span) != 2:
        return None
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int) or end != start + 1:
        return None
    return start


def leaf_at_position(root: Any, position: int) -> Optional[Any]:
    """Return the unique leaf at ``position``; malformed duplicates fail closed."""

    matches = [leaf for leaf in iter_leaves(root) if leaf_position(leaf) == position]
    return matches[0] if len(matches) == 1 else None


def path_to_position(root: Any, position: int) -> Optional[List[Any]]:
    """Nodes from ``root`` down to the leaf at ``position``, or ``None``."""

    if root is None:
        return None
    if bool(getattr(root, "is_leaf", False)):
        return [root] if leaf_position(root) == position else None
    for child in children(root):
        child_path = path_to_position(child, position)
        if child_path is not None:
            return [root, *child_path]
    return None


# --- AUTO span trees ----------------------------------------------------------------------------------


class _AutoCursor:
    __slots__ = ("s", "i", "n")

    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def get(self) -> str:
        ch = self.peek()
        self.i += 1
        return ch

    def skip_ws(self) -> None:
        while self.peek() and self.peek().isspace():
            self.i += 1


def _read_angle_tokens(cur: _AutoCursor) -> list[str]:
    """Read tokens inside a '<...>' block (leading '<' already consumed)."""
    tokens: list[str] = []
    token_chars: list[str] = []
    while True:
        ch = cur.get()
        if not ch:
            raise ValueError("Unclosed '<...>' block")
        if ch == ">":
            if token_chars:
                tokens.append("".join(token_chars))
            break
        if ch.isspace():
            if token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
            continue
        token_chars.append(ch)
    if not tokens:
        raise ValueError("Empty '<...>' block")
    return tokens


class SpanNode:
    """One AUTO node: category, children, 1-based inclusive leaf span and its exact AUTO substring."""

    __slots__ = ("cat", "children", "start", "end", "auto")

    def __init__(
        self,
        cat: str,
        children: list["SpanNode"],
        start: int,
        end: int,
        auto: str = "",
    ):
        self.cat = cat
        self.children = children
        self.start = start
        self.end = end
        self.auto = auto


def parse_auto_tree_with_spans(derivation: str) -> SpanNode:
    """Parse an AUTO derivation string into a span-annotated tree (1-indexed leaves).

    The holccg AUTO reader returns an immutable node tree without the original
    substring of each node, but the PARTIAL expansion has to re-emit every
    fragment exactly as the parser wrote it, so this keeps its own reader.
    """
    cur = _AutoCursor(derivation)
    leaf_index = 0

    def parse_node() -> SpanNode:
        nonlocal leaf_index
        cur.skip_ws()
        node_start = cur.i
        if cur.get() != "(":
            raise ValueError("Expected '(' at node start")
        cur.skip_ws()
        if cur.get() != "<":
            raise ValueError("Expected '<' after '('")

        block = _read_angle_tokens(cur)
        kind, payload = block[0], block[1:]
        if kind == "T":
            if not payload:
                raise ValueError(f"Malformed T header: {block}")
            cat = payload[0]
            num_child: Optional[int] = None
            if len(payload) >= 3 and payload[2].isdigit():
                num_child = int(payload[2])

            children: list[SpanNode] = []
            if num_child is None:
                while True:
                    cur.skip_ws()
                    ch = cur.peek()
                    if ch == "(":
                        children.append(parse_node())
                        continue
                    if ch == ")":
                        cur.get()
                        break
                    raise ValueError(
                        f"Unexpected char inside T node: {ch!r} at pos {cur.i}"
                    )
            else:
                for _ in range(num_child):
                    children.append(parse_node())
                cur.skip_ws()
                if cur.get() != ")":
                    raise ValueError("Expected ')' at end of T node")

            if not children:
                raise ValueError(f"T node with no children: {block}")
            return SpanNode(
                cat,
                children,
                children[0].start,
                children[-1].end,
                derivation[node_start:cur.i],
            )

        if kind == "L":
            if not payload:
                raise ValueError(f"Malformed L header: {block}")
            cat = payload[0]
            leaf_index += 1
            cur.skip_ws()
            if cur.get() != ")":
                raise ValueError("Expected ')' after L node")
            return SpanNode(
                cat,
                [],
                leaf_index,
                leaf_index,
                derivation[node_start:cur.i],
            )

        raise ValueError(f"Unknown node kind: {kind}")

    root = parse_node()
    cur.skip_ws()
    if cur.i != cur.n:
        trailing = derivation[cur.i:cur.i + 40]
        raise ValueError(f"Unconsumed trailing chars: {trailing!r}")
    return root


def extract_partial_subtrees(
    root: SpanNode, partial_root_cat: str
) -> list[SpanNode]:
    """Return the real subtrees represented by a synthetic PARTIAL root.

    Stage 1 joins skimmer fragments with one or more synthetic PARTIAL nodes.
    Walking those nodes recursively is deliberately more permissive than
    assuming one particular left-branching shape, and also supports a unary
    PARTIAL wrapper.
    """
    if root.cat != partial_root_cat:
        raise ValueError(f"Root cat {root.cat!r} != {partial_root_cat!r}")

    subtrees: list[SpanNode] = []

    def visit(node: SpanNode) -> None:
        if node.cat == partial_root_cat:
            if not node.children:
                raise ValueError("PARTIAL node has no children")
            for child in node.children:
                visit(child)
            return
        subtrees.append(node)

    visit(root)
    if not subtrees:
        raise ValueError("PARTIAL tree contains no real subtrees")

    expected_start = root.start
    for subtree in subtrees:
        if subtree.start != expected_start:
            raise ValueError(
                "PARTIAL subtree spans are not contiguous: "
                f"expected start {expected_start}, got {subtree.start}"
            )
        expected_start = subtree.end + 1
    if expected_start != root.end + 1:
        raise ValueError(
            "PARTIAL subtree spans do not cover the full sentence: "
            f"covered through {expected_start - 1}, root ends at {root.end}"
        )
    return subtrees
