r"""Supertag-Constrained Parsing and the choice of the ambiguous preposition (paper Sec. IV-C1).

A TAB PP prompt is parsed once without constraints. The preposition whose
attachment is ambiguous is then chosen from the grammar alone:

* **Schema pairs.** Every lexical category of the parser is peeled down to a
  modifier core ``NP\NP`` (noun modifier) or ``(S\NP)\(S\NP)`` (verb-phrase
  modifier). Two categories with the same argument chain and different cores
  form a pair, e.g. ``(NP\NP)/NP`` and ``((S\NP)\(S\NP))/NP``.
* **Anchoring.** Each token's category in the unconstrained parse is peeled
  until a self-modifier ``X\X``/``X/X``, a non-nominal atomic category or a
  type-change source that the grammar sends to both cores; the peeled chain
  names the one pair tried at that token.
* **Realizability.** A token is a candidate site when Supertag-Constrained
  Parsing with each category of the pair keeps the forced leaf and derives the
  root category of the unconstrained parse.

The decision cascade (tiers ``t0/a/b1/b2/b3``): one realizable site wins
(``single_site``). With none, a governor whose category strands a trailing
argument in both constrained parses is re-forced with that slot deleted
(``frame_repair``). With several, a site nested inside another site's flipped
constituent is dropped (``nesting``), then sites whose two interpretations give
the preposition the same resolved dependencies (``vacuity``); if none is left,
the ``N\N``-core variant of the noun-modifier category is retried
(``nominal_core_variant``).

The two constrained derivations of the chosen site are the interpretations
``np_attach`` and ``vp_attach``. ``setup_mapping_evidence`` builds the
syntax evidence that ``evaluation.tab.map_tab_item_readings_to_setups`` maps to
the TAB setups. Hol-CCG (``parsing.holccg``) is imported only by the functions
that parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from ccg_diffusion.parsing.ccg import CCGCategory, MarkedDependency, parse_ccg_category
from ccg_diffusion.parsing.derivation import (
    NOMINAL_MODIFIER_CATEGORY,
    VERBAL_MODIFIER_CATEGORY,
    iter_leaves,
    leaf_at_position,
    leaf_position,
    node_category,
    parse_auto_tree_with_spans,
)

NP_MODIFIER_CORE = NOMINAL_MODIFIER_CATEGORY
VP_MODIFIER_CORE = VERBAL_MODIFIER_CATEGORY
MODIFIER_CORES: Mapping[str, str] = {NP_MODIFIER_CORE: "np_attach", VP_MODIFIER_CORE: "vp_attach"}
CORE_BY_READING: Mapping[str, str] = {"np_attach": NP_MODIFIER_CORE, "vp_attach": VP_MODIFIER_CORE}
READINGS: tuple[str, str] = ("np_attach", "vp_attach")
NOMINAL_CORE_VARIANT = "N\\N"
NOMINAL_PROJECTION_ATOMS = frozenset({"N", "NP"})
TYPECHANGE_SEPARATOR = "-->"
TIERS: tuple[str, ...] = ("t0", "a", "b1", "b2", "b3")  # recorded tier names; all are always on
RESOLUTION_MAX_DEPTH = 5
PARSE_GRAMMAR = "treebank"
AUTO_ID = "ID=prompts.{index} PARSER=HOLCCG NUMPARSE=1"  # index: 1-based PP item index (recorded in traces)
_LEAF_CATEGORY_PATTERN = re.compile(r"\(<L ([^ ]+) ")

PD_FIELDS = ("exist", "corr", "leak", "possession")
TWO_SIDED_HOLDER_BRANCH = "two_sided_holder_branch"
ONE_SIDED_MISSING_SLOT = "one_sided_missing_slot"
NO_SITE_PAIR_BRANCH = "no_site_pair_in_either_reading"

# (auto lines with ID lines interleaved, raw sentence lines) -> one marked-dependency block per sentence.
DependencyGenerator = Callable[[Sequence[str], Sequence[str]], list[list[MarkedDependency]]]


# --- schema pairs -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaPair:
    """One noun-modifier / verb-phrase-modifier category pair sharing an argument chain."""

    chain_id: str
    chain: tuple[tuple[str, str], ...]
    np_category: str
    vp_category: str
    rightward_argument_count: int
    leftward_argument_count: int

    def category(self, reading: str) -> str:
        """The forced category this pair supplies for ``reading``."""

        return self.np_category if reading == "np_attach" else self.vp_category


@dataclass(frozen=True)
class Grammar:
    """The parser's schema pairs (by argument chain), type-change stops and lexical vocabulary."""

    pairs_by_chain: Mapping[tuple[tuple[str, str], ...], SchemaPair]
    typechange_stops: frozenset[str]
    vocabulary: frozenset[str]


def render_category(category: CCGCategory) -> str:
    """Render a parsed category back to its canonical bracketed string."""

    if category.is_atomic:
        return str(category.atom)
    result, argument = category.result, category.argument
    if result is None or argument is None:
        raise ValueError(f"malformed CCG category node: {category!r}")
    left = render_category(result)
    right = render_category(argument)
    if not result.is_atomic:
        left = f"({left})"
    if not argument.is_atomic:
        right = f"({right})"
    return f"{left}{category.slash}{right}"


def _split_modifier_core(category: CCGCategory) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """Peel arguments until a modifier core is exposed: ``(core, chain innermost first)`` or ``None``."""

    chain: list[tuple[str, str]] = []
    current = category
    while not current.is_atomic:
        text = render_category(current)
        if text in MODIFIER_CORES:
            return text, tuple(reversed(chain))
        argument = current.argument
        if argument is None or current.result is None:
            return None
        chain.append((str(current.slash), render_category(argument)))
        current = current.result
    return None


def _chain_id(chain: Sequence[tuple[str, str]]) -> str:
    """A short readable identifier of one argument chain."""

    if not chain:
        return "(bare)"
    return "|".join(f"{slash}{argument}" for slash, argument in chain)


def _discover_schema_pairs(parser: Any) -> list[SchemaPair]:
    """Every schema pair of the parser's lexical vocabulary, sorted by chain length and noun category."""

    vocabulary = list(parser.word_category2index)
    by_chain: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for text in vocabulary:
        if TYPECHANGE_SEPARATOR in text:
            continue
        try:
            parsed = parse_ccg_category(text)
            rendered = render_category(parsed)
        except Exception:  # noqa: BLE001 - an unreadable vocabulary entry is not a candidate
            continue
        if rendered != text.replace(" ", ""):
            continue
        split = _split_modifier_core(parsed)
        if split is None:
            continue
        core, chain = split
        by_chain.setdefault(chain, {})[core] = text

    pairs: list[SchemaPair] = []
    for chain, cores in by_chain.items():
        if NP_MODIFIER_CORE in cores and VP_MODIFIER_CORE in cores:
            pairs.append(
                SchemaPair(
                    chain_id=_chain_id(chain),
                    chain=chain,
                    np_category=cores[NP_MODIFIER_CORE],
                    vp_category=cores[VP_MODIFIER_CORE],
                    rightward_argument_count=sum(1 for slash, _ in chain if slash == "/"),
                    leftward_argument_count=sum(1 for slash, _ in chain if slash == "\\"),
                )
            )
    pairs.sort(key=lambda pair: (len(pair.chain), pair.np_category))
    return pairs


def _typechange_stops(parser: Any) -> frozenset[str]:
    """Categories that the grammar's ``SRC-->DST`` type changes send to both modifier cores."""

    by_source: dict[str, set[str]] = {}
    for _key, parents in parser.binary_rule.items():
        for parent in parents:
            if TYPECHANGE_SEPARATOR not in parent:
                continue
            steps = parent.split(TYPECHANGE_SEPARATOR)
            for source, target in zip(steps, steps[1:]):
                if target in MODIFIER_CORES:
                    by_source.setdefault(source, set()).add(target)
    return frozenset(source for source, targets in by_source.items() if len(targets) >= 2)


def load_grammar(parser: Any) -> Grammar:
    """Set the treebank parse grammar and read the schema pairs and type-change stops off ``parser``."""

    parser.set_parse_grammar(PARSE_GRAMMAR)
    pairs = _discover_schema_pairs(parser)
    return Grammar({pair.chain: pair for pair in pairs}, _typechange_stops(parser),
                   frozenset(parser.word_category2index))


# --- the chosen site ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRepair:
    """The governor whose category lost one argument slot in both constrained parses (tier ``a``)."""

    governor: int
    governor_word: str
    governor_category: str
    repaired_category: str
    deleted_argument: str


@dataclass(frozen=True)
class PrepositionSite:
    """The ambiguous preposition: 0-based token position, its categories and the deciding tier."""

    position: int
    word: str
    base_category: str
    chain_id: str
    np_category: str
    vp_category: str
    root_category: str
    tier: str
    nominal_core: str | None
    frame_repair: FrameRepair | None

    @property
    def source_index(self) -> int:
        return self.position + 1

    def forced_supertags(self, reading: str) -> dict[int, str]:
        """``{position: category}`` of one interpretation (plus the repaired governor)."""

        forced = {self.position: self.np_category if reading == "np_attach" else self.vp_category}
        if self.frame_repair is not None:
            forced[self.frame_repair.governor] = self.frame_repair.repaired_category
        return forced


@dataclass(frozen=True)
class ConstrainedDerivation:
    """The Supertag-Constrained derivation of one interpretation (AUTO derivation line)."""

    reading: str
    forced_supertags: Mapping[int, str]
    derivation: str
    root_category: str


# --- constrained parses and trials --------------------------------------------------------------------


def _forced_probe(
    parser: Any, sentence: str, forced: Mapping[int, str], site_position: int, base_root_category: str,
) -> dict[str, Any]:
    """Force ``forced`` (position -> category) and classify the parse against realizability."""

    slot: dict[str, Any] = {
        "forced_category": forced[site_position],
        "forced_positions": {str(pos): cat for pos, cat in sorted(forced.items())},
        "status": None,
        "leaf_category": None,
        "leaf_kept": False,
        "root_category": None,
        "root_consistent": False,
        "error": None,
        "auto": None,
    }
    from ccg_diffusion.parsing.holccg import supertag_constrained_parse

    try:
        result = supertag_constrained_parse(parser, sentence, forced)
    except Exception as exc:  # noqa: BLE001 - a rejected forcing is data
        slot["status"] = "error"
        slot["error"] = f"{type(exc).__name__}: {exc}"
        return slot
    status = str(result.get("status"))
    root = result.get("root_cat")
    if status != "ok" or root is None or not result.get("auto"):
        slot["status"] = "fail"
        return slot
    kept = True
    for position, category in forced.items():
        leaf = leaf_at_position(root, position)
        leaf_category = str(getattr(leaf, "cat", None)) if leaf is not None else None
        if position == site_position:
            slot["leaf_category"] = leaf_category
        if leaf_category != category:
            kept = False
    root_category = str(getattr(root, "cat", None))
    slot.update(
        {
            "status": "ok",
            "leaf_kept": kept,
            "root_category": root_category,
            "root_consistent": root_category == base_root_category,
            "auto": str(result["auto"]).strip(),
        }
    )
    return slot


def _trial_outcome(np_slot: Mapping[str, Any], vp_slot: Mapping[str, Any]) -> str:
    if np_slot["status"] != "ok" or vp_slot["status"] != "ok":
        return "parse_failed"
    if not np_slot["leaf_kept"] or not vp_slot["leaf_kept"]:
        return "leaf_not_kept"
    if not np_slot["root_consistent"] or not vp_slot["root_consistent"]:
        return "root_mismatch"
    return "realizable"


def _run_trial(
    parser: Any, sentence: str, *, position: int, pair: SchemaPair, tokens: Sequence[str],
    base_categories: Mapping[int, str], base_root_category: str,
) -> dict[str, Any]:
    """Force both categories of ``pair`` at ``position`` and classify the trial."""

    slots = {
        reading: _forced_probe(parser, sentence, {position: pair.category(reading)}, position, base_root_category)
        for reading in READINGS
    }
    return {
        "position": position,
        "word": tokens[position],
        "base_category": base_categories.get(position),
        "chain_id": pair.chain_id,
        "np_category": pair.np_category,
        "vp_category": pair.vp_category,
        "outcome": _trial_outcome(slots["np_attach"], slots["vp_attach"]),
        "readings": slots,
    }


def _is_self_modifier(category: CCGCategory) -> bool:
    """Whether ``category`` is ``X\\X`` or ``X/X``."""

    if category.is_atomic:
        return False
    result, argument = category.result, category.argument
    if result is None or argument is None:
        return False
    return render_category(result) == render_category(argument)


def _anchor_chain(base_category: str, typechange_stops: frozenset[str]) -> tuple[tuple[str, str], ...] | None:
    """The argument chain peeled off ``base_category`` before the first stop (``None``: no stop)."""

    current = parse_ccg_category(base_category)
    peeled: list[tuple[str, str]] = []
    while True:
        text = render_category(current)
        if (_is_self_modifier(current) or (current.is_atomic and current.base_atom not in NOMINAL_PROJECTION_ATOMS)
                or text in typechange_stops):
            return tuple(reversed(peeled))
        result, argument = current.result, current.argument
        if result is None or argument is None:
            return None
        peeled.append((str(current.slash), render_category(argument)))
        current = result


def _search_sites(
    parser: Any, sentence: str, *, tokens: Sequence[str], base_root_category: str,
    base_categories: Mapping[int, str], grammar: Grammar,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(realizable trials, root-mismatch trials)`` of the anchored sweep over all tokens."""

    token_count = len(tokens)
    realizable: list[dict[str, Any]] = []
    root_mismatch: list[dict[str, Any]] = []
    for position in range(token_count):
        base_category = base_categories.get(position)
        if base_category is None:
            continue
        try:
            chain = _anchor_chain(base_category, grammar.typechange_stops)
        except Exception:  # noqa: BLE001 - an unreadable base category is not a candidate
            continue
        if chain is None:
            continue
        pair = grammar.pairs_by_chain.get(chain)
        if pair is None or position == 0 or pair.rightward_argument_count > token_count - position - 1:
            continue
        trial = _run_trial(parser, sentence, position=position, pair=pair, tokens=tokens,
                           base_categories=base_categories, base_root_category=base_root_category)
        if trial["outcome"] == "realizable":
            realizable.append(trial)
        elif trial["outcome"] == "root_mismatch":
            for slot in trial["readings"].values():
                slot.pop("auto", None)
            root_mismatch.append(trial)
    return realizable, root_mismatch


# --- tier a: frame repair -----------------------------------------------------------------------------


def _needs_parens(text: str) -> bool:
    return not parse_ccg_category(text).is_atomic


def rebuild_category(remainder: str, chain_outermost_first: Sequence[tuple[str, str]]) -> str:
    """Reassemble a category from a remainder and outermost-first arguments."""

    text = remainder
    for slash, argument in reversed(list(chain_outermost_first)):
        left = f"({text})" if _needs_parens(text) else text
        right = f"({argument})" if _needs_parens(argument) else argument
        text = f"{left}{slash}{right}"
    return text


def peel_all_arguments(category: str) -> tuple[str, list[tuple[str, str]]]:
    """The innermost result and the outermost-first argument list of ``category``."""

    current = parse_ccg_category(category)
    peeled: list[tuple[str, str]] = []
    while not current.is_atomic and current.result is not None and current.argument is not None:
        peeled.append((str(current.slash), render_category(current.argument)))
        current = current.result
    return render_category(current), peeled


def stranded_arguments(root_category: str, base_root_category: str) -> list[tuple[str, str]] | None:
    """The trailing arguments ``root_category`` adds over the base root (``None``: not of that form)."""

    if root_category == base_root_category:
        return []
    try:
        current = parse_ccg_category(root_category)
    except Exception:  # noqa: BLE001 - an unreadable root is simply not repairable
        return None
    peeled: list[tuple[str, str]] = []
    while True:
        if render_category(current) == base_root_category:
            return peeled
        if current.is_atomic or current.result is None or current.argument is None:
            return None
        peeled.append((str(current.slash), render_category(current.argument)))
        current = current.result


def delete_argument_slot(category: str, argument: str) -> str | None:
    """Delete the outermost argument slot matching ``argument`` from ``category``."""

    remainder, chain = peel_all_arguments(category)
    for index, (_slash, argument_text) in enumerate(chain):
        if argument_text == argument:
            del chain[index]
            return rebuild_category(remainder, chain)
    return None


def _frame_repair(
    parser: Any, sentence: str, *, tokens: Sequence[str], base_categories: Mapping[int, str],
    base_root_category: str, root_mismatch_sites: Sequence[Mapping[str, Any]], vocabulary: frozenset[str],
) -> list[dict[str, Any]]:
    """Re-force the governor of each stranded argument with the slot deleted, together with the site."""

    repaired: list[dict[str, Any]] = []
    for trial in root_mismatch_sites:
        roots = {reading: trial["readings"][reading]["root_category"] for reading in READINGS}
        if len(set(roots.values())) != 1:
            continue
        stranded = stranded_arguments(next(iter(roots.values())), base_root_category)
        if not stranded:
            continue
        site_position = int(trial["position"])
        for _slash, stranded_argument in stranded:
            for governor, governor_category in sorted(base_categories.items()):
                if governor == site_position or governor_category is None:
                    continue
                repaired_category = delete_argument_slot(governor_category, stranded_argument)
                if repaired_category is None or repaired_category not in vocabulary:
                    continue
                slots = {
                    reading: _forced_probe(
                        parser, sentence,
                        {site_position: str(trial[f"{reading.split('_')[0]}_category"]),
                         governor: repaired_category},
                        site_position, base_root_category)
                    for reading in READINGS
                }
                if _trial_outcome(slots["np_attach"], slots["vp_attach"]) != "realizable":
                    continue
                repaired.append({
                    "position": site_position,
                    "word": trial["word"],
                    "base_category": trial["base_category"],
                    "chain_id": trial["chain_id"],
                    "np_category": trial["np_category"],
                    "vp_category": trial["vp_category"],
                    "outcome": "realizable",
                    "readings": slots,
                    "repair": FrameRepair(governor, tokens[governor], governor_category, repaired_category,
                                          stranded_argument),
                })
    return repaired


# --- tier b1: nesting ---------------------------------------------------------------------------------


def modifier_projection_span(auto: str, *, position: int, core: str) -> tuple[int, int] | None:
    """Span (1-based, inclusive) of the smallest ``core`` node starting at the site."""

    try:
        tree = parse_auto_tree_with_spans(auto)
    except Exception:  # noqa: BLE001 - an unreadable derivation just yields no span
        return None
    leaf_index = position + 1
    best: tuple[int, int] | None = None

    def visit(node: Any) -> None:
        nonlocal best
        if node.start > leaf_index or node.end < leaf_index:
            return
        if node.cat == core and node.start == leaf_index:
            if best is None or (node.end - node.start) < (best[1] - best[0]):
                best = (node.start, node.end)
        for child in node.children:
            visit(child)

    visit(tree)
    return best


def _site_spans(trial: Mapping[str, Any]) -> dict[str, tuple[int, int] | None]:
    return {
        reading: modifier_projection_span(
            str(trial["readings"][reading]["auto"]), position=int(trial["position"]), core=CORE_BY_READING[reading])
        for reading in READINGS
    }


def _strictly_inside(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and (outer[0] < inner[0] or inner[1] < outer[1])


def _apply_nesting(sites: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Drop sites strictly nested inside another site's flipped constituent in every reading."""

    spans = [_site_spans(trial) for trial in sites]
    survivors: list[Mapping[str, Any]] = []
    for index, trial in enumerate(sites):
        nested = False
        for other_index in range(len(sites)):
            if other_index == index:
                continue
            comparable = [
                reading for reading in READINGS
                if spans[index][reading] is not None and spans[other_index][reading] is not None
            ]
            if not comparable:
                continue
            inside_everywhere = all(
                _strictly_inside(spans[index][reading], spans[other_index][reading]) for reading in comparable)
            outside_somewhere = any(
                _strictly_inside(spans[other_index][reading], spans[index][reading]) for reading in comparable)
            if inside_everywhere and not outside_somewhere:
                nested = True
                break
        if not nested:
            survivors.append(trial)
    return survivors


# --- tier b2: vacuity ---------------------------------------------------------------------------------


def _leaf_categories_from_auto(auto: str) -> list[str]:
    return _LEAF_CATEGORY_PATTERN.findall(auto)


def _is_nominal_leaf(category: str) -> bool:
    try:
        parsed = parse_ccg_category(category)
    except Exception:  # noqa: BLE001 - unreadable leaves resolve to themselves
        return False
    return parsed.is_atomic and parsed.base_atom in NOMINAL_PROJECTION_ATOMS


def resolve_endpoint(
    index: int, *, dependencies_by_head: Mapping[int, Sequence[MarkedDependency]], leaf_categories: Sequence[str],
) -> int:
    """Chase slot-1 dependencies until the endpoint is a nominal leaf."""

    current = index
    seen = {current}
    for _ in range(RESOLUTION_MAX_DEPTH):
        category = leaf_categories[current - 1] if 0 < current <= len(leaf_categories) else None
        if category is not None and _is_nominal_leaf(category):
            return current
        subjects = [
            dependency.argument_index
            for dependency in dependencies_by_head.get(current, ())
            if dependency.slot == 1
        ]
        if len(subjects) != 1 or subjects[0] in seen:
            return current
        current = subjects[0]
        seen.add(current)
    return current


def site_dependency_signature(
    dependencies: Sequence[MarkedDependency], *, site_index: int, leaf_categories: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Sorted ``(slot, resolved endpoint)`` pairs of the site token's dependencies."""

    by_head: dict[int, list[MarkedDependency]] = {}
    for dependency in dependencies:
        by_head.setdefault(dependency.head_index, []).append(dependency)
    signature = [
        (dependency.slot,
         resolve_endpoint(dependency.argument_index, dependencies_by_head=by_head, leaf_categories=leaf_categories))
        for dependency in by_head.get(site_index, ())
    ]
    return tuple(sorted(signature))


def _site_classification(
    trial: Mapping[str, Any], sentence: str, generate_dependencies: DependencyGenerator,
) -> str:
    """``discriminating``, ``vacuous`` (same resolved site dependencies), ``degenerate`` or ``error``."""

    site_position = int(trial["position"])
    signatures: dict[str, tuple[tuple[int, int], ...]] = {}
    for reading in READINGS:
        auto = str(trial["readings"][reading]["auto"])
        try:
            dependencies = generate_dependencies(_auto_corpus([auto], first_index=1), [sentence])[0]
        except Exception:  # noqa: BLE001 - a tooling failure keeps the site
            return "error"
        signatures[reading] = site_dependency_signature(
            dependencies, site_index=site_position + 1, leaf_categories=_leaf_categories_from_auto(auto))
    if any(not signatures[reading] for reading in READINGS):
        return "degenerate"
    if len({signatures[reading] for reading in READINGS}) == 1:
        return "vacuous"
    return "discriminating"


def _apply_vacuity(
    sites: Sequence[Mapping[str, Any]], sentence: str, generate_dependencies: DependencyGenerator,
) -> list[Mapping[str, Any]]:
    """Keep the sites whose two readings resolve to differing site dependencies (or whose check failed)."""

    return [trial for trial in sites
            if _site_classification(trial, sentence, generate_dependencies) in ("discriminating", "error")]


# --- tier b3: nominal-core variant --------------------------------------------------------------------


def _nominal_core_variant_category(np_category: str) -> str | None:
    """The ``N\\N``-core sibling of one noun-modifier category, textually exact."""

    if np_category == NP_MODIFIER_CORE:
        return NOMINAL_CORE_VARIANT
    prefix = f"({NP_MODIFIER_CORE})"
    if np_category.startswith(prefix):
        return f"({NOMINAL_CORE_VARIANT}){np_category[len(prefix):]}"
    return None


def _nominal_core_variants(
    parser: Any, sentence: str, *, sites: Sequence[Mapping[str, Any]], tokens: Sequence[str],
    base_categories: Mapping[int, str], base_root_category: str, vocabulary: frozenset[str],
    generate_dependencies: DependencyGenerator,
) -> list[dict[str, Any]]:
    """Retry each eliminated site with the ``N\\N``-core noun reading."""

    variants: list[dict[str, Any]] = []
    for trial in sites:
        variant_np = _nominal_core_variant_category(str(trial["np_category"]))
        if variant_np is None or variant_np not in vocabulary:
            continue
        variant_pair = SchemaPair(
            chain_id=str(trial["chain_id"]), chain=(), np_category=variant_np,
            vp_category=str(trial["vp_category"]), rightward_argument_count=0, leftward_argument_count=0)
        variant_trial = _run_trial(
            parser, sentence, position=int(trial["position"]), pair=variant_pair, tokens=tokens,
            base_categories=base_categories, base_root_category=base_root_category)
        if variant_trial["outcome"] != "realizable":
            continue
        if _site_classification(variant_trial, sentence, generate_dependencies) != "discriminating":
            continue
        variant_trial["nominal_core"] = NOMINAL_CORE_VARIANT
        variants.append(variant_trial)
    return variants


# --- preposition selection ----------------------------------------------------------------------------


def _leaf_categories(root: Any) -> dict[int, str]:
    """``{0-based token position: leaf category}`` of one chart derivation."""

    categories: dict[int, str] = {}
    for leaf in iter_leaves(root):
        position = leaf_position(leaf)
        if position is not None:
            categories[int(position)] = node_category(leaf)
    return categories


def select_preposition(
    parser: Any, prompt: str, grammar: Grammar, *, generate_dependencies: DependencyGenerator,
) -> tuple[PrepositionSite, dict[str, ConstrainedDerivation]]:
    """Choose the ambiguous preposition of ``prompt`` and return its two constrained derivations.

    ``prompt`` is the whitespace-tokenized TAB prompt. Raises ``ValueError``
    when the cascade does not end with exactly one site.
    """

    from ccg_diffusion.parsing.holccg import site_parse

    tokens = prompt.split()
    base = site_parse(parser, prompt)
    root = base.get("root_cat")
    if str(base.get("status")) != "ok" or root is None:
        raise ValueError(f"unconstrained parse failed: {prompt!r}")
    base_root_category = str(getattr(root, "cat", None))
    base_categories = _leaf_categories(root)
    realizable, root_mismatch = _search_sites(
        parser, prompt, tokens=tokens, base_root_category=base_root_category,
        base_categories=base_categories, grammar=grammar)

    selected: Optional[Mapping[str, Any]] = None
    tier: str | None = None
    if len(realizable) == 1:
        selected, tier = realizable[0], "single_site"
    elif not realizable:
        repaired = _frame_repair(
            parser, prompt, tokens=tokens, base_categories=base_categories,
            base_root_category=base_root_category, root_mismatch_sites=root_mismatch,
            vocabulary=grammar.vocabulary)
        if len(repaired) == 1:
            selected, tier = repaired[0], "frame_repair"
    else:
        survivors = _apply_nesting(realizable)
        tier = "nesting"
        if len(survivors) >= 2:
            survivors = _apply_vacuity(survivors, prompt, generate_dependencies)
            tier = "vacuity"
        if not survivors:
            survivors = _nominal_core_variants(
                parser, prompt, sites=realizable, tokens=tokens, base_categories=base_categories,
                base_root_category=base_root_category, vocabulary=grammar.vocabulary,
                generate_dependencies=generate_dependencies)
            tier = "nominal_core_variant"
        if len(survivors) == 1:
            selected = survivors[0]
    if selected is None or tier is None:
        raise ValueError(f"no unique ambiguous preposition: {prompt!r}")

    site = PrepositionSite(
        position=int(selected["position"]), word=selected["word"], base_category=selected["base_category"],
        chain_id=selected["chain_id"], np_category=selected["np_category"], vp_category=selected["vp_category"],
        root_category=base_root_category, tier=tier, nominal_core=selected.get("nominal_core"),
        frame_repair=selected.get("repair"))
    derivations = {
        reading: ConstrainedDerivation(
            reading, {int(position): category
                      for position, category in selected["readings"][reading]["forced_positions"].items()},
            selected["readings"][reading]["auto"], selected["readings"][reading]["root_category"])
        for reading in READINGS
    }
    return site, derivations


def _auto_corpus(derivations: Sequence[str], *, first_index: int) -> list[str]:
    lines: list[str] = []
    for offset, derivation in enumerate(derivations):
        lines.append(AUTO_ID.format(index=first_index + offset))
        lines.append(derivation.strip())
    return lines


def auto_lines(derivations: Sequence[ConstrainedDerivation], *, first_index: int = 1) -> list[str]:
    """AUTO corpus lines (``ID=prompts.<n>`` line, derivation line) of one interpretation per item."""

    return _auto_corpus([derivation.derivation for derivation in derivations], first_index=first_index)


# --- reading -> setup mapping evidence ----------------------------------------------------------------


def _word_by_source(alignment: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {int(entry["source_index"]): str(entry["text"]) for entry in alignment}


def _render_pair(holder: int, object_: int, words: Mapping[int, str]) -> str:
    return f"{words.get(holder, '?')}_{holder}->{words.get(object_, '?')}_{object_}"


def _classify_possession_pairs(
    trace: Mapping[str, Any], *, words: Mapping[int, str], site_source: Optional[int],
) -> list[dict[str, Any]]:
    """Each source-level possession pair with what justifies it (site attachment, other attachment, lemma)."""

    attachment_heads: dict[tuple[int, int], list[int]] = {}
    for frame in trace.get("attachment_possession_frames", ()):
        head_source = int(frame["head_source"])
        for pair in frame.get("pairs", ()):
            attachment_heads.setdefault((int(pair[0]), int(pair[1])), []).append(head_source)
    lemma_predicates: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for frame in trace.get("possession_frames", ()):
        predicate_source = int(frame["predicate_source"])
        for pair in frame.get("pairs", ()):
            lemma_predicates.setdefault((int(pair[0]), int(pair[1])), []).append(
                {"predicate_source": predicate_source, "predicate_word": words.get(predicate_source),
                 "lemma": frame.get("lemma")})

    classified: list[dict[str, Any]] = []
    for raw_pair in trace.get("source_indices", {}).get("possession", ()):
        holder, object_ = int(raw_pair[0]), int(raw_pair[1])
        key = (holder, object_)
        heads = sorted(set(attachment_heads.get(key, ())))
        lemmas = lemma_predicates.get(key, [])
        from_site = site_source is not None and site_source in heads
        from_other_attachment = any(head != site_source for head in heads)
        from_lemma = bool(lemmas)
        origins = [
            name
            for name, present in (
                ("site_attachment", from_site),
                ("other_attachment", from_other_attachment),
                ("possession_lemma", from_lemma),
            )
            if present
        ]
        classified.append({
            "holder_source": holder,
            "holder_word": words.get(holder),
            "object_source": object_,
            "object_word": words.get(object_),
            "text": _render_pair(holder, object_, words),
            "from_site_attachment": from_site,
            "from_other_attachment": from_other_attachment,
            "from_possession_lemma": from_lemma,
            "attachment_head_sources": heads,
            "attachment_head_words": [words.get(head) for head in heads],
            "possession_lemma_frames": lemmas,
            "origin": "+".join(origins) if origins else "unattributed",
        })
    return classified


def _changed_fields(np_program: Mapping[str, Any], vp_program: Mapping[str, Any]) -> list[str]:
    """The guidance-target fields whose CLIP-level values differ between the two interpretations."""

    changed: list[str] = []
    for field in PD_FIELDS:
        np_values = {tuple(value) if isinstance(value, list) else int(value) for value in np_program.get(field, ())}
        vp_values = {tuple(value) if isinstance(value, list) else int(value) for value in vp_program.get(field, ())}
        if np_values != vp_values:
            changed.append(field)
    return changed


def setup_mapping_evidence(
    tab_id: str, prompt: str, site: PrepositionSite, conversion_traces: Mapping[str, Mapping[str, Any]],
    alignments: Mapping[str, Sequence[Mapping[str, Any]]], programs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Syntax evidence of one item for ``evaluation.tab.map_tab_item_readings_to_setups``.

    Per reading: the conversion trace, the CLIP alignment and the guidance
    targets ``{"pd_indices", "program_sha256"}`` with the attachment possession.
    """

    site_source = site.source_index
    possession: dict[str, list[dict[str, Any]]] = {}
    site_pair_texts: dict[str, list[str]] = {}
    site_frame_reasons: dict[str, list[Optional[str]]] = {}
    for reading in READINGS:
        trace = conversion_traces[reading]
        pairs = _classify_possession_pairs(trace, words=_word_by_source(alignments[reading]), site_source=site_source)
        possession[reading] = pairs
        frames = [frame for frame in trace.get("attachment_possession_frames", ())
                  if int(frame["head_source"]) == site_source]
        site_pair_texts[reading] = sorted({pair["text"] for pair in pairs if pair["from_site_attachment"]})
        site_frame_reasons[reading] = [frame.get("reason") for frame in frames]
    differs = programs["np_attach"]["program_sha256"] != programs["vp_attach"]["program_sha256"]
    site_branch_kind: Optional[str] = None
    if differs:
        if site_pair_texts["np_attach"] and site_pair_texts["vp_attach"]:
            site_branch_kind = TWO_SIDED_HOLDER_BRANCH
        elif site_pair_texts["np_attach"] or site_pair_texts["vp_attach"]:
            site_branch_kind = ONE_SIDED_MISSING_SLOT
        else:
            site_branch_kind = NO_SITE_PAIR_BRANCH
    return {
        "tab_id": tab_id,
        "prompt": prompt,
        "site": {"position": site.position, "source_index": site_source, "word": site.word},
        "differs": differs,
        "changed_fields": _changed_fields(programs["np_attach"]["pd_indices"], programs["vp_attach"]["pd_indices"]),
        "possession_pairs": possession,
        "site_branch_kind": site_branch_kind,
        "site_frame_reasons": site_frame_reasons,
    }
