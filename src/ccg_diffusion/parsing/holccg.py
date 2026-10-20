"""Stage 1: Hol-CCG loading, Supertag-Constrained Parsing and corpus parsing to AUTO.

* :func:`resolve_checkpoint` and :func:`load_holccg` load the Hol-CCG checkpoint
  at its pinned revision.
* :class:`SupertagConstrainedHolCCG` adds Supertag-Constrained Parsing (paper
  Sec. IV-C1) to the public ``holccg`` parser: ``parse(..., forced_supertags=
  {position: category})`` restricts the chart cell of a token, e.g. the
  preposition of a TAB prompt, to one category, so the CKY search returns the
  best derivation of one PP interpretation. :func:`supertag_constrained_parse`
  and :func:`site_parse` call it with the TAB parse contracts.
* :func:`parse_corpus` parses a ``.raw`` corpus (one tokenized sentence per line)
  into CCGbank AUTO and writes the Stage 1 metadata next to it.

Mechanism of the forced supertags
---------------------------------
:class:`holccg.parsing.decoder._HolCCGDecoder` resolves attribute reads against
``vars(model)`` *before* its own implementations (a documented instrumentation
hook). Class-level overrides on a :class:`HolCCG` subclass therefore never reach
the decoder's internal ``self._build_leaf_chart(...)`` calls, but an entry in the
model's instance ``__dict__`` does. :class:`SupertagConstrainedHolCCG` installs
such an entry for the duration of one ``parse`` call: the hook builds the
pristine leaf chart via an unbound call (so there is no re-entry), then replaces
the leaf cell of every forced position with a single-category cell.

``_build_leaf_chart`` is the only leaf-construction point in the public decoder
and is reached by the single-sentence, per-sentence batched, and chunk-merged
paths alike, so one hook covers every ``parse`` shape.

Because the hook lives on the model instance, a ``forced_supertags`` parse is
not re-entrant and not thread-safe; concurrent forced parses on one model raise
:class:`RuntimeError` rather than silently interleaving.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import torch

import holccg as _holccg_pkg
from holccg.models.holccg import HolCCG
from holccg.parsing.chart import ChartCategory, ChartCell
from holccg.parsing.decoder import _HolCCGDecoder
from holccg.parsing.inference import HolCCGParser, InferenceOptions, ParsedSentence, ParseStatus
from holccg.parsing.render import auto_wrap_binary, fallback_right_branching

from ccg_diffusion import config
from ccg_diffusion.config import HolCCGParseContract
from ccg_diffusion.provenance import artifact_record, atomic_write_json, repository_root

# Upstream versions this hook has been validated against. The decoder internals
# it relies on are private, so a version bump must be re-checked by hand.
TESTED_HOLCCG_VERSIONS = frozenset({"1.0.0"})
_LEAF_CHART_PARAMS = ("self", "words_raw", "word_vectors", "stag_threshold")

# Forced-supertag probability handling: below ``_FORCED_PROB_EPS`` the
# classifier probability is replaced by ``_FORCED_PROB_FLOOR`` so downstream
# log-likelihoods stay finite. ``_LL_EPS`` matches the decoder's log offset.
_FORCED_PROB_EPS = 1e-6
_FORCED_PROB_FLOOR = 1e-3
_LL_EPS = 1e-12

_LeafChart = Dict[Tuple[int, int], ChartCell]
_LeafChartHook = Callable[[List[str], torch.Tensor, float], _LeafChart]


def check_holccg_compat() -> None:
    """Fail fast at import when the decoder internals this module hooks moved.

    Verifies (1) the installed ``holccg`` version is on the validated list,
    (2) ``_HolCCGDecoder`` still prefers instance-dict overrides over its own
    implementations, and (3) ``_build_leaf_chart`` still takes the positional
    arguments the hook is called with.
    """
    version = getattr(_holccg_pkg, "__version__", None)
    if version not in TESTED_HOLCCG_VERSIONS:
        raise RuntimeError(
            f"ccg_diffusion forced_supertags support is validated against holccg "
            f"{sorted(TESTED_HOLCCG_VERSIONS)}, found {version!r}. Re-check "
            f"ccg_diffusion/parsing/holccg.py against the new decoder before "
            f"widening the allowlist."
        )

    class _Probe:
        """Minimal stand-in for a parameter-owning model."""

    probe = _Probe()
    sentinel = object()
    probe.__dict__["_build_leaf_chart"] = sentinel
    if _HolCCGDecoder(probe)._build_leaf_chart is not sentinel:
        raise RuntimeError(
            "holccg _HolCCGDecoder no longer resolves instance-dict overrides "
            "ahead of its own implementations; the forced_supertags leaf hook "
            "would be silently ignored."
        )

    params = tuple(inspect.signature(_HolCCGDecoder._build_leaf_chart).parameters)
    if params != _LEAF_CHART_PARAMS:
        raise RuntimeError(
            f"holccg _build_leaf_chart signature changed to {params}; the "
            f"forced_supertags hook expects {_LEAF_CHART_PARAMS}."
        )


check_holccg_compat()


def _supertag_probs(decoder: Any, word_vectors: torch.Tensor) -> torch.Tensor:
    """Recompute the word-classifier probabilities used for forced positions.

    Mirrors the softmax stage of ``_build_leaf_chart`` — including the candc
    word-mask ``-inf`` fill — so a forced category's probability is exactly the
    one the pristine chart would have scored it with.
    """
    logits = decoder.word_category_classifier(word_vectors)
    if decoder._parse_grammar_mode == "candc" and decoder._word_mask_ids:
        mask = decoder._word_mask_tensor
        if mask is None or mask.device != logits.device:
            mask = torch.as_tensor(
                decoder._word_mask_ids, dtype=torch.long, device=logits.device
            )
        logits = logits.index_fill(-1, mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _forced_leaf_cell(
    decoder: Any,
    position: int,
    cat_str: str,
    words_raw: List[str],
    word_vectors: torch.Tensor,
    probs: torch.Tensor,
) -> ChartCell:
    """Build the single-category leaf cell for one forced position.

    The cell holds exactly ``cat_str`` (no top-1, no above-threshold
    candidates), scored with the classifier probability when it is at least
    ``_FORCED_PROB_EPS`` and with ``_FORCED_PROB_FLOOR`` otherwise.
    """
    forced_idx = decoder.word_category2index[cat_str]
    classifier_p = float(probs[position][forced_idx].item())
    p_used = classifier_p if classifier_p >= _FORCED_PROB_EPS else _FORCED_PROB_FLOOR

    index2category = decoder.word_index2category
    vocab_cat = index2category[forced_idx] if forced_idx < len(index2category) else None
    if vocab_cat is None:
        raise RuntimeError(
            f"Failed to add forced category {cat_str!r} at position {position}"
        )

    span = (position, position + 1)
    cat_ll = math.log(p_used + _LL_EPS)
    category = ChartCategory(
        span=span,
        cat=vocab_cat,
        type="leaf",
        is_leaf=True,
        vector=word_vectors[position],
        total_ll=cat_ll,
        cat_ll=cat_ll,
        span_ll=0.0,
        num_child=0,
        left_child=None,
        right_child=None,
        head=position,
        word=words_raw[position],
    )
    category.cat_id = decoder._cat_id(category.cat)
    cell = ChartCell(span)
    cell.add_category(category)
    return cell


def _make_forced_leaf_hook(model: HolCCG, forced: Dict[int, str]) -> _LeafChartHook:
    """Return the ``_build_leaf_chart`` replacement that applies ``forced``.

    Out-of-range positions are ignored: the forced mapping is consulted per token
    position, so entries past the sentence end never fire.
    """

    def _hook(
        words_raw: List[str], word_vectors: torch.Tensor, stag_threshold: float
    ) -> _LeafChart:
        decoder = _HolCCGDecoder(model)
        # Unbound call: the pristine implementation never re-reads
        # ``self._build_leaf_chart``, so this cannot recurse into the hook.
        chart = _HolCCGDecoder._build_leaf_chart(
            decoder, words_raw, word_vectors, stag_threshold
        )
        positions = [
            position for position in forced if 0 <= position < len(words_raw)
        ]
        if not positions:
            return chart
        probs = _supertag_probs(decoder, word_vectors)
        for position in positions:
            chart[(position, position + 1)] = _forced_leaf_cell(
                decoder, position, forced[position], words_raw, word_vectors, probs
            )
        return chart

    return _hook


def align_encoder_dtype(model: HolCCG, dtype: torch.dtype = torch.float32) -> HolCCG:
    """Cast ``model``'s text encoder to ``dtype`` so it matches the Hol-CCG heads.

    transformers v5 defaults ``from_pretrained(dtype=...)`` to ``"auto"``, which
    honours the ``dtype`` entry in the checkpoint's ``config.json``. A bundle
    that declares a non-fp32 encoder would then load in bf16 while Hol-CCG's own
    projection and classifier heads stay fp32, and the first matmul would raise
    "mat1 and mat2 must have the same dtype". transformers 4.51.3 ignored that
    entry and upcast the stored weights to fp32, so restoring fp32 reproduces
    the historical numerics exactly (widening bf16 to fp32 is lossless).

    Returns ``model`` for call-site chaining. Heads are left untouched.
    """
    model.encoder.to(dtype)
    return model


class SupertagConstrainedHolCCG(HolCCG):
    """:class:`HolCCG` plus the ``forced_supertags`` parse option (Supertag-Constrained Parsing).

    ``from_pretrained`` needs no override: ``PersistenceOps.from_pretrained``
    instantiates ``cls(...)``, so it returns this subclass.
    """

    def parse(
        self,
        sentence: Union[str, List[str]],
        *,
        forced_supertags: Optional[Dict[int, str]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse ``sentence``, optionally pinning lexical categories.

        Args:
            sentence: Sentence, or list of sentences, as accepted by
                :meth:`holccg.HolCCG.parse`.
            forced_supertags: Optional ``{token_position: category_str}``. Each
                listed position gets a leaf cell holding only ``category_str``,
                skipping the usual top-1 + above-threshold candidates. For a
                list input the same mapping applies to every sentence.
                ``None`` / empty takes the stock parse path unchanged.
            **kwargs: Forwarded verbatim to :meth:`holccg.HolCCG.parse`.

        Raises:
            ValueError: A forced category is not in ``word_category2index``.
            RuntimeError: A forced parse is already in flight on this model.
        """
        if not forced_supertags:
            return super().parse(sentence, **kwargs)

        for position, cat_str in forced_supertags.items():
            if cat_str not in self.word_category2index:
                raise ValueError(
                    f"forced_supertags category {cat_str!r} at position {position} "
                    f"is not in word_category2index"
                )

        if "_build_leaf_chart" in self.__dict__:
            raise RuntimeError(
                "a forced_supertags parse is already active on this model; "
                "re-entrant or concurrent forced parses are not supported"
            )
        # Assign through __dict__ directly: nn.Module.__setattr__ would reject /
        # mis-route a plain callable, and the decoder reads ``vars(model)``.
        self.__dict__["_build_leaf_chart"] = _make_forced_leaf_hook(
            self, dict(forced_supertags)
        )
        try:
            return super().parse(sentence, **kwargs)
        finally:
            self.__dict__.pop("_build_leaf_chart", None)


# --- loading ------------------------------------------------------------------------------------------

_HF_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def resolve_checkpoint(
    model_dir: Union[str, Path] = config.HOLCCG_CHECKPOINT,
    *,
    revision: Optional[str] = config.HOLCCG_REVISION,
    local_files_only: bool = True,
) -> Union[str, Path]:
    """Resolve an ``owner/name`` Hugging Face repo id to its snapshot directory.

    An existing local directory passes through unchanged and never triggers a
    Hub lookup. A repo-id-shaped value is resolved with ``snapshot_download`` at
    ``revision`` (the pinned Hol-CCG revision by default), from the local cache
    unless ``local_files_only`` is false.
    """
    path = Path(model_dir)
    if path.is_dir():
        return path
    if _HF_REPO_ID_RE.match(str(model_dir)):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(model_dir), revision=revision, local_files_only=local_files_only))
    return model_dir


def load_holccg(
    model_dir: Union[str, Path] = config.HOLCCG_CHECKPOINT,
    device: Union[str, torch.device] = "cpu",
    *,
    local_files_only: bool = True,
) -> SupertagConstrainedHolCCG:
    """Load the Hol-CCG checkpoint as a :class:`SupertagConstrainedHolCCG`.

    The model is put in eval mode on ``device`` with ``requires_grad=False``; the
    parser weights are never updated. The encoder is loaded in fp32 so that it
    matches the fp32 Hol-CCG heads (see :func:`align_encoder_dtype`).
    """
    model = SupertagConstrainedHolCCG.from_pretrained(
        str(resolve_checkpoint(model_dir, local_files_only=local_files_only)),
        encoder_kwargs={"dtype": torch.float32},
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def supertag_constrained_parse(
    parser: SupertagConstrainedHolCCG,
    sentence: str,
    supertags: Mapping[int, str],
    *,
    contract: HolCCGParseContract = config.HOLCCG_SUPERTAG_CONSTRAINED_PARSE,
) -> dict[str, Any]:
    """Parse ``sentence`` with the token at each position of ``supertags`` fixed to its category."""
    return parser.parse(
        sentence,
        forced_supertags=dict(supertags),
        **contract.parsing_kwargs(),
        return_chart=False,
        force_eval=True,
        use_no_grad=True,
    )


def site_parse(
    parser: SupertagConstrainedHolCCG,
    sentence: str,
    *,
    contract: HolCCGParseContract = config.HOLCCG_TAB_SITE_PARSE,
) -> dict[str, Any]:
    """Unconstrained TAB parse used to locate the preposition before Supertag-Constrained Parsing."""
    return parser.parse(
        sentence,
        **contract.parsing_kwargs(),
        return_chart=False,
        force_eval=True,
        use_no_grad=True,
    )


# --- Stage 1 corpus parsing ---------------------------------------------------------------------------

# CKY parse-time grammar. "treebank" (the model's own extracted binary_rule
# table) reproduces every recorded parse; "candc" is the public package's own
# default and stays opt-in.
SUPPORTED_GRAMMARS = ("treebank", "candc")

# Sentences per batched supertag/CKY call. 1 keeps every sentence in its own
# encoder pass, which is the only setting that cannot change a parse: batching
# pads sentences together, and the resulting float differences flip near-tie
# supertag decisions (measured: 1 of 800 ABC-6K prompts at chunk_size=32).
DEFAULT_PARSE_CHUNK_SIZE = 1


def resolve_grammar(value: Any) -> str:
    """Validate the optional ``parsing.grammar`` key (default ``treebank``)."""
    grammar = "treebank" if value is None else str(value)
    if grammar not in SUPPORTED_GRAMMARS:
        raise ValueError(
            f"parsing.grammar must be one of {SUPPORTED_GRAMMARS}, got {grammar!r}"
        )
    return grammar


def resolve_chunk_size(value: Any) -> int:
    """Validate the optional ``parsing.chunk_size`` key (default 1)."""
    chunk_size = DEFAULT_PARSE_CHUNK_SIZE if value is None else int(value)
    if chunk_size < 1:
        raise ValueError(f"parsing.chunk_size must be positive, got {chunk_size}")
    return chunk_size


def inference_options(contract: HolCCGParseContract, *, retain_root_cat: bool = False) -> InferenceOptions:
    """The ``holccg`` inference options of a parse contract."""
    return InferenceOptions(
        stag_threshold=contract.stag_threshold,
        phrase_threshold=contract.phrase_threshold,
        span_threshold=contract.span_threshold,
        apply_skimmer=contract.apply_skimmer,
        min_rule_freq=contract.min_rule_freq,
        grammar=contract.grammar,
        chunk_size=contract.chunk_size,
        retain_root_cat=retain_root_cat,
    )


def combine_partial_autos(autos: List[str], cat: str) -> str:
    """Combine skimmer fragments into one AUTO tree under synthetic ``cat`` roots.

    This keeps output strictly one parse per sentence (NUMPARSE=1). A single
    fragment is wrapped in a unary ``cat`` node as well, so downstream code never
    mistakes a skimmer result for a complete parse.
    """
    if not autos:
        return ""
    if len(autos) == 1:
        return f"(<T {cat} 0 1> {autos[0]} )"
    tree = autos[0]
    for a in autos[1:]:
        tree = auto_wrap_binary(cat, tree, a, head=0)
    return tree


def render_auto(
    record: ParsedSentence,
    *,
    partial_root_cat: str,
    fail_root_cat: str,
    fail_pos: str,
) -> str:
    """Render one parse record as a single-line AUTO derivation.

    ``holccg.parsing.render.render_legacy_auto`` is deliberately not used: it
    leaves a lone skimmer fragment unwrapped, whereas this wraps it in a unary
    ``PARTIAL`` node (see :func:`combine_partial_autos`).
    """
    tokens = list(record.tokens)
    if record.status is ParseStatus.SUCCESS and record.auto:
        return record.auto.strip()
    if record.status is ParseStatus.PARTIAL and record.partial_autos:
        combined = combine_partial_autos(list(record.partial_autos), partial_root_cat)
        if combined:
            return combined
    return fallback_right_branching(tokens, fail_root_cat, fail_pos)


def count_parse_record(record: ParsedSentence, counters: Dict[str, int]) -> None:
    """Increment the ok/partial/fail/error counters for one record.

    The historical accounting quirk is preserved: a ``partial`` model result
    whose skimmer fragments combine to an empty tree counts as both ``partial``
    and ``fail``.
    """
    if record.status is ParseStatus.SUCCESS:
        counters["ok"] += 1
        return
    if record.status is ParseStatus.PARTIAL:
        counters["partial"] += 1
        return
    if record.status is ParseStatus.ERROR:
        counters["error"] += 1
        return
    if record.model_status is ParseStatus.PARTIAL:
        counters["partial"] += 1
    counters["fail"] += 1


def tokenizer_max_length(tokenizer: Any) -> Optional[int]:
    """Return a real position limit (e.g. CLIP's 77) or None if unbounded.

    The Hugging Face sentinel 1e30 means no limit; anything above 1M is treated the same.
    """
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None:
        return None
    try:
        max_len = int(max_len)
    except (TypeError, ValueError):
        return None
    if max_len >= 1_000_000:
        return None
    return max_len


def would_truncate_words(tokenizer: Any, tokens: List[str], max_length: int) -> bool:
    """Return True iff tokenizing ``tokens`` at ``max_length`` drops any word."""
    try:
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        word_ids = enc.word_ids()
    except Exception:
        return False
    surviving = {wid for wid in word_ids if wid is not None}
    return len(surviving) != len(tokens)


def _iter_raw_sentences(path: str, encoding: str, skip_empty: bool) -> Iterator[str]:
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            s = line.rstrip("\n").strip()
            if skip_empty and not s:
                continue
            yield s


def load_raw_sentences(
    path: str, encoding: str, skip_empty: bool, max_sentences: Optional[int]
) -> List[str]:
    sentences: List[str] = []
    for sentence in _iter_raw_sentences(path, encoding=encoding, skip_empty=skip_empty):
        if max_sentences is not None and len(sentences) >= max_sentences:
            break
        sentences.append(sentence)
    return sentences


def make_sentence_id(prefix: str, index: int) -> str:
    """CCGbank style sentence ID ``<prefix>.<index>``."""
    return f"{prefix}.{index}"


def auto_meta_path(output_auto_path: str | Path) -> Path:
    """``<auto>.meta.json``: the Stage 1 metadata next to an AUTO file."""
    path = Path(output_auto_path)
    return path.with_name(path.name + ".meta.json")


def build_parse_meta(
    *,
    model_dir: str,
    options: InferenceOptions,
    input_raw_path: str,
    output_auto_path: str,
    counters: Mapping[str, int],
    n_overlen: int,
    n_sentences: int,
    resolved_revision: str | None,
) -> Dict[str, Any]:
    """Build the Stage 1 metadata that ties one AUTO file to its checkpoint.

    ``model_dir`` is recorded exactly as configured (Hugging Face repo id or local
    path). ``resolved_revision`` is the snapshot the repo id resolved to (None for
    a local directory). Paths are repository-relative when inside the checkout.
    """
    root = repository_root()
    raw = artifact_record(input_raw_path).to_dict(relative_to=root)
    auto = artifact_record(output_auto_path).to_dict(relative_to=root)
    return {
        "artifact": "holccg_parse_auto",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": model_dir,
        "resolved_revision": resolved_revision,
        "holccg_version": importlib.metadata.version("holccg"),
        "parsing": {
            "stag_threshold": options.stag_threshold,
            "phrase_threshold": options.phrase_threshold,
            "span_threshold": options.span_threshold,
            "apply_skimmer": options.apply_skimmer,
            "min_rule_freq": options.min_rule_freq,
            "grammar": options.grammar,
            "chunk_size": options.chunk_size,
        },
        "input_raw_path": raw["path"],
        "input_raw_sha256": raw["sha256"],
        "output_auto_path": auto["path"],
        "output_auto_sha256": auto["sha256"],
        "sentence_counts": {
            "total": n_sentences,
            "ok": counters["ok"],
            "partial": counters["partial"],
            "fail": counters["fail"],
            "error": counters["error"],
            "overlen": n_overlen,
        },
    }


def parse_corpus(
    *,
    model_dir: str,
    input_raw_path: str,
    output_auto_path: str,
    options: InferenceOptions,
    device: str,
    revision: str | None = config.HOLCCG_REVISION,
    local_files_only: bool = True,
    id_prefix: str | None = None,
    id_start: int = 1,
    parser_name: str = "HOLCCG",
    partial_root_cat: str = "PARTIAL",
    fail_root_cat: str = "FAIL",
    fail_pos: str = "POS",
    overlen_root_cat: str = "OVERLEN",
    encoding: str = "utf-8",
    skip_empty: bool = True,
    max_sentences: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Parse a ``.raw`` corpus into ``output_auto_path`` and write ``<auto>.meta.json``.

    Each sentence becomes ``ID=<prefix>.<n> PARSER=<parser_name> NUMPARSE=1``
    plus one derivation line. Skimmer fragments are wrapped in ``PARTIAL`` roots;
    failed sentences become right-branching ``FAIL`` trees; a sentence whose
    sub-word tokenization would exceed the encoder's positional limit is not
    parsed and becomes an ``OVERLEN`` tree. Returns the metadata.
    """
    from tqdm import tqdm

    resolved = resolve_checkpoint(model_dir, revision=revision, local_files_only=local_files_only)
    # A repo id resolves to <cache>/snapshots/<commit>; a local directory has no revision.
    resolved_revision = None if Path(model_dir).is_dir() else Path(resolved).name
    # strict=False skips the public bundle-manifest validation: the checkpoint
    # predates that manifest.
    parser = HolCCGParser.from_pretrained(str(resolved), options=options, device=device, strict=False)
    # HolCCGParser.from_pretrained exposes no encoder_kwargs, so the fp32 pin is
    # applied to the loaded encoder instead (bit-identical to a dtype=float32 load).
    align_encoder_dtype(parser.model)

    sentences = load_raw_sentences(input_raw_path, encoding, skip_empty, max_sentences)
    token_lists = [sentence.split() for sentence in sentences]
    prefix = id_prefix if id_prefix is not None else Path(input_raw_path).stem
    sent_ids = [make_sentence_id(prefix, id_start + offset) for offset in range(len(sentences))]

    # CLIP's 77-token limit silently drops late words, which breaks Hol-CCG's
    # leaf alignment. Detect such sentences up front and emit an OVERLEN
    # fallback tree so downstream stages can drop them cleanly.
    encoder_max_len = tokenizer_max_length(parser.model.tokenizer)
    overlen = [
        encoder_max_len is not None
        and bool(tokens)
        and would_truncate_words(parser.model.tokenizer, tokens, encoder_max_len)
        for tokens in token_lists
    ]

    # Records arrive in input order, so the OVERLEN sentences skipped above can
    # be interleaved back while streaming the output file.
    parse_positions = [index for index, flag in enumerate(overlen) if not flag]
    record_stream = parser.iter_parse_chunks(
        (token_lists[index] for index in parse_positions),
        id_start=id_start,
        sentence_ids=[sent_ids[index] for index in parse_positions],
    )
    if progress:
        record_stream = tqdm(record_stream, desc="Parsing", unit="sent", total=len(parse_positions))
    records = iter(record_stream)

    counters = {"ok": 0, "partial": 0, "fail": 0, "error": 0}
    n_overlen = 0
    Path(output_auto_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_auto_path, "w", encoding="utf-8") as auto_file:
        for position, (sent_id, tokens) in enumerate(zip(sent_ids, token_lists)):
            if overlen[position]:
                n_overlen += 1
                auto_out = fallback_right_branching(tokens, overlen_root_cat, fail_pos)
            else:
                record = next(records)
                count_parse_record(record, counters)
                auto_out = render_auto(
                    record,
                    partial_root_cat=partial_root_cat,
                    fail_root_cat=fail_root_cat,
                    fail_pos=fail_pos,
                )
            # Exactly 2 lines per sentence (CCGbank AUTO style).
            auto_file.write(f"ID={sent_id} PARSER={parser_name} NUMPARSE=1\n")
            auto_file.write(auto_out.strip() + "\n")

    meta = build_parse_meta(
        model_dir=model_dir,
        options=options,
        input_raw_path=input_raw_path,
        output_auto_path=output_auto_path,
        counters=counters,
        n_overlen=n_overlen,
        n_sentences=len(sentences),
        resolved_revision=resolved_revision,
    )
    atomic_write_json(auto_meta_path(output_auto_path), meta)
    return meta
