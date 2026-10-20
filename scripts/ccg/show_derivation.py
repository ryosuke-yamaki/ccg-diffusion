#!/usr/bin/env python3
"""Print the CCG derivation and the marked dependencies of one prompt.

Without ``--force-supertag`` the prompt goes through the single-prompt Stage 1-2
front end (``parsing.frontend.CCGFrontEnd``, the T2I parse contract). With
``--force-supertag INDEX CATEGORY`` (repeatable; INDEX is the 0-based token
position) the prompt is parsed by Supertag-Constrained Parsing
(``parsing.holccg.supertag_constrained_parse``), which fixes those tokens to the
given categories, e.g. the two attachments of the PP "with a telescope" (to
"the girl" and to the verb phrase)::

    python -m scripts.ccg.show_derivation --prompt "the boy looks at the girl with a telescope" \\
        --force-supertag 6 "(NP\\NP)/NP"
    python -m scripts.ccg.show_derivation --prompt "the boy looks at the girl with a telescope" \\
        --force-supertag 6 "((S\\NP)\\(S\\NP))/NP"

Near-tie supertags can differ on hardware other than the paper's, so this is
not a paper-exact reproduction tool.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any, Sequence

from ccg_diffusion.parsing.ccg import parse_auto_derivation
from ccg_diffusion.parsing.sentences import split_and_tokenize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="one English sentence")
    parser.add_argument("--force-supertag", nargs=2, action="append", default=[], metavar=("INDEX", "CATEGORY"),
                        help="fix the token at 0-based INDEX to CATEGORY (Supertag-Constrained Parsing)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--allow-download", action="store_true",
                        help="download the pinned Hol-CCG checkpoint if it is not in the Hugging Face cache")
    return parser


def _tree_lines(node: Any, depth: int = 0) -> list[str]:
    indent = "  " * depth
    if not node.children:
        return [f"{indent}{node.leaf_category}  {node.word}  [{node.start - 1}]"]
    lines = [f"{indent}{node.category}"]
    for child in node.children:
        lines.extend(_tree_lines(child, depth + 1))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sentences = split_and_tokenize(args.prompt)
    if len(sentences) != 1:
        raise SystemExit(f"expected one sentence, found {len(sentences)}")
    tokens = [token.text for token in sentences[0].tokens]
    forced = {int(index): category for index, category in args.force_supertag}
    if forced:
        from ccg_diffusion.parsing.dependencies import generate_marked_dependencies
        from ccg_diffusion.parsing.holccg import load_holccg, supertag_constrained_parse

        parser = load_holccg(device=args.device, local_files_only=not args.allow_download)
        result = supertag_constrained_parse(parser, " ".join(tokens), forced)
        if result.get("status") != "ok" or not result.get("auto"):
            raise SystemExit(f"Supertag-Constrained Parsing found no derivation: status={result.get('status')!r}")
        id_line = "ID=prompt.1 PARSER=HOLCCG NUMPARSE=1"
        derivation = str(result["auto"]).strip()
        with contextlib.redirect_stdout(io.StringIO()):  # silence the per-call "+ generate ..." lines
            dependencies = generate_marked_dependencies([id_line, derivation], [" ".join(tokens)])[0]
    else:
        from ccg_diffusion.parsing.frontend import CCGFrontEnd

        front = CCGFrontEnd(args.device, local_files_only=not args.allow_download).parse(
            args.prompt, sentence_id="prompt.1")
        derivation, dependencies = front.derivation, front.dependencies

    print("tokens: " + " ".join(f"{index}:{token}" for index, token in enumerate(tokens)))
    if forced:
        print("forced supertags: " + ", ".join(f"{index}={category}" for index, category in sorted(forced.items())))
    print("AUTO: " + derivation)
    print("derivation:")
    print("\n".join(_tree_lines(parse_auto_derivation(derivation), 1)))
    print("marked dependencies (head [category, slot] -> argument):")
    for dependency in dependencies:
        print(f"  {dependency.head_word}_{dependency.head_index} [{dependency.marked_category}, "
              f"{dependency.slot}] -> {dependency.argument_word}_{dependency.argument_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
