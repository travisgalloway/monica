"""Stream the cleaned jsonl.gz corpus (local or R2), and/or the Phase A' multi-domain
distillation-extension sources (code/math/docs/wiki/conversation/reasoning/code_problems/
code_instruct, #65), into build_distill_corpus.

The cleaned jsonl(.gz) shards are already normalized/filtered/deduped text, so `--in` skips
the datatrove re-clean and feeds `iter_jsonl_texts -> Record -> build_distill_corpus` unchanged
(clean[no-op]->qwen3 tokenize->uint32 pack), writing the poc-distill/corpus layout.

Setting any `--code-source`/`--math-source`/`--docs-source`/`--wiki-source`/
`--conversation-sources`/`--reasoning-sources`/`--code-problem-sources`/
`--code-instruct-sources` domain flag additionally (or instead) streams the new-source extension
records via `distill_sources.build_extension_records` and, because those sources are curated
(not raw web scrape), applies the documented A' cleaning policy: `quality=False,
license_filter=True, drop_minified=True, drop_autogen=True, scrub=True` (see
`.claude/plans/issue-65.md`, "Cleaning policy"). Plain `--in`-only runs keep the original
no-cleaning-kwargs behavior byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from src.data.corpus import Record, iter_jsonl_texts
from src.data.distill_corpus import build_distill_corpus
from src.data.distill_sources import DEFAULT_CODE_LANGS, build_extension_records


def _records(uri: str, source: str, lang: str = "en", license: str = "odc-by"):
    for text in iter_jsonl_texts(uri):
        yield Record(text=text, source=source, lang=lang, license=license)


def _chain(*iterables: Iterator[Record]) -> Iterator[Record]:
    for it in iterables:
        yield from it


def _extension_cfg(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.code_source != "none":
        cfg["code"] = {"source": args.code_source, "langs": args.code_langs,
                      "tokens_per_lang": args.code_tokens_per_lang, "tokens": args.code_tokens}
    if args.math_source != "none":
        cfg["math"] = {"source": args.math_source, "tokens": args.math_tokens}
    if args.docs_source != "none":
        cfg["docs"] = {"source": args.docs_source,
                      "langs": args.docs_langs or args.code_langs,
                      "tokens": args.docs_tokens}
    if args.wiki_source != "none":
        cfg["wiki"] = {"source": args.wiki_source, "tokens": args.wiki_tokens}
    if args.conversation_sources:
        cfg["conversation"] = {"sources": args.conversation_sources,
                               "tokens": args.conversation_tokens}
    if args.reasoning_sources:
        cfg["reasoning"] = {"sources": args.reasoning_sources, "tokens": args.reasoning_tokens}
    if args.code_problem_sources:
        cfg["code_problems"] = {"sources": args.code_problem_sources,
                                "langs": args.code_problem_langs or args.code_langs,
                                "tokens": args.code_problem_tokens}
    if args.code_instruct_sources:
        cfg["code_instruct"] = {"sources": args.code_instruct_sources,
                               "tokens": args.code_instruct_tokens}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=None,
                    help="cleaned jsonl(.gz) dir or file; local path or s3://... (optional if "
                         "a domain source flag below is set)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--source-name", default="reserve-pretrain")
    ap.add_argument("--tokenizer", default="qwen3",
                    choices=("qwen3", "qwen25", "olmo", "starcoder2"))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--shard-size-mb", type=int, default=512)
    ap.add_argument("--clean-shard-size-mb", type=int, default=512)
    ap.add_argument("--byte-fallback", action="store_true")
    ap.add_argument("--lang", default="en",
                    help="BCP-47 language tag written into each --in Record (default: en)")
    ap.add_argument("--license", default="odc-by",
                    help="SPDX license written into each --in Record (default: odc-by)")
    # Phase A' multi-domain extension sources (#65)
    ap.add_argument("--code-source", choices=("the-stack-dedup", "the-stack-smol", "none"),
                    default="none")
    ap.add_argument("--code-langs", nargs="+", default=list(DEFAULT_CODE_LANGS),
                    help="the-stack `data/<lang>` directories (default: the curated ~30-lang set)")
    ap.add_argument("--code-tokens", type=int, default=None,
                    help="pooled token budget shared across all --code-langs")
    ap.add_argument("--code-tokens-per-lang", type=int, default=None,
                    help="per-language token budget (equal cap/lang); takes precedence over "
                         "--code-tokens when both are set")
    ap.add_argument("--math-source", choices=("open-web-math", "none"), default="none")
    ap.add_argument("--math-tokens", type=int, default=None)
    ap.add_argument("--docs-source",
                    choices=("starcoder2-documentation", "library-documentation", "none"),
                    default="none",
                    help="starcoder2-documentation is multilingual (48 langs, filtered by "
                         "--docs-langs); library-documentation is Python-only and kept for "
                         "back-compat (#65: starcoder2-documentation replaces it as the "
                         "recommended choice, 2026-07-04)")
    ap.add_argument("--docs-langs", nargs="+", default=None,
                    help="language filter for --docs-source starcoder2-documentation "
                         "(default: --code-langs); ignored by library-documentation")
    ap.add_argument("--docs-tokens", type=int, default=None)
    ap.add_argument("--wiki-source", choices=("structured-wikipedia", "none"), default="none",
                    help="wikimedia/structured-wikipedia, English config only (#65)")
    ap.add_argument("--wiki-tokens", type=int, default=None)
    ap.add_argument("--conversation-sources", nargs="+", choices=("ultrachat", "oasst1"),
                    default=None)
    ap.add_argument("--conversation-tokens", type=int, default=None)
    ap.add_argument("--reasoning-sources", nargs="+",
                    choices=("mot", "openthoughts", "openthoughts2"), default=None,
                    help="use openthoughts2 (OpenThoughts2-1M), not openthoughts "
                         "(OpenThoughts-114k), for a real build -- 2-1M supersets 114k, so "
                         "blending both duplicates traces (#65, 2026-07-04)")
    ap.add_argument("--reasoning-tokens", type=int, default=None)
    ap.add_argument("--code-problem-sources", nargs="+",
                    choices=("opencodereasoning", "rosetta-code", "mceval", "kodcode"),
                    default=None,
                    help="competitive-programming / multilingual problem+solution sources "
                         "(#65); kodcode (CC BY-NC 4.0) and rosetta-code (GFDL) carry "
                         "non-permissive licenses, accepted per the project's 2026-07-04 "
                         "decision (#182) that this open-source research project keeps such "
                         "licenses rather than dropping them")
    ap.add_argument("--code-problem-langs", nargs="+", default=None,
                    help="language filter for the multilingual code-problem sources "
                         "(rosetta-code/mceval; default: --code-langs); opencodereasoning/"
                         "kodcode are Python-only and ignore this")
    ap.add_argument("--code-problem-tokens", type=int, default=None)
    ap.add_argument("--code-instruct-sources", nargs="+",
                    choices=("opencodeinstruct", "codefeedback"), default=None,
                    help="instruction -> code-solution sources (#65); opencodeinstruct is "
                         "~5M rows -- set --code-instruct-tokens to cap it")
    ap.add_argument("--code-instruct-tokens", type=int, default=None)
    ap.add_argument("--push", default=None,
                    help="after building, mirror <out-root> to this fsspec URI / R2 prefix "
                         "(e.g. s3://monica-training/poc-distill); R2 endpoint from "
                         "AWS_ENDPOINT_URL_S3 (#80)")
    args = ap.parse_args()

    ext_cfg = _extension_cfg(args)
    if not args.inp and not ext_cfg:
        ap.error("either --in or a --*-source/--*-sources domain flag is required")

    provenance: dict = {}
    streams = []
    if args.inp:
        streams.append(_records(args.inp, args.source_name, lang=args.lang, license=args.license))
    if ext_cfg:
        ext_stream, provenance = build_extension_records(ext_cfg)
        streams.append(ext_stream)
    records = _chain(*streams)

    # Curated-source cleaning policy (quality gate targets raw web scrape and would wrongly
    # drop reasoning/chat/math text) — only applied when the extension sources are in play.
    clean_kwargs = ({"quality": False, "license_filter": True, "drop_minified": True,
                    "drop_autogen": True, "scrub": True} if ext_cfg else {})

    m = build_distill_corpus(
        records, args.out_root,
        tokenizer=args.tokenizer, model_id=args.model_id, seq_len=args.seq_len,
        byte_fallback=args.byte_fallback, shard_size_mb=args.shard_size_mb,
        clean_shard_size_mb=args.clean_shard_size_mb, **clean_kwargs)
    print(f"distill corpus: {m['n_tokens']} tokens ({m['dtype']}, {m['n_documents']} docs, "
          f"{m['n_sequences']} seq x {m['seq_len']}, tok={m['tokenizer']}) -> {m['tokenized_dir']}")

    if provenance:
        prov_path = Path(args.out_root) / "corpus" / "provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(provenance, indent=2))
        print(f"provenance -> {prov_path}")

    if args.push:
        from src.data.r2_sync import upload_dir
        written = upload_dir(args.out_root, args.push)
        print(f"pushed {len(written)} file(s): {args.out_root} -> {args.push}")


if __name__ == "__main__":
    main()
