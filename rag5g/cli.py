"""Command line: ingest | ask | make-eval | eval

    python -m rag5g.cli ingest                 # PDFs in ./specs -> chunks -> Chroma
    python -m rag5g.cli ask "What does the AMF do?"
    python -m rag5g.cli ask                    # interactive
    python -m rag5g.cli make-eval              # 20 questions with known gold chunks
    python -m rag5g.cli eval [--answers]       # hit rate @5 (+ optional citation check)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from pathlib import Path

from . import config


def cmd_ingest(args, embedder=None):
    from .chunking import chunk_blocks
    from .pdf import load_pdf
    from .store import Embedder, add_chunks, get_collection

    pdfs = sorted(Path(args.specs).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {args.specs}. See README for where to download the specs.")
    embedder = embedder or Embedder()
    if args.chunk_tokens + 2 > embedder.max_seq_length:
        print(f"WARNING: chunks of {args.chunk_tokens} tokens exceed {embedder.name}'s limit of "
              f"{embedder.max_seq_length}; the tail of each chunk will be ignored when embedding.")

    collection = get_collection(Path(args.db), reset=True)
    total = []
    for pdf in pdfs:
        doc, blocks = load_pdf(pdf)
        chunks = chunk_blocks(blocks, doc, pdf.name, embedder.count_tokens, args.chunk_tokens, args.overlap)
        add_chunks(collection, chunks, embedder)
        sizes = [c.n_tokens for c in chunks]
        print(f"{pdf.name}: {doc}, {len(blocks)} blocks -> {len(chunks)} chunks "
              f"(tokens median {statistics.median(sizes):.0f}, max {max(sizes)})")
        total += chunks
    print(f"Stored {len(total)} chunks in {args.db} (collection '{config.COLLECTION}').")
    return total


def _print_answer(a: dict):
    print("\n" + textwrap.fill(a["answer"], 100, replace_whitespace=False) + "\n")
    print("Sources:")
    for s in a["sources"]:
        mark = "*" if s["n"] in a["cited"] else " "
        print(f" {mark}[{s['n']}] {s['citation']}  (sim {s['score']:.3f}, id {s['id']})")
    if not a["cited"]:
        print("  (the model did not cite any source)")
    print("  * = cited in the answer")


def cmd_ask(args, embedder=None, llm=None):
    from .llm import OllamaLLM, answer
    from .store import Embedder, get_collection, retrieve

    embedder = embedder or Embedder()
    llm = llm or OllamaLLM(args.model)
    collection = get_collection(Path(args.db))
    def run(q):
        a = answer(llm, q, retrieve(collection, embedder, q, k=args.k))
        print(json.dumps(a, indent=2)) if args.json else _print_answer(a)
        return a

    if args.question:
        return run(" ".join(args.question))
    while (q := input("\nQuestion (blank to quit): ").strip()):
        run(q)


def cmd_make_eval(args, llm=None):
    from .evaluation import make_questions, save_jsonl
    from .llm import OllamaLLM
    from .store import get_collection

    llm = llm or OllamaLLM(args.model)
    qs = make_questions(get_collection(Path(args.db)), llm, n=args.n, seed=args.seed)
    out = Path(args.out)
    save_jsonl(qs, out)
    print(f"Wrote {len(qs)} questions to {out}. Read them over: drop vague ones, reword any that "
          f"copy the spec's wording, and keep `evidence` verbatim.")
    if len(qs) < args.n:
        print(f"Only {len(qs)}/{args.n}: either the model failed to quote verbatim evidence or the corpus ran out "
              f"of usable chunks. Try another --seed, a larger model, or add hand-written questions.")
    return qs


def cmd_eval(args, embedder=None, llm=None):
    from .evaluation import evaluate, load_jsonl, report_markdown
    from .store import Embedder, get_collection

    embedder = embedder or Embedder()
    if args.answers and llm is None:
        from .llm import OllamaLLM
        llm = OllamaLLM(args.model)
    questions = load_jsonl(Path(args.questions))
    result = evaluate(get_collection(Path(args.db)), embedder, questions, k=args.k, llm=llm if args.answers else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    md = report_markdown(result)
    (out_dir / "report.md").write_text(md)
    print(md)
    print(f"Saved {out_dir/'report.md'} and {out_dir/'results.json'}")
    return result


def build_parser():
    p = argparse.ArgumentParser(prog="rag5g")
    p.add_argument("--db", default=str(config.CHROMA_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--specs", default=str(config.SPECS_DIR))
    s.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    s.add_argument("--overlap", type=int, default=config.OVERLAP_TOKENS)

    s = sub.add_parser("ask")
    s.add_argument("question", nargs="*")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--model", default=config.OLLAMA_MODEL)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("make-eval")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--model", default=config.OLLAMA_MODEL)
    s.add_argument("--out", default=str(config.EVAL_DIR / "questions.jsonl"))

    s = sub.add_parser("eval")
    s.add_argument("--questions", default=str(config.EVAL_DIR / "questions.jsonl"))
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--answers", action="store_true", help="also generate answers and check they cite the gold chunk")
    s.add_argument("--model", default=config.OLLAMA_MODEL)
    s.add_argument("--out-dir", default=str(config.EVAL_DIR))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"ingest": cmd_ingest, "ask": cmd_ask, "make-eval": cmd_make_eval, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
