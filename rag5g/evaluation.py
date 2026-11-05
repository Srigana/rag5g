"""Retrieval evaluation: does the chunk that holds the answer land in the top k?

Each question in eval/questions.jsonl carries an `evidence` string: a short
verbatim span from the spec that answers it. The gold chunk(s) are whichever
stored chunks contain that span, resolved at evaluation time. That keeps the
gold labels valid if you re-chunk with different sizes, and it counts a hit
when the same span sits in the overlap of two neighbouring chunks.

Two ways to fill questions.jsonl:
  * `make-eval` has the local LLM write one question per sampled chunk and
    quote the sentence that answers it (gold known by construction).
  * write questions by hand: {"question": ..., "evidence": "<verbatim span>"}.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .llm import parse_json
from .store import retrieve


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("­", "")
    s = re.sub(r"-\s+", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def all_chunks(collection) -> tuple[list[str], list[str], list[dict]]:
    got = collection.get(include=["documents", "metadatas"])
    return got["ids"], got["documents"], got["metadatas"]


def resolve_gold(evidence: str, ids, docs) -> list[str]:
    e = norm(evidence)
    return [i for i, d in zip(ids, docs) if e in norm(d)]


# ---------------------------------------------------------------- question generation

GEN_SYSTEM = (
    "You write evaluation questions for a search system over 3GPP 5G specifications. "
    "Given one excerpt, write ONE specific factual question that this excerpt answers. "
    "The question must make sense on its own (name the procedure, entity or parameter; "
    "never say 'this excerpt' or 'the text'). Do not copy long phrases from the excerpt into the question; "
    "paraphrase, the way an engineer would ask it. "
    'Reply with JSON: {"question": "...", "answer": "short answer", '
    '"evidence": "an exact verbatim quote of 8 to 30 words copied from the excerpt that contains the answer"}'
)


def _good_candidate(text: str, meta: dict) -> bool:
    if meta.get("n_tokens", 0) < 150:
        return False
    clause = (meta.get("clause") or "").lower()
    # References lists, forewords, IPR and change history make poor questions.
    if re.match(r"^2(\s|\.\d)", clause) or any(w in clause for w in ("foreword", "intellectual property", "change history", "references")):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text)) > 0.6  # skip flattened tables full of numbers


def make_questions(collection, llm, n: int = 20, seed: int = 7, max_attempts_per_q: int = 4) -> list[dict]:
    ids, docs, metas = all_chunks(collection)
    by_doc = defaultdict(list)
    for i, d, m in zip(ids, docs, metas):
        if _good_candidate(d, m):
            by_doc[m["doc"]].append((i, d, m))
    if not by_doc:
        raise SystemExit("No candidate chunks found. Did ingest run?")
    rng = random.Random(seed)
    for v in by_doc.values():
        rng.shuffle(v)

    docs_order = sorted(by_doc)
    questions, attempts, used = [], 0, set()
    while len(questions) < n and attempts < n * max_attempts_per_q:
        doc = docs_order[attempts % len(docs_order)]  # round-robin across specs
        attempts += 1
        pool = [c for c in by_doc[doc] if c[0] not in used]
        if not pool:
            continue
        cid, text, meta = pool[0]
        used.add(cid)
        out = parse_json(llm.chat(GEN_SYSTEM, f"Excerpt from {meta['doc']} ({meta.get('clause','')}):\n\n{text}", json_mode=True, temperature=0.3))
        if not out or not all(k in out for k in ("question", "evidence")):
            continue
        ev = str(out["evidence"]).strip().strip('"')
        if len(ev.split()) < 6 or norm(ev) not in norm(text):
            continue  # model paraphrased instead of quoting: gold can't be verified
        questions.append(
            {
                "id": f"q{len(questions)+1:02d}",
                "question": str(out["question"]).strip(),
                "answer": str(out.get("answer", "")).strip(),
                "evidence": ev,
                "source_chunk": cid,
                "doc": meta["doc"],
                "clause": meta.get("clause", ""),
                "origin": "llm-generated",
            }
        )
    return questions


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------------------------------------------------------------- evaluation

def evaluate(collection, embedder, questions: list[dict], k: int = 5, llm=None) -> dict:
    ids, docs, _ = all_chunks(collection)
    rows, skipped = [], []
    for q in questions:
        gold = resolve_gold(q["evidence"], ids, docs) if q.get("evidence") else []
        if not gold and q.get("source_chunk") in ids:
            gold = [q["source_chunk"]]
        if not gold:
            skipped.append({"id": q.get("id"), "question": q["question"], "reason": "evidence not found in any chunk"})
            continue
        hits = retrieve(collection, embedder, q["question"], k=k)
        ranks = [r for r, h in enumerate(hits, start=1) if h.id in gold]
        row = {
            "id": q.get("id"),
            "question": q["question"],
            "doc": q.get("doc", ""),
            "gold": gold,
            "rank": ranks[0] if ranks else None,
            "hit": bool(ranks),
            "retrieved": [{"id": h.id, "citation": h.citation(), "score": round(h.score, 4)} for h in hits],
        }
        if llm is not None:
            from .llm import answer

            a = answer(llm, q["question"], hits)
            row["answer"] = a["answer"]
            row["cited"] = a["cited"]
            row["cited_gold"] = any(n in ranks for n in a["cited"])
        rows.append(row)

    n = len(rows)
    summary = {
        "k": k,
        "n_questions": n,
        "n_skipped": len(skipped),
        f"hit_rate@{k}": round(sum(r["hit"] for r in rows) / n, 3) if n else None,
        "hit_rate@1": round(sum(r["rank"] == 1 for r in rows) / n, 3) if n else None,
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3) if n else None,
    }
    if llm is not None and n:
        summary["answer_cites_gold"] = round(sum(r["cited_gold"] for r in rows) / n, 3)
        summary["answer_cites_any"] = round(sum(bool(r["cited"]) for r in rows) / n, 3)
    per_doc = defaultdict(list)
    for r in rows:
        per_doc[r["doc"] or "?"].append(r["hit"])
    summary["by_doc"] = {d: {"n": len(v), f"hit_rate@{k}": round(sum(v) / len(v), 3)} for d, v in sorted(per_doc.items())}
    return {"summary": summary, "rows": rows, "skipped": skipped}


def report_markdown(result: dict) -> str:
    s, k = result["summary"], result["summary"]["k"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Questions scored: **{s['n_questions']}** (skipped: {s['n_skipped']})",
        f"- **Right chunk in top {k}: {s[f'hit_rate@{k}']:.0%}** ({sum(r['hit'] for r in result['rows'])}/{s['n_questions']})",
        f"- Right chunk ranked first: {s['hit_rate@1']:.0%}",
        f"- MRR@{k}: {s['mrr']:.3f}",
    ]
    if "answer_cites_gold" in s:
        lines.append(f"- LLM answer cites the gold chunk: {s['answer_cites_gold']:.0%} (cites anything: {s['answer_cites_any']:.0%})")
    lines += ["", "| Spec | n | hit@%d |" % k, "|---|---|---|"]
    for d, v in s["by_doc"].items():
        lines.append(f"| {d} | {v['n']} | {v[f'hit_rate@{k}']:.0%} |")
    lines += ["", "| # | Question | Gold rank | Top hit |", "|---|---|---|---|"]
    for r in result["rows"]:
        rank = str(r["rank"]) if r["rank"] else f"miss (>{k})"
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {r['id']} | {q} | {rank} | {r['retrieved'][0]['citation']} |")
    if result["skipped"]:
        lines += ["", "Skipped (evidence text not found in any chunk; fix the evidence span):"]
        lines += [f"- {x['id']}: {x['question']}" for x in result["skipped"]]
    return "\n".join(lines) + "\n"
