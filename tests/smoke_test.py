"""Offline smoke test of the whole pipeline.

Uses a generated fixture PDF (laid out like an ETSI spec: running header,
contents page, numbered clauses) and fake stand-ins for the embedder and the
LLM, so it runs with no model downloads. It checks the plumbing (PDF cleaning,
chunk sizes, Chroma storage, retrieval, prompting and citations, eval scoring),
not retrieval quality. Retrieval quality needs the real model: run the README steps.

    python tests/smoke_test.py
"""
import hashlib
import random
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag5g import cli  # noqa: E402
from rag5g.pdf import load_pdf  # noqa: E402

TOPICS = {
    "5.1 Registration management": "registration area tracking AMF UE context mobility periodic timer",
    "5.2 Session management": "PDU session SMF UPF anchor QoS flow establishment release",
    "5.3 Network slicing": "S-NSSAI slice service type differentiator NSSF selection allowed",
    "5.4 Security anchor function": "SEAF SUCI SUPI concealment home network public key AUSF",
    "6.1 Radio resource control": "RRC_INACTIVE RNA update gNB resume suspend paging",
    "6.2 Bandwidth part adaptation": "BWP switching DCI timer active downlink uplink numerology",
}


def make_fixture_pdf(path: Path, seed: int = 0):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    rng = random.Random(seed)
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    page = [1]

    def header():
        c.setFont("Helvetica", 9)
        c.drawString(50, H - 30, "ETSI TS 123 501 V18.8.0 (2024-12)")
        c.drawString(W - 80, H - 30, str(page[0]))
        c.drawString(50, 25, "3GPP TS 23.501 version 18.8.0 Release 18")

    def new_page():
        c.showPage()
        page[0] += 1
        header()
        return H - 60

    header()
    c.setFont("Helvetica", 11)
    c.drawString(50, H - 80, "Fixture spec for smoke testing (not real 3GPP text)")
    y = new_page()
    c.setFont("Helvetica", 10)
    for t in TOPICS:  # contents page with dot leaders: must be dropped
        c.drawString(50, y, f"{t} " + "." * 60 + f" {rng.randint(10, 99)}")
        y -= 14
    y = new_page()

    for title, words in TOPICS.items():
        w = words.split()
        lines = [title]
        for s in range(40):
            k = rng.sample(w, 4)
            lines.append(f"The {k[0]} shall interact with the {k[1]} when the {k[2]} changes, "
                         f"as described for {k[3]} in sentence {s}.")
        for ln in lines:
            if y < 60:
                y = new_page()
            c.setFont("Helvetica-Bold" if ln == title else "Helvetica", 10)
            # wrap long lines
            for part in textwrap.wrap(ln, 95):
                c.drawString(50, y, part)
                y -= 13
        y -= 8
    c.save()


class FakeEmbedder:
    """Hashed bag of words. Lexical only: good enough to exercise the wiring."""
    name, max_seq_length, dim = "fake-hash", 512, 512

    def count_tokens(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _vec(self, text):
        v = np.zeros(self.dim)
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def chat(self, system, user, json_mode=False, temperature=0.0):
        if json_mode:  # question generation: quote one sentence of the excerpt
            body = user.split("\n\n", 1)[1]
            sents = [s for s in re.split(r"(?<=\.)\s+", body) if len(s.split()) >= 8]
            ev = sents[len(sents) // 2]
            words = re.findall(r"[A-Za-z_\-]{3,}", ev)
            return '{"question": "How does %s relate to %s and %s?", "answer": "x", "evidence": "%s"}' % (
                words[1], words[3], words[5], ev)
        return "The AMF handles registration [1]. Slices are selected by the NSSF [3]."


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        specs = tmp / "specs"
        specs.mkdir()
        make_fixture_pdf(specs / "fixture_ts23501.pdf")

        doc, blocks = load_pdf(specs / "fixture_ts23501.pdf")
        assert doc == "TS 23.501", doc
        alltext = " ".join(b.text for b in blocks)
        assert "ETSI TS 123 501 V18" not in alltext, "running header not stripped"
        assert "....." not in alltext, "contents page not dropped"
        clauses = {b.clause for b in blocks}
        assert set(TOPICS) <= clauses, clauses
        print(f"pdf ok: {doc}, {len(blocks)} blocks, clauses detected: {len(clauses)}")

        emb, llm = FakeEmbedder(), FakeLLM()
        parser = cli.build_parser()
        db = str(tmp / "db")
        chunks = cli.cmd_ingest(parser.parse_args(["--db", db, "ingest", "--specs", str(specs)]), embedder=emb)
        sizes = [c.n_tokens for c in chunks]
        assert max(sizes) <= 500 and min(sizes[:-1]) > 300, sizes
        # Overlap: consecutive chunks share text.
        assert chunks[1].text[:60] in chunks[0].text, "no overlap between chunks"
        print(f"chunk ok: sizes {sizes}")

        a = cli.cmd_ask(parser.parse_args(["--db", db, "ask", "Which", "function", "selects", "the", "S-NSSAI", "slice?"]),
                        embedder=emb, llm=llm)
        assert a["cited"] == [1, 3] and len(a["sources"]) == 5
        assert "§5.3" in a["sources"][0]["citation"], a["sources"][0]

        eval_dir = tmp / "eval"
        qs = cli.cmd_make_eval(parser.parse_args(["--db", db, "make-eval", "--out", str(eval_dir / "questions.jsonl")]), llm=llm)
        assert len(qs) == len({q["source_chunk"] for q in qs}) and len(qs) >= 8, len(qs)
        # Add one hand-written question whose evidence doesn't exist -> must be skipped, not crash.
        with open(eval_dir / "questions.jsonl", "a") as f:
            f.write('{"id": "h01", "question": "bogus?", "evidence": "this sentence is not in the spec at all"}\n')
        res = cli.cmd_eval(parser.parse_args(["--db", db, "eval", "--questions", str(eval_dir / "questions.jsonl"),
                                              "--out-dir", str(eval_dir), "--answers"]), embedder=emb, llm=llm)
        s = res["summary"]
        assert s["n_skipped"] == 1 and s["n_questions"] == len(qs)
        # Recompute the hit rate by brute force (cosine over all chunks) to check the Chroma path.
        from rag5g.evaluation import all_chunks, resolve_gold
        from rag5g.store import get_collection
        ids, docs, _ = all_chunks(get_collection(Path(db)))
        M = np.array(emb.embed_passages(docs))
        bf_hits = 0
        for q in qs:
            gold = set(resolve_gold(q["evidence"], ids, docs))
            assert q["source_chunk"] in gold
            top = np.argsort(-(M @ np.array(emb.embed_query(q["question"]))))[:5]
            bf_hits += bool(gold & {ids[t] for t in top})
        assert abs(bf_hits / len(qs) - s["hit_rate@5"]) < 0.051, (bf_hits, s)
        print("eval ok:", {k: v for k, v in s.items() if k != "by_doc"})
    finally:
        shutil.rmtree(tmp)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
