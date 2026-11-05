"""Pack text blocks into ~N-token chunks with a small overlap.

Token counts come from the embedding model's own tokenizer, so "500 tokens"
means 500 tokens the embedder will actually see (nothing gets truncated).
Chunks break at sentence boundaries where possible.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from .pdf import Block

SENT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\[\"'])|\n")


@dataclass
class Chunk:
    id: str
    doc: str
    source_file: str
    text: str
    page_start: int
    page_end: int
    clause: str
    n_tokens: int
    clauses: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "doc": self.doc,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "clause": self.clause,
            "clauses": " | ".join(self.clauses)[:1000],
            "n_tokens": self.n_tokens,
        }

    def citation(self) -> str:
        return format_citation(self.metadata())


def format_citation(meta: dict) -> str:
    """'TS 23.501 §5.15.2, p.152' or 'TS 38.300 Annex B (informative), pp.40-41'."""
    ps, pe = meta["page_start"], meta["page_end"]
    pages = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
    clause = meta.get("clause") or ""
    if clause.startswith("Annex"):
        where = " " + clause.split(":")[0][:30]
    elif clause:
        where = " §" + clause.split(" ")[0]
    else:
        where = ""
    return f"{meta['doc']}{where}, {pages}"


@dataclass
class _Unit:
    text: str
    page: int
    clause: str
    n: int


def _units(blocks: list[Block], count: Callable[[str], int], max_tokens: int) -> list[_Unit]:
    units = []
    for b in blocks:
        for sent in SENT_RE.split(b.text):
            sent = sent.strip()
            if not sent:
                continue
            n = count(sent)
            if n <= max_tokens:
                units.append(_Unit(sent, b.page, b.clause, n))
                continue
            # A "sentence" longer than a chunk (tables flattened by PDF extraction):
            # cut it into word windows that fit.
            words, cur = sent.split(), []
            for w in words:
                cur.append(w)
                if len(cur) % 20 == 0 and count(" ".join(cur)) > max_tokens - 20:
                    units.append(_Unit(" ".join(cur), b.page, b.clause, count(" ".join(cur))))
                    cur = []
            if cur:
                units.append(_Unit(" ".join(cur), b.page, b.clause, count(" ".join(cur))))
    return units


def chunk_blocks(
    blocks: list[Block],
    doc: str,
    source_file: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    units = _units(blocks, count_tokens, max_tokens)
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and total + units[j].n <= max_tokens:
            total += units[j].n
            j += 1
        if j == i:  # safety: always make progress
            j = i + 1
        group = units[i:j]
        text = " ".join(u.text for u in group)
        clauses = list(dict.fromkeys(u.clause for u in group if u.clause))
        chunk_no = len(chunks)
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        chunks.append(
            Chunk(
                id=f"{doc.replace(' ', '')}-{chunk_no:04d}-{digest}",
                doc=doc,
                source_file=source_file,
                text=text,
                page_start=group[0].page,
                page_end=group[-1].page,
                clause=group[0].clause or (clauses[0] if clauses else ""),
                n_tokens=count_tokens(text),
                clauses=clauses,
            )
        )
        if j >= len(units):
            break
        # Step back so the next chunk starts with ~overlap_tokens of context:
        # whole trailing sentences until the budget is reached, allowing one
        # sentence to overshoot up to 2x the budget so long sentences still overlap.
        back, k = 0, j
        while overlap_tokens > 0 and k - 1 > i and back < overlap_tokens and back + units[k - 1].n <= 2 * overlap_tokens:
            k -= 1
            back += units[k].n
        i = k
    return chunks
