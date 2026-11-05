"""Turn a 3GPP/ETSI spec PDF into clean text blocks tagged with page and clause.

3GPP specs downloaded as PDF (normally the ETSI TS 1xx xxx editions) carry a
running header/footer on every page, a long table of contents, and numbered
clause headings. We drop the first two and use the third for citations.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Block:
    text: str
    page: int          # 1-based PDF page
    clause: str        # e.g. "5.15.2 Network slice selection" ("" before the first heading)


# "5.15.2 Network Slice Selection", "4.2.3 Service-based ...", "A.2 Something", "Annex B (informative): ..."
HEADING_RE = re.compile(
    r"^(?:(?P<num>(?:\d{1,2}|[A-H])(?:\.\d{1,3}){0,6}[a-z]?)\s{1,4}(?P<title>[A-Z][^\n]{2,110}?)"
    r"|(?P<annex>Annex [A-Z][^\n]{0,100}))\s*$"
)
TOC_LINE_RE = re.compile(r"(\.{4,}|…{2,}|\s\.\s\.\s\.)\s*\d+\s*$")
SPEC_RE_3GPP = re.compile(r"3GPP\s+TS\s+(\d{2}\.\d{3})")
SPEC_RE_ETSI = re.compile(r"ETSI\s+TS\s+1(\d{2})\s?(\d{3})")


def spec_name(first_pages_text: str, fallback: str) -> str:
    m = SPEC_RE_3GPP.search(first_pages_text)
    if m:
        return f"TS {m.group(1)}"
    m = SPEC_RE_ETSI.search(first_pages_text)
    if m:  # ETSI TS 123 501 is 3GPP TS 23.501
        return f"TS {m.group(1)}.{m.group(2)}"
    return fallback


def _is_heading(line: str) -> bool:
    m = HEADING_RE.match(line)
    if not m:
        return False
    if m.group("annex"):
        return True
    title = m.group("title")
    num = m.group("num")
    # Single-number lines like "5 GHz band is used" are usually prose, not headings.
    # Accept single numbers only for short title-like lines with no sentence ending.
    if "." not in num and (len(title) > 60 or title.endswith((".", ",", ";", ":"))):
        return False
    if title.endswith((".", ",", ";")):
        return False
    return True


def _norm_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def load_pdf(path: Path) -> tuple[str, list[Block]]:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    name = spec_name("\n".join(pages[:3]), path.stem)

    # Running headers/footers: short lines near the top/bottom that repeat on
    # many pages once digits are masked (page numbers, version strings).
    edge = Counter()
    split_pages = []
    for t in pages:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        split_pages.append(lines)
        for ln in set(lines[:3] + lines[-3:]):
            if len(ln) < 120:
                edge[_norm_key(ln)] += 1
    threshold = max(3, int(0.4 * len(pages)))
    boiler = {k for k, c in edge.items() if c >= threshold}

    blocks: list[Block] = []
    clause = ""
    for pno, lines in enumerate(split_pages, start=1):
        lines = [ln for ln in lines if _norm_key(ln) not in boiler]
        if not lines:
            continue
        # Table-of-contents pages: mostly dot-leader lines. Skip the whole page.
        toc_hits = sum(bool(TOC_LINE_RE.search(ln)) for ln in lines)
        if toc_hits >= max(3, 0.3 * len(lines)):
            continue
        buf: list[str] = []

        def flush():
            if buf:
                text = _join_lines(buf)
                if text:
                    blocks.append(Block(text, pno, clause))
                buf.clear()

        for ln in lines:
            if TOC_LINE_RE.search(ln):
                continue
            if _is_heading(ln):
                flush()
                clause = ln
                buf.append(ln)       # keep the heading in the text: it helps retrieval
                flush()
            else:
                buf.append(ln)
        flush()
    return name, _merge_small(blocks)


def _join_lines(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        if text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln            # de-hyphenate "conn-\nection"
        else:
            text = f"{text} {ln}" if text else ln
    return re.sub(r"\s+", " ", text).strip()


def _merge_small(blocks: list[Block]) -> list[Block]:
    """Glue a heading block onto the paragraph that follows it."""
    out: list[Block] = []
    for b in blocks:
        if out and out[-1].clause == b.clause and len(out[-1].text) < 120 and out[-1].page == b.page:
            out[-1] = Block(out[-1].text + "\n" + b.text, b.page, b.clause)
        else:
            out.append(b)
    return out
