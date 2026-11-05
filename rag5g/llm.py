"""Answer generation with a small local model served by Ollama."""
from __future__ import annotations

import json
import re

from . import config

SYSTEM = (
    "You answer questions about 3GPP 5G specifications using ONLY the numbered sources provided. "
    "After every claim, cite the source number in square brackets, e.g. [2] or [1][3]. "
    "If the sources do not contain the answer, say \"The provided excerpts do not answer this.\" "
    "Do not use outside knowledge. Be concise: at most 6 sentences."
)


class OllamaLLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format="json" if json_mode else "",
            options={"temperature": temperature, "num_ctx": 8192},
        )
        return resp["message"]["content"]


def build_prompt(question: str, hits) -> str:
    parts = []
    for n, h in enumerate(hits, start=1):
        parts.append(f"[{n}] ({h.citation()})\n{h.text}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer with citations:"


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    nums = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_sources)


def answer(llm, question: str, hits) -> dict:
    text = llm.chat(SYSTEM, build_prompt(question, hits)).strip()
    cited = cited_numbers(text, len(hits))
    return {
        "question": question,
        "answer": text,
        "cited": cited,
        "sources": [
            {"n": n, "id": h.id, "citation": h.citation(), "score": round(h.score, 4)}
            for n, h in enumerate(hits, start=1)
        ],
    }


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
