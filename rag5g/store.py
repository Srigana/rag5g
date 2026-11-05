"""Embedding model + Chroma collection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from . import config
from .chunking import format_citation


class Embedder:
    """Sentence-Transformers wrapper. Also exposes the model's tokenizer for chunking."""

    def __init__(self, model_name: str = config.EMBED_MODEL, query_prefix: str = config.QUERY_PREFIX):
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.query_prefix = query_prefix
        self.max_seq_length = self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 64).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True).tolist()[0]


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float  # cosine similarity

    def citation(self) -> str:
        return format_citation(self.meta)


def get_collection(path: Path = config.CHROMA_DIR, name: str = config.COLLECTION, reset: bool = False):
    client = chromadb.PersistentClient(path=str(path))
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks(collection, chunks, embedder, batch: int = 256) -> None:
    for s in range(0, len(chunks), batch):
        part = chunks[s : s + batch]
        collection.add(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.metadata() for c in part],
            embeddings=embedder.embed_passages([c.text for c in part]),
        )


def retrieve(collection, embedder, question: str, k: int = config.TOP_K) -> list[Hit]:
    res = collection.query(
        query_embeddings=[embedder.embed_query(question)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Hit(id=i, text=d, meta=m, score=1.0 - dist)
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
