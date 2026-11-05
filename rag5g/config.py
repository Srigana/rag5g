"""Central settings. Override any of these with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPECS_DIR = Path(os.getenv("RAG5G_SPECS_DIR", ROOT / "specs"))
CHROMA_DIR = Path(os.getenv("RAG5G_CHROMA_DIR", ROOT / "chroma_db"))
EVAL_DIR = Path(os.getenv("RAG5G_EVAL_DIR", ROOT / "eval"))
COLLECTION = os.getenv("RAG5G_COLLECTION", "specs_5g")

# Embedding model. bge-small-en-v1.5 accepts 512 tokens, so a 500-token chunk
# is embedded whole. (all-MiniLM-L6-v2 stops at 256 tokens and would silently
# drop the second half of every chunk.)
EMBED_MODEL = os.getenv("RAG5G_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 retrieves better when queries (not passages) carry this prefix.
QUERY_PREFIX = os.getenv(
    "RAG5G_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

CHUNK_TOKENS = int(os.getenv("RAG5G_CHUNK_TOKENS", 500))     # measured with the embedder's tokenizer
OVERLAP_TOKENS = int(os.getenv("RAG5G_OVERLAP_TOKENS", 50))
TOP_K = int(os.getenv("RAG5G_TOP_K", 5))

OLLAMA_MODEL = os.getenv("RAG5G_OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
