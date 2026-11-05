# RAG Q&A over 3GPP 5G specs

Local retrieval-augmented Q&A over 3GPP 5G specification PDFs:

| Step | What | Default |
|---|---|---|
| Load + chunk | pypdf, running headers/footers and contents pages stripped, ~500-token chunks with 50-token overlap, split at sentence boundaries, each chunk tagged with spec, clause and pages | 500 tokens, measured with the embedder's own tokenizer |
| Embed + store | Sentence-Transformers → Chroma (persistent, cosine) | `BAAI/bge-small-en-v1.5` (512-token window, so a 500-token chunk is embedded whole) |
| Answer | top-5 chunks → small Ollama model, which must cite `[n]`. Sources are printed as e.g. `TS 23.501 §5.15.2, p.152` | `llama3.2:3b` |
| Evaluate | 20 questions with known gold chunks → how often the gold chunk is in the top 5 (also hit@1, MRR, optional citation check) | `eval/report.md` |

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ollama (https://ollama.com/download), then:
ollama pull llama3.2:3b          # ~2 GB; qwen2.5:3b or phi3.5 also work
```

The embedding model (~130 MB) downloads from Hugging Face on first use.

## 2. Get two or three specs into `specs/`

Good choices that cover different parts of 5G:

- **TS 23.501**: 5G system architecture (AMF, SMF, UPF, slicing, QoS)
- **TS 38.300**: NR and NG-RAN overall description (RRC states, BWP, mobility)
- **TS 33.501**: 5G security (SUCI/SUPI, 5G-AKA, SEAF)

3GPP specs are published as PDF by ETSI under a `1` prefix (TS 23.501 = **ETSI TS 123 501**). Search the number at
<https://www.etsi.org/standards-search>, download the latest PDF, and drop it in `specs/`.
3GPP's own archive (<https://www.3gpp.org/ftp/Specs/archive/>) has .docx files inside zips. Convert them with
`soffice --headless --convert-to pdf file.docx`.

## 3. Run

```bash
python -m rag5g.cli ingest                       # parse, chunk, embed, store (a few minutes on CPU)
python -m rag5g.cli ask "What is the role of the SEAF?"
python -m rag5g.cli ask                          # interactive

python -m rag5g.cli make-eval                    # writes eval/questions.jsonl (20 questions)
#   → read it over and edit before trusting the numbers (see below)
python -m rag5g.cli eval                         # writes eval/report.md + eval/results.json
python -m rag5g.cli eval --answers               # also checks the answer cites the gold chunk
```

Example `ask` output shape:

```
The SEAF acts as the security anchor in the serving network ... [1]. It receives the anchor key from the AUSF [1][4].

Sources:
 *[1] TS 33.501 §6.1.3.2, p.48  (sim 0.812, id TS33.501-0131-1f3a9c2e)
  [2] TS 33.501 §5.2.3, p.31   (sim 0.774, ...)
 ...
  * = cited in the answer
```

## 4. How the evaluation knows the right chunk

Every question carries an `evidence` field: a verbatim span from the spec that answers it. At eval time the gold
chunk is whichever stored chunk contains that span (whitespace, case and quote styles normalised). So the labels
stay valid if you change chunk size and re-ingest, and a span that falls in the overlap of two chunks counts
either one as correct.

`make-eval` builds the 20 questions: it samples chunks round-robin across the specs, skipping references lists,
forewords, change history and number-heavy tables. It asks the LLM for a question plus an exact quote from that chunk.
Any question whose quote isn't verbatim in the chunk is thrown away, so every kept question's gold chunk is known.

Before reporting the number:

- **Review `eval/questions.jsonl`.** Small models sometimes write vague questions ("What does the procedure do?")
  or copy the spec's own phrasing, which makes retrieval look better than it is. Reword those questions and leave `evidence` alone.
- **Add hand-written questions** in the same file. Only `question` and `evidence` are required:
  ```json
  {"id": "h01", "question": "Which network function selects the set of network slice instances for a UE?", "evidence": "<copy a sentence from the PDF>"}
  ```
  If an evidence span isn't found in any chunk (e.g. it was copied across a page break), the question is
  listed as skipped in the report rather than counted as a miss.

`report.md` gives the headline (right chunk in top 5), hit@1, MRR, a per-spec breakdown and a per-question table
with the gold rank and the top hit, so misses can be inspected one by one.

## Tuning knobs

All in `rag5g/config.py`, overridable by env var: `RAG5G_EMBED_MODEL` (e.g. `BAAI/bge-base-en-v1.5` for higher
quality), `RAG5G_CHUNK_TOKENS`, `RAG5G_OVERLAP_TOKENS`, `RAG5G_TOP_K`, `RAG5G_OLLAMA_MODEL`, `OLLAMA_HOST`.
If you switch to a model with a short window (e.g. `all-MiniLM-L6-v2`, 256 tokens), lower `RAG5G_CHUNK_TOKENS`.
Ingest warns when chunks exceed the model's window.

## Tests

`python tests/smoke_test.py` runs the full pipeline offline against a generated fixture PDF, with fake
embedder and LLM classes. It checks the plumbing: header/contents stripping, clause detection, chunk sizes and overlap,
Chroma storage vs. brute-force cosine retrieval, citation formatting, eval scoring and skipped questions. It says nothing about
retrieval quality.

## Layout

```
rag5g/
  config.py      settings
  pdf.py         PDF → cleaned blocks with page + clause
  chunking.py    blocks → ~500-token chunks with overlap
  store.py       Sentence-Transformers embedder, Chroma collection, retrieve()
  llm.py         Ollama client, prompt, citation parsing
  evaluation.py  question generation, gold resolution, hit@k / MRR, report
  cli.py         ingest | ask | make-eval | eval
tests/smoke_test.py
```
