# Retrieval-Augmented Question Answering over 3GPP 5G Specifications

## 1. Overview

This project answers questions about 3GPP 5G specifications using retrieval-augmented generation. It runs locally.

The pipeline has four stages:

1. Specification PDFs are parsed and split into chunks of about 500 tokens.
2. Each chunk is embedded with a Sentence-Transformers model and stored in a Chroma vector database.
3. For each question, the five most similar chunks are retrieved and passed to a small open LLM served by Ollama. The model answers from those chunks only and cites them.
4. Retrieval is evaluated on 20 questions whose answer location is known in advance. The main metric is the share of questions where the correct chunk appears among the top five results.

## 2. Corpus

The evaluation used three specifications:

| Specification | Subject |
|---|---|
| TS 23.501 | System architecture for the 5G System |
| TS 38.300 | NR and NG-RAN overall description |
| TS 33.501 | Security architecture and procedures for 5G |

The PDFs are not included in this repository. ETSI publishes 3GPP specifications as PDFs, with a leading "1" added to the number (TS 23.501 is published as ETSI TS 123 501). They can be downloaded from the ETSI standards search at https://www.etsi.org/standards-search. Place them in the `specs/` directory.

## 3. Method

### 3.1 Text Extraction

PDFs are read with `pypdf`. The extractor removes lines that recur at the top or bottom of most pages (running headers, footers and page numbers). It drops table-of-contents pages, which it recognizes by dot leaders. It also detects numbered clause headings, so each chunk can be traced back to its clause.

### 3.2 Chunking

Text is split at sentence boundaries and packed into chunks of up to 500 tokens. Consecutive chunks overlap by about 50 tokens. Token counts come from the embedding model's own tokenizer, so a chunk never exceeds what the model can read. Each chunk stores its specification number, clause, and page range.

### 3.3 Embedding and Storage

Chunks are embedded with `BAAI/bge-small-en-v1.5`. This model was chosen partly for its 512-token input window, which fits a 500-token chunk in full. Several common alternatives, such as `all-MiniLM-L6-v2`, truncate input at 256 tokens. Vectors are stored in a persistent Chroma collection using cosine distance.

### 3.4 Answer Generation

The five highest-scoring chunks are numbered and placed in the prompt, each with its citation (for example, `TS 33.501 §6.1.3.2, p.48`). The model, `llama3.2:3b` by default, is told to answer only from these sources, to cite them by number, and to say so when they do not contain the answer.

### 3.5 Evaluation

Each evaluation question records a short verbatim passage from the specification that answers it. At evaluation time, the correct chunk is whichever stored chunk contains that passage. The labels therefore stay valid if the chunk size changes and the corpus is re-indexed.

The 20 questions were drafted by the local LLM from chunks sampled evenly across the three specifications. A question was kept only if the passage it quoted appears word for word in its source chunk.

## 4. Results

| Metric | Value |
|---|---|
| Correct chunk in top 5 | 75% (15 of 20) |
| Correct chunk ranked first | 60% (12 of 20) |
| Mean reciprocal rank (top 5) | 0.654 |

| Specification | Questions | Correct chunk in top 5 |
|---|---|---|
| TS 23.501 | 6 | 83% |
| TS 33.501 | 8 | 62% |
| TS 38.300 | 6 | 83% |

The five misses were reviewed by hand:

- Two were poorly formed questions. In one, the question asked for the purpose of the N9 interface, while the recorded passage only states that N9 may be intra- or inter-PLMN. In the other, the passage was a label from a message-flow figure rather than prose.
- One was a correct retrieval scored as a miss. The top result (TS 33.501 §A.3) answers the question directly, but it is not the chunk the question was generated from.
- Two were genuine retrieval failures. These were a question on NF selection across PLMNs, and one on how NH and NCC values are handled when a new NAS security context is activated.

Excluding the two invalid questions and counting the misattributed one as a hit, 16 of the remaining 18 questions (89%) retrieved a correct chunk within the top five. The strict figure of 75% is the one reported above.

## 5. Installation

Python 3.10 or later and Ollama (https://ollama.com/download) are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

The embedding model, about 130 MB, is downloaded from Hugging Face on first use.

## 6. Usage

```bash
python -m rag5g.cli ingest                              # build the index from specs/
python -m rag5g.cli ask "What is the role of the SEAF?" # answer a single question
python -m rag5g.cli ask                                 # interactive mode
python -m rag5g.cli make-eval                           # generate eval/questions.jsonl
python -m rag5g.cli eval                                # write eval/report.md and eval/results.json
python -m rag5g.cli eval --answers                      # also check that answers cite the correct chunk
```

Questions can also be written by hand in `eval/questions.jsonl`. Each entry needs a `question` and an `evidence` field, and the evidence must be copied exactly from the PDF. If an evidence passage cannot be found in any chunk, that question is reported as skipped rather than counted as a miss.

## 7. Configuration

Settings are defined in `rag5g/config.py`, and each can be overridden with an environment variable:

| Variable | Default | Purpose |
|---|---|---|
| `RAG5G_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RAG5G_CHUNK_TOKENS` | `500` | Maximum chunk length in tokens |
| `RAG5G_OVERLAP_TOKENS` | `50` | Overlap between consecutive chunks |
| `RAG5G_TOP_K` | `5` | Number of chunks retrieved per question |
| `RAG5G_OLLAMA_MODEL` | `llama3.2:3b` | Model used for answers and question generation |
| `RAG5G_CHROMA_DIR` | `chroma_db` | Location of the vector database |

If the embedding model is changed, the index must be rebuilt, and the same model must be set when running `ask` and `eval`.

## 8. Limitations

- Tables and figures do not survive PDF text extraction well. Content that exists only in a figure or a dense table is hard to retrieve.
- PDF extraction sometimes joins adjacent words (for example, "accessto"). This may reduce retrieval accuracy.
- Questions written by an LLM tend to reuse the wording of their source passage, which can make retrieval look better than it would with real user questions. The question set was reviewed by hand for this reason, but it is still small.
- Only one passage is treated as correct for each question. Retrieving a different passage that also answers the question still counts as a miss.

## 9. Testing

```bash
python tests/smoke_test.py
```

This test runs the full pipeline against a generated PDF, with simple stand-ins for the embedding model and the LLM. It checks parsing, chunking, storage, retrieval, citation formatting and scoring. It does not measure retrieval quality.

## 10. Repository Structure

```
rag5g/
  config.py       Settings
  pdf.py          PDF parsing and cleanup
  chunking.py     Token-based chunking with overlap
  store.py        Embedding model, Chroma collection, retrieval
  llm.py          Ollama client, prompt construction, citation parsing
  evaluation.py   Question generation, scoring, report
  cli.py          Command-line interface
tests/
  smoke_test.py   Offline end-to-end test
eval/
  questions.jsonl Evaluation questions
  report.md       Evaluation report
```
