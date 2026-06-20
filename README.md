# Recipe Bot — RAG Cooking Assistant

A **Flask REST microservice** that answers Kenyan-cuisine cooking questions using
**Retrieval-Augmented Generation (RAG)** over a recipe knowledge base. Built with
an application-factory + blueprint structure, a JSON API, a lazy-loaded RAG
pipeline with a **hallucination guard**, source citations, an evaluation harness,
tests and Docker.

> Portfolio project demonstrating practical, production-minded RAG engineering.

---

## RAG Architecture

```
user message  (+ optional chat history)
        │
        ▼
History-aware retriever        rewrites follow-ups into standalone queries
        │                       using the conversation history
        ▼
HuggingFace embeddings ──►  Pinecone vector store
(all-MiniLM-L6-v2, 384-dim)        │  MMR search (relevant + diverse chunks)
        │                          │
        ├──────────────────────────┘
        ▼
Grounding guard   ──(best score < threshold)──►  refuse ("out of scope")
        │  (relevant context found)
        ▼
ChatOpenAI (gpt-4o-mini) + recipe-aware, strictly-grounded prompt
        ▼
answer  +  source citations (document + page)  +  grounded flag
```

### Design decisions worth calling out

| Concern | Approach | Why |
| --- | --- | --- |
| **Hallucination** | A grounding guard scores the top retrieval; below a configurable threshold the bot refuses instead of inventing a recipe. | A recipe bot that makes up ingredients is worse than one that says "I don't know." |
| **Multi-turn** | `create_history_aware_retriever` reformulates follow-ups into standalone queries before retrieval. | "How long do I boil *it*?" needs the prior turn to retrieve correctly. |
| **Retrieval quality** | MMR (Maximal Marginal Relevance) with configurable `k` / `fetch_k`. | Returns relevant *and* non-redundant context. |
| **Trust** | Every answer cites its source document + page. | Users (and reviewers) can verify the source. |
| **Robustness** | The pipeline is lazy-loaded and cached; the app and tests boot without ML deps/keys and return a clean `503` when unconfigured. | Fast tests, graceful degradation, cheap container health checks. |
| **Quality measurement** | A small evaluation harness (`scripts/evaluate.py`) scores grounding, keyword recall and the guard. | RAG quality is measured, not assumed. |

Pipeline: [app/rag.py](app/rag.py).

---

## Tech Stack

Python 3.12 · Flask 3 (app factory + blueprints) · LangChain (Pinecone + OpenAI +
HuggingFace) · Gunicorn · pytest · Docker.

---

## Project Layout

```
.
├── app/
│   ├── __init__.py     # create_app() factory
│   ├── config.py       # env-based config
│   ├── rag.py          # RAG pipeline + grounding guard
│   └── api/            # blueprint: /api/health, /api/chat
├── src/                # RAG helpers (PDF load, chunk, embeddings)
├── Data/               # recipe knowledge base (PDF)
├── scripts/evaluate.py # RAG evaluation harness
├── store_index.py      # one-off: build the Pinecone index
├── tests/              # pytest suite
├── wsgi.py             # gunicorn entrypoint
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start (Docker)

```bash
git clone <repo-url>
cd Recipe-Bot
cp .env.example .env          # add OPENAI_API_KEY + PINECONE_API_KEY to enable the bot
docker compose up --build
```

Health check: <http://localhost:8080/api/health>

---

## Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # installs the local src/ package too (-e .)
cp .env.example .env

python store_index.py                  # one-off: build the vector index (needs keys)
python wsgi.py                         # dev server on :8080
pytest                                 # run tests
python scripts/evaluate.py             # RAG evaluation (needs a configured index)
```

---

## API

### `GET /api/health` → `{ "status": "ok" }`

### `POST /api/chat`

```json
{
  "message": "How do I make chapati?",
  "history": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello! Ask me about Kenyan recipes."}
  ]
}
```

`history` is optional (multi-turn context). Response:

```json
{
  "answer": "Chapati is a soft flatbread...\n\nIngredients:\n- ...\n\nSteps:\n1. ...",
  "sources": [{"source": "Kenya Recipe Book 2018.pdf", "page": 23}],
  "grounded": true
}
```

Out-of-scope questions return `"grounded": false` with a polite refusal (no LLM
hallucination). Status codes: `400` (missing message), `502` (model failure),
`503` (not configured).

Full reference: [docs/API.md](docs/API.md).

---

## Enabling the Chatbot

1. Set `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`.
2. Build the index once: `python store_index.py`.
3. `POST /api/chat`. Until configured, `/api/chat` returns `503`; `/api/health`
   works regardless.
