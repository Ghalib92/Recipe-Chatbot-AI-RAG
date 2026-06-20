# API Reference

Base URL: `http://localhost:8080`

JSON request/response bodies.

---

## `GET /api/health`

Liveness probe.

**200 OK** → `{ "status": "ok" }`

---

## `POST /api/chat`

Ask the recipe assistant a question.

**Request**

```json
{
  "message": "What ingredients do I need for pilau?",
  "history": [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hi! Ask me about Kenyan recipes."}
  ]
}
```

- `message` (required) — the user's question.
- `history` (optional) — prior turns for multi-turn context.

**200 OK**

```json
{
  "answer": "Pilau is a spiced rice dish...\n\nIngredients:\n- rice\n- pilau masala\n...\n\nSteps:\n1. ...",
  "sources": [{"source": "Kenya Recipe Book 2018.pdf", "page": 41}],
  "grounded": true
}
```

**Out-of-scope** — the grounding guard refuses rather than hallucinating:

```json
{
  "answer": "I can only help with recipes from my Kenyan recipe knowledge base, and I couldn't find anything relevant to that...",
  "sources": [],
  "grounded": false
}
```

**Errors**

| Status | Meaning |
| --- | --- |
| 400 | `message` is missing or empty |
| 502 | Upstream model/retrieval failure |
| 503 | Chatbot not configured (missing keys / index) |

---

## Evaluation

`scripts/evaluate.py` runs probe questions against a configured instance and
reports grounding, keyword recall and whether the guard correctly refuses an
out-of-scope question. See the README for usage.
