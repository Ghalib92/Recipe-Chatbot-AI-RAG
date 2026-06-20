"""
Retrieval-Augmented Generation pipeline for the Kenyan-recipe assistant.

What makes this production-grade rather than a toy:

* Lazy + cached    - components are built on first use and cached, so the Flask
  app and tests boot without the heavy ML deps or API keys. Misconfiguration
  surfaces as ChatbotUnavailable -> a clean HTTP 503.
* Chat model       - ChatOpenAI (gpt-4o-mini) instead of the legacy completion API.
* History-aware    - follow-up questions are reformulated into standalone queries
  using the conversation history before retrieval.
* MMR retrieval    - Maximal Marginal Relevance returns relevant *and* diverse
  recipe chunks, reducing near-duplicate context.
* Grounding guard  - if the best retrieval score is below a threshold, the bot
  refuses ("out of scope") instead of hallucinating a recipe.
* Citations        - the source document + page of every retrieved chunk is
  returned alongside the answer.
"""

from functools import lru_cache

from flask import current_app


class ChatbotUnavailable(Exception):
    """Raised when the RAG pipeline cannot be built (missing keys or deps)."""


@lru_cache(maxsize=1)
def _build():
    """Build and cache (vector store, rag_chain). Raises ChatbotUnavailable."""
    cfg = current_app.config
    if not cfg["OPENAI_API_KEY"] or not cfg["PINECONE_API_KEY"]:
        raise ChatbotUnavailable(
            "Chatbot is not configured. Set OPENAI_API_KEY and PINECONE_API_KEY, "
            "and build the index with `python store_index.py`."
        )

    try:
        import os

        from langchain.chains import (
            create_history_aware_retriever,
            create_retrieval_chain,
        )
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI
        from langchain_pinecone import PineconeVectorStore

        from src.helper import download_hugging_face_embeddings
        from src.prompt import contextualize_q_system_prompt, system_prompt
    except ImportError as exc:  # pragma: no cover - optional ML deps
        raise ChatbotUnavailable(f"Chatbot dependencies are not installed: {exc}") from exc

    os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = cfg["PINECONE_API_KEY"]

    embeddings = download_hugging_face_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=cfg["PINECONE_INDEX_NAME"], embedding=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg["RAG_RETRIEVER_K"], "fetch_k": cfg["RAG_FETCH_K"]},
    )

    llm = ChatOpenAI(model=cfg["OPENAI_CHAT_MODEL"], temperature=0.3, max_tokens=600)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return vector_store, rag_chain


def _to_messages(history):
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for turn in history or []:
        role, content = turn.get("role"), turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai", "bot"):
            messages.append(AIMessage(content=content))
    return messages


def _format_sources(documents):
    seen, sources = set(), []
    for doc in documents:
        meta = doc.metadata or {}
        source = str(meta.get("source", "knowledge base")).split("/")[-1]
        page = meta.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page})
    return sources


def _is_grounded(vector_store, message) -> bool:
    """True if the knowledge base has anything relevant to the message."""
    threshold = current_app.config["RAG_MIN_SCORE"]
    try:
        scored = vector_store.similarity_search_with_relevance_scores(message, k=1)
    except Exception:  # pragma: no cover - scoring not critical to availability
        return True
    return bool(scored) and scored[0][1] >= threshold


def answer_question(message: str, history=None) -> dict:
    """
    Answer a user message. Returns:
        {"answer": str, "sources": list, "grounded": bool}

    On a first-turn question with no relevant context, refuses instead of
    hallucinating (grounding guard).
    """
    from src.prompt import OUT_OF_SCOPE_MESSAGE

    vector_store, rag_chain = _build()

    # Only guard first-turn questions; follow-ups rely on the history-aware
    # retriever (the raw follow-up text is often not self-contained).
    if not history and not _is_grounded(vector_store, message):
        return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": [], "grounded": False}

    result = rag_chain.invoke({"input": message, "chat_history": _to_messages(history)})
    return {
        "answer": result["answer"],
        "sources": _format_sources(result.get("context", [])),
        "grounded": True,
    }


def reset_cache():
    """Clear the cached pipeline (used in tests)."""
    _build.cache_clear()
