"""Application configuration, loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Config:
    # Chatbot / RAG
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "recipebot2")
    OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    RAG_RETRIEVER_K = _int("RAG_RETRIEVER_K", 4)
    RAG_FETCH_K = _int("RAG_FETCH_K", 20)
    # Below this best-match similarity score, treat the question as out-of-scope
    # and refuse rather than hallucinate (0..1, cosine similarity).
    RAG_MIN_SCORE = _float("RAG_MIN_SCORE", 0.30)

    JSON_SORT_KEYS = False


class TestConfig(Config):
    TESTING = True
    OPENAI_API_KEY = ""
    PINECONE_API_KEY = ""
