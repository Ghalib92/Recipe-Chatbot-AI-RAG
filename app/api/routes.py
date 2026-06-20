import logging

from flask import jsonify, request

from app.rag import ChatbotUnavailable, answer_question
from . import api_bp

logger = logging.getLogger(__name__)


@api_bp.get("/health")
def health():
    return jsonify(status="ok")


@api_bp.post("/chat")
def chat():
    """
    Ask the recipe assistant a question.

    Body: {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
    Returns the answer, source citations and whether it was grounded in context.
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="`message` is required."), 400

    history = data.get("history") if isinstance(data.get("history"), list) else None

    try:
        result = answer_question(message, history)
    except ChatbotUnavailable as exc:
        return jsonify(error=str(exc)), 503
    except Exception:
        logger.exception("Chatbot failed to answer")
        return jsonify(error="The assistant failed to answer. Please try again later."), 502

    return jsonify(result)
