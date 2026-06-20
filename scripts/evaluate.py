"""
Lightweight RAG evaluation harness.

Runs a small set of probe questions against the live pipeline and reports:
  * grounding   - did the bot answer from context (vs. refuse)?
  * keyword recall - fraction of expected keywords present in the answer
  * guard       - does an out-of-scope question get correctly refused?

Requires OPENAI_API_KEY + PINECONE_API_KEY and a built index. Run:

    python scripts/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from app.rag import answer_question  # noqa: E402

# (question, expected keywords that a grounded answer should mention)
PROBES = [
    ("How do I cook ugali?", ["maize", "water", "stir"]),
    ("What ingredients are in chapati?", ["flour", "water"]),
    ("Tell me about githeri.", ["maize", "beans"]),
]

# Should be refused by the grounding guard (not in a recipe book).
OUT_OF_SCOPE = "How do I replace a flat car tyre?"


def keyword_recall(answer: str, keywords) -> float:
    text = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in text)
    return hits / len(keywords) if keywords else 0.0


def main():
    app = create_app()
    with app.app_context():
        print("=== Grounded probes ===")
        recalls = []
        for question, keywords in PROBES:
            result = answer_question(question)
            recall = keyword_recall(result["answer"], keywords)
            recalls.append(recall)
            pages = ", ".join(str(s.get("page")) for s in result["sources"]) or "-"
            print(f"\nQ: {question}")
            print(f"  grounded={result['grounded']}  keyword_recall={recall:.2f}  pages=[{pages}]")
            print(f"  A: {result['answer'][:160]}...")

        print("\n=== Guard probe (should refuse) ===")
        guard = answer_question(OUT_OF_SCOPE)
        guard_ok = guard["grounded"] is False
        print(f"Q: {OUT_OF_SCOPE}\n  refused={guard_ok}")

        print("\n=== Summary ===")
        print(f"  mean keyword recall: {sum(recalls) / len(recalls):.2f}")
        print(f"  guard correct:       {guard_ok}")


if __name__ == "__main__":
    main()
