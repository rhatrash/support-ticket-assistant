"""Smoke tests. They use the offline TF-IDF embedder so they run without downloading models."""
import os

os.environ["EMBED_BACKEND"] = "tfidf"

from src.make_data import build  # noqa: E402
from src.data import detect_language, split  # noqa: E402
from src.pipeline import TicketAssistant  # noqa: E402


def test_split_has_no_leakage():
    train, test = split(build(n=200))
    assert set(train["text"]).isdisjoint(set(test["text"]))


def test_language_detection():
    assert detect_language("طلبي ما وصل") == "ar"
    assert detect_language("My order is late") == "en"


def test_pipeline_end_to_end():
    train, _ = split(build(n=200))
    assistant = TicketAssistant(train)
    out = assistant.analyze("I was charged twice for my subscription. This is urgent.")
    assert out["category"] == "billing"
    assert out["priority"] in {"high", "medium", "low"}
    assert len(out["similar_cases"]) == 3
    assert out["draft_reply"]
