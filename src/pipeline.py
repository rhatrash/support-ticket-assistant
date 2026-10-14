"""End-to-end system: ticket text -> language, intent, category, priority, similar cases, draft reply."""
from typing import Dict

import pandas as pd

from .config import CANDIDATES_K, TOP_K
from .data import detect_language
from .embedder import Embedder, VectorIndex
from .models import TARGETS, EmbLR
from .rag import draft_reply

try:
    from .schemas import TicketAnalysis
except ImportError:  # pydantic missing -> return plain dicts
    TicketAnalysis = None


class TicketAssistant:
    def __init__(self, train_df: pd.DataFrame, embedder: Embedder = None):
        self.df = train_df.reset_index(drop=True)
        texts = self.df["text"].tolist()
        self.embedder = embedder or Embedder().fit(texts)
        self.vecs = self.embedder.encode(texts)
        self.index = VectorIndex(self.vecs.shape[1])
        self.index.add(self.vecs)
        labels = {t: self.df[t].values for t in TARGETS}
        self.classifier = EmbLR(self.embedder).fit(texts, labels, vecs=self.vecs)

    def analyze(self, text: str) -> Dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty ticket text")
        language = detect_language(text)
        qv = self.embedder.encode([text])

        preds, conf = self.classifier.predict_vecs(qv)
        category, priority, intent = (str(preds[t][0]) for t in ("category", "priority", "intent"))

        scores, idx = self.index.search(qv, CANDIDATES_K)
        cases = []
        for s, i in zip(scores[0], idx[0]):
            row = self.df.iloc[int(i)]
            cases.append(
                {
                    "id": int(row["id"]),
                    "text": row["text"],
                    "category": row["category"],
                    "language": row["language"],
                    "resolution": row["resolution"],
                    "score": round(float(s), 4),
                }
            )

        reply, source, grounded_on = draft_reply(text, language, category, priority, cases)
        result = {
            "text": text,
            "language": language,
            "intent": intent,
            "category": category,
            "priority": priority,
            "confidence": {t: round(float(conf[t][0]), 3) for t in TARGETS},
            "similar_cases": [
                {k: c[k] for k in ("id", "text", "category", "score")} for c in cases[:TOP_K]
            ],
            "draft_reply": reply,
            "reply_source": source,
            "grounded_on": grounded_on,
        }
        if TicketAnalysis is not None:
            result = TicketAnalysis(**result).model_dump()
        return result
