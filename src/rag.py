"""Grounded reply drafting: retrieve similar historical cases, then answer from their resolutions."""
from typing import List, Tuple

from . import llm

GREETING = {"en": "Hello,", "ar": "أهلا بك،"}
REFS = {"en": "Based on similar resolved cases:", "ar": "بناء على حالات مشابهة تم حلها:"}
CLOSING = {"en": "Best regards, Support Team", "ar": "مع تحياتنا، فريق الدعم"}


def pick_grounding_case(cases: List[dict], category: str, language: str) -> dict:
    """Prefer a neighbour with the predicted category and the ticket's language."""
    for c in cases:
        if c["category"] == category and c["language"] == language:
            return c
    for c in cases:
        if c["category"] == category:
            return c
    return cases[0]


def draft_reply(ticket: str, language: str, category: str, priority: str, cases: List[dict]) -> Tuple[str, str, List[int]]:
    """Returns (reply, source, grounded_on_case_ids)."""
    out = llm.generate_reply(ticket, language, category, priority, cases[:5])
    if out is not None:
        return out["reply"], "llm", out["used_case_ids"]

    best = pick_grounding_case(cases, category, language)
    refs = ", ".join(f"#{c['id']}" for c in cases[:3])
    reply = (
        f"{GREETING[language]}\n{best['resolution']}\n\n"
        f"({REFS[language]} {refs})\n{CLOSING[language]}"
    )
    return reply, "template", [best["id"]]
