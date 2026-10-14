"""Compare the decision models, evaluate retrieval, and write an error analysis.

Run:  python -m src.evaluate
Outputs (in results/): metrics.json, results.md, errors.csv, classification_report_category.txt
"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import DATA_DIR, RESULTS_DIR
from .data import load_tickets, split
from .embedder import Embedder
from .models import TARGETS, EmbLR, KNNVote, TfidfLR


def score(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def retrieval_metrics(knn: KNNVote, embedder: Embedder, test: pd.DataFrame, train: pd.DataFrame, k: int = 5):
    """A retrieved ticket is 'relevant' if it has the same category as the query."""
    _, idx = knn.index.search(embedder.encode(test["text"].tolist()), k)
    train_cat = train["category"].values
    hits = {1: 0, 3: 0, 5: 0}
    rr = []
    for q_cat, row in zip(test["category"].values, idx):
        rel = [train_cat[i] == q_cat for i in row]
        for kk in hits:
            hits[kk] += any(rel[:kk])
        rr.append(next((1 / (r + 1) for r, ok in enumerate(rel) if ok), 0.0))
    n = len(test)
    return {"hit@1": round(hits[1] / n, 4), "hit@3": round(hits[3] / n, 4),
            "hit@5": round(hits[5] / n, 4), "mrr@5": round(float(np.mean(rr)), 4)}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_tickets()
    train, test = split(df)
    print(f"train={len(train)}  test={len(test)}")

    train_texts, test_texts = train["text"].tolist(), test["text"].tolist()
    labels = {t: train[t].values for t in TARGETS}

    embedder = Embedder().fit(train_texts)
    Xtr = embedder.encode(train_texts)

    systems = [
        TfidfLR().fit(train_texts, labels),
        EmbLR(embedder).fit(train_texts, labels, vecs=Xtr),
        KNNVote(embedder, k=5).fit(train_texts, labels, vecs=Xtr),
    ]

    results, all_preds = [], {}
    for s in systems:
        t0 = time.perf_counter()
        preds = s.predict(test_texts)
        ms = (time.perf_counter() - t0) / len(test) * 1000
        all_preds[s.name] = preds
        row = {"system": s.name, "latency_ms_per_ticket": round(ms, 2)}
        for t in TARGETS:
            m = score(test[t].values, preds[t])
            row[t] = m
        results.append(row)
        print(s.name, {t: row[t]["macro_f1"] for t in TARGETS}, f"{ms:.2f} ms")

    retrieval = retrieval_metrics(systems[2], embedder, test, train)
    print("retrieval:", retrieval)

    # ---------- error analysis on the final-pipeline model (Embeddings + LR)
    final = systems[1]
    preds, conf = final.predict_with_conf(test_texts)
    err = test.copy()
    for t in TARGETS:
        err[f"pred_{t}"] = preds[t]
        err[f"conf_{t}"] = np.round(conf[t], 3)
    wrong = err[(err["category"] != err["pred_category"]) | (err["priority"] != err["pred_priority"])
                | (err["intent"] != err["pred_intent"])]
    cols = ["id", "language", "is_hard", "text"] + [c for t in TARGETS for c in (t, f"pred_{t}", f"conf_{t}")]
    wrong[cols].to_csv(RESULTS_DIR / "errors.csv", index=False, encoding="utf-8-sig")

    by_lang = {lang: score(g["category"], g["pred_category"]) for lang, g in err.groupby("language")}
    by_hard = {("hard" if h else "regular"): score(g["category"], g["pred_category"]) for h, g in err.groupby("is_hard")}
    confusions = Counter(
        (a, b) for a, b in zip(err["category"], err["pred_category"]) if a != b
    ).most_common(5)
    report = classification_report(err["category"], err["pred_category"], zero_division=0)
    (RESULTS_DIR / "classification_report_category.txt").write_text(report, encoding="utf-8")

    # ---------- out-of-template challenge set (hand-written paraphrases never seen in the templates)
    challenge = pd.read_csv(DATA_DIR / "challenge.csv")
    ch_texts = challenge["text"].tolist()
    challenge_results = []
    for s in systems:
        p = s.predict(ch_texts)
        challenge_results.append({"system": s.name,
                                  **{t: score(challenge[t].values, p[t]) for t in ("category", "priority")}})
        print("challenge", s.name, {t: challenge_results[-1][t]["macro_f1"] for t in ("category", "priority")})
    cp, cc = final.predict_with_conf(ch_texts)
    ch_err = challenge.copy()
    for t in ("category", "priority"):
        ch_err[f"pred_{t}"] = cp[t]
        ch_err[f"conf_{t}"] = np.round(cc[t], 3)
    ch_err = ch_err[(ch_err["category"] != ch_err["pred_category"]) | (ch_err["priority"] != ch_err["pred_priority"])]
    ch_err.to_csv(RESULTS_DIR / "challenge_errors.csv", index=False, encoding="utf-8-sig")

    metrics = {"n_train": len(train), "n_test": len(test), "systems": results, "retrieval": retrieval,
               "challenge_set": {"n": len(challenge), "systems": challenge_results},
               "error_analysis": {"n_wrong_any_target": int(len(wrong)),
                                  "category_by_language": by_lang, "category_by_difficulty": by_hard,
                                  "top_category_confusions": [[a, b, n] for (a, b), n in confusions]}}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- human-readable summary
    lines = [
        "# Evaluation results", "",
        f"Train: {len(train)} tickets, Test: {len(test)} tickets (stratified split, fixed seed).", "",
        "## Model comparison (macro-F1 on the test set)", "",
        "| System | Category F1 | Priority F1 | Intent F1 | ms / ticket |", "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['system']} | {r['category']['macro_f1']} | {r['priority']['macro_f1']} | "
                     f"{r['intent']['macro_f1']} | {r['latency_ms_per_ticket']} |")
    lines += ["", f"## Challenge set ({len(challenge)} hand-written tickets with wording NOT seen in the templates)", "",
              "| System | Category F1 | Priority F1 | Category acc | Priority acc |", "|---|---|---|---|---|"]
    for r in challenge_results:
        lines.append(f"| {r['system']} | {r['category']['macro_f1']} | {r['priority']['macro_f1']} | "
                     f"{r['category']['accuracy']} | {r['priority']['accuracy']} |")
    lines += ["", f"Wrong on the challenge set (final model): **{len(ch_err)} / {len(challenge)}** (see `challenge_errors.csv`)."]
    lines += ["", "## Retrieval quality (similar-case search, relevant = same category)", "",
              "| Hit@1 | Hit@3 | Hit@5 | MRR@5 |", "|---|---|---|---|",
              f"| {retrieval['hit@1']} | {retrieval['hit@3']} | {retrieval['hit@5']} | {retrieval['mrr@5']} |", "",
              "## Error analysis (final model: Embeddings + LR)", "",
              f"- Tickets with at least one wrong prediction: **{len(wrong)} / {len(test)}** (see `errors.csv`)",
              "- Category accuracy by language: " + ", ".join(f"{k}={v['accuracy']}" for k, v in by_lang.items()),
              "- Category accuracy by difficulty: " + ", ".join(f"{k}={v['accuracy']}" for k, v in by_hard.items()),
              "- Most common category confusions (true -> predicted): "
              + (", ".join(f"{a}->{b} ({n})" for (a, b), n in confusions) or "none"), "",
              "```", report, "```", ""]
    (RESULTS_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
