# Evaluation results

Train: 304 tickets, Test: 76 tickets (stratified split, fixed seed).

## Model comparison (macro-F1 on the test set)

| System | Category F1 | Priority F1 | Intent F1 | ms / ticket |
|---|---|---|---|---|
| TF-IDF (char n-gram) + LR | 1.0 | 1.0 | 1.0 | 0.38 |
| Embeddings + LR | 1.0 | 0.8408 | 0.9852 | 1.07 |
| kNN retrieval vote (k=5) | 0.8944 | 0.3913 | 0.8801 | 1.0 |

## Challenge set (20 hand-written tickets with wording NOT seen in the templates)

| System | Category F1 | Priority F1 | Category acc | Priority acc |
|---|---|---|---|---|
| TF-IDF (char n-gram) + LR | 0.4361 | 0.7313 | 0.45 | 0.75 |
| Embeddings + LR | 0.6406 | 0.5 | 0.65 | 0.5 |
| kNN retrieval vote (k=5) | 0.5202 | 0.0702 | 0.55 | 0.1 |

Wrong on the challenge set (final model): **14 / 20** (see `challenge_errors.csv`).

## Retrieval quality (similar-case search, relevant = same category)

| Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|
| 0.9605 | 0.9737 | 0.9868 | 0.9704 |

## Error analysis (final model: Embeddings + LR)

- Tickets with at least one wrong prediction: **10 / 76** (see `errors.csv`)
- Category accuracy by language: ar=1.0, en=1.0
- Category accuracy by difficulty: regular=1.0, hard=1.0
- Most common category confusions (true -> predicted): none

```
              precision    recall  f1-score   support

     account       1.00      1.00      1.00        11
     billing       1.00      1.00      1.00        16
      refund       1.00      1.00      1.00        11
    shipping       1.00      1.00      1.00        23
   technical       1.00      1.00      1.00        15

    accuracy                           1.00        76
   macro avg       1.00      1.00      1.00        76
weighted avg       1.00      1.00      1.00        76

```
