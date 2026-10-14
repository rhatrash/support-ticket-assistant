"""Fine-tune a multilingual encoder (full fine-tuning, or LoRA with --lora) for ticket classification.

Needs a GPU for comfortable speed (Colab T4 is enough; ~2-5 minutes on this dataset).

Examples:
    python -m src.finetune --target category            # full fine-tuning
    python -m src.finetune --target category --lora     # parameter-efficient (LoRA via PEFT)
    python -m src.finetune --target priority

Outputs: results/finetune_<target>[_lora].json / .md (test + challenge-set metrics, compared with the
baseline from `python -m src.evaluate`), and the model in models/ (git-ignored).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import DATA_DIR, RESULTS_DIR, ROOT, SEED
from .data import load_tickets, split

FT_MODEL = os.getenv("FT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=128)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def baseline_numbers(target: str):
    """Read the 'Embeddings + LR' numbers produced by src.evaluate (if it was run)."""
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    base = next(s for s in m["systems"] if s["system"].startswith("Embeddings"))
    ch = next(s for s in m["challenge_set"]["systems"] if s["system"].startswith("Embeddings"))
    return {"test_macro_f1": base[target]["macro_f1"], "challenge_macro_f1": ch[target]["macro_f1"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["category", "priority"], default="category")
    ap.add_argument("--lora", action="store_true", help="use LoRA (PEFT) instead of full fine-tuning")
    ap.add_argument("--epochs", type=int, default=None, help="default 10 (full) / 30 (LoRA)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-5 (full) / 1e-3 (LoRA)")
    ap.add_argument("--patience", type=int, default=None, help="early-stopping patience; default 2 (full) / 6 (LoRA)")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = ap.parse_args()
    set_seed(SEED)

    tag = f"{args.target}{'_lora' if args.lora else ''}"
    lr = args.lr or (1e-3 if args.lora else 3e-5)
    epochs = args.epochs or (30 if args.lora else 10)
    patience = args.patience or (6 if args.lora else 2)

    # ---- data: the SAME test split as src.evaluate; validation is carved out of train only
    train_full, test = split(load_tickets())
    train, val = train_test_split(
        train_full, test_size=0.12, stratify=train_full[args.target], random_state=SEED
    )
    classes = sorted(train_full[args.target].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    print(f"train={len(train)} val={len(val)} test={len(test)} classes={classes}")

    tokenizer = AutoTokenizer.from_pretrained(FT_MODEL)
    mk = lambda df: TextDataset(df["text"], df[args.target].map(label2id), tokenizer)  # noqa: E731
    ds_train, ds_val, ds_test = mk(train), mk(val), mk(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        FT_MODEL, num_labels=len(classes), label2id=label2id, id2label=id2label
    )
    if args.lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(task_type="SEQ_CLS", r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.1,
                         target_modules=["query", "key", "value", "dense"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    out_dir = ROOT / "models" / tag
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],  # stop when val F1 stops improving
    )
    trainer.train()
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    # ---- evaluation: held-out test set + out-of-template challenge set
    t0 = time.perf_counter()
    test_pred = trainer.predict(ds_test).predictions.argmax(-1)
    ms = (time.perf_counter() - t0) / len(test) * 1000
    y_test = test[args.target].map(label2id).values

    challenge = pd.read_csv(DATA_DIR / "challenge.csv")
    ch_pred = trainer.predict(
        TextDataset(challenge["text"], challenge[args.target].map(label2id), tokenizer)
    ).predictions.argmax(-1)
    y_ch = challenge[args.target].map(label2id).values

    res = {
        "model": FT_MODEL,
        "method": f"LoRA (r={args.rank})" if args.lora else "full fine-tuning",
        "target": args.target,
        "learning_rate": lr,
        "test": {"accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
                 "macro_f1": round(float(f1_score(y_test, test_pred, average="macro")), 4)},
        "challenge": {"accuracy": round(float(accuracy_score(y_ch, ch_pred)), 4),
                      "macro_f1": round(float(f1_score(y_ch, ch_pred, average="macro")), 4)},
        "latency_ms_per_ticket": round(ms, 2),
        "baseline_embeddings_lr": baseline_numbers(args.target),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"finetune_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    base = res["baseline_embeddings_lr"]
    lines = [
        f"# Fine-tuning results: {tag}", "",
        f"Model: `{FT_MODEL}`  |  method: {res['method']}  |  lr: {lr}", "",
        "| System | Test macro-F1 | Challenge macro-F1 | ms / ticket |", "|---|---|---|---|",
        f"| Fine-tuned ({res['method']}) | {res['test']['macro_f1']} | {res['challenge']['macro_f1']} | {res['latency_ms_per_ticket']} |",
    ]
    if base:
        lines.append(f"| Embeddings + LR (baseline) | {base['test_macro_f1']} | {base['challenge_macro_f1']} | - |")
    (RESULTS_DIR / f"finetune_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
