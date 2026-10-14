# Fine-tuning results: priority_lora

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  |  method: LoRA (r=16)  |  lr: 0.001

| System | Test macro-F1 | Challenge macro-F1 | ms / ticket |
|---|---|---|---|
| Fine-tuned (LoRA (r=16)) | 1.0 | 0.537 | 1.0 |
| Embeddings + LR (baseline) | 0.8408 | 0.5 | - |
