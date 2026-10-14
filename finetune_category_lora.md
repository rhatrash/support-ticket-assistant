# Fine-tuning results: category_lora

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  |  method: LoRA (r=16)  |  lr: 0.001

| System | Test macro-F1 | Challenge macro-F1 | ms / ticket |
|---|---|---|---|
| Fine-tuned (LoRA (r=16)) | 0.9157 | 0.6324 | 0.97 |
| Embeddings + LR (baseline) | 1.0 | 0.6406 | - |
