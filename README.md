# Intelligent Support Ticket Assistant (English + Arabic)

Capstone project for the **Applied NLP Engineering** bootcamp, AI Engineering Track, SDAIA Academy.

- SDAIA Academy on GitHub: https://github.com/SDAIAAcademy
- Course page: https://sdaiaacademy.github.io/applied-nlp/

**Author:** Ghala Khalid Al-Harbi (individual project)

> **ملخص بالعربي:** نظام يستقبل رسالة عميل (عربي أو إنجليزي) ويحدد نيّتها وفئتها وأولويتها، ويبحث عن حالات مشابهة سابقة، ثم يكتب ردًّا مقترحًا مبنيًّا على حلول تلك الحالات. يغطي المشروع التمثيل (Embeddings)، التوليد المقيَّد بالمصادر (RAG)، المقارنة بين النماذج، وتحليل الأخطاء.

## What it does

Given a customer message in English or Arabic, the system:

1. detects the **language**,
2. predicts **intent**, **category** (billing, technical, account, shipping, refund) and **priority** (high, medium, low),
3. retrieves the **most similar historical tickets** (semantic search),
4. drafts a **reply grounded in those cases' resolutions** and lists which cases it relied on.

```
ticket ──► language detect ──► multilingual embedding ──┬─► Logistic Regression ─► intent / category / priority
                                                        └─► FAISS search ─► similar cases ─► grounded reply
                                                                                   (template, or LLM + Pydantic validation)
```

## Repository layout

```
app.py                 Gradio demo
run_all.sh             data -> evaluation -> example outputs
src/make_data.py       synthetic bilingual dataset (380 tickets, ~12% deliberately hard)
src/data.py            loading, stratified split (no leakage), language detection
src/embedder.py        embeddings (Sentence-Transformers) + FAISS index (NumPy fallback)
src/models.py          three decision models compared in the evaluation
src/rag.py, llm.py     grounded reply drafting (template by default, optional LLM)
src/schemas.py         Pydantic schemas for outputs and LLM JSON validation
src/pipeline.py        TicketAssistant: the end-to-end system
src/finetune.py        full fine-tuning / LoRA of the encoder (GPU)
src/evaluate.py        model comparison, retrieval metrics, challenge set, error analysis
src/run_examples.py    runs sample inputs and saves examples/sample_outputs.json
data/                  tickets.csv (generated) and challenge.csv (20 hand-written tickets)
examples/              sample_inputs.json (+ sample_outputs.json after running)
results/               metrics.json, results.md, errors.csv (generated)
tests/                 smoke tests
```

## Setup and run

### Google Colab (recommended)

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt
!bash run_all.sh          # data, evaluation, examples
!python app.py            # Gradio demo (set demo.launch(share=True) for a public link)
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
python app.py
pytest -q                 # smoke tests (offline, no model download)
```

Offline / quick mode (no model download): `EMBED_BACKEND=tfidf bash run_all.sh`.
Optional LLM for the draft reply (any OpenAI-compatible endpoint): set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. Without them, a grounded template reply is used.

## Design decisions (model selection)

| Component | Choice | Why |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | small, fast on CPU, supports Arabic and English in one space |
| Search | FAISS `IndexFlatIP` on normalized vectors | exact cosine search; dataset is tiny |
| Decision model | Logistic Regression on embeddings | cheap, fast, gives calibrated-ish confidence scores |
| Baselines | char n-gram TF-IDF + LR, and kNN vote | show whether embeddings and a learned head actually help |
| Reply generation | retrieval-grounded template (default) or LLM with JSON schema + case-id check | answers use only retrieved resolutions; ungrounded LLM output is rejected |

**Adaptation: fine-tuning and LoRA.** `src/finetune.py` fine-tunes the multilingual encoder for classification with the Hugging Face `Trainer` (validation split carved from train only, early stopping on validation macro-F1, same held-out test set as the baselines). Two modes:

```bash
python -m src.evaluate                                # baselines first (needed for the comparison table)
python -m src.finetune --target category              # full fine-tuning
python -m src.finetune --target category --lora       # LoRA via PEFT (r=16 on attention + dense layers, lr 1e-3, up to 30 epochs)
python -m src.finetune --target priority
```

Results go to `results/finetune_*.md` and are compared with the frozen-embeddings + Logistic Regression baseline on both the test set and the challenge set. A GPU (Colab T4) is recommended. I chose an encoder rather than an LLM for LoRA because the task is classification, the data is small, and an encoder is far cheaper and faster to serve.

## Evaluation

`python -m src.evaluate` writes everything to `results/`:

- model comparison on a stratified held-out test set (macro-F1, accuracy, latency per ticket),
- retrieval quality (Hit@1/3/5, MRR@5),
- a **challenge set** of 20 hand-written tickets whose wording does not appear in the templates,
- error analysis: `errors.csv`, per-language and per-difficulty accuracy, top confusions, per-class report.

**Results:** after running, copy the tables from `results/results.md` here.

<!-- Paste results/results.md tables below -->

## Limitations (please read)

- The dataset is **synthetic and template-based**, so test-set scores are optimistic. The challenge set is the more honest measure of generalization; expect lower numbers there.
- Labels for hard tickets follow one rule ("what the customer most needs resolved") and are subjective.
- Historical "resolutions" are invented, one per category and language; the grounded reply is only as good as those resolutions.
- Priority is predicted from wording only (urgency cues), not from customer tier or SLA data.
- The template reply mode does not personalize beyond the retrieved resolution.

## Acknowledgements

Built following the labs of the Applied NLP Engineering course, SDAIA Academy: https://github.com/SDAIAAcademy
