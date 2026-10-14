#!/usr/bin/env bash
# Reproduce everything: data -> evaluation -> example outputs.  (Add EMBED_BACKEND=tfidf to run fully offline.)
set -e
python -m src.make_data
python -m src.evaluate
python -m src.run_examples
