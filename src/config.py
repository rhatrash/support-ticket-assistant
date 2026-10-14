"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EXAMPLES_DIR = ROOT / "examples"

SEED = 42
TOP_K = 3          # similar cases shown to the user
CANDIDATES_K = 8   # neighbours fetched before choosing the grounding case

# "st" = Sentence-Transformers (multilingual, supports Arabic)
# "tfidf" = lightweight offline fallback (char n-grams + SVD), useful for tests / no internet
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "st")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Optional OpenAI-compatible LLM for the draft reply. If not set, a grounded template reply is used.
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
LLM_MODEL = os.getenv("LLM_MODEL")
