"""Chemins et paramètres partagés par tout le projet."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RE2020_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
INDEX_DIR = DATA_DIR / "index"
QDRANT_DIR = INDEX_DIR / "qdrant"
EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = ROOT / "results"

# Embeddings : multilingual-e5-small (licence MIT), export ONNX quantifié int8 publié par Xenova.
EMBED_MODEL = "Xenova/multilingual-e5-small"
EMBED_MODEL_FILE = "onnx/model_quantized.onnx"
EMBED_DIM = 384
EMBED_BATCH = 16
QDRANT_COLLECTION = "re2020"

# Découpage
CHUNK_TARGET_CHARS = 1100
CHUNK_MAX_CHARS = 1700

# Fusion
RRF_K = 60
CANDIDATES_PER_RETRIEVER = 50

# Modèles Anthropic (identifiants vérifiés dans la documentation du SDK, septembre 2026).
AGENT_MODEL = os.environ.get("RE2020_AGENT_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("RE2020_JUDGE_MODEL", "claude-sonnet-5")

# Prix publics en dollars par million de jetons (entrée, sortie).
PRICES_USD_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.10
