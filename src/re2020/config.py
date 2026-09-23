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

# Génération locale par Ollama : aucune clé, aucun appel sortant, coût nul.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Modèle retenu après mesure sur ce poste : un 3 milliards de paramètres pagine (3,6 jetons/s en
# sortie), le 1,5 milliard quantifié tient en mémoire (218 jetons/s en lecture, 10,7 en sortie).
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
AGENT_NUM_PREDICT = 400      # plafond de jetons générés par tour, pour borner la latence sur processeur
JUDGE_NUM_PREDICT = 400
AGENT_NUM_CTX = 4096         # fenêtre de contexte : consigne, outils et cinq passages
KEEP_ALIVE = "30m"           # garde le modèle chargé entre les questions (30 s de chargement évitées)
PASSAGES_PAR_RECHERCHE = 5
