"""Recherche lexicale (BM25), vectorielle (e5 + Qdrant local) et hybride (Reciprocal Rank Fusion)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from rank_bm25 import BM25Okapi

from re2020 import config
from re2020.chunking import Chunk, load_chunks
from re2020.text import tokenize

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


def passage_text(chunk: Chunk) -> str:
    """Texte indexé : titre du document et de la section, puis le passage."""
    return f"{chunk.doc_titre}. {chunk.section}. {chunk.text}"


def rrf(rankings: Iterable[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion : score(d) = somme sur les classements de 1 / (k + rang)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(passage_text(c)) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]


def _embedder():
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if config.EMBED_MODEL not in known:
        TextEmbedding.add_custom_model(
            model=config.EMBED_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=config.EMBED_MODEL),
            dim=config.EMBED_DIM,
            model_file=config.EMBED_MODEL_FILE,
            license="mit",
        )
    return TextEmbedding(model_name=config.EMBED_MODEL, threads=2)


class VectorIndex:
    """Index Qdrant en mode local (fichiers sur disque, sans serveur)."""

    def __init__(self, path=None):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=str(path or config.QDRANT_DIR))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _embedder()
        return self._model

    def build(self, chunks: list[Chunk]) -> None:
        from qdrant_client import models

        if self.client.collection_exists(config.QDRANT_COLLECTION):
            self.client.delete_collection(config.QDRANT_COLLECTION)
        self.client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
        )
        texts = [f"passage: {passage_text(c)}" for c in chunks]
        batch = config.EMBED_BATCH
        for start in range(0, len(texts), batch):
            vectors = list(self.model.embed(texts[start:start + batch], batch_size=batch))
            self.client.upsert(
                config.QDRANT_COLLECTION,
                points=[
                    models.PointStruct(id=start + j, vector=v.tolist(), payload={"chunk_id": chunks[start + j].chunk_id})
                    for j, v in enumerate(vectors)
                ],
            )
            print(f"  vecteurs {min(start + batch, len(texts))}/{len(texts)}", flush=True)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = next(iter(self.model.embed([f"query: {query}"]))).tolist()
        res = self.client.query_points(config.QDRANT_COLLECTION, query=vec, limit=k, with_payload=True)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def close(self) -> None:
        self.client.close()


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, vector_index: VectorIndex | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self._vector = vector_index

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index(self.chunks)

    @property
    def vector(self) -> VectorIndex:
        if self._vector is None:
            self._vector = VectorIndex()
        return self._vector

    def ranked_ids(self, query: str, mode: Mode, k: int) -> list[tuple[str, float]]:
        n = max(k, config.CANDIDATES_PER_RETRIEVER)
        if mode == "bm25":
            return self.bm25.search(query, k)
        if mode == "vector":
            return self.vector.search(query, k)
        if mode == "hybrid":
            lex = [cid for cid, _ in self.bm25.search(query, n)]
            sem = [cid for cid, _ in self.vector.search(query, n)]
            return rrf([lex, sem])[:k]
        raise ValueError(f"mode inconnu : {mode}")

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        return [
            Hit(self.by_id[cid], score, rank)
            for rank, (cid, score) in enumerate(self.ranked_ids(query, mode, k), start=1)
        ]


def build_vector_index() -> None:
    chunks = load_chunks()
    idx = VectorIndex()
    try:
        idx.build(chunks)
    finally:
        idx.close()
