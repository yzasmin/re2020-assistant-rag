import pytest

from re2020.chunking import Chunk, load_chunks
from re2020.config import CHUNKS_PATH
from re2020.evaluation import gold_chunk_ids, load_questions
from re2020.sources import SOURCES, SOURCES_BY_ID

besoin_corpus = pytest.mark.skipif(
    not CHUNKS_PATH.exists(), reason="corpus absent : lancer `uv run python -m re2020 indexer`")


def test_citation_lisible():
    c = Chunk("id", "guide-re2020", "Guide", "3.1 Champ d'application", 34, 35, "https://x", "texte")
    assert c.citation() == "[guide-re2020, 3.1 Champ d'application, p. 34-35]"
    c2 = Chunk("id", "arrete-2021", "Arrêté", "Article 19", None, None, "https://x", "texte")
    assert c2.citation() == "[arrete-2021, Article 19]"


def test_sources_documentees():
    assert len(SOURCES) == len(SOURCES_BY_ID)
    for src in SOURCES:
        assert src.licence and src.url.startswith("https://") and src.kind in {"pdf", "html"}


@besoin_corpus
def test_corpus_indexe():
    chunks = load_chunks()
    assert len(chunks) > 300
    assert {c.doc_id for c in chunks} <= set(SOURCES_BY_ID)
    assert all(c.text.strip() for c in chunks)
    assert all(len(c.text) <= 2000 for c in chunks)


@besoin_corpus
def test_chaque_source_attendue_existe_dans_un_passage():
    chunks = load_chunks()
    manquantes = []
    for q in load_questions():
        for source, ids in zip(q.sources, gold_chunk_ids(q, chunks), strict=True):
            if not ids:
                manquantes.append((q.id, source["contient"][:60]))
    assert not manquantes, f"phrases de référence introuvables : {manquantes}"
