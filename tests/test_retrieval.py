import pytest

from re2020.chunking import Chunk
from re2020.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from re2020.retrieval import BM25Index, Retriever, rrf
from re2020.text import normalize, tokenize


def chunk(cid, texte, doc="doc", section="Article 1"):
    return Chunk(cid, doc, "Document de test", section, 1, 1, "https://exemple.fr", texte)


CORPUS = [
    chunk("a", "La perméabilité à l'air Q4Pa-surf est inférieure ou égale à 0,60 m3/(h.m2) en maison individuelle."),
    chunk("b", "Le coefficient de conversion de l'électricité en énergie primaire est égal à 2,3."),
    chunk("c", "Les baies s'ouvrent sur au moins 30 % de leur surface totale."),
]


def test_normalisation_francaise():
    assert normalize("Étanchéité à l'air") == "etancheite a l'air"
    assert normalize("HSP moy  2,5 m") == "hsp moy 2,5 m"


def test_tokenisation_retire_mots_vides_et_garde_les_identifiants():
    tokens = tokenize("La perméabilité de l'enveloppe Q4Pa-surf vaut 0,60 m3")
    assert "la" not in tokens and "de" not in tokens
    assert "0,60" in tokens
    assert any(t.startswith("permeabil") for t in tokens)


def test_bm25_trouve_le_passage_attendu():
    index = BM25Index(CORPUS)
    assert index.search("perméabilité à l'air maison individuelle", 1)[0][0] == "a"
    assert index.search("coefficient électricité énergie primaire", 1)[0][0] == "b"


def test_bm25_sans_correspondance_renvoie_une_liste_vide():
    assert BM25Index(CORPUS).search("kangourou", 3) == []


def test_rrf_favorise_un_document_bien_classe_par_les_deux_moteurs():
    fusion = dict(rrf([["a", "b", "c"], ["b", "a", "c"]]))
    assert fusion["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert max(fusion, key=fusion.get) in {"a", "b"}
    assert fusion["c"] < fusion["a"]


def test_retriever_mode_inconnu():
    with pytest.raises(ValueError):
        Retriever(CORPUS).ranked_ids("test", "magique", 3)


def test_recall_compte_les_sources_couvertes():
    groupes = [["a", "a2"], ["b"]]
    assert recall_at_k(groupes, ["a", "z"], 1) == 0.5
    assert recall_at_k(groupes, ["a2", "b"], 3) == 1.0
    assert recall_at_k(groupes, ["z"], 10) == 0.0


def test_mrr_et_ndcg():
    groupes = [["a"], ["b"]]
    assert reciprocal_rank(groupes, ["z", "a", "b"]) == 0.5
    assert reciprocal_rank(groupes, ["z"]) == 0.0
    assert ndcg_at_k([["a"]], ["a"], 10) == 1.0
    assert ndcg_at_k([["a"]], ["z", "a"], 10) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k([["a"], ["b"]], ["a", "b"], 10) == 1.0
