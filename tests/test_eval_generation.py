"""Tests de l'évaluation de génération et du juge, avec un client simulé (aucun appel réseau)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from re2020.agent import Trace
from re2020.chunking import Chunk
from re2020.eval_generation import citations_valides, passages_cites, refus_correct
from re2020.evaluation import Question, load_questions
from re2020.judge import Judge

CHUNKS = {
    "arrete-2021#0": Chunk("arrete-2021#0", "arrete-2021", "Arrêté", "Article 19", None, None, "u", "0,60 m3/(h.m2)"),
    "guide-re2020#0": Chunk("guide-re2020#0", "guide-re2020", "Guide", "3.1 Champ", 34, 34, "u", "1er janvier 2022"),
}


def trace(reponse, vus, citations):
    t = Trace(question="q", reponse=reponse)
    t.passages_vus = vus
    t.citations = citations
    return t


def test_passages_cites_se_limite_aux_documents_cites():
    t = trace("texte [arrete-2021, Article 19]", list(CHUNKS), [("arrete-2021", "Article 19")])
    extraits = passages_cites(t, CHUNKS)
    assert len(extraits) == 1 and "0,60" in extraits[0]


def test_passages_cites_retombe_sur_tous_les_passages_vus():
    t = trace("texte sans citation", list(CHUNKS), [])
    assert len(passages_cites(t, CHUNKS)) == 2


def test_citation_vers_un_document_non_consulte_est_invalide():
    t = trace("texte [guide-re2020, 3.1 Champ] et [inconnu, Article 1]", ["arrete-2021#0"],
              [("guide-re2020", "3.1 Champ"), ("inconnu", "Article 1")])
    assert citations_valides(t, CHUNKS) == (0, 2)


def test_citation_vers_un_document_consulte_est_valide():
    t = trace("texte [arrete-2021, Article 19]", ["arrete-2021#0"], [("arrete-2021", "Article 19")])
    assert citations_valides(t, CHUNKS) == (1, 1)


def test_refus_correct_selon_le_perimetre():
    hors = Question("q33", "question", "hors-perimetre", "refus attendu", [], "", True)
    dans = Question("q01", "question", "seuil", "réponse", [{"doc": "x", "contient": "y"}], "", False)
    assert refus_correct(hors, trace("Hors périmètre : rien dans les textes.", [], []))
    assert not refus_correct(hors, trace("La valeur est 3.", [], []))
    assert refus_correct(dans, trace("La valeur est 3 [arrete-2021, Article 9].", [], []))
    assert not refus_correct(dans, trace("Hors périmètre : je ne sais pas.", [], []))


class FauxClientJuge:
    def __init__(self, verdict):
        payload = json.dumps(verdict, ensure_ascii=False)
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 1000, "output_tokens": 200})
        self.message = SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)], usage=usage)
        self.requetes = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requetes.append(kwargs)
        return self.message


def test_juge_calcule_la_fidelite_et_compte_le_cout():
    verdict_simule = {
        "affirmations": [
            {"affirmation": "0,60 m3/(h.m2)", "soutenue": True, "justification": "passage 1"},
            {"affirmation": "obligation depuis 2020", "soutenue": False, "justification": "absent"},
        ],
        "exactitude": "partielle",
        "commentaire": "un chiffre non soutenu",
    }
    client = FauxClientJuge(verdict_simule)
    juge = Judge(client=client, model="claude-sonnet-5")
    verdict = juge.evaluer("question", "réponse", ["[arrete-2021, Article 19] 0,60"], "référence")
    assert verdict["fidelite"] == 0.5
    assert verdict["exactitude"] == "partielle"
    # claude-sonnet-5 : 2 $ / Mjetons en entrée, 10 $ en sortie
    assert juge.cout_total_usd == pytest.approx(1000 / 1e6 * 2 + 200 / 1e6 * 10)
    assert client.requetes[0]["output_config"]["format"]["type"] == "json_schema"


def test_juge_sans_affirmation_ne_fabrique_pas_de_score():
    juge = Judge(client=FauxClientJuge({"affirmations": [], "exactitude": "correcte", "commentaire": ""}))
    assert juge.evaluer("q", "r", [], "ref")["fidelite"] is None


def test_jeu_de_questions_complet_et_coherent():
    questions = load_questions()
    assert len(questions) >= 30
    assert sum(q.hors_perimetre for q in questions) >= 3
    assert sum(q.type == "calcul" for q in questions) >= 3
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.reponse_reference.strip()
        assert bool(q.sources) != q.hors_perimetre
