"""Tests de la boucle d'agent avec un client Anthropic simulé.

Aucun appel réseau : le client simulé rejoue des réponses figées. Ces tests ne publient aucun chiffre
de performance, ils vérifient la mécanique (appels d'outils, citations, refus, comptage du coût).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from re2020.agent import MARQUEUR_REFUS, Agent, cout_usd, formater_passages
from re2020.chunking import Chunk
from re2020.retrieval import Hit, Retriever

CORPUS = [
    Chunk("doc#0", "arrete-2021", "Arrêté du 4 août 2021", "Article 19", None, None, "https://exemple.fr",
          "La perméabilité Q4Pa-surf est inférieure ou égale à 0,60 m3/(h.m2) en maison individuelle."),
    Chunk("doc#1", "arrete-2021", "Arrêté du 4 août 2021", "Article 9", None, None, "https://exemple.fr",
          "Le coefficient de conversion en énergie primaire de l'électricité vaut 2,3."),
    # BM25 attribue une pondération nulle à un terme présent dans la moitié d'un corpus de deux documents :
    # un troisième passage rend le petit corpus de test représentatif.
    Chunk("doc#2", "guide-re2020", "Guide RE2020", "1.5 Attestations", 24, 24, "https://exemple.fr",
          "L'attestation d'achèvement est établie par un architecte ou un bureau de contrôle."),
]


def bloc_texte(texte):
    return SimpleNamespace(type="text", text=texte)


def bloc_outil(nom, entree, identifiant="tu_1"):
    return SimpleNamespace(type="tool_use", name=nom, input=entree, id=identifiant)


class FauxClient:
    """Rejoue une liste de réponses ; enregistre les requêtes reçues."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.requetes = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requetes.append(kwargs)
        return self.reponses.pop(0)


def reponse(contenu, stop_reason="end_turn", entree=100, sortie=50, cache_lecture=0):
    usage = SimpleNamespace(model_dump=lambda: {
        "input_tokens": entree, "output_tokens": sortie,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": cache_lecture,
    })
    return SimpleNamespace(content=contenu, stop_reason=stop_reason, usage=usage)


def agent_simule(reponses):
    return Agent(retriever=Retriever(CORPUS), client=FauxClient(reponses), model="claude-haiku-4-5", mode="bm25")


def test_appel_de_recherche_puis_reponse_citee():
    agent = agent_simule([
        reponse([bloc_outil("rechercher_reglementation", {"requete": "perméabilité maison individuelle"})],
                stop_reason="tool_use"),
        reponse([bloc_texte("La limite est 0,60 m3/(h.m2) [arrete-2021, Article 19].")]),
    ])
    trace = agent.answer("Quelle perméabilité pour une maison individuelle ?")
    assert trace.appels_outils[0]["outil"] == "rechercher_reglementation"
    assert trace.citations == [("arrete-2021", "Article 19")]
    assert "doc#0" in trace.passages_vus
    assert not trace.refus
    assert trace.tours == 2


def test_outil_de_calcul_et_resultat_transmis():
    client = FauxClient([
        reponse([bloc_outil("calculer", {"expression": "0,55 * 1,2"})], stop_reason="tool_use"),
        reponse([bloc_texte("0,66 m3/(h.m2), donc non conforme [arrete-2021, Article 19].")]),
    ])
    trace = Agent(retriever=Retriever(CORPUS), client=client, mode="bm25").answer("0,55 par échantillonnage, conforme ?")
    resultats = client.requetes[1]["messages"][-1]["content"]
    assert resultats[0]["content"] == "0.66"
    assert trace.appels_outils[0]["erreur"] is False


def test_expression_refusee_remonte_une_erreur_d_outil():
    client = FauxClient([
        reponse([bloc_outil("calculer", {"expression": "__import__('os')"})], stop_reason="tool_use"),
        reponse([bloc_texte(f"{MARQUEUR_REFUS} : calcul impossible.")]),
    ])
    trace = Agent(retriever=Retriever(CORPUS), client=client, mode="bm25").answer("calcule ceci")
    resultats = client.requetes[1]["messages"][-1]["content"]
    assert resultats[0]["is_error"] is True
    assert trace.appels_outils[0]["erreur"] is True
    assert trace.refus


def test_refus_detecte_sur_question_hors_perimetre():
    agent = agent_simule([
        reponse([bloc_outil("rechercher_reglementation", {"requete": "MaPrimeRénov"})], stop_reason="tool_use"),
        reponse([bloc_texte(f"{MARQUEUR_REFUS} : les textes indexés ne traitent pas des aides financières.")]),
    ])
    trace = agent.answer("Quel montant de MaPrimeRénov ?")
    assert trace.refus and trace.citations == []


def test_outil_inconnu_signale_sans_interrompre():
    agent = agent_simule([
        reponse([bloc_outil("outil_fantome", {})], stop_reason="tool_use"),
        reponse([bloc_texte("Réponse finale.")]),
    ])
    trace = agent.answer("test")
    assert trace.appels_outils[0]["erreur"] is True


def test_arret_apres_le_nombre_maximal_de_tours():
    reponses = [reponse([bloc_outil("rechercher_reglementation", {"requete": "x"})], stop_reason="tool_use")
                for _ in range(3)]
    agent = Agent(retriever=Retriever(CORPUS), client=FauxClient(reponses), max_tours=3, mode="bm25")
    trace = agent.answer("question sans fin")
    assert trace.erreur is not None and trace.tours == 3


def test_refus_de_securite_du_modele():
    agent = agent_simule([reponse([], stop_reason="refusal")])
    trace = agent.answer("question sensible")
    assert trace.erreur == "refus de sécurité du modèle"


def test_consigne_et_outils_envoyes_avec_marquage_de_cache():
    agent = agent_simule([reponse([bloc_texte("Réponse.")])])
    agent.answer("question")
    requete = agent._client.requetes[0]
    assert requete["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in requete["tools"]} == {"rechercher_reglementation", "calculer"}
    assert requete["model"] == "claude-haiku-4-5"


def test_cumul_du_cout_sur_plusieurs_tours():
    agent = agent_simule([
        reponse([bloc_outil("rechercher_reglementation", {"requete": "x"})], stop_reason="tool_use",
                entree=1000, sortie=100),
        reponse([bloc_texte("Fini.")], entree=2000, sortie=200),
    ])
    trace = agent.answer("question")
    assert trace.usage["input_tokens"] == 3000
    assert trace.usage["output_tokens"] == 300
    # claude-haiku-4-5 : 1 $ / Mjetons en entrée, 5 $ en sortie
    assert trace.cout_usd == pytest.approx(3000 / 1e6 * 1 + 300 / 1e6 * 5)


def test_cout_tient_compte_du_cache():
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert cout_usd("claude-haiku-4-5", usage) == pytest.approx(1.25 + 0.10)


def test_formatage_des_passages_expose_la_citation_et_la_source():
    texte = formater_passages([Hit(CORPUS[0], 1.0, 1)])
    assert "[arrete-2021, Article 19]" in texte
    assert "https://exemple.fr" in texte
    assert formater_passages([]).startswith("Aucun passage")


def test_entete_espace_de_travail_ajoute_quand_la_variable_existe(monkeypatch):
    from re2020.api import client_anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "cle-de-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    assert client_anthropic().default_headers["anthropic-workspace-id"] == "wrkspc_test"

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    assert "anthropic-workspace-id" not in client_anthropic().default_headers
