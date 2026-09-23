"""Tests de la boucle d'agent avec un client Ollama simulé.

Aucun appel réseau : le client simulé rejoue des réponses figées. Ces tests ne publient aucun chiffre
de performance, ils vérifient la mécanique (appels d'outils, citations, refus, comptage des jetons).
"""

from __future__ import annotations

from re2020.agent import MARQUEUR_REFUS, Agent, formater_passages
from re2020.chunking import Chunk
from re2020.ollama_client import ChatResponse, json_depuis_texte
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


def appel(nom, arguments):
    return {"function": {"name": nom, "arguments": arguments}}


def reponse(contenu="", tool_calls=None, entree=100, sortie=50):
    return ChatResponse(content=contenu, tool_calls=tool_calls or [], duree_s=1.0,
                        jetons_entree=entree, jetons_sortie=sortie)


class FauxClient:
    """Rejoue une liste de réponses ; enregistre les requêtes reçues."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.requetes = []

    def chat(self, model, messages, tools=None, format=None, options=None):
        self.requetes.append({"model": model, "messages": [dict(m) for m in messages],
                              "tools": tools, "format": format, "options": options})
        return self.reponses.pop(0)


def appels_modele(trace):
    """Appels décidés par le modèle, hors recherche initiale déterministe."""
    return [a for a in trace.appels_outils if not a.get("automatique")]


def agent_simule(reponses, **kwargs):
    return Agent(retriever=Retriever(CORPUS), client=FauxClient(reponses),
                 model="qwen2.5:3b-instruct-q4_K_M", mode="bm25", **kwargs)


def test_appel_de_recherche_puis_reponse_citee():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "perméabilité maison individuelle"})]),
        reponse("La limite est 0,60 m3/(h.m2) [arrete-2021, Article 19]."),
    ])
    trace = agent.answer("Quelle perméabilité pour une maison individuelle ?")
    assert trace.appels_outils[0] == {"outil": "rechercher_reglementation", "erreur": False,
                                      "automatique": True,
                                      "entree": {"requete": "Quelle perméabilité pour une maison individuelle ?"}}
    assert appels_modele(trace)[0]["outil"] == "rechercher_reglementation"
    assert trace.citations == [("arrete-2021", "Article 19")]
    assert "doc#0" in trace.passages_vus
    assert not trace.refus
    assert trace.tours == 2


def test_resultat_outil_transmis_au_modele_avec_le_role_tool():
    agent = agent_simule([
        reponse(tool_calls=[appel("calculer", {"expression": "0,55 * 1,2"})]),
        reponse("0,66 m3/(h.m2), donc non conforme [arrete-2021, Article 19]."),
    ])
    agent.answer("0,55 par échantillonnage, conforme ?")
    dernier = agent._client.requetes[1]["messages"][-1]
    assert dernier["role"] == "tool" and dernier["content"] == "0.66"


def test_arguments_transmis_sous_forme_de_chaine():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", '{"requete": "étanchéité"}')]),
        reponse("Réponse [arrete-2021, Article 19]."),
    ])
    trace = agent.answer("question")
    assert appels_modele(trace)[0]["entree"] == {"requete": "étanchéité"}


def test_expression_refusee_remonte_une_erreur_d_outil():
    agent = agent_simule([
        reponse(tool_calls=[appel("calculer", {"expression": "__import__('os')"})]),
        reponse(f"{MARQUEUR_REFUS} : calcul impossible."),
    ])
    trace = agent.answer("calcule ceci")
    assert appels_modele(trace)[0]["erreur"] is True
    assert agent._client.requetes[1]["messages"][-1]["content"].startswith("Calcul refusé")
    assert trace.refus


def test_requete_vide_signalee_sans_recherche():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", {})]),
        reponse("Réponse finale."),
    ])
    trace = agent.answer("question")
    assert appels_modele(trace)[0]["erreur"] is True


def test_refus_detecte_sur_question_hors_perimetre():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "MaPrimeRénov"})]),
        reponse(f"{MARQUEUR_REFUS} : les textes indexés ne traitent pas des aides financières."),
    ])
    trace = agent.answer("Quel montant de MaPrimeRénov ?")
    assert trace.refus and trace.citations == []


def test_outil_inconnu_signale_sans_interrompre():
    agent = agent_simule([
        reponse(tool_calls=[appel("outil_fantome", {})]),
        reponse("Réponse finale."),
    ])
    trace = agent.answer("test")
    assert appels_modele(trace)[0]["erreur"] is True and trace.reponse == "Réponse finale."


def test_outils_retires_au_dernier_tour_pour_forcer_une_conclusion():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "x"})]),
        reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "y"})]),
        reponse("Conclusion [arrete-2021, Article 9]."),
    ], max_tours=3)
    trace = agent.answer("question")
    requetes = agent._client.requetes
    assert requetes[0]["tools"] is not None and requetes[2]["tools"] is None
    assert trace.erreur is None and trace.tours == 3


def test_arret_apres_le_nombre_maximal_de_tours():
    agent = agent_simule([reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "x"})])
                          for _ in range(3)], max_tours=3)
    trace = agent.answer("question sans fin")
    assert trace.erreur is not None and trace.tours == 3


def test_passages_joints_a_la_question_des_le_premier_appel():
    agent = agent_simule([reponse("Réponse [arrete-2021, Article 19].")])
    trace = agent.answer("perméabilité maison individuelle")
    contenu = agent._client.requetes[0]["messages"][1]["content"]
    assert contenu.startswith("Question : perméabilité maison individuelle")
    assert "[arrete-2021, Article 19]" in contenu
    assert trace.passages_vus and trace.tours == 1


def test_consigne_et_outils_envoyes_au_modele():
    agent = agent_simule([reponse("Réponse.")])
    agent.answer("question")
    requete = agent._client.requetes[0]
    assert requete["messages"][0]["role"] == "system"
    assert {o["function"]["name"] for o in requete["tools"]} == {"rechercher_reglementation", "calculer"}
    assert requete["options"]["temperature"] == 0
    assert requete["model"] == "qwen2.5:3b-instruct-q4_K_M"


def test_cumul_des_jetons_sur_plusieurs_tours():
    agent = agent_simule([
        reponse(tool_calls=[appel("rechercher_reglementation", {"requete": "x"})], entree=1000, sortie=100),
        reponse("Fini.", entree=2000, sortie=200),
    ])
    trace = agent.answer("question")
    assert trace.jetons == {"entree": 3000, "sortie": 300}


def test_formatage_des_passages_expose_la_citation_et_la_source():
    texte = formater_passages([Hit(CORPUS[0], 1.0, 1)])
    assert "[arrete-2021, Article 19]" in texte
    assert "https://exemple.fr" in texte
    assert formater_passages([]).startswith("Aucun passage")


def test_lecture_json_tolerante_aux_blocs_de_code():
    assert json_depuis_texte('```json\n{"a": 1}\n```') == {"a": 1}
    assert json_depuis_texte('Voici : {"a": [1, 2]} fin') == {"a": [1, 2]}
