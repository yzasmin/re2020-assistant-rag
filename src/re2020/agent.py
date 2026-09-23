"""Agent outillé sur un modèle local (Ollama) : recherche hybride, calcul sûr, réponses citées ou refus."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from re2020 import config
from re2020.calculator import CalculError, calculate
from re2020.ollama_client import OllamaClient, modele_par_defaut
from re2020.retrieval import Hit, Retriever

MARQUEUR_REFUS = "Hors périmètre"

SYSTEM_PROMPT = f"""Tu es un assistant spécialisé dans la réglementation environnementale française RE2020.
Tu réponds à des professionnels du bâtiment et à des particuliers, en français, en trois phrases au plus.

Règles impératives :
1. Tu ne réponds qu'à partir des passages renvoyés par l'outil rechercher_reglementation. Tu n'utilises
   jamais tes connaissances générales pour affirmer un chiffre, une date ou une obligation.
2. Chaque affirmation tirée des textes est suivie de sa source au format [identifiant_du_document, section],
   recopiée telle quelle depuis l'en-tête du passage utilisé, par exemple [arrete-2021, Article 19].
3. Si, et seulement si, aucun passage ne contient la réponse, tu commences ta réponse par
   « {MARQUEUR_REFUS} : » et tu expliques en une phrase ce qui manque. Dès qu'un passage donne la
   réponse, tu réponds directement, sans employer cette formule, et tu ne proposes jamais de réponse
   approximative tirée de tes connaissances générales.
4. Une première recherche est déjà faite pour toi : ses passages sont joints à la question. Tu peux
   lancer une recherche complémentaire avec l'outil rechercher_reglementation si ces passages ne
   suffisent pas, puis tu réponds.
5. Pour tout calcul arithmétique, tu utilises l'outil calculer, et tu rappelles la formule et les valeurs
   réglementaires utilisées avec leurs sources.
6. Les seuils RE2020 dépendent de l'usage du bâtiment, de l'année de dépôt du permis de construire et de
   la zone climatique : précise ces conditions quand le passage les donne.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher_reglementation",
            "description": (
                "Recherche des passages dans les textes officiels RE2020 indexés (arrêté du 4 août 2021 "
                "consolidé et ses annexes, annexe de l'article R. 172-4 du code de la construction et de "
                "l'habitation, guide RE2020 du ministère). Renvoie des passages numérotés avec leur source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {
                        "type": "string",
                        "description": ("Question ou mots-clés en français, par exemple "
                                        "« seuil Icconstruction logements collectifs 2025 »"),
                    },
                },
                "required": ["requete"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer",
            "description": (
                "Évalue une expression arithmétique (opérateurs + - * / ** et fonctions min, max, abs, "
                "round, sqrt). La virgule décimale française est acceptée. Aucun accès au système."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Par exemple : 63 * 1,15 - 3,6"},
                },
                "required": ["expression"],
            },
        },
    },
]

_CITATION = re.compile(r"\[([a-zA-Z0-9\-]+)\s*,\s*([^\]]{2,160})\]")


@dataclass
class Trace:
    question: str
    reponse: str = ""
    appels_outils: list[dict] = field(default_factory=list)
    passages_vus: list[str] = field(default_factory=list)
    citations: list[tuple[str, str]] = field(default_factory=list)
    jetons: dict = field(default_factory=dict)
    latence_s: float = 0.0
    tours: int = 0
    erreur: str | None = None

    @property
    def refus(self) -> bool:
        return self.reponse.strip().lower().startswith(MARQUEUR_REFUS.lower())

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["citations"] = [list(c) for c in self.citations]
        d["refus"] = self.refus
        return d


def formater_passages(hits: list[Hit]) -> str:
    blocs = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocs.append(f"Passage {i} {c.citation()} (source : {c.url})\n{c.text}")
    return "\n\n".join(blocs) if blocs else "Aucun passage trouvé pour cette requête."


class Agent:
    """Boucle d'appel d'outils sur l'API /api/chat d'Ollama."""

    def __init__(self, retriever: Retriever | None = None, client: OllamaClient | None = None,
                 model: str | None = None, max_tours: int = 3,
                 k: int = config.PASSAGES_PAR_RECHERCHE, mode: str = "hybrid"):
        self.retriever = retriever if retriever is not None else Retriever()
        self.model = model or modele_par_defaut()
        self.max_tours = max_tours
        self.k = k
        self.mode = mode
        self._client = client

    @property
    def client(self) -> OllamaClient:
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    def _executer_outil(self, nom: str, entree: dict, trace: Trace) -> tuple[str, bool]:
        if nom == "rechercher_reglementation":
            requete = str(entree.get("requete") or entree.get("query") or "").strip()
            if not requete:
                return "Erreur : le paramètre requete est obligatoire.", True
            hits = self.retriever.search(requete, self.k, mode=self.mode)
            trace.passages_vus.extend(h.chunk.chunk_id for h in hits)
            return formater_passages(hits), False
        if nom == "calculer":
            try:
                return f"{calculate(str(entree.get('expression', ''))):.10g}", False
            except CalculError as exc:
                return f"Calcul refusé : {exc}", True
        return f"Outil inconnu : {nom}", True

    def answer(self, question: str) -> Trace:
        trace = Trace(question=question)
        debut = time.perf_counter()
        # Première recherche déterministe : un modèle local de 1,5 milliard de paramètres n'émet pas
        # d'appel d'outil fiable au premier tour. Les outils restent disponibles ensuite.
        hits = self.retriever.search(question, self.k, mode=self.mode)
        trace.passages_vus.extend(h.chunk.chunk_id for h in hits)
        trace.appels_outils.append({"outil": "rechercher_reglementation", "entree": {"requete": question},
                                    "erreur": False, "automatique": True})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question : {question}\n\nPassages trouvés :\n{formater_passages(hits)}"},
        ]
        options = {"temperature": 0, "num_ctx": config.AGENT_NUM_CTX,
                   "num_predict": config.AGENT_NUM_PREDICT}

        for tour in range(1, self.max_tours + 1):
            trace.tours = tour
            # Au dernier tour, les outils sont retirés : le modèle doit conclure.
            outils = TOOLS if tour < self.max_tours else None
            reponse = self.client.chat(self.model, messages, tools=outils, options=options)
            self._cumuler_jetons(trace, reponse)

            if not reponse.tool_calls:
                trace.reponse = reponse.content.strip()
                break

            messages.append({"role": "assistant", "content": reponse.content,
                             "tool_calls": reponse.tool_calls})
            for appel in reponse.tool_calls:
                fonction = appel.get("function", {})
                nom = fonction.get("name", "")
                entree = fonction.get("arguments") or {}
                if isinstance(entree, str):
                    try:
                        entree = json.loads(entree)
                    except json.JSONDecodeError:
                        entree = {"requete": entree}
                contenu, erreur = self._executer_outil(nom, entree, trace)
                trace.appels_outils.append({"outil": nom, "entree": entree, "erreur": erreur})
                messages.append({"role": "tool", "name": nom, "content": contenu})
        else:
            trace.erreur = f"nombre maximal de tours atteint ({self.max_tours})"

        trace.citations = [(d, s.strip()) for d, s in _CITATION.findall(trace.reponse)]
        trace.latence_s = round(time.perf_counter() - debut, 2)
        return trace

    def _cumuler_jetons(self, trace: Trace, reponse) -> None:
        trace.jetons["entree"] = trace.jetons.get("entree", 0) + reponse.jetons_entree
        trace.jetons["sortie"] = trace.jetons.get("sortie", 0) + reponse.jetons_sortie


def repondre(question: str, mode_json: bool = False) -> str:
    """Point d'entrée de la commande `ask`."""
    agent = Agent()
    trace = agent.answer(question)
    if mode_json:
        return json.dumps(trace.to_dict(), ensure_ascii=False, indent=2)
    return (f"{trace.reponse}\n\n({trace.tours} tours, {len(trace.appels_outils)} appels d'outils, "
            f"{trace.latence_s} s, modèle local {agent.model}, coût 0 $)")
