"""Agent Claude outillé : recherche hybride, calcul sûr, réponses citées ou refus."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from re2020 import config
from re2020.calculator import CalculError, calculate
from re2020.retrieval import Hit, Retriever

MARQUEUR_REFUS = "Hors périmètre"

SYSTEM_PROMPT = f"""Tu es un assistant spécialisé dans la réglementation environnementale française RE2020.
Tu réponds à des professionnels du bâtiment et à des particuliers, en français, de façon brève et précise.

Règles impératives :
1. Tu ne réponds qu'à partir des passages renvoyés par l'outil rechercher_reglementation. Tu n'utilises
   jamais tes connaissances générales pour affirmer un chiffre, une date ou une obligation.
2. Chaque affirmation tirée des textes est suivie de sa source au format [identifiant_du_document, section],
   exactement tel qu'il apparaît en tête du passage utilisé.
3. Si les passages trouvés ne couvrent pas la question, ou ne permettent pas de répondre avec certitude,
   tu commences ta réponse par « {MARQUEUR_REFUS} : » puis tu expliques en une phrase ce qui manque et vers
   quel type de source se tourner. Tu ne proposes aucune réponse approximative.
4. Tu lances au moins une recherche avant de répondre, et tu peux en lancer plusieurs avec des formulations
   différentes si la première ne suffit pas.
5. Pour tout calcul arithmétique, tu utilises l'outil calculer plutôt que de calculer de tête, et tu
   rappelles la formule et les valeurs réglementaires utilisées, avec leurs sources.
6. Les seuils RE2020 dépendent de l'usage du bâtiment, de l'année de dépôt du permis de construire, de la
   zone climatique et de coefficients de modulation : précise toujours ces conditions quand elles existent.
"""

TOOLS = [
    {
        "name": "rechercher_reglementation",
        "description": (
            "Recherche des passages dans les textes officiels RE2020 indexés (arrêté du 4 août 2021 "
            "consolidé et ses annexes, annexe de l'article R. 172-4 du code de la construction et de "
            "l'habitation, guide RE2020 du ministère). Renvoie des passages numérotés avec leur source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requete": {
                    "type": "string",
                    "description": ("Question ou mots-clés en français, par exemple "
                                    "« seuil Icconstruction logements collectifs 2025 »"),
                },
                "nombre_resultats": {
                    "type": "integer",
                    "description": "Nombre de passages souhaités, entre 1 et 10 (5 par défaut)",
                },
            },
            "required": ["requete"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculer",
        "description": (
            "Évalue une expression arithmétique (opérateurs + - * / ** et fonctions min, max, abs, round, "
            "sqrt). La virgule décimale française est acceptée. Aucun accès au système, aucune variable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Par exemple : 63 * 1,15 - 3,6"},
            },
            "required": ["expression"],
            "additionalProperties": False,
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
    usage: dict = field(default_factory=dict)
    cout_usd: float = 0.0
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


def cout_usd(model: str, usage: dict) -> float:
    """Coût d'un appel, en dollars, à partir des prix publics par million de jetons."""
    entree, sortie = config.PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
    ecriture = usage.get("cache_creation_input_tokens", 0) * entree * config.CACHE_WRITE_FACTOR
    lecture = usage.get("cache_read_input_tokens", 0) * entree * config.CACHE_READ_FACTOR
    brut = usage.get("input_tokens", 0) * entree + usage.get("output_tokens", 0) * sortie
    return (brut + ecriture + lecture) / 1_000_000


def formater_passages(hits: list[Hit]) -> str:
    blocs = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocs.append(f"Passage {i} {c.citation()} (source : {c.url})\n{c.text}")
    return "\n\n".join(blocs) if blocs else "Aucun passage trouvé pour cette requête."


class Agent:
    """Boucle d'appel d'outils sur l'API Messages d'Anthropic."""

    def __init__(self, retriever: Retriever | None = None, client=None,
                 model: str = config.AGENT_MODEL, max_tours: int = 6, k: int = 5, mode: str = "hybrid"):
        self.retriever = retriever if retriever is not None else Retriever()
        self.model = model
        self.max_tours = max_tours
        self.k = k
        self.mode = mode
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from re2020.api import client_anthropic
            self._client = client_anthropic()
        return self._client

    def _executer_outil(self, nom: str, entree: dict, trace: Trace) -> tuple[str, bool]:
        if nom == "rechercher_reglementation":
            k = int(entree.get("nombre_resultats") or self.k)
            hits = self.retriever.search(str(entree["requete"]), max(1, min(k, 10)), mode=self.mode)
            trace.passages_vus.extend(h.chunk.chunk_id for h in hits)
            return formater_passages(hits), False
        if nom == "calculer":
            try:
                return f"{calculate(str(entree['expression'])):.10g}", False
            except CalculError as exc:
                return f"Calcul refusé : {exc}", True
        return f"Outil inconnu : {nom}", True

    def answer(self, question: str) -> Trace:
        trace = Trace(question=question)
        debut = time.perf_counter()
        messages = [{"role": "user", "content": question}]
        # Le préfixe stable (consigne + outils) est marqué pour le cache. Sur claude-haiku-4-5 le préfixe
        # minimal cachable est de 4096 jetons : la mise en cache ne s'active que si la consigne grossit,
        # ce que confirme usage.cache_creation_input_tokens dans les traces.
        systeme = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

        for tour in range(1, self.max_tours + 1):
            trace.tours = tour
            reponse = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=systeme,
                tools=TOOLS,
                messages=messages,
            )
            self._cumuler_usage(trace, reponse)

            if reponse.stop_reason == "refusal":
                trace.erreur = "refus de sécurité du modèle"
                break

            blocs_outils = [b for b in reponse.content if b.type == "tool_use"]
            if not blocs_outils:
                trace.reponse = "".join(b.text for b in reponse.content if b.type == "text").strip()
                break

            messages.append({"role": "assistant", "content": reponse.content})
            resultats = []
            for bloc in blocs_outils:
                contenu, erreur = self._executer_outil(bloc.name, dict(bloc.input), trace)
                trace.appels_outils.append({"outil": bloc.name, "entree": dict(bloc.input), "erreur": erreur})
                resultats.append({
                    "type": "tool_result",
                    "tool_use_id": bloc.id,
                    "content": contenu,
                    **({"is_error": True} if erreur else {}),
                })
            messages.append({"role": "user", "content": resultats})
        else:
            trace.erreur = f"nombre maximal de tours atteint ({self.max_tours})"

        trace.citations = [(d, s.strip()) for d, s in _CITATION.findall(trace.reponse)]
        trace.latence_s = round(time.perf_counter() - debut, 2)
        return trace

    def _cumuler_usage(self, trace: Trace, reponse) -> None:
        usage = reponse.usage.model_dump() if hasattr(reponse.usage, "model_dump") else dict(reponse.usage)
        champs = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        courant = {c: (usage.get(c) or 0) for c in champs}
        for champ, valeur in courant.items():
            trace.usage[champ] = trace.usage.get(champ, 0) + valeur
        trace.cout_usd = round(trace.cout_usd + cout_usd(self.model, courant), 6)


def repondre(question: str, mode_json: bool = False) -> str:
    """Point d'entrée de la commande `ask`."""
    agent = Agent()
    trace = agent.answer(question)
    if mode_json:
        return json.dumps(trace.to_dict(), ensure_ascii=False, indent=2)
    lignes = [trace.reponse, ""]
    lignes.append(f"({trace.tours} tours, {len(trace.appels_outils)} appels d'outils, "
                  f"{trace.latence_s} s, {trace.cout_usd:.4f} $)")
    return "\n".join(lignes)
