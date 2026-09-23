"""Juge local : fidélité des réponses aux passages cités et exactitude face à la réponse de référence.

Le juge tourne sur le même modèle local que l'agent. C'est une limite assumée : un petit modèle qui
s'évalue lui-même est moins fiable qu'un juge plus capable, et les chiffres de fidélité publiés doivent
être lus avec cette réserve. Les métriques qui ne dépendent pas du juge (citations valides, refus
corrects) servent de garde-fou.
"""

from __future__ import annotations

from re2020 import config
from re2020.ollama_client import OllamaClient, json_depuis_texte, modele_par_defaut

SCHEMA = {
    "type": "object",
    "properties": {
        "affirmations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "affirmation": {"type": "string"},
                    "soutenue": {"type": "boolean"},
                },
                "required": ["affirmation", "soutenue"],
            },
        },
        "exactitude": {"type": "string", "enum": ["correcte", "partielle", "incorrecte"]},
        "commentaire": {"type": "string"},
    },
    "required": ["affirmations", "exactitude", "commentaire"],
}

CONSIGNE = """Tu évalues la réponse d'un assistant réglementaire RE2020. Tu réponds uniquement en JSON.

1. Découpe la réponse évaluée en affirmations vérifiables (chiffres, seuils, dates, obligations).
   Ignore les formules de politesse et les reformulations de la question.
2. Pour chaque affirmation, mets soutenue à true si un des passages fournis l'établit, false sinon.
   Une affirmation plausible mais absente des passages n'est pas soutenue.
3. Donne ensuite exactitude : « correcte », « partielle » ou « incorrecte » par rapport à la réponse de
   référence. Une différence de formulation n'est pas une erreur ; un chiffre faux ou manquant en est une.
Tu ne t'appuies que sur les passages et la référence fournis."""


class Judge:
    def __init__(self, client: OllamaClient | None = None, model: str | None = None):
        self.model = model or modele_par_defaut()
        self._client = client
        self.echecs = 0

    @property
    def client(self) -> OllamaClient:
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    def evaluer(self, question: str, reponse: str, passages: list[str], reference: str) -> dict:
        contenu = (
            f"Question posée :\n{question}\n\n"
            f"Réponse de référence :\n{reference}\n\n"
            "Passages fournis à l'assistant :\n" + "\n\n".join(passages) + "\n\n"
            f"Réponse évaluée :\n{reponse}"
        )
        sortie = self.client.chat(
            self.model,
            [{"role": "system", "content": CONSIGNE}, {"role": "user", "content": contenu}],
            format=SCHEMA,
            options={"temperature": 0, "num_ctx": config.AGENT_NUM_CTX,
                     "num_predict": config.JUDGE_NUM_PREDICT},
        )
        verdict = json_depuis_texte(sortie.content)
        affirmations = [a for a in verdict.get("affirmations", []) if isinstance(a, dict)]
        soutenues = sum(1 for a in affirmations if a.get("soutenue"))
        verdict["affirmations"] = affirmations
        verdict["fidelite"] = soutenues / len(affirmations) if affirmations else None
        verdict["latence_s"] = sortie.duree_s
        if verdict.get("exactitude") not in {"correcte", "partielle", "incorrecte"}:
            verdict["exactitude"] = None
        return verdict
