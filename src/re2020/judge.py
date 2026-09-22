"""Juge LLM : fidélité des réponses aux passages cités et exactitude face à la réponse de référence."""

from __future__ import annotations

import json

from re2020 import config
from re2020.agent import cout_usd

SCHEMA = {
    "type": "object",
    "properties": {
        "affirmations": {
            "type": "array",
            "description": "Une entrée par affirmation vérifiable de la réponse évaluée",
            "items": {
                "type": "object",
                "properties": {
                    "affirmation": {"type": "string"},
                    "soutenue": {"type": "boolean", "description": "Vrai si un passage fourni l'établit"},
                    "justification": {"type": "string"},
                },
                "required": ["affirmation", "soutenue", "justification"],
                "additionalProperties": False,
            },
        },
        "exactitude": {
            "type": "string",
            "enum": ["correcte", "partielle", "incorrecte"],
            "description": "Comparaison du fond de la réponse avec la réponse de référence",
        },
        "commentaire": {"type": "string"},
    },
    "required": ["affirmations", "exactitude", "commentaire"],
    "additionalProperties": False,
}

CONSIGNE = """Tu évalues la réponse d'un assistant réglementaire RE2020.

Travail demandé :
1. Découpe la réponse évaluée en affirmations vérifiables (chiffres, seuils, dates, obligations).
   Ignore les formules de politesse et les reformulations de la question.
2. Pour chaque affirmation, dis si elle est soutenue par au moins un des passages fournis. Une affirmation
   plausible mais absente des passages n'est pas soutenue.
3. Compare ensuite le fond de la réponse à la réponse de référence : correcte, partielle ou incorrecte.
   Une différence de formulation n'est pas une erreur ; un chiffre faux ou manquant en est une.
Tu ne t'appuies que sur les passages et la référence fournis."""


class Judge:
    def __init__(self, client=None, model: str = config.JUDGE_MODEL):
        self.model = model
        self._client = client
        self.cout_total_usd = 0.0

    @property
    def client(self):
        if self._client is None:
            from re2020.api import client_anthropic
            self._client = client_anthropic()
        return self._client

    def evaluer(self, question: str, reponse: str, passages: list[str], reference: str) -> dict:
        contenu = (
            f"Question posée :\n{question}\n\n"
            f"Réponse de référence :\n{reference}\n\n"
            f"Passages fournis à l'assistant :\n" + "\n\n".join(passages) + "\n\n"
            f"Réponse évaluée :\n{reponse}"
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=CONSIGNE,
            messages=[{"role": "user", "content": contenu}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        usage = message.usage.model_dump() if hasattr(message.usage, "model_dump") else dict(message.usage)
        self.cout_total_usd += cout_usd(self.model, usage)
        texte = "".join(b.text for b in message.content if b.type == "text")
        verdict = json.loads(texte)
        affirmations = verdict["affirmations"]
        soutenues = sum(1 for a in affirmations if a["soutenue"])
        verdict["fidelite"] = soutenues / len(affirmations) if affirmations else None
        return verdict
