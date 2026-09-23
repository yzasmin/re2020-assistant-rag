"""Client minimal pour l'API locale d'Ollama (aucune clé, aucun appel sortant)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

from re2020.config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, KEEP_ALIVE, ROOT


class OllamaError(RuntimeError):
    """Erreur réseau ou réponse inattendue du serveur Ollama."""


@dataclass
class ChatResponse:
    """Réponse de /api/chat, réduite à ce dont l'agent a besoin."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    duree_s: float = 0.0
    jetons_entree: int = 0
    jetons_sortie: int = 0
    brut: dict = field(default_factory=dict)


def _hote() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def modele_par_defaut() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


class OllamaClient:
    """Appels à /api/chat, /api/tags et /api/ps."""

    def __init__(self, hote: str | None = None, timeout: float = 900.0):
        self.hote = (hote or _hote()).rstrip("/")
        self.timeout = timeout

    def _post(self, chemin: str, charge: dict) -> dict:
        try:
            reponse = requests.post(f"{self.hote}{chemin}", json=charge, timeout=self.timeout)
            reponse.raise_for_status()
            return reponse.json()
        except requests.RequestException as exc:
            raise OllamaError(f"appel {chemin} impossible sur {self.hote} : {exc}") from exc

    def _get(self, chemin: str) -> dict:
        try:
            reponse = requests.get(f"{self.hote}{chemin}", timeout=30)
            reponse.raise_for_status()
            return reponse.json()
        except requests.RequestException as exc:
            raise OllamaError(f"appel {chemin} impossible sur {self.hote} : {exc}") from exc

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             format: dict | None = None, options: dict | None = None) -> ChatResponse:
        charge: dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                  "keep_alive": KEEP_ALIVE,
                                  "options": options or {"temperature": 0}}
        if tools:
            charge["tools"] = tools
        if format:
            charge["format"] = format
        donnees = self._post("/api/chat", charge)
        message = donnees.get("message") or {}
        return ChatResponse(
            content=message.get("content", "") or "",
            tool_calls=list(message.get("tool_calls") or []),
            duree_s=round(donnees.get("total_duration", 0) / 1e9, 2),
            jetons_entree=donnees.get("prompt_eval_count", 0) or 0,
            jetons_sortie=donnees.get("eval_count", 0) or 0,
            brut=donnees,
        )

    def modeles(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def disponible(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except OllamaError:
            return False

    def empreinte_memoire(self) -> dict:
        """Mémoire occupée par les modèles chargés, d'après /api/ps."""
        charges = self._get("/api/ps").get("models", [])
        return {
            "modeles_charges": [m.get("name") for m in charges],
            "octets_resident": sum(m.get("size", 0) for m in charges),
            "octets_sur_gpu": sum(m.get("size_vram", 0) for m in charges),
        }

    def details_modele(self, model: str) -> dict:
        """Famille, nombre de paramètres et quantification, pour documenter les mesures."""
        donnees = self._post("/api/show", {"model": model})
        details = donnees.get("details", {})
        return {
            "modele": model,
            "famille": details.get("family"),
            "parametres": details.get("parameter_size"),
            "quantification": details.get("quantization_level"),
        }


def json_depuis_texte(texte: str) -> dict:
    """Analyse une réponse JSON, en tolérant un bloc de code autour."""
    brut = texte.strip()
    if brut.startswith("```"):
        brut = brut.split("```")[1]
        brut = brut[4:] if brut.lower().startswith("json") else brut
    debut, fin = brut.find("{"), brut.rfind("}")
    if debut < 0 or fin <= debut:
        raise OllamaError(f"réponse sans objet JSON : {texte[:200]}")
    return json.loads(brut[debut:fin + 1])
