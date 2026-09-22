"""Construction du client Anthropic (clé et espace de travail lus dans l'environnement ou .env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from re2020.config import ROOT


def client_anthropic():
    """Client Anthropic prêt à l'emploi.

    Une clé non rattachée à un espace de travail impose l'en-tête `anthropic-workspace-id` : il est
    ajouté quand ANTHROPIC_WORKSPACE_ID est défini, sinon l'API répond 400 sur chaque appel.
    """
    import anthropic

    load_dotenv(ROOT / ".env")
    espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    entetes = {"anthropic-workspace-id": espace} if espace else None
    return anthropic.Anthropic(default_headers=entetes)
