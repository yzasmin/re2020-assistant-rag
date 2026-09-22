"""Jeu d'évaluation et métriques de récupération."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from re2020.chunking import Chunk
from re2020.config import QUESTIONS_PATH
from re2020.text import normalize


@dataclass
class Question:
    id: str
    question: str
    type: str
    reponse_reference: str
    sources: list[dict] = field(default_factory=list)
    profil: str = ""
    hors_perimetre: bool = False


def load_questions(path=QUESTIONS_PATH) -> list[Question]:
    with open(path, encoding="utf-8") as fh:
        return [Question(**json.loads(line)) for line in fh if line.strip()]


def gold_chunk_ids(question: Question, chunks: list[Chunk]) -> list[list[str]]:
    """Pour chaque source attendue, la liste des passages qui contiennent la phrase de référence.

    Les passages sont repérés par leur contenu, pas par un identifiant figé : le jeu de questions
    reste valable si le découpage change.
    """
    groupes = []
    for source in question.sources:
        besoin = normalize(source["contient"])
        groupes.append([
            c.chunk_id for c in chunks
            if c.doc_id == source["doc"] and besoin in normalize(c.text)
        ])
    return groupes


def recall_at_k(groupes: list[list[str]], classement: list[str], k: int) -> float:
    """Part des sources attendues dont au moins un passage figure dans les k premiers résultats."""
    if not groupes:
        return float("nan")
    tete = set(classement[:k])
    trouves = sum(1 for g in groupes if tete & set(g))
    return trouves / len(groupes)


def reciprocal_rank(groupes: list[list[str]], classement: list[str]) -> float:
    """1 / rang du premier passage attendu, 0 si aucun n'est remonté."""
    attendus = {cid for g in groupes for cid in g}
    for rang, cid in enumerate(classement, start=1):
        if cid in attendus:
            return 1.0 / rang
    return 0.0


def ndcg_at_k(groupes: list[list[str]], classement: list[str], k: int = 10) -> float:
    """nDCG binaire : un passage attendu compte 1, les autres 0, une seule fois par source."""
    if not groupes:
        return float("nan")
    restants = [set(g) for g in groupes]
    dcg = 0.0
    for rang, cid in enumerate(classement[:k], start=1):
        for groupe in restants:
            if cid in groupe:
                dcg += 1.0 / math.log2(rang + 1)
                groupe.clear()
                break
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(groupes), k) + 1))
    return dcg / ideal if ideal else float("nan")
