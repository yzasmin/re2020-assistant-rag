"""Normalisation du texte français pour l'indexation lexicale (BM25)."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

STOPWORDS = frozenset(
    """a au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur leurs lui ma mais me
    meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une
    vos votre vous c d j l m n s t y ete etre est sont sera seront a ont avait etait fait faut peut doit
    ci ca cela celui celle ceux celles dont ainsi aussi alors si sans sous entre vers chez tout tous toute toutes
    plus moins tres quel quelle quels quelles quoi comment combien quand est-ce lorsque lorsqu puis selon
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_,.][a-z0-9]+)*")
_STEMMER = snowballstemmer.stemmer("french")


def normalize(text: str) -> str:
    """NFKC (lettres mathématiques des PDF vers ASCII), minuscules, sans accents, espaces réduits."""
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("œ", "oe")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    if any(ch.isdigit() for ch in token) or "_" in token:
        return token  # identifiants réglementaires (Bbio_max, R.172-4, 2025) gardés tels quels
    return _STEMMER.stemWord(token)


def tokenize(text: str) -> list[str]:
    """Jetons BM25 : normalisation, élision (l', d'), mots vides retirés, racinisation Snowball."""
    norm = normalize(text).replace("'", " ")
    tokens = []
    for tok in _TOKEN_RE.findall(norm):
        tok = tok.strip(".,")
        if len(tok) < 2 and not tok.isdigit():
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(_stem(tok))
    return tokens
