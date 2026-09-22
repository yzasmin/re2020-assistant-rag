"""Extraction du texte et découpage en passages citables (document, section, page, URL)."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import asdict, dataclass

import pymupdf
from bs4 import BeautifulSoup

from re2020.config import CHUNK_MAX_CHARS, CHUNK_TARGET_CHARS, CHUNKS_PATH, RAW_DIR
from re2020.sources import SOURCES, Source


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_titre: str
    section: str
    page_debut: int | None
    page_fin: int | None
    url: str
    text: str

    def citation(self) -> str:
        page = ""
        if self.page_debut:
            page = f", p. {self.page_debut}" if self.page_debut == self.page_fin else f", p. {self.page_debut}-{self.page_fin}"
        return f"[{self.doc_id}, {self.section}{page}]"


# Titres de section reconnus dans les PDF.
_NUM_ALONE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?$")
_NUM_TITLE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+([A-ZÉÈÀÂÎÔÛÇ«].{10,110})$")
_ENDS_WITH_PAGE = re.compile(r"\s\d{1,3}$")
_CHAPITRE = re.compile(r"^(Chapitre\s+[IVXL]+(?:er)?\b.{0,110}|Titre\s+[IVXL]+(?:er)?\b.{0,110}|Annexe\s+[IVXL]+\b.{0,110})$")
_BULLET_HEAD = re.compile(r"^[\uf000-\uf8ff]?\s*((?:Coefficients de modulation de l|Valeurs de ).{5,140})$")
_TOC_LINE = re.compile(r"\.{5,}")
_PAGE_NUM = re.compile(r"^\d{1,3}$")
_NOISE = {"Guide RE 2020"}


def _clean_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    return re.sub(r"\s+", " ", line).strip()


def _pdf_lines(path) -> list[tuple[int, str]]:
    doc = pymupdf.open(path)
    out: list[tuple[int, str]] = []
    for pno, page in enumerate(doc, start=1):
        first = True
        for raw in page.get_text().split("\n"):
            line = _clean_line(raw)
            if not line or line in _NOISE or _TOC_LINE.search(line):
                continue
            if first and _PAGE_NUM.match(line):
                first = False
                continue  # numéro de page imprimé en tête de page
            first = False
            out.append((pno, line))
    doc.close()
    return out


def _heading(lines: list[tuple[int, str]], i: int) -> tuple[str | None, int]:
    """Renvoie (titre, nombre de lignes consommées) si la ligne i ouvre une section."""
    _, line = lines[i]
    if m := _CHAPITRE.match(line):
        return m.group(1).rstrip(" :."), 1
    if (m := _NUM_TITLE.match(line)) and not _ENDS_WITH_PAGE.search(line):
        return f"{m.group(1)} {m.group(2)}".rstrip(" :."), 1
    if (m := _BULLET_HEAD.match(line)) and "prend" not in line:
        return m.group(1).rstrip(" :."), 1
    if (m := _NUM_ALONE.match(line)) and i + 1 < len(lines):
        nxt = lines[i + 1][1]
        dotted = "." in m.group(1)
        majuscules = 12 <= len(nxt) <= 110 and nxt.upper() == nxt and " " in nxt and "," not in nxt
        long_enough = 15 <= len(nxt) <= 110 if dotted else majuscules
        if long_enough and nxt[:1].isupper() and not nxt.endswith(";") and not _ENDS_WITH_PAGE.search(nxt):
            return f"{m.group(1)} {nxt}".rstrip(" :."), 2
    return None, 0


class _Builder:
    def __init__(self, src: Source):
        self.src = src
        self.chunks: list[Chunk] = []
        self.buf: list[str] = []
        self.pages: list[int] = []
        self.section = "Préambule"

    def flush(self) -> None:
        text = " ".join(self.buf).strip()
        if len(text) >= 40:
            pages = [p for p in self.pages if p is not None]
            self.chunks.append(Chunk(
                chunk_id=f"{self.src.doc_id}#{len(self.chunks):04d}",
                doc_id=self.src.doc_id,
                doc_titre=self.src.titre,
                section=self.section,
                page_debut=min(pages) if pages else None,
                page_fin=max(pages) if pages else None,
                url=self.src.reference or self.src.url,
                text=text,
            ))
        self.buf, self.pages = [], []

    def add(self, line: str, page: int | None) -> None:
        size = sum(len(x) + 1 for x in self.buf)
        if size >= CHUNK_MAX_CHARS or (size >= CHUNK_TARGET_CHARS and self.buf and self.buf[-1].endswith((".", ";", ":"))):
            self.flush()
        self.buf.append(line)
        self.pages.append(page)

    def new_section(self, title: str) -> None:
        self.flush()
        self.section = title[:140]
        self.buf.append(title)
        self.pages.append(None)


def chunk_pdf(src: Source) -> list[Chunk]:
    lines = _pdf_lines(RAW_DIR / src.fichier)
    b = _Builder(src)
    i = 0
    while i < len(lines):
        title, used = _heading(lines, i)
        if title:
            b.new_section(title)
            b.pages[-1] = lines[i][0]
            i += used
            continue
        b.add(lines[i][1], lines[i][0])
        i += 1
    b.flush()
    return b.chunks


_ARTICLE = re.compile(r"(Article \d+(?:-\d+)?(?:er)? de l'arrêté du 4 août 2021)")
_TITRE_CHAP = re.compile(r"((?:Titre|Chapitre) [IVXL]+(?:er)? : [^:]{3,160}?)(?= (?:Chapitre|Article|Titre) )")


def _html_text(path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = html.unescape(soup.get_text(" "))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    start = text.find("Arrêtent :")
    end = text.find("Annexes (Arrêté du")
    if start < 0 or end < 0:
        raise ValueError(f"Structure inattendue dans {path.name}")
    return text[start + len("Arrêtent :"):end].strip()


def chunk_html_arrete(src: Source) -> list[Chunk]:
    body = _html_text(RAW_DIR / src.fichier)
    b = _Builder(src)
    parts = _ARTICLE.split(body)
    # parts = [avant, "Article 1er de ...", texte, "Article 2 de ...", texte, ...]
    b.section = "Arrêté du 4 août 2021"
    for i in range(1, len(parts), 2):
        num = parts[i].split(" de l'")[0]
        texte = _TITRE_CHAP.sub("", parts[i + 1]).strip()
        b.new_section(num)
        b.buf[-1] = f"{num} de l'arrêté du 4 août 2021."
        for sentence in re.split(r"(?<=[.;:])\s+", texte):
            b.add(sentence, None)
        b.flush()
    return b.chunks


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for src in SOURCES:
        chunks.extend(chunk_html_arrete(src) if src.kind == "html" else chunk_pdf(src))
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return chunks


def load_chunks() -> list[Chunk]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
