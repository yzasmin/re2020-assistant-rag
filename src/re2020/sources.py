"""Documents officiels indexés et leur téléchargement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date

import requests

from re2020.config import MANIFEST_PATH, RAW_DIR

RTRE = "https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf"
LICENCE_ETALAB = "Licence Ouverte / Etalab 2.0"
LICENCE_TEXTE_OFFICIEL = (
    "Texte officiel (acte réglementaire publié au JORF), librement réutilisable ; "
    "version consolidée reproduite par AIDA (Ineris)"
)


@dataclass(frozen=True)
class Source:
    doc_id: str
    titre: str
    url: str
    fichier: str
    licence: str
    kind: str  # "pdf" ou "html"
    reference: str = ""  # URL de référence juridique quand la copie vient d'un miroir


SOURCES: tuple[Source, ...] = (
    Source(
        "arrete-2021",
        "Arrêté du 4 août 2021 (version consolidée)",
        "https://aida.ineris.fr/reglementation/arrete-040821-relatif-exigences-performance-energetique-environnementale",
        "arrete_4_aout_2021_consolide.html",
        LICENCE_TEXTE_OFFICIEL,
        "html",
        "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000043936431",
    ),
    Source(
        "cch-annexe-r172-4",
        "Annexe à l'article R. 172-4 du CCH, chapitres I à III (version au 1er juillet 2026)",
        f"{RTRE}/chapitre1a3_annexe_r172-4_post-rivaton-i.pdf",
        "chapitre1a3_annexe_r172-4_post-rivaton-i.pdf",
        LICENCE_ETALAB,
        "pdf",
    ),
    Source("arrete-annexe-1", "Arrêté du 4 août 2021, annexe I (définitions)",
           f"{RTRE}/annexei_arrete_4_aout_2021.pdf", "annexei_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-2", "Arrêté du 4 août 2021, annexe II (règles générales de calcul)",
           f"{RTRE}/annexeii_arrete_4_aout_2021.pdf", "annexeii_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-5", "Arrêté du 4 août 2021, annexe V (autocontrôle et approbation)",
           f"{RTRE}/annexev_arrete_4_aout_2021.pdf", "annexev_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-6", "Arrêté du 4 août 2021, annexe VI (contenu du RSEE)",
           f"{RTRE}/annexevi_arrete_4_aout_2021.pdf", "annexevi_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-7", "Arrêté du 4 août 2021, annexe VII (étanchéité à l'air)",
           f"{RTRE}/annexevii_arrete_4_aout_2021.pdf", "annexevii_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-8", "Arrêté du 4 août 2021, annexe VIII (vérification de la ventilation)",
           f"{RTRE}/annexeviii_arrete_4_aout_2021.pdf", "annexeviii_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-9", "Arrêté du 4 août 2021, annexe IX (applications simplifiées)",
           f"{RTRE}/annexeix_arrete_4_aout_2021.pdf", "annexeix_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-10", "Arrêté du 4 août 2021, annexe X (cas particuliers)",
           f"{RTRE}/annexex_arrete_4_aout_2021.pdf", "annexex_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source("arrete-annexe-11", "Arrêté du 4 août 2021, annexe XI (performances forfaitaires)",
           f"{RTRE}/annexexi_arrete_4_aout_2021.pdf", "annexexi_arrete_4_aout_2021.pdf", LICENCE_ETALAB, "pdf"),
    Source(
        "guide-re2020",
        "Guide RE 2020 du ministère (Cerema, janvier 2024)",
        "https://www.ecologie.gouv.fr/sites/default/files/documents/guide_re2020_version_janvier_2024.pdf",
        "guide_re2020_version_janvier_2024.pdf",
        "Publication administrative, réutilisation libre (CRPA art. L. 321-1)",
        "pdf",
    ),
)

SOURCES_BY_ID = {s.doc_id: s for s in SOURCES}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_all(force: bool = False) -> list[dict]:
    """Télécharge chaque source dans data/raw et écrit un manifeste (taille, empreinte, date)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in SOURCES:
        path = RAW_DIR / src.fichier
        if force or not path.exists():
            resp = requests.get(src.url, timeout=120, headers={"User-Agent": "re2020-assistant-rag/0.1"})
            resp.raise_for_status()
            path.write_bytes(resp.content)
            print(f"téléchargé  {src.fichier} ({len(resp.content)} octets)")
        else:
            print(f"déjà là     {src.fichier}")
        data = path.read_bytes()
        manifest.append({**asdict(src), "octets": len(data), "sha256": _sha256(data)})
    MANIFEST_PATH.write_text(
        json.dumps({"date_telechargement": date.today().isoformat(), "documents": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
