"""Mesure de la qualité de récupération : BM25 seul, vectoriel seul, hybride."""

from __future__ import annotations

import json
import statistics
import time
from datetime import date

from re2020 import config
from re2020.chunking import load_chunks
from re2020.evaluation import gold_chunk_ids, load_questions, ndcg_at_k, recall_at_k, reciprocal_rank
from re2020.retrieval import Retriever

MODES = ("bm25", "vector", "hybrid")
KS = (1, 3, 5, 10)
LIBELLES = {"bm25": "BM25 seul", "vector": "Vectoriel seul", "hybrid": "Hybride (RRF)"}


def run() -> dict:
    chunks = load_chunks()
    questions = [q for q in load_questions() if not q.hors_perimetre]
    retriever = Retriever(chunks)
    resultats: dict[str, dict] = {}
    detail: dict[str, dict] = {}

    for mode in MODES:
        mesures = {f"recall@{k}": [] for k in KS}
        mesures["mrr"], mesures["ndcg@10"] = [], []
        latences = []
        detail[mode] = {}
        for q in questions:
            groupes = gold_chunk_ids(q, chunks)
            debut = time.perf_counter()
            classement = [cid for cid, _ in retriever.ranked_ids(q.question, mode, max(KS))]
            latences.append((time.perf_counter() - debut) * 1000)
            for k in KS:
                mesures[f"recall@{k}"].append(recall_at_k(groupes, classement, k))
            mesures["mrr"].append(reciprocal_rank(groupes, classement))
            mesures["ndcg@10"].append(ndcg_at_k(groupes, classement, 10))
            detail[mode][q.id] = {
                "rang_premier_passage_attendu": next(
                    (r for r, cid in enumerate(classement, 1) if any(cid in g for g in groupes)), None),
                "top3": classement[:3],
            }
        resultats[mode] = {nom: round(statistics.fmean(vals), 4) for nom, vals in mesures.items()}
        resultats[mode]["latence_ms_mediane"] = round(statistics.median(latences), 1)
        print(f"{LIBELLES[mode]:16s} " + "  ".join(f"{n}={v}" for n, v in resultats[mode].items()))

    sortie = {
        "date": date.today().isoformat(),
        "passages_indexes": len(chunks),
        "questions_evaluees": len(questions),
        "questions_hors_perimetre": len(load_questions()) - len(questions),
        "modele_embeddings": config.EMBED_MODEL,
        "fusion": f"Reciprocal Rank Fusion, k={config.RRF_K}, {config.CANDIDATES_PER_RETRIEVER} candidats par moteur",
        "metriques": resultats,
        "detail": detail,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "retrieval_metrics.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")
    return sortie


def graphique(metriques: dict, chemin) -> None:
    """Graphique comparatif en thème sombre (1600x900)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fond, texte, accent = "#0D0F12", "#ECE8DF", "#FF5C8F"
    noms = [f"recall@{k}" for k in KS] + ["mrr", "ndcg@10"]
    etiquettes = [f"Rappel@{k}" for k in KS] + ["MRR", "nDCG@10"]
    couleurs = {"bm25": "#6E7B8B", "vector": "#3FD1BE", "hybrid": accent}

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(fond)
    ax.set_facecolor(fond)
    largeur = 0.26
    for i, mode in enumerate(MODES):
        valeurs = [metriques[mode][n] for n in noms]
        x = [j + (i - 1) * largeur for j in range(len(noms))]
        barres = ax.bar(x, valeurs, largeur, label=LIBELLES[mode], color=couleurs[mode])
        for rect, val in zip(barres, valeurs, strict=True):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.015, f"{val:.2f}".replace(".", ","),
                    ha="center", color=texte, fontsize=13)
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels(etiquettes, color=texte, fontsize=17)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"], color=texte, fontsize=15)
    ax.set_title("Récupération RE2020 : BM25, vectoriel et fusion hybride", color=texte, fontsize=26, pad=28)
    ax.tick_params(colors=texte, length=0)
    for cote in ax.spines.values():
        cote.set_visible(False)
    ax.grid(axis="y", color="#2A2F36", linewidth=1)
    ax.set_axisbelow(True)
    legende = ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=17, bbox_to_anchor=(0.5, -0.07))
    for t in legende.get_texts():
        t.set_color(texte)
    fig.tight_layout()
    fig.savefig(chemin, facecolor=fond)
    plt.close(fig)


def main() -> None:
    sortie = run()
    graphique(sortie["metriques"], config.RESULTS_DIR / "retrieval_comparison.png")
    print(f"écrit : {config.RESULTS_DIR / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
