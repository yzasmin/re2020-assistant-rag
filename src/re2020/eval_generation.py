"""Évaluation de la génération : fidélité, exactitude, citations, refus, latence, mémoire.

Tout tourne en local sur Ollama : aucune clé, aucun appel sortant, coût nul. La contrepartie est la
latence, mesurée et publiée telle quelle.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import date

from re2020 import config
from re2020.agent import Agent, Trace
from re2020.chunking import load_chunks
from re2020.evaluation import Question, load_questions
from re2020.judge import Judge
from re2020.ollama_client import OllamaClient, OllamaError, modele_par_defaut
from re2020.retrieval import Retriever
from re2020.sources import SOURCES_BY_ID


def passages_cites(trace: Trace, par_id: dict) -> list[str]:
    """Textes des passages effectivement cités ; à défaut, tous les passages consultés."""
    vus = [par_id[cid] for cid in dict.fromkeys(trace.passages_vus) if cid in par_id]
    cites = [c for c in vus if any(d == c.doc_id and s[:30] in c.section for d, s in trace.citations)]
    retenus = cites or vus
    return [f"{c.citation()}\n{c.text}" for c in retenus]


def citations_valides(trace: Trace, par_id: dict) -> tuple[int, int]:
    """(citations pointant vers un document indexé et réellement consulté, total des citations)."""
    docs_vus = {par_id[cid].doc_id for cid in trace.passages_vus if cid in par_id}
    bonnes = sum(1 for doc, _ in trace.citations if doc in SOURCES_BY_ID and doc in docs_vus)
    return bonnes, len(trace.citations)


def refus_correct(question: Question, trace: Trace) -> bool:
    return trace.refus if question.hors_perimetre else not trace.refus


def run(limite: int | None = None) -> dict:
    chunks = load_chunks()
    par_id = {c.chunk_id: c for c in chunks}
    questions = load_questions()[:limite]
    client = OllamaClient()
    modele = modele_par_defaut()
    agent = Agent(Retriever(chunks), client=client, model=modele)
    juge = Judge(client=client, model=modele)

    traces, verdicts, memoires = [], [], []
    for i, q in enumerate(questions, start=1):
        trace = agent.answer(q.question)
        verdict = None
        if not trace.refus and trace.reponse:
            try:
                verdict = juge.evaluer(q.question, trace.reponse, passages_cites(trace, par_id),
                                       q.reponse_reference)
            except (OllamaError, ValueError) as exc:
                juge.echecs += 1
                verdict = {"erreur": str(exc)[:200], "fidelite": None, "exactitude": None}
        bonnes, total = citations_valides(trace, par_id)
        memoires.append(client.empreinte_memoire())
        traces.append({
            "id": q.id, "type": q.type, "hors_perimetre": q.hors_perimetre,
            **trace.to_dict(),
            "citations_valides": bonnes, "citations_total": total,
            "refus_correct": refus_correct(q, trace),
            "verdict_juge": verdict,
        })
        verdicts.append(verdict)
        print(f"[{i}/{len(questions)}] {q.id} refus={trace.refus} {trace.latence_s} s "
              f"citations={bonnes}/{total}", flush=True)

    fidelites = [v["fidelite"] for v in verdicts if v and v.get("fidelite") is not None]
    exactitudes = Counter(v["exactitude"] for v in verdicts if v and v.get("exactitude"))
    bonnes = sum(t["citations_valides"] for t in traces)
    total_citations = sum(t["citations_total"] for t in traces)
    hors = [t for t in traces if t["hors_perimetre"]]
    dans = [t for t in traces if not t["hors_perimetre"]]
    latences = sorted(t["latence_s"] for t in traces)
    jugees = len([v for v in verdicts if v and v.get("exactitude")])

    metriques = {
        "date": date.today().isoformat(),
        "modele": client.details_modele(modele),
        "juge": f"{modele} (même modèle local que l'agent, voir les limites)",
        "questions": len(questions),
        "fidelite_moyenne": round(statistics.fmean(fidelites), 4) if fidelites else None,
        "reponses_jugees": len(fidelites),
        "exactitude": {k: exactitudes.get(k, 0) for k in ("correcte", "partielle", "incorrecte")},
        "part_correcte_ou_partielle": round(
            (exactitudes.get("correcte", 0) + exactitudes.get("partielle", 0)) / jugees, 4) if jugees else None,
        "taux_citations_valides": round(bonnes / total_citations, 4) if total_citations else None,
        "citations_total": total_citations,
        "reponses_sans_citation": sum(1 for t in dans if t["citations_total"] == 0),
        "taux_refus_correct_hors_perimetre": round(
            sum(t["refus_correct"] for t in hors) / len(hors), 4) if hors else None,
        "taux_reponse_dans_perimetre": round(
            sum(t["refus_correct"] for t in dans) / len(dans), 4) if dans else None,
        "latence_mediane_s": round(statistics.median(latences), 1),
        "latence_moyenne_s": round(statistics.fmean(latences), 1),
        "latence_p90_s": round(latences[int(0.9 * (len(latences) - 1))], 1),
        "duree_totale_min": round(sum(latences) / 60, 1),
        "jetons_entree_moyens": round(statistics.fmean(t["jetons"].get("entree", 0) for t in traces)),
        "jetons_sortie_moyens": round(statistics.fmean(t["jetons"].get("sortie", 0) for t in traces)),
        "appels_outils_moyens": round(statistics.fmean(len(t["appels_outils"]) for t in traces), 2),
        "memoire_modele_max_mo": round(max((m["octets_resident"] for m in memoires), default=0) / 1e6, 1),
        "memoire_gpu_max_mo": round(max((m["octets_sur_gpu"] for m in memoires), default=0) / 1e6, 1),
        "verdicts_juge_illisibles": juge.echecs,
        "cout_usd": 0.0,
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "generation_metrics.json").write_text(
        json.dumps(metriques, ensure_ascii=False, indent=2), encoding="utf-8")
    with (config.RESULTS_DIR / "generation_traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    return metriques


def main(limite: int | None = None) -> None:
    client = OllamaClient()
    if not client.disponible():
        print("Serveur Ollama injoignable sur http://localhost:11434 : lancer `ollama serve`.",
              file=sys.stderr)
        raise SystemExit(1)
    modele = modele_par_defaut()
    if modele not in client.modeles():
        print(f"Modèle {modele} absent : lancer `ollama pull {modele}`.", file=sys.stderr)
        raise SystemExit(1)
    metriques = run(limite)
    print(json.dumps(metriques, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
