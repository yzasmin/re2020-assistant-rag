"""Évaluation de la génération : fidélité, exactitude, citations, refus, coût, latence.

Cette évaluation appelle l'API Anthropic : elle exige ANTHROPIC_API_KEY (fichier .env ou variable
d'environnement). Sans clé, la commande s'arrête sans rien écrire et sans produire de chiffre.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from datetime import date

from dotenv import load_dotenv

from re2020 import config
from re2020.agent import Agent, Trace
from re2020.chunking import load_chunks
from re2020.evaluation import Question, load_questions
from re2020.judge import Judge
from re2020.retrieval import Retriever
from re2020.sources import SOURCES_BY_ID


def cle_presente() -> bool:
    load_dotenv(config.ROOT / ".env")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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


def run() -> dict:
    chunks = load_chunks()
    par_id = {c.chunk_id: c for c in chunks}
    questions = load_questions()
    agent = Agent(Retriever(chunks))
    juge = Judge()

    traces, verdicts = [], []
    for i, q in enumerate(questions, start=1):
        trace = agent.answer(q.question)
        verdict = None
        if not trace.refus and trace.reponse:
            verdict = juge.evaluer(q.question, trace.reponse, passages_cites(trace, par_id), q.reponse_reference)
        bonnes, total = citations_valides(trace, par_id)
        traces.append({
            "id": q.id, "type": q.type, "hors_perimetre": q.hors_perimetre,
            **trace.to_dict(),
            "citations_valides": bonnes, "citations_total": total,
            "refus_correct": refus_correct(q, trace),
            "verdict_juge": verdict,
        })
        verdicts.append(verdict)
        print(f"[{i}/{len(questions)}] {q.id} refus={trace.refus} coût={trace.cout_usd:.4f} $", flush=True)

    fidelites = [v["fidelite"] for v in verdicts if v and v["fidelite"] is not None]
    exactitudes = Counter(v["exactitude"] for v in verdicts if v)
    bonnes = sum(t["citations_valides"] for t in traces)
    total_citations = sum(t["citations_total"] for t in traces)
    hors = [t for t in traces if t["hors_perimetre"]]
    dans = [t for t in traces if not t["hors_perimetre"]]
    cout_agent = sum(t["cout_usd"] for t in traces)

    metriques = {
        "date": date.today().isoformat(),
        "modele_agent": agent.model,
        "modele_juge": juge.model,
        "questions": len(questions),
        "fidelite_moyenne": round(statistics.fmean(fidelites), 4) if fidelites else None,
        "reponses_jugees": len(fidelites),
        "exactitude": {k: exactitudes.get(k, 0) for k in ("correcte", "partielle", "incorrecte")},
        "taux_citations_valides": round(bonnes / total_citations, 4) if total_citations else None,
        "citations_total": total_citations,
        "taux_refus_correct_hors_perimetre": round(sum(t["refus_correct"] for t in hors) / len(hors), 4) if hors else None,
        "taux_reponse_dans_perimetre": round(sum(t["refus_correct"] for t in dans) / len(dans), 4) if dans else None,
        "cout_total_usd": round(cout_agent + juge.cout_total_usd, 4),
        "cout_agent_usd": round(cout_agent, 4),
        "cout_juge_usd": round(juge.cout_total_usd, 4),
        "cout_moyen_par_question_usd": round((cout_agent + juge.cout_total_usd) / len(questions), 5),
        "latence_mediane_s": round(statistics.median(t["latence_s"] for t in traces), 2),
        "appels_outils_moyens": round(statistics.fmean(len(t["appels_outils"]) for t in traces), 2),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "generation_metrics.json").write_text(
        json.dumps(metriques, ensure_ascii=False, indent=2), encoding="utf-8")
    with (config.RESULTS_DIR / "generation_traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    return metriques


JETONS_PAR_CARACTERE = 0.25  # approximation usuelle en français, faute de pouvoir appeler count_tokens


def estimation_cout(k: int = 5) -> dict:
    """Estime le coût de l'évaluation complète sans appeler l'API.

    Les tailles sont mesurées sur les vrais prompts et les vrais passages ; seule la conversion
    caractères vers jetons est approchée. C'est une estimation, pas une mesure.
    """
    import json as _json

    from re2020.agent import SYSTEM_PROMPT, TOOLS
    from re2020.judge import CONSIGNE

    chunks = load_chunks()
    questions = load_questions()
    taille_passage = statistics.fmean(len(c.text) for c in chunks)
    def jetons(n: float) -> float:
        return n * JETONS_PAR_CARACTERE


    socle = jetons(len(SYSTEM_PROMPT) + len(_json.dumps(TOOLS, ensure_ascii=False)))
    passages = jetons(taille_passage * k)
    question_moyenne = jetons(statistics.fmean(len(q.question) for q in questions))
    sortie_agent = 400  # réponse citée courte, ordre de grandeur

    # Deux appels par question : recherche, puis réponse avec les passages.
    entree_agent = (socle + question_moyenne) + (socle + question_moyenne + passages + sortie_agent)
    prix_agent = config.PRICES_USD_PER_MTOK[config.AGENT_MODEL]
    cout_agent = (entree_agent * prix_agent[0] + sortie_agent * prix_agent[1]) / 1e6

    jugees = sum(1 for q in questions if not q.hors_perimetre)
    entree_juge = jetons(len(CONSIGNE)) + question_moyenne + jetons(600) + passages * 0.6 + sortie_agent
    sortie_juge = 600
    prix_juge = config.PRICES_USD_PER_MTOK[config.JUDGE_MODEL]
    cout_juge = (entree_juge * prix_juge[0] + sortie_juge * prix_juge[1]) / 1e6

    total = cout_agent * len(questions) + cout_juge * jugees
    return {
        "hypotheses": f"{JETONS_PAR_CARACTERE} jeton par caractère, 2 appels d'agent par question, "
                      f"{k} passages par recherche, {sortie_agent} jetons de réponse",
        "questions": len(questions),
        "reponses_jugees_estimees": jugees,
        "cout_agent_usd": round(cout_agent * len(questions), 3),
        "cout_juge_usd": round(cout_juge * jugees, 3),
        "cout_total_estime_usd": round(total, 3),
    }


def main() -> None:
    if not cle_presente():
        print(
            "ANTHROPIC_API_KEY absente : l'évaluation de la génération ne peut pas être exécutée.\n"
            "Copier .env.example en .env et y placer la clé, puis relancer cette commande.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    metriques = run()
    print(json.dumps(metriques, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
