"""Interface en ligne de commande : uv run python -m re2020 <commande>."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from re2020 import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="re2020", description="Assistant RAG sur la RE2020")
    sous = parser.add_subparsers(dest="commande", required=True)

    sous.add_parser("telecharger", help="télécharge les documents officiels dans data/raw")
    sous.add_parser("indexer", help="découpe les documents et construit les index BM25 et vectoriel")
    p_ask = sous.add_parser("ask", help="pose une question à l'agent (nécessite Ollama en local)")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="sortie complète en JSON (trace, coût, citations)")
    p_rech = sous.add_parser("chercher", help="recherche seule, sans appel au modèle")
    p_rech.add_argument("requete")
    p_rech.add_argument("-k", type=int, default=5)
    p_rech.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="hybrid")
    sous.add_parser("eval-retrieval", help="mesure la récupération et écrit results/retrieval_metrics.json")
    p_gen = sous.add_parser("eval-generation", help="mesure la génération (nécessite Ollama en local)")
    p_gen.add_argument("--limite", type=int, default=None, help="n'évaluer que les N premières questions")

    args = parser.parse_args(argv)
    load_dotenv(config.ROOT / ".env")

    if args.commande == "telecharger":
        from re2020.sources import download_all
        download_all()
    elif args.commande == "indexer":
        from re2020.chunking import build_chunks
        from re2020.retrieval import build_vector_index
        chunks = build_chunks()
        print(f"{len(chunks)} passages écrits dans {config.CHUNKS_PATH}")
        build_vector_index()
        print(f"index vectoriel écrit dans {config.QDRANT_DIR}")
    elif args.commande == "ask":
        from re2020.agent import repondre
        print(repondre(args.question, mode_json=args.json))
    elif args.commande == "chercher":
        from re2020.retrieval import Retriever
        for hit in Retriever().search(args.requete, args.k, args.mode):
            print(f"{hit.rank}. {hit.chunk.citation()}  score={hit.score:.4f}")
            print(f"   {hit.chunk.text[:220]}...")
    elif args.commande == "eval-retrieval":
        from re2020.eval_retrieval import main as run_eval
        run_eval()
    elif args.commande == "eval-generation":
        from re2020.eval_generation import main as run_eval_gen
        run_eval_gen(args.limite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
