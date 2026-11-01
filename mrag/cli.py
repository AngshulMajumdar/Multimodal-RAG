"""Command line: python -m mrag {ingest,ask,stats,reset}"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mrag", description="Multimodal RAG with free local models")
    ap.add_argument("--config", help="JSON config file (env vars MRAG_* override)")
    ap.add_argument("--data-dir")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="index files / folders (pdf, docx, images, txt/md)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true", help="re-index unchanged files")

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--backend", choices=["ollama", "openai", "extractive"])
    p.add_argument("--model")
    p.add_argument("--url")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--show-context", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("stats", help="index statistics")
    sub.add_parser("reset", help="delete the index and extracted assets")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    over = {"data_dir": a.data_dir}
    if a.cmd == "ask":
        over.update(llm_backend=a.backend, llm_model=a.model, llm_url=a.url)
    cfg = Config.load(a.config, **over)

    if a.cmd == "reset":
        d = Path(cfg.data_dir)
        if d.exists():
            shutil.rmtree(d)
        print(f"removed {d}")
        return 0

    from .pipeline import MultimodalRAG

    rag = MultimodalRAG(cfg)
    try:
        if a.cmd == "ingest":
            stats = rag.ingest(a.paths, force=a.force, progress=lambda m: print(m, file=sys.stderr))
            print(json.dumps(stats, indent=2))
            return 1 if stats["errors"] else 0
        if a.cmd == "stats":
            print(json.dumps(rag.stats(), indent=2))
            return 0
        stream = not (a.no_stream or a.json)
        res = rag.ask(a.question, stream=stream)
        if a.json:
            print(json.dumps({
                "answer": res["answer"],
                "texts": [vars(t) for t in res["texts"]],
                "images": [vars(i) for i in res["images"]],
            }, indent=2, default=str))
            return 0
        if a.show_context:
            for i, t in enumerate(res["texts"], 1):
                print(f"[{i}] {Path(t.source).name} p.{t.page} ({t.kind}): {t.text[:160]}...")
            for i, im in enumerate(res["images"], 1):
                print(f"[Img {i}] {Path(im.source).name} p.{im.page} {im.why} -> {im.path}")
            print("-" * 60)
        if stream:
            for piece in res["answer"]:
                print(piece, end="", flush=True)
            print()
        else:
            print(res["answer"])
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
