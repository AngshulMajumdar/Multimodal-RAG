"""End-to-end with real models: WordLlama / sentence-transformers text embeddings, OpenCLIP, RapidOCR, Qdrant."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, model_cfg

TEXT_CASES = [  # question, expected file, expected page, expected kind, fact that must be in the top hit
    ("How long is the warranty on the R2?", "report.pdf", 3, "ocr", "36-month"),
    ("What was Asia-Pacific revenue in FY2025?", "report.pdf", 2, "ocr", "41.2"),
    ("Who is the CEO of Nimbus Robotics?", "report.pdf", 1, "text", "Meera Iyer"),
    ("What is the maximum load of the electric forklift?", "handbook.docx", None, "table", "2500 kg"),
    ("When is the server room door code changed?", "notice_board.png", None, "ocr", "Monday"),
]

IMAGE_CASES = [  # question, expected top image file (None = no image should be returned)
    ("show me a picture of a cat", "office_cat.jpg"),
    ("photo of a rocket launch", "launch.jpg"),
    ("an astronaut in a space suit", "visitor.png"),
    ("a cup of coffee", "handbook.docx"),
    ("What was Asia-Pacific revenue in FY2025?", "report.pdf"),
    ("a giraffe in the savanna", None),
    ("What is the maximum load of the electric forklift?", None),
]


def test_ingest_counts(rag):
    s = rag.stats()
    assert s["files"] == 7
    assert s["mrag_image"] == 7          # chart, scanned page, docx photo, 4 standalone images
    assert s["mrag_text"] == 8           # 5 text/table chunks + 3 OCR chunks


@pytest.mark.parametrize("q,fname,page,kind,fact", TEXT_CASES)
def test_text_retrieval_top1(rag, q, fname, page, kind, fact):
    texts, _ = rag.retrieve(q)
    top = texts[0]
    assert Path(top.source).name == fname and top.page == page and top.kind == kind
    assert fact in top.text


@pytest.mark.parametrize("q,fname", IMAGE_CASES)
def test_image_retrieval(rag, q, fname):
    _, images = rag.retrieve(q)
    if fname is None:
        assert images == []
    else:
        assert images and Path(images[0].source).name == fname
        assert Path(images[0].path).exists()


def test_chart_image_reached_by_both_clip_and_ocr(rag):
    _, images = rag.retrieve("What was Asia-Pacific revenue in FY2025?")
    assert images[0].page == 2 and "ocr-text match" in images[0].why and images[0].why[0].startswith("clip=")


@pytest.mark.parametrize("q,fact", [("How long is the warranty on the R2?", "36-month"),
                                    ("Who is the CEO of Nimbus Robotics?", "Meera Iyer"),
                                    ("When is the server room door code changed?", "every Monday")])
def test_extractive_answer(rag, q, fact):
    ans = rag.ask(q)["answer"]
    assert fact in ans and "[1]" in ans


def test_reingest_is_idempotent(rag, corpus):
    before = rag.stats()
    assert rag.ingest([corpus])["skipped"] == 7
    s = rag.ingest([corpus / "report.pdf"], force=True)
    assert s["files"] == 1 and rag.stats() == before


def test_model_mismatch_is_detected(rag, corpus):
    from mrag.store import Store

    st = Store.__new__(Store)  # only exercise the metadata check
    st.meta_path = Path(rag.cfg.data_dir) / "index_meta.json"
    with pytest.raises(RuntimeError, match="reset"):
        Store.ensure(st, 999, 512, "other", "clip")


def test_cli_roundtrip(corpus, tmp_path):
    env = {**os.environ, "MRAG_DATA_DIR": str(tmp_path / "idx"),
           "MRAG_TEXT_BACKEND": os.environ.get("MRAG_TEXT_BACKEND", "wordllama")}
    run = lambda *a: subprocess.run([sys.executable, "-m", "mrag", *a], cwd=ROOT, env=env,  # noqa: E731
                                    capture_output=True, text=True, timeout=900)
    r = run("ingest", str(corpus / "images"))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["images"] == 4
    r = run("ask", "show me a picture of a cat", "--backend", "extractive", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert Path(out["images"][0]["source"]).name == "office_cat.jpg"
    assert run("reset").returncode == 0 and not (tmp_path / "idx").exists()
