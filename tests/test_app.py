"""Streamlit UI driven headlessly with streamlit.testing.AppTest."""
import pytest

from conftest import ROOT, model_cfg


def test_app_answers_and_shows_images(corpus, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from mrag.pipeline import MultimodalRAG

    cfg = model_cfg(tmp_path / "idx")
    r = MultimodalRAG(cfg)
    r.ingest([corpus / "images", corpus / "report.pdf"])
    r.close()  # embedded Qdrant allows one client per folder

    monkeypatch.setenv("MRAG_DATA_DIR", cfg.data_dir)
    monkeypatch.setenv("MRAG_TEXT_BACKEND", cfg.text_backend)
    monkeypatch.setenv("MRAG_LLM_BACKEND", "extractive")
    at = AppTest.from_file(str(ROOT / "mrag" / "streamlit_app.py"), default_timeout=600).run()
    assert not at.exception
    at.text_input[0].input("How long is the warranty on the R2?").run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "36-month" in page
    assert at.subheader[0].value == "Retrieved images"
