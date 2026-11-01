"""Streamlit UI:  streamlit run mrag/streamlit_app.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mrag.config import Config  # noqa: E402
from mrag.pipeline import MultimodalRAG  # noqa: E402

st.set_page_config(page_title="Multimodal RAG", layout="wide")


@st.cache_resource
def get_rag() -> MultimodalRAG:  # one instance per server: Qdrant embedded mode allows a single client
    return MultimodalRAG(Config.load())


rag = get_rag()
cfg = rag.cfg

with st.sidebar:
    st.header("Index")
    files = st.file_uploader("Add documents", accept_multiple_files=True,
                             type=["pdf", "docx", "png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff",
                                   "txt", "md"])
    if st.button("Ingest", disabled=not files):
        upload_dir = Path(cfg.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            p = upload_dir / Path(f.name).name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        with st.spinner("Parsing, OCR, embedding..."):
            s = rag.ingest(paths)
        st.success(f"{s['files']} file(s): {s['text_chunks']} text chunks, {s['images']} images, "
                   f"{s['ocr_chunks']} OCR chunks")
        for e in s["errors"]:
            st.error(e)
    st.caption(f"Index: {rag.stats()}")
    st.header("Generator")
    cfg.llm_backend = st.selectbox("Backend", ["ollama", "openai", "extractive"],
                                   index=["ollama", "openai", "extractive"].index(cfg.llm_backend))
    if cfg.llm_backend != "extractive":
        cfg.llm_model = st.text_input("Model", cfg.llm_model)
        cfg.llm_url = st.text_input("Server URL", cfg.llm_url)
        cfg.llm_vision = st.checkbox("Send retrieved images to the model", cfg.llm_vision)

st.title("Multimodal RAG")
q = st.text_input("Ask about your documents")
if q:
    try:
        res = rag.ask(q, stream=True)
        st.write_stream(res["answer"])
    except Exception as e:
        st.error(str(e))
        texts, images = rag.retrieve(q)
        res = {"texts": texts, "images": images}
    if res["images"]:
        st.subheader("Retrieved images")
        cols = st.columns(len(res["images"]))
        for i, (c, im) in enumerate(zip(cols, res["images"]), 1):
            c.image(im.path, caption=f"[Img {i}] {Path(im.source).name} p.{im.page} {', '.join(im.why)}")
    with st.expander("Retrieved text"):
        for i, t in enumerate(res["texts"], 1):
            st.markdown(f"**[{i}] {Path(t.source).name} p.{t.page}** ({t.kind})")
            st.write(t.text)
