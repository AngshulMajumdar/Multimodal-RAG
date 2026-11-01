"""Hugging Face Space (Gradio, ZeroGPU): multimodal RAG over PDFs, Word files and images.

CPU: parsing, OCR, BGE + CLIP embeddings, Qdrant (one in-memory index per browser session).
GPU (ZeroGPU, only while answering): Qwen3-VL reads the retrieved text and images and answers with citations.
"""
from __future__ import annotations

import spaces  # noqa: I001  (import before torch on ZeroGPU)

import logging
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "examples"))

from mrag.config import Config  # noqa: E402
from mrag.generate import Generator, SYSTEM_PROMPT, build_prompt, load_image  # noqa: E402
from mrag.models import Models  # noqa: E402
from mrag.pipeline import MultimodalRAG  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("space")

MAX_FILES = int(os.environ.get("MRAG_MAX_FILES", 10))
MAX_FILE_MB = int(os.environ.get("MRAG_MAX_FILE_MB", 25))

BASE_CFG = Config.load(
    qdrant_url=":memory:",
    llm_backend=os.environ.get("MRAG_LLM_BACKEND", "transformers"),
    max_pdf_pages=int(os.environ.get("MRAG_MAX_PDF_PAGES", 40)),
    device="cpu",  # retrieval callbacks never have a ZeroGPU allocation
)
if BASE_CFG.llm_backend not in {"transformers", "extractive"}:
    raise ValueError("This Space supports MRAG_LLM_BACKEND=transformers or extractive.")

# ---- models: loaded once per process -------------------------------------------------
MODELS = Models(BASE_CFG)
_ = MODELS.text, MODELS.clip, MODELS.ocr          # CPU models, warm at startup

VLM = None
if BASE_CFG.llm_backend == "transformers":
    from mrag.hf_vlm import HFVLM

    VLM = HFVLM(BASE_CFG.vlm_id)                  # device "cuda" on ZeroGPU (attached per call)
    log.info("VLM %s on %s", BASE_CFG.vlm_id, VLM.device)

DEMO_DIR = ROOT / "demo_docs"
if not DEMO_DIR.exists():
    import make_demo_corpus

    make_demo_corpus.build(DEMO_DIR)


# ---- per-session state ------------------------------------------------------------------
class Session:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="mrag_session_")
        self.rag = MultimodalRAG(replace(BASE_CFG, data_dir=self.dir), models=MODELS, vlm=VLM)
        self.busy = False

    def close(self):
        self.rag.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _cleanup(s):
    if isinstance(s, Session):
        s.close()


def _session(s):
    return s if isinstance(s, Session) else Session()


def _idle(s):
    if isinstance(s, Session) and s.busy:
        raise gr.Error("An answer is still running. Wait for it to finish.")


def finish_answer(s):
    """Runs in the web process, including after a GPU error or quota rejection."""
    if isinstance(s, Session):
        s.busy = False
    return s, "", [], "", *[gr.update(interactive=True) for _ in range(4)]


def _status(s: Session) -> str:
    st = s.rag.stats()
    names = sorted(Path(p).name for p in s.rag._manifest())
    return (f"**Indexed:** {st['files']} file(s) · {st['mrag_text']} text chunks · {st['mrag_image']} images"
            + (f"\n\n{', '.join(names)}" if names else ""))


# ---- GPU part -------------------------------------------------------------------------
def _gpu_seconds(prompt, image_paths, max_new_tokens):
    return int(min(120, 20 + 4 * len(image_paths) + max_new_tokens / 12))


@spaces.GPU(duration=lambda prompt, image_paths, max_new_tokens, history:
            _gpu_seconds(prompt, image_paths, max_new_tokens))
def vlm_answer(prompt: str, image_paths: list[str], max_new_tokens: int, history: list[dict]):
    """Gradio-bound GPU callback: stream an answer directly into the chat history.

    Hugging Face ZeroGPU must see @spaces.GPU on the ACTUAL registered event
    handler (the second step in the Gradio event chain below), not on a helper
    invoked by an undecorated event handler.
    """
    if not prompt or not history or history[-1].get("role") != "assistant":
        raise gr.Error("Submit a question before requesting an answer.")
    images = [load_image(p, BASE_CFG.max_image_side) for p in image_paths]
    messages = [dict(item) for item in (history or [])]
    output = ""
    for piece in VLM.stream(SYSTEM_PROMPT, prompt, images, int(max_new_tokens), BASE_CFG.llm_temperature):
        output += piece
        messages[-1] = {"role": "assistant", "content": output}
        yield messages
    if not output.strip():
        messages[-1] = {"role": "assistant", "content": "The model returned an empty response."}
        yield messages


# ---- handlers -------------------------------------------------------------------------
def index_files(files, s, progress=gr.Progress()):
    _idle(s)
    files = files or []
    if len(files) > MAX_FILES:
        raise gr.Error(f"At most {MAX_FILES} files per session.")
    paths = []
    for f in files:
        p = Path(f if isinstance(f, str) else f.name)
        if p.stat().st_size > MAX_FILE_MB * 2**20:
            raise gr.Error(f"{p.name} is larger than {MAX_FILE_MB} MB.")
        paths.append(p)
    if not paths:
        raise gr.Error("Choose some files first.")
    existing = set(s.rag._manifest()) if isinstance(s, Session) else set()
    if len(existing | {str(p.resolve()) for p in paths}) > MAX_FILES:
        raise gr.Error(f"At most {MAX_FILES} indexed files per session. Clear the index first.")
    s = _session(s)
    progress(0, desc="Parsing, OCR and embedding")
    stats = s.rag.ingest(paths)
    if stats["errors"]:
        gr.Warning("; ".join(stats["errors"])[:500])
    return s, _status(s), None


def load_demo(s):
    _idle(s)
    from mrag.ingest import iter_files

    paths = list(iter_files([DEMO_DIR]))
    existing = set(s.rag._manifest()) if isinstance(s, Session) else set()
    if len(existing | {str(p.resolve()) for p in paths}) > MAX_FILES:
        raise gr.Error(f"Demo would exceed {MAX_FILES} indexed files. Clear the index first.")
    s = _session(s)
    stats = s.rag.ingest(paths)
    if stats["errors"]:
        gr.Warning("; ".join(stats["errors"])[:500])
    return s, _status(s)


def reset(s):
    _idle(s)
    _cleanup(s)
    s = Session()
    return s, _status(s), [], [], "", "", [], ""


def prepare_answer(question, history, s, top_k_text, top_k_images, send_images):
    """CPU-only retrieval. Returns inputs for the separate ZeroGPU event."""
    _idle(s)
    if not question or not question.strip():
        raise gr.Error("Enter a question first.")
    if len(question) > 4000:
        raise gr.Error("Keep the question within 4000 characters.")
    if not isinstance(s, Session):
        raise gr.Error("Index some files first (or click 'Load demo documents').")
    if s.rag.stats()["files"] == 0:
        raise gr.Error("Index some files first (or click 'Load demo documents').")
    s.rag.cfg.top_k_text, s.rag.cfg.top_k_images = int(top_k_text), int(top_k_images)
    texts, images = s.rag.retrieve(question)
    selected_images = images if send_images else []
    gallery = [(im.path, f"[Img {i}] {Path(im.source).name}" + (f" p.{im.page}" if im.page else ""))
               for i, im in enumerate(images, 1)]
    sources = "\n\n".join(
        f"**[{i}] {Path(t.source).name}{f' p.{t.page}' if t.page else ''}** "
        f"<small>({t.kind})</small>\n\n{t.text[:700]}" for i, t in enumerate(texts, 1)) or "_none_"
    history = list(history or []) + [
        {"role": "user", "content": question}, {"role": "assistant", "content": ""},
    ]
    s.busy = True
    return (s, history, gallery, sources, gr.update(value="", interactive=False), build_prompt(question, texts, selected_images),
            [im.path for im in selected_images], question,
            *[gr.update(interactive=False) for _ in range(3)])


def cpu_answer(prompt, image_paths, max_new_tokens, history, s, question):
    """Optional non-GPU local fallback. Not used for a standard ZeroGPU Space."""
    messages = [dict(item) for item in history]
    output = ""
    texts, images = s.rag.retrieve(question)
    if not image_paths:
        images = []
    cfg = replace(s.rag.cfg, max_new_tokens=int(max_new_tokens))
    for piece in Generator(cfg, MODELS.text).answer(question, texts, images, stream=True):
        output += piece
        messages[-1] = {"role": "assistant", "content": output}
        yield messages
    if not output:
        messages[-1] = {"role": "assistant", "content": "No evidence found."}
        yield messages


# ---- UI -------------------------------------------------------------------------------
EXAMPLES = [
    "How long is the warranty on the Nimbus R2?",
    "Which region had the highest revenue in FY2025, and how much?",
    "What is the maximum load of the electric forklift?",
    "When is the server room door code changed?",
    "Show me the photo of the office cat.",
]

with gr.Blocks(title="Multimodal RAG", delete_cache=(3600, 10800)) as demo:
    gr.Markdown(
        "# Multimodal RAG\n"
        "Ask questions over **PDFs, Word files, images, charts and scanned pages**. "
        "Retrieval: BGE-small + BM25 for text, OpenCLIP for images, RapidOCR for text inside images, Qdrant. "
        f"Answering: **{Path(BASE_CFG.vlm_id).name}** on ZeroGPU, which reads the retrieved text *and* images. "
        "Each browser session gets a separate temporary index. Avoid uploading confidential data.")
    state = gr.State(None, delete_callback=_cleanup, time_to_live=3 * 3600)
    pending_prompt = gr.State("")
    pending_images = gr.State([])
    pending_question = gr.State("")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            files = gr.File(label=f"Upload (≤{MAX_FILES} files, ≤{MAX_FILE_MB} MB each)", file_count="multiple",
                            file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
                                        ".tif", ".tiff", ".txt", ".md"])
            btn_index = gr.Button("Index files", variant="primary")
            btn_demo = gr.Button("Load demo documents")
            btn_reset = gr.Button("Clear index")
            status = gr.Markdown("**Indexed:** nothing yet")
            with gr.Accordion("Settings", open=False):
                k_text = gr.Slider(1, 12, BASE_CFG.top_k_text, step=1, label="Text passages")
                k_img = gr.Slider(0, 6, BASE_CFG.top_k_images, step=1, label="Images")
                max_tokens = gr.Slider(64, 1024, BASE_CFG.max_new_tokens, step=32, label="Max new tokens")
                send_imgs = gr.Checkbox(True, label="Send retrieved images to the VLM")
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Answer", height=420)
            q = gr.Textbox(label="Question", placeholder="Ask about your documents", submit_btn=True)
            gr.Examples(EXAMPLES, inputs=q)
            gallery = gr.Gallery(label="Retrieved images", columns=3, height=260)
            with gr.Accordion("Retrieved text", open=False):
                src = gr.Markdown()

    cpu_queue = dict(concurrency_id="retrieval", concurrency_limit=1)
    btn_index.click(index_files, [files, state], [state, status, files], **cpu_queue)
    btn_demo.click(load_demo, [state], [state, status], **cpu_queue)
    btn_reset.click(reset, [state], [state, status, chat, gallery, src,
                                  pending_prompt, pending_images, pending_question], **cpu_queue)
    # A two-step Gradio event chain is necessary for ZeroGPU detection.
    # Only successful retrieval can trigger generation; .then also runs on failure.
    prep = q.submit(prepare_answer,
                    inputs=[q, chat, state, k_text, k_img, send_imgs],
                    outputs=[state, chat, gallery, src, q, pending_prompt, pending_images, pending_question,
                             btn_index, btn_demo, btn_reset], **cpu_queue)
    if BASE_CFG.llm_backend == "transformers":
        answer = prep.success(fn=vlm_answer,
                  inputs=[pending_prompt, pending_images, max_tokens, chat],
                  outputs=chat, concurrency_id="vlm", concurrency_limit=1)
    else:
        answer = prep.success(fn=cpu_answer,
                  inputs=[pending_prompt, pending_images, max_tokens, chat, state, pending_question],
                  outputs=chat, **cpu_queue)

    answer.then(finish_answer, inputs=state,
                outputs=[state, pending_prompt, pending_images, pending_question,
                         q, btn_index, btn_demo, btn_reset], **cpu_queue)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch(max_file_size=f"{MAX_FILE_MB}mb")
