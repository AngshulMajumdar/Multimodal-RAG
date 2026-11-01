---
title: Multimodal RAG
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
startup_duration_timeout: 1h
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ask questions over PDFs, Word files, images and charts
models:
  - Qwen/Qwen3-VL-4B-Instruct
  - BAAI/bge-small-en-v1.5
---

# Multimodal RAG on ZeroGPU

Upload PDFs, Word files, images, screenshots or scanned pages and ask questions about them. The answer cites the text passages and images it used, and the retrieved images appear in a gallery.

| Stage | Model / tool | Runs on |
|---|---|---|
| Parsing | PyMuPDF: text, embedded images, vector-drawn pages rendered, scanned pages rasterised. python-docx: text, tables, images | CPU |
| OCR (text inside images, charts, scans) | RapidOCR (PP-OCR in ONNX) | CPU |
| Text retrieval | `BAAI/bge-small-en-v1.5` + BM25, fused with Reciprocal Rank Fusion | CPU |
| Image retrieval | OpenCLIP ViT-B/32 (`laion400m_e32`) text→image, OCR-linked images and images of vector-drawn pages linked to retrieved native text | CPU |
| Vector store | Qdrant in-memory, one index per browser session | CPU |
| Answer | `Qwen/Qwen3-VL-4B-Instruct`, which sees the numbered passages **and the retrieved images** | ZeroGPU (only while answering) |

Each visitor receives a temporary session-isolated vector index with a 3-hour state TTL. Gradio may retain uploaded-file cache copies beyond this cleanup; avoid uploading confidential documents to a public Space.

## Deploy

1. On huggingface.co, go to **New Space**, choose **SDK: Gradio** and **Hardware: ZeroGPU**. Eligible free personal accounts (verified email, account older than 30 days, good standing) can host up to two ZeroGPU Spaces; PRO is not required. Check current eligibility at https://huggingface.co/docs/hub/spaces-zerogpu.
2. Extract the ZIP and upload its **contents** to the Space repository root. `app.py`, `README.md`, `requirements.txt`, `packages.txt`, `mrag/` and `examples/` must sit at the root; do not upload only the ZIP or put everything inside an extra enclosing folder. Alternatively, from the extracted folder:
   ```bash
   pip install -U huggingface_hub
   hf auth login
   hf upload <user>/<space-name> . --repo-type=space
   ```
3. Wait for the Space to show **Running**. The first boot downloads Qwen3-VL-4B (about 9 GB), BGE-small and CLIP. The startup timeout is one hour to allow a cold download. A demo corpus is included (and can be regenerated), so visitors can click **Load demo documents** and ask straight away.
4. Ask the five example questions. Expected evidence: warranty **36 months** (scanned PDF p.3); **Asia-Pacific, USD 41.2 million** (chart on PDF p.2); forklift **2500 kg** (Word table); code rotation **Monday at 09:00** (screenshot); office-cat photo (image retrieval). These are synthetic documents.
5. In a second browser session, verify that asking before indexing raises an error. Upload a small file there and verify it does not retrieve documents from the first session. Clear an index and confirm that answering requires indexing again.

No API key, Ollama service or external Qdrant server is required. Select ZeroGPU in the actual Space hardware settings; README metadata alone does not provision it.

### Space variables (Settings → Variables), all optional

| Variable | Default | |
|---|---|---|
| `MRAG_VLM_ID` | `Qwen/Qwen3-VL-4B-Instruct` | e.g. `Qwen/Qwen3-VL-8B-Instruct` for better answers, `Qwen/Qwen3-VL-2B-Instruct` for speed |
| `MRAG_TEXT_MODEL` | `BAAI/bge-small-en-v1.5` | `BAAI/bge-m3` for multilingual documents |
| `MRAG_MAX_FILES` / `MRAG_MAX_FILE_MB` / `MRAG_MAX_PDF_PAGES` | 10 / 25 / 40 | abuse limits for the public demo |
| `MRAG_TOP_K_TEXT` / `MRAG_TOP_K_IMAGES` | 6 / 3 | default context size (also adjustable in the UI) |
| `MRAG_LLM_BACKEND` | `transformers` | Keep this value on ZeroGPU. `extractive` is for local/CPU-only copies with no LLM. |

The Gradio event chain runs retrieval on CPU, then calls the **directly registered `@spaces.GPU` handler** for streaming, **only if retrieval succeeds**. CPU callbacks share a one-task queue, GPU generation has concurrency one, and a session cannot index or clear its files during an answer. On either generation success or failure, a web-process callback unlocks the controls and clears the pending prompt and image paths.

GPU time per answer is requested dynamically: 20 s, plus 4 s per image, plus 1 s per 12 tokens of the "Max new tokens" setting, capped at 120 s. This is an **unbenchmarked estimate**, not a measured runtime guarantee. Start with 256 output tokens if visitors hit a quota or duration error, then tune after representative hosted measurements. Lower requests get better queue priority.

The file count is cumulative across uploads and demo loading. PDFs are limited to their first 40 pages by default; the rest are not indexed. Upload size is checked at both the Gradio upload endpoint and indexing handler. The UI answers each question independently; the displayed transcript is not passed as conversational memory to retrieval or the model.

## Run locally

```bash
pip install -r requirements.txt
python app.py                                  # same Gradio app; uses CUDA/MPS if present, else CPU
```

The package also works without Gradio:
- CLI: `python -m mrag ingest docs/`, then `python -m mrag ask "..." --backend ollama|openai|transformers|extractive`
- Streamlit UI: `streamlit run mrag/streamlit_app.py`

All settings are in `mrag/config.py` and can be overridden with `MRAG_<FIELD>` environment variables.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The bundled test suite covers:
- loaders, OCR and retrieval on real models (correct top-1 text and images, plus negative queries that must return no image)
- re-ingest and the CLI
- the Ollama and OpenAI-compatible clients, against local servers that speak their wire protocols
- the transformers VLM path, using a tiny Qwen3-VL built on disk (same processor, image-token expansion, vision tower and streaming)
- this Gradio app driven over its API with `gradio_client`: demo documents, streamed answers, uploads, and isolation between sessions

To run the VLM and app tests against a real checkpoint, set `MRAG_VLM_ID=Qwen/Qwen3-VL-2B-Instruct`.

## Changes from the original Claude ZIP

See `REVIEW_AND_TEST_REPORT.md` for the latest corrections, measured validation results and remaining hosted checks. Full hosted inference still needs to be tested on an actual ZeroGPU Space.
