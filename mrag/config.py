"""Configuration. Every field can be overridden with an env var MRAG_<FIELD> (upper-case)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # storage
    data_dir: str = "mrag_data"            # Qdrant (embedded mode) + extracted image assets
    qdrant_url: str = ""                    # http://host:6333 for a Qdrant server, ":memory:" for in-RAM

    # text embeddings
    text_backend: str = "sentence-transformers"   # "sentence-transformers" | "wordllama"
    text_model: str = "BAAI/bge-small-en-v1.5"    # any sentence-transformers model id or local path
    query_prefix: str = "Represent this sentence for searching relevant passages: "  # BGE query instruction
    device: str = "cpu"                            # "cpu" | "cuda" | "mps"

    # image-text embeddings (OpenCLIP)
    clip_model: str = "ViT-B-32-quickgelu"
    clip_pretrained: str = "laion400m_e32"         # OpenCLIP tag or path to a local .pt checkpoint
    clip_min_score: float = 0.22                   # cosine floor for text->image matches

    # OCR on images / scanned pages (RapidOCR, ONNX models ship inside the pip wheel)
    ocr: bool = True
    min_image_px: int = 64                         # skip icons / decorations smaller than this
    scanned_page_chars: int = 30                   # PDF page with fewer chars is treated as scanned
    max_pdf_pages: int = 0                         # 0 = no limit

    # chunking
    chunk_words: int = 180
    chunk_overlap: int = 40

    # retrieval
    top_k_text: int = 6
    top_k_images: int = 3
    candidates: int = 25                           # per-retriever candidates before fusion
    rrf_k: int = 60
    ocr_link_rank: int = 2                         # attach an image if its OCR text ranks in the top-N text hits

    # generation
    llm_backend: str = "ollama"                    # "ollama" | "openai" | "transformers" | "extractive"
    llm_model: str = "qwen3-vl:4b"                 # any Ollama tag; vision model => images are sent
    llm_url: str = "http://localhost:11434"        # Ollama; for "openai" e.g. http://localhost:8080 (llama-server)
    llm_api_key: str = ""                          # only for OpenAI-compatible servers that want one
    llm_vision: bool = True                        # send retrieved images to the model
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout: int = 600
    max_image_side: int = 1024                     # downscale images before sending to the VLM
    vlm_id: str = "Qwen/Qwen3-VL-4B-Instruct"     # "transformers" backend: HF model id or local path
    max_new_tokens: int = 512

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> "Config":
        values: dict = {}
        if path:
            with open(path, encoding="utf-8") as f:
                values.update(json.load(f))
        for f in fields(cls):
            env = os.environ.get(f"MRAG_{f.name.upper()}")
            if env is not None:
                values[f.name] = env
        values.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls()
        for f in fields(cls):
            if f.name in values:
                setattr(cfg, f.name, _coerce(values[f.name], type(getattr(cfg, f.name))))
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(v, typ):
    if isinstance(v, typ):
        return v
    if typ is bool:
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    return typ(v)
