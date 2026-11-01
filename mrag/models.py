"""Free, local models: text embedder, OpenCLIP image/text embedder, RapidOCR."""
from __future__ import annotations

import logging
from functools import cached_property
from threading import Lock

import numpy as np
from PIL import Image

from .config import Config

log = logging.getLogger(__name__)


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class TextEmbedder:
    """Dense text embeddings. Backends: sentence-transformers (default BGE-small) or WordLlama."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.text_backend
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(cfg.text_model, device=cfg.device)
            get_dim = getattr(self._m, "get_embedding_dimension", None) or self._m.get_sentence_embedding_dimension
            self.dim = int(get_dim())
            self.name = f"st:{cfg.text_model}"
        elif self.backend == "wordllama":
            from wordllama import WordLlama

            self._m = WordLlama.load()
            self.dim = int(self._m.embed(["probe"]).shape[-1])
            self.name = "wordllama:default"
        else:
            raise ValueError(f"unknown text_backend {self.backend!r}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        if self.backend == "sentence-transformers":
            v = self._m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        else:
            v = self._m.embed(texts)
        return _l2norm(v)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.cfg.query_prefix if self.backend == "sentence-transformers" else ""
        return self._encode([prefix + text])[0]


class ClipEmbedder:
    """OpenCLIP: images and text in one 512-d space (text->image retrieval)."""

    def __init__(self, cfg: Config):
        import open_clip
        import torch

        self.torch = torch
        self.device = cfg.device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained, device=cfg.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model)
        self.dim = int(self.embed_text(["probe"]).shape[-1])
        self.name = f"clip:{cfg.clip_model}:{cfg.clip_pretrained.split('/')[-1]}"

    def embed_images(self, images: list[Image.Image], batch: int = 16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                x = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + batch]])
                out.append(self.model.encode_image(x.to(self.device)).float().cpu().numpy())
        return _l2norm(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tok = self.tokenizer(texts).to(self.device)
            return _l2norm(self.model.encode_text(tok).float().cpu().numpy())


class OCR:
    """RapidOCR (PaddleOCR models exported to ONNX; bundled in the wheel, no download)."""

    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self.min_score = min_score

    def __call__(self, image: Image.Image) -> str:
        arr = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR (OpenCV convention)
        result, _ = self._engine(arr)
        if not result:
            return ""
        # sort top-to-bottom, left-to-right
        result.sort(key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
        return " ".join(t for _, t, s in result if float(s) >= self.min_score).strip()


class Models:
    """Lazy holder so nothing heavy loads until it is needed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ocr_lock = Lock()

    @cached_property
    def text(self) -> TextEmbedder:
        log.info("loading text embedder (%s)", self.cfg.text_backend)
        return TextEmbedder(self.cfg)

    @cached_property
    def clip(self) -> ClipEmbedder:
        log.info("loading OpenCLIP %s/%s", self.cfg.clip_model, self.cfg.clip_pretrained)
        return ClipEmbedder(self.cfg)

    @cached_property
    def ocr(self) -> OCR | None:
        return OCR() if self.cfg.ocr else None

    def ocr_text(self, key: str, image: Image.Image) -> str:
        """OCR with a content-hash cache (same image in many sessions/files is read once)."""
        if self.ocr is None:
            return ""
        # The OCR engine and its cache are shared by every browser session.
        with self._ocr_lock:
            cache = self.__dict__.setdefault("_ocr_cache", {})
            if key not in cache:
                if len(cache) >= 512:
                    cache.clear()
                cache[key] = self.ocr(image)
            return cache[key]
