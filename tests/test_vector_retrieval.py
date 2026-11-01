"""The CPU retrieval stage can link vector-drawn page images via native PDF text."""
from pathlib import Path

import numpy as np

from mrag.config import Config
from mrag.pipeline import MultimodalRAG
from mrag.store import IMAGE_COL, TEXT_COL


class _TextModel:
    def embed_query(self, q):
        return np.array([1.0, 0.0], dtype=np.float32)


class _ClipModel:
    def embed_text(self, qs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class _Models:
    text = _TextModel()
    clip = _ClipModel()


class _Store:
    def search(self, col, vector, k):
        if col == TEXT_COL:
            return [("text1", 0.9, {"text": "Figure 2: temperature trend", "source": "/x/demo.pdf", "page": 4, "kind": "text"})]
        return []  # CLIP found nothing, so the same-page link is vital

    def bm25_search(self, q, k):
        return []

    def all_payloads(self, col):
        return [("vector-page", {"path": "/x/chart.png", "source": "/x/demo.pdf", "page": 4,
                                  "page_render": True, "ocr": ""})]

    def get(self, col, ids):
        return {}


def test_vector_plot_retrieved_without_clip_match():
    r = MultimodalRAG.__new__(MultimodalRAG)
    r._ready = True
    r.cfg = Config(top_k_text=4, top_k_images=2)
    r.store = _Store()
    r.models = _Models()
    texts, images = r.retrieve("What is plotted in Figure 2?")
    assert texts and images and images[0].page == 4
    assert "same-page text match" in images[0].why
