"""Qdrant vector store (embedded on-disk mode by default, or a Qdrant server) + BM25 keyword index."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import numpy as np

TEXT_COL = "mrag_text"
IMAGE_COL = "mrag_image"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def point_id(*parts) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(map(str, parts))))


class Store:
    def __init__(self, data_dir: str, qdrant_url: str = ""):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.dir = Path(data_dir)
        self.assets = self.dir / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "index_meta.json"
        if qdrant_url == ":memory:":          # per-session index (e.g. a public demo)
            self.client = QdrantClient(location=":memory:")
        elif qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            self.client = QdrantClient(path=str(self.dir / "qdrant"))
        self._bm25 = None

    # ---------- schema ----------
    def ensure(self, text_dim: int, image_dim: int, text_model: str, image_model: str):
        meta = {"text_model": text_model, "text_dim": text_dim, "image_model": image_model, "image_dim": image_dim}
        if self.meta_path.exists():
            old = json.loads(self.meta_path.read_text())
            if old != meta:
                raise RuntimeError(
                    f"Index was built with {old}, current models are {meta}. "
                    f"Run `python -m mrag reset` (or use another data_dir) before re-ingesting.")
        for name, dim in ((TEXT_COL, text_dim), (IMAGE_COL, image_dim)):
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    name, vectors_config=self.models.VectorParams(size=dim, distance=self.models.Distance.COSINE))
        self.meta_path.write_text(json.dumps(meta, indent=2))

    # ---------- writes ----------
    def upsert(self, col: str, ids: list[str], vecs: np.ndarray, payloads: list[dict], batch: int = 256):
        P = self.models.PointStruct
        for i in range(0, len(ids), batch):
            self.client.upsert(col, points=[P(id=pid, vector=v.tolist(), payload=pl) for pid, v, pl in
                                            zip(ids[i:i + batch], vecs[i:i + batch], payloads[i:i + batch])])
        if col == TEXT_COL:
            self._bm25 = None

    def delete_source(self, source: str):
        f = self.models.Filter(must=[self.models.FieldCondition(
            key="source", match=self.models.MatchValue(value=source))])
        for col in (TEXT_COL, IMAGE_COL):
            if self.client.collection_exists(col):
                self.client.delete(col, points_selector=self.models.FilterSelector(filter=f))
        self._bm25 = None

    # ---------- reads ----------
    def search(self, col: str, vec: np.ndarray, k: int) -> list[tuple[str, float, dict]]:
        if not self.client.collection_exists(col):
            return []
        res = self.client.query_points(col, query=vec.tolist(), limit=k, with_payload=True)
        return [(str(p.id), float(p.score), p.payload) for p in res.points]

    def all_payloads(self, col: str) -> list[tuple[str, dict]]:
        if not self.client.collection_exists(col):
            return []
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(col, limit=512, offset=offset, with_payload=True, with_vectors=False)
            out.extend((str(p.id), p.payload) for p in pts)
            if offset is None:
                return out

    def get(self, col: str, ids: list[str]) -> dict[str, dict]:
        if not ids or not self.client.collection_exists(col):
            return {}
        return {str(p.id): p.payload for p in self.client.retrieve(col, ids=ids, with_payload=True)}

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float, dict]]:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            docs = self.all_payloads(TEXT_COL)
            self._bm25 = (BM25Okapi([tokenize(p["text"]) or ["_"] for _, p in docs]), docs) if docs else (None, [])
        bm, docs = self._bm25
        q = tokenize(query)
        if bm is None or not q:
            return []
        scores = bm.get_scores(q)
        top = np.argsort(-scores)[:k]
        return [(docs[i][0], float(scores[i]), docs[i][1]) for i in top if scores[i] > 0]

    def count(self) -> dict:
        c = {}
        for col in (TEXT_COL, IMAGE_COL):
            c[col] = self.client.count(col).count if self.client.collection_exists(col) else 0
        return c

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
