"""End-to-end multimodal RAG: ingest -> hybrid retrieval (dense + BM25 + CLIP) -> grounded generation."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .config import Config
from .generate import Generator, ImageHit, TextHit
from .ingest import TextItem, chunk_text, iter_files, load_any
from .models import Models
from .store import IMAGE_COL, TEXT_COL, Store, point_id

log = logging.getLogger(__name__)


def rrf(rankings: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


class MultimodalRAG:
    def __init__(self, cfg: Config | None = None, models: Models | None = None, vlm=None):
        self.cfg = cfg or Config.load()
        self.models = models or Models(self.cfg)   # share one Models across sessions to load weights once
        self.vlm = vlm                              # for llm_backend="transformers" (mrag.hf_vlm.HFVLM)
        self.store = Store(self.cfg.data_dir, self.cfg.qdrant_url)
        self._ready = False
        self.manifest_path = Path(self.cfg.data_dir) / "manifest.json"

    # ------------------------------------------------------------------ setup
    def _ensure(self):
        if not self._ready:
            t, c = self.models.text, self.models.clip
            self.store.ensure(t.dim, c.dim, t.name, c.name)
            self._ready = True

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    # ------------------------------------------------------------------ ingest
    def ingest(self, paths: list[str | Path], force: bool = False,
               progress: Callable[[str], None] | None = None) -> dict:
        self._ensure()
        cfg, m = self.cfg, self.models
        manifest = self._manifest()
        stats = {"files": 0, "skipped": 0, "text_chunks": 0, "images": 0, "ocr_chunks": 0, "errors": []}
        for f in iter_files(paths):
            src = str(f.resolve())
            digest = hashlib.sha1(f.read_bytes()).hexdigest()
            if not force and manifest.get(src) == digest:
                stats["skipped"] += 1
                continue
            if progress:
                progress(f"ingesting {f.name}")
            try:
                ex = load_any(f, cfg.min_image_px, cfg.scanned_page_chars, cfg.max_pdf_pages)
            except Exception as e:
                log.exception("failed to load %s", f)
                stats["errors"].append(f"{f}: {e}")
                continue
            self.store.delete_source(src)  # re-ingest replaces old points for this file

            text_items: list[TextItem] = []
            for t in ex.texts:
                for c in chunk_text(t.text, cfg.chunk_words, cfg.chunk_overlap):
                    text_items.append(TextItem(c, src, t.page, t.kind))

            img_ids, img_vecs, img_payloads = [], [], []
            if ex.images:
                vecs = m.clip.embed_images([i.image for i in ex.images])
                for item, v in zip(ex.images, vecs):
                    path = self.store.assets / f"{item.sha1}.png"
                    if not path.exists():
                        item.image.save(path)
                    ocr = ("" if item.meta.get("page_render")
                           else m.ocr_text(item.sha1, item.image))
                    # Same illustration can occur on different pages; each citation
                    # must retain its own page instead of clobbering the Qdrant ID.
                    iid = point_id(src, "img", item.page, item.sha1)
                    img_ids.append(iid)
                    img_vecs.append(v)
                    img_payloads.append({"path": str(path.resolve()), "source": src, "page": item.page,
                                         "ocr": ocr, "sha1": item.sha1, **item.meta})
                    if len(ocr.split()) >= 3:
                        for c in chunk_text(ocr, cfg.chunk_words, cfg.chunk_overlap):
                            text_items.append(TextItem(c, src, item.page, "ocr", iid))
                            stats["ocr_chunks"] += 1
                self.store.upsert(IMAGE_COL, img_ids, np.stack(img_vecs), img_payloads)

            if text_items:
                vecs = m.text.embed_docs([t.text for t in text_items])
                ids = [point_id(src, t.page, t.kind, i, hashlib.sha1(t.text.encode()).hexdigest())
                       for i, t in enumerate(text_items)]
                payloads = [{"text": t.text, "source": src, "page": t.page, "kind": t.kind,
                             "image_id": t.image_id} for t in text_items]
                self.store.upsert(TEXT_COL, ids, vecs, payloads)

            manifest[src] = digest
            self.manifest_path.write_text(json.dumps(manifest, indent=1))
            stats["files"] += 1
            stats["text_chunks"] += len(text_items) - sum(t.kind == "ocr" for t in text_items)
            stats["images"] += len(ex.images)
        return stats

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, question: str) -> tuple[list[TextHit], list[ImageHit]]:
        self._ensure()
        cfg, m = self.cfg, self.models
        C = cfg.candidates

        dense = self.store.search(TEXT_COL, m.text.embed_query(question), C)
        sparse = self.store.bm25_search(question, C)
        payload = {pid: p for pid, _, p in dense + sparse}
        dscore = {pid: s for pid, s, _ in dense}
        fused = rrf([[pid for pid, _, _ in dense], [pid for pid, _, _ in sparse]], cfg.rrf_k)
        texts = [TextHit(pid, payload[pid]["text"], payload[pid]["source"], payload[pid].get("page"),
                         payload[pid].get("kind", "text"), dscore.get(pid, 0.0), payload[pid].get("image_id"))
                 for pid, _ in fused[:cfg.top_k_text]]

        # images: CLIP text->image similarity, plus images whose OCR text was retrieved
        clip_hits = [h for h in self.store.search(IMAGE_COL, m.clip.embed_text([question])[0], C)
                     if h[1] >= cfg.clip_min_score]
        ocr_linked = list(dict.fromkeys(t.image_id for t in texts[:cfg.ocr_link_rank] if t.image_id))
        # A born-digital PDF's vector-drawn chart cannot have an OCR image_id.
        # Connect relevant native text to the rendered image of its own page.
        relevant_pages = {(t.source, t.page) for t in texts[:cfg.ocr_link_rank] if t.page is not None}
        page_payloads = [(pid, p) for pid, p in self.store.all_payloads(IMAGE_COL)
                         if p.get("page_render") and (p.get("source"), p.get("page")) in relevant_pages]
        page_linked = [pid for pid, _ in page_payloads]
        img_payload = {pid: p for pid, _, p in clip_hits}
        img_payload.update(dict(page_payloads))
        img_payload.update(self.store.get(IMAGE_COL, [i for i in ocr_linked if i not in img_payload]))
        cscore = {pid: s for pid, s, _ in clip_hits}
        images = []
        for pid, _ in rrf([[h[0] for h in clip_hits], ocr_linked, page_linked], cfg.rrf_k)[:cfg.top_k_images]:
            p = img_payload.get(pid)
            if p is None:
                continue
            why = ([f"clip={cscore[pid]:.3f}"] if pid in cscore else []) + (["ocr-text match"] if pid in ocr_linked else []) + (["same-page text match"] if pid in page_linked else [])
            images.append(ImageHit(pid, p["path"], p["source"], p.get("page"), p.get("ocr", ""),
                                   cscore.get(pid, 0.0), why))
        return texts, images

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, stream: bool = False) -> dict:
        texts, images = self.retrieve(question)
        gen = Generator(self.cfg, self.models.text, self.vlm)
        return {"question": question, "answer": gen.answer(question, texts, images, stream=stream),
                "texts": texts, "images": images}

    def stats(self) -> dict:
        return {**self.store.count(), "files": len(self._manifest())}

    def close(self):
        self.store.close()
