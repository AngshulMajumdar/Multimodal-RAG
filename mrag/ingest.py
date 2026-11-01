"""Document loaders: PDF (text, embedded images, scanned pages), DOCX (text, tables, images),
images, and plain text/markdown. Everything is normalised into TextItem / ImageItem records."""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv"}
SUPPORTED = IMAGE_EXT | TEXT_EXT | {".pdf", ".docx"}


@dataclass
class TextItem:
    text: str
    source: str
    page: int | None = None
    kind: str = "text"            # "text" | "table" | "ocr"
    image_id: str | None = None   # set when the text came from an image (OCR)


@dataclass
class ImageItem:
    image: Image.Image
    source: str
    page: int | None = None
    sha1: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Extracted:
    texts: list[TextItem] = field(default_factory=list)
    images: list[ImageItem] = field(default_factory=list)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _clean(s: str) -> str:
    s = s.replace("­", "")                 # soft hyphen
    s = re.sub(r"-\n(?=[a-z])", "", s)          # de-hyphenate line breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_words: int = 180, overlap: int = 40) -> list[str]:
    """Paragraph-aware word-window chunking."""
    if chunk_words <= 0 or not 0 <= overlap < chunk_words:
        raise ValueError("require chunk_words > 0 and 0 <= overlap < chunk_words")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    step = max(1, chunk_words - overlap)
    for p in paras:
        words = p.split()
        if len(words) > chunk_words:            # long paragraph: flush, then slide a window
            if buf:
                chunks.append(" ".join(buf))
                buf = []
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + chunk_words]))
                if i + chunk_words >= len(words):
                    break
            continue
        if len(buf) + len(words) > chunk_words and buf:
            chunks.append(" ".join(buf))
            # Preserve only overlap that fits beside the next whole paragraph.
            keep = min(overlap, chunk_words - len(words))
            buf = buf[-keep:] if keep else []
        buf.extend(words)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _img_from_bytes(b: bytes) -> Image.Image | None:
    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        return im.convert("RGB")
    except Exception as e:  # unsupported codec etc.
        log.warning("could not decode image: %s", e)
        return None


def load_pdf(path: Path, min_px: int, scanned_chars: int, max_pages: int = 0) -> Extracted:
    import pymupdf as fitz

    out = Extracted()
    doc = fitz.open(path)
    seen: set[str] = set()
    for pno, page in enumerate(doc, start=1):
        if max_pages and pno > max_pages:
            log.warning("%s: only the first %d pages were indexed", path.name, max_pages)
            break
        text = _clean(page.get_text("text"))
        if len(text) >= scanned_chars:
            out.texts.append(TextItem(text, str(path), pno))
        else:  # scanned / image-only page: rasterise it, OCR will supply the text
            pix = page.get_pixmap(dpi=150)
            png = pix.tobytes("png")
            im = _img_from_bytes(png)
            if im is not None:
                out.images.append(ImageItem(im, str(path), pno, sha1_bytes(png), {"scanned_page": True}))
            continue
        # Vector graphics (plots, flowcharts, PDF-drawn tables) are not returned by
        # get_images(). Rasterise pages that contain drawing commands so retrieval
        # and the VLM can inspect diagrams even in born-digital scientific PDFs.
        # Do not OCR this extra rendering: its native text layer is already indexed.
        if page.get_drawings():
            pix = page.get_pixmap(dpi=120, alpha=False)
            png = pix.tobytes("png")
            im = _img_from_bytes(png)
            if im is not None:
                out.images.append(ImageItem(im, str(path), pno, sha1_bytes(png),
                                            {"page_render": True}))

        for info in page.get_images(full=True):
            xref = info[0]
            try:
                raw = doc.extract_image(xref)["image"]
                im = _img_from_bytes(raw)
            except Exception:
                raw, im = b"", None
            if im is None:
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    raw = pix.tobytes("png")
                    im = _img_from_bytes(raw)
                except Exception as e:
                    log.warning("%s p%d: skipping image xref %d (%s)", path.name, pno, xref, e)
                    continue
            if im is None or min(im.size) < min_px:
                continue
            h = sha1_bytes(raw)
            if f"{pno}:{h}" in seen:
                continue
            seen.add(f"{pno}:{h}")
            out.images.append(ImageItem(im, str(path), pno, h))
    return out


def load_docx(path: Path, min_px: int) -> Extracted:
    import docx
    from docx.oxml.ns import qn

    out = Extracted()
    d = docx.Document(str(path))
    body: list[str] = []
    # walk body in document order so tables stay where they were
    for el in d.element.body.iterchildren():
        if el.tag == qn("w:p"):
            t = "".join(n.text or "" for n in el.iter(qn("w:t")))
            if t.strip():
                body.append(t.strip())
        elif el.tag == qn("w:tbl"):
            rows = []
            for tr in el.iter(qn("w:tr")):
                cells = []
                for tc in tr.iter(qn("w:tc")):
                    cells.append(" ".join("".join(n.text or "" for n in p.iter(qn("w:t")))
                                          for p in tc.iter(qn("w:p"))).strip())
                rows.append(" | ".join(cells))
            if rows:
                out.texts.append(TextItem("Table:\n" + "\n".join(rows), str(path), None, "table"))
    if body:
        out.texts.insert(0, TextItem(_clean("\n\n".join(body)), str(path)))
    seen: set[str] = set()
    for rel in d.part.rels.values():
        if "image" in rel.reltype and not rel.is_external:
            blob = rel.target_part.blob
            h = sha1_bytes(blob)
            if h in seen:
                continue
            seen.add(h)
            im = _img_from_bytes(blob)
            if im is not None and min(im.size) >= min_px:
                out.images.append(ImageItem(im, str(path), None, h))
    return out


def load_image(path: Path, min_px: int) -> Extracted:
    raw = path.read_bytes()
    im = _img_from_bytes(raw)
    out = Extracted()
    if im is not None and min(im.size) >= min_px:
        out.images.append(ImageItem(im, str(path), None, sha1_bytes(raw)))
    return out


def load_text(path: Path) -> Extracted:
    return Extracted(texts=[TextItem(_clean(path.read_text(encoding="utf-8", errors="replace")), str(path))])


def load_any(path: Path, min_px: int = 64, scanned_chars: int = 30, max_pages: int = 0) -> Extracted:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, min_px, scanned_chars, max_pages)
    if ext == ".docx":
        return load_docx(path, min_px)
    if ext in IMAGE_EXT:
        return load_image(path, min_px)
    if ext in TEXT_EXT:
        return load_text(path)
    raise ValueError(f"unsupported file type: {path}")


def iter_files(paths: list[str | Path]):
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED and not any(
                        part.startswith(".") for part in f.relative_to(p).parts):
                    yield f
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED:
                yield p
            else:
                log.warning("skipping unsupported file %s", p)
        else:
            raise FileNotFoundError(p)
