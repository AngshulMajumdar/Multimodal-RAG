"""Fast tests: no neural models except the bundled OCR."""
from pathlib import Path

from mrag.config import Config
from mrag.generate import ImageHit, TextHit, build_prompt
from mrag.ingest import chunk_text, iter_files, load_any
from mrag.pipeline import rrf


def test_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("MRAG_TOP_K_TEXT", "9")
    monkeypatch.setenv("MRAG_LLM_VISION", "false")
    monkeypatch.setenv("MRAG_CLIP_MIN_SCORE", "0.3")
    c = Config.load(llm_model="gemma3:4b")
    assert c.top_k_text == 9 and c.llm_vision is False and c.clip_min_score == 0.3 and c.llm_model == "gemma3:4b"


def test_chunking_covers_all_words_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_words=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunking_paragraphs():
    text = "\n\n".join(" ".join(["p%d" % i] * 30) for i in range(10))
    chunks = chunk_text(text, chunk_words=100, overlap=0)
    assert len(chunks) == 4 and sum(len(c.split()) for c in chunks) == 300


def test_chunking_overlap_with_a_full_paragraph():
    text = " ".join(["a"] * 170) + "\n\n" + " ".join(["b"] * 180)
    chunks = chunk_text(text, chunk_words=180, overlap=40)
    assert all(len(c.split()) <= 180 for c in chunks)
    assert sum(c.split().count("b") for c in chunks) == 180


def test_rrf_prefers_agreement():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"], ["b"]])
    assert fused[0][0] == "b"


def test_pdf_loader_text_chart_and_scanned_page(corpus):
    ex = load_any(corpus / "report.pdf")
    assert [t.page for t in ex.texts] == [1, 2]                  # page 3 has no text layer
    assert "Meera Iyer" in ex.texts[0].text
    assert "41.2" not in " ".join(t.text for t in ex.texts)       # chart value only exists in the image
    pages = sorted((i.page, i.meta.get("scanned_page", False)) for i in ex.images)
    assert (2, False) in pages and (3, True) in pages


def test_docx_loader_table_and_image(corpus):
    ex = load_any(corpus / "handbook.docx")
    kinds = {t.kind for t in ex.texts}
    assert kinds == {"text", "table"}
    table = next(t.text for t in ex.texts if t.kind == "table")
    assert "Electric forklift | 2500 kg" in table
    assert len(ex.images) == 1 and ex.images[0].image.size == (600, 400)


def test_iter_files_skips_hidden(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.txt").write_text("secret")
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.xyz").write_text("nope")
    assert [p.name for p in iter_files([tmp_path])] == ["a.txt"]


def test_ocr_reads_scanned_page_and_chart(corpus):
    from mrag.models import OCR

    ocr = OCR()
    ex = load_any(corpus / "report.pdf")
    by_page = {i.page: ocr(i.image) for i in ex.images}
    assert "36-month" in by_page[3] and "warranty" in by_page[3].lower()
    assert "41.2" in by_page[2] and "Asia-Pacific" in by_page[2]
    notice = load_any(corpus / "images" / "notice_board.png").images[0].image
    assert "Monday" in ocr(notice)


def test_prompt_numbering():
    p = build_prompt("q?", [TextHit("1", "alpha", "/x/a.pdf", 2, "text", 0.5),
                            TextHit("2", "beta", "/x/b.png", None, "ocr", 0.4)],
                     [ImageHit("i", "/tmp/i.png", "/x/b.png", None, "beta", 0.3)])
    assert "[1] (a.pdf, p.2)\nalpha" in p and "[2] (b.png, OCR of an image)" in p
    assert "[Img 1] b.png | text in image: beta" in p and p.rstrip().endswith("Answer:")


def test_pdf_loader_keeps_vector_drawn_figures(tmp_path):
    """A plotted line is vector artwork, invisible to page.get_images()."""
    import pymupdf
    pdf = tmp_path / "vector.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((70, 70), "Plot A demonstrates the relationship between pressure and temperature.")
    page.draw_line((75, 300), (400, 300), color=(0, 0, 0), width=2)
    page.draw_line((75, 300), (75, 120), color=(0, 0, 0), width=2)
    page.draw_line((75, 280), (350, 140), color=(0, 0, 1), width=3)
    assert page.get_drawings() and not page.get_images(full=True)
    doc.save(pdf)
    doc.close()
    out = load_any(pdf)
    assert out.texts and "Plot A demonstrates" in out.texts[0].text
    assert any(im.meta.get("page_render") and im.page == 1 for im in out.images)
