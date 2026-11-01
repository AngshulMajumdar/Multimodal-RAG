"""Builds a small mixed-modality corpus for trying / testing the pipeline.

Facts are planted so that each one is only answerable through a specific modality:
  * report.pdf p1  - plain text layer
  * report.pdf p2  - Asia-Pacific revenue figure exists ONLY inside an embedded chart image (needs OCR)
  * report.pdf p3  - scanned page, no text layer (needs page rasterisation + OCR)
  * handbook.docx  - paragraphs, a table, and an embedded photo
  * images/        - real photographs (scikit-image sample data) + a text screenshot
  * notes.md       - unrelated distractor text

usage: python examples/make_demo_corpus.py OUT_DIR
"""
from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = font_manager.findfont("DejaVu Sans")


def _text_image(lines: list[str], size=(1240, 1754), font_size=34, margin=90) -> Image.Image:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, font_size)
    y = margin
    for line in lines:
        for w in textwrap.wrap(line, 52) or [""]:
            d.text((margin, y), w, fill="black", font=f)
            y += int(font_size * 1.5)
        y += font_size // 2
    return im


def _png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def chart_png() -> bytes:
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    vals = [38.5, 27.1, 41.2, 9.8]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    bars = ax.bar(regions, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}", ha="center", fontsize=13)
    ax.set_title("Revenue by region FY2025 (USD million)", fontsize=14)
    ax.set_ylim(0, 50)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    return b.getvalue()


def make_pdf(path: Path):
    import pymupdf as fitz

    doc = fitz.open()
    p = doc.new_page()
    p.insert_textbox(fitz.Rect(60, 60, 540, 780), (
        "Nimbus Robotics Annual Report 2025\n\n"
        "Nimbus Robotics designs autonomous warehouse robots. In fiscal year 2025 total revenue reached "
        "116.6 million US dollars, up 22 percent from the previous year. The company employs 640 people "
        "across offices in Pune, Rotterdam and Austin.\n\n"
        "The flagship product is the Nimbus R2, a mobile picking robot with a payload of 25 kilograms. "
        "The chief executive officer is Meera Iyer, who founded the company in 2016."), fontsize=11)
    p = doc.new_page()
    p.insert_textbox(fitz.Rect(60, 60, 540, 200), (
        "Regional performance\n\nGrowth was strongest in one region this year; the chart below breaks down "
        "revenue by geography."), fontsize=11)
    p.insert_image(fitz.Rect(60, 210, 540, 500), stream=chart_png())
    p = doc.new_page()  # scanned page: image only, no text layer
    scan = _text_image([
        "Appendix B - Service and Warranty",
        "Every Nimbus R2 unit ships with a 36-month limited warranty covering drive motors and the lidar module.",
        "Battery packs are covered for 18 months or 4000 charge cycles, whichever comes first.",
    ])
    p.insert_image(p.rect, stream=_png(scan))
    doc.save(path)


def make_docx(path: Path, photo: Image.Image):
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.add_heading("Warehouse Operations Handbook", 1)
    d.add_paragraph("All staff must complete forklift certification before operating lifting equipment. "
                    "Night shifts start at 22:00 and end at 06:00.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Equipment", "Maximum load"
    for k, v in [("Electric forklift", "2500 kg"), ("Pallet jack", "1800 kg"), ("Nimbus R2 robot", "25 kg")]:
        r = t.add_row().cells
        r[0].text, r[1].text = k, v
    d.add_paragraph("The staff break room on level 2 has a coffee machine; the photo below shows it.")
    b = io.BytesIO()
    photo.save(b, format="PNG")
    b.seek(0)
    d.add_picture(b, width=Inches(3))
    d.save(path)


def build(out: str | Path) -> Path:
    from skimage import data as D

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    make_pdf(out / "report.pdf")
    make_docx(out / "handbook.docx", Image.fromarray(D.coffee()))
    Image.fromarray(D.chelsea()).save(out / "images" / "office_cat.jpg", quality=92)
    Image.fromarray(D.rocket()).save(out / "images" / "launch.jpg", quality=92)
    Image.fromarray(D.astronaut()).save(out / "images" / "visitor.png")
    _text_image(["NOTICE", "The server room door code is rotated every Monday at 09:00.",
                 "Contact facilities on extension 4417."], size=(1100, 520), font_size=36, margin=50
                ).save(out / "images" / "notice_board.png")
    (out / "notes.md").write_text(
        "# Reading notes\n\nPhotosynthesis converts light energy into chemical energy in chloroplasts. "
        "The Calvin cycle fixes carbon dioxide into sugars.\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "demo_corpus"))
