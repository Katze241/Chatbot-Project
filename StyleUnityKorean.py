!pip install pymupdf
!apt-get update -qq
!apt-get install -qq fonts-nanum fonts-nanum-extra fonts-noto-cjk

import fitz
from google.colab import files

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_NAME = "NanumGothicCustom"

def unify_pdf_style(
    input_pdf_path: str,
    output_pdf_path: str,
    fontfile: str = FONT_PATH,
    fontsize: float = 10.0,
    color: tuple = (0, 0, 0)
):
    if max(color) > 1:
        color = tuple(c/255 for c in color)

    src = fitz.open(input_pdf_path)
    dst = fitz.open()

    for page in src:
        new_page = dst.new_page(width=page.rect.width, height=page.rect.height)

        new_page.insert_font(fontname=FONT_NAME, fontfile=fontfile)

        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    new_page.insert_text(
                        span["origin"],
                        span["text"],
                        fontname=FONT_NAME,
                        fontsize=fontsize,
                        color=color
                    )

    dst.save(output_pdf_path)
    src.close(); dst.close()

uploaded = files.upload()
input_fname = next(iter(uploaded.keys()))
output_fname = "styled_output_fixed.pdf"

unify_pdf_style(
    input_pdf_path=input_fname,
    output_pdf_path=output_fname,
    fontfile=FONT_PATH,
    fontsize=10.0,
    color=(0, 0, 0)
)

files.download(output_fname)
