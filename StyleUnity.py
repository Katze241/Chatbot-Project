!pip install pymupdf
import fitz
from google.colab import files


uploaded = files.upload()
filename = next(iter(uploaded))
output_pdf_path = "styled_output.pdf"

font_name = "helv"
font_size = 10
text_color = (0, 0, 0)

doc = fitz.open(filename)
styled_pdf = fitz.open()

for page in doc:
    blocks = page.get_text("dict")["blocks"]

    new_page = styled_pdf.new_page(width=page.rect.width, height=page.rect.height)

    for block in blocks:
        if block['type'] != 0:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                x, y = span["origin"]

                new_page.insert_text((x, y), text,
                                 fontname=font_name,
                                 fontsize=font_size,
                                 color=text_color)

styled_pdf.save(output_pdf_path)
styled_pdf.close()
doc.close()
files.download(output_pdf_path)
