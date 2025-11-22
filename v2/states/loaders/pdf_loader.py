import fitz  # PyMuPDF
import pandas as pd
from llama_index.core import Document
import os
from PIL import Image
import io
from states.loaders.lvm import analyze_image_with_lvm


# -------------------------
# Table Extraction
# -------------------------
def extract_pdf_tables(pdf_path):
    """Extract tables using Camelot if installed."""
    tables = []
    try:
        import camelot
        t = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for table in t:
            tables.append(table.df)
    except Exception:
        pass  # ignore if camelot not installed
    return tables


# -------------------------
# Safe Pixmap Save
# -------------------------
def save_pixmap_as_png(pix, out_path):
    """Fix CMYK images by converting to RGB."""
    if pix.n < 5:  # RGB
        pix.save(out_path)
    else:
        pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(out_path)


# -------------------------
# FULL PDF LOADER
# -------------------------
def load_pdf(path, state=None):
    """
    Loads:
      ✓ PDF text
      ✓ PDF tables (-> markdown)
      ✓ PDF images (-> LVM description + insights)
    Returns: list[Document] for LlamaIndex
    """

    docs = []
    pdf = fitz.open(path)

    # -------------------------------------------------------
    # 1. TEXT EXTRACTION
    # -------------------------------------------------------
    text = ""
    for page in pdf:
        text += page.get_text()

    docs.append(Document(text=text, metadata={"source": path}))

    # -------------------------------------------------------
    # 2. TABLE EXTRACTION
    # -------------------------------------------------------
    tables = extract_pdf_tables(path)

    extracted_tables_for_state = []

    for idx, df in enumerate(tables):
        table_markdown = df.to_markdown()

        docs.append(Document(
            text=table_markdown,
            metadata={
                "source": path,
                "type": "table",
                "table_index": idx
            }
        ))

        extracted_tables_for_state.append(df.to_dict())

    # -------------------------------------------------------
    # 3. IMAGE EXTRACTION + Vision LLM
    # -------------------------------------------------------
    images_folder = "./extracted_images"
    os.makedirs(images_folder, exist_ok=True)

    extracted_images = []
    image_descriptions = []
    image_insights = []

    for page_idx, page in enumerate(pdf):
        for img_idx, img in enumerate(page.get_images()):
            xref = img[0]
            pix = fitz.Pixmap(pdf, xref)

            img_path = f"{images_folder}/{os.path.basename(path)}_{page_idx}_{img_idx}.png"
            save_pixmap_as_png(pix, img_path)

            extracted_images.append(img_path)

            # --- Load into PIL for LLM ---
            pil_image = Image.open(img_path)

            # --- LVM Vision Description & Insights ---
            description, insights = analyze_image_with_lvm(pil_image)

            image_descriptions.append(description)
            image_insights.append(insights)

            # Add as LlamaIndex document
            docs.append(
                Document(
                    text=f"[IMAGE_DESCRIPTION]\n{description}\n\n[INSIGHTS]\n{insights}",
                    metadata={
                        "source_pdf": path,
                        "source_image": img_path,
                        "type": "image_description"
                    }
                )
            )


    if state:
        state.extracted_tables = extracted_tables_for_state
        state.extracted_images = extracted_images
        state.image_descriptions = image_descriptions
        state.image_insights = image_insights

    return docs
