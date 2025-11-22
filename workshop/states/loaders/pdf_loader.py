from __future__ import annotations

import io
from typing import Any, Dict, List

import fitz  # PyMuPDF
from PIL import Image
from llama_index.core import Document

from states.loaders.utils import (
    analyze_image_with_lvm,
    page_to_pil,
    run_ocr_on_pil,
    save_pil_image,
)

try:
    import camelot  # type: ignore

    _HAS_CAMELOT = True
except Exception:
    camelot = None
    _HAS_CAMELOT = False


def load_pdf(path: str, filename: str, artifacts: Dict[str, List[Any]] | None = None) -> List[Document]:
    docs: List[Document] = []
    try:
        pdf = fitz.open(path)
    except Exception as e:
        print(f"Failed to open PDF {filename}: {e}")
        return docs

    full_text_chunks: List[str] = []
    for page_num in range(len(pdf)):
        try:
            page = pdf[page_num]
            page_text = page.get_text("text") or ""
            if not page_text.strip():
                pil_page = page_to_pil(page)
                if pil_page:
                    ocr_text = run_ocr_on_pil(pil_page)
                    page_text = ocr_text or page_text
            full_text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page_text.strip()}")
        except Exception as e:
            print(f"PDF page read error {filename} page {page_num + 1}: {e}")
            continue

        try:
            images_info = page.get_images(full=True)
        except Exception:
            images_info = []

        for img_index, img in enumerate(images_info):
            try:
                xref = img[0]
                base = pdf.extract_image(xref)
                image_bytes = base.get("image")
                if not image_bytes:
                    continue
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_path = save_pil_image(pil_img, f"{filename}_p{page_num + 1}_i{img_index}")
                caption, insights = analyze_image_with_lvm(pil_img)
                ocr_text = run_ocr_on_pil(pil_img)

                if artifacts is not None:
                    artifacts["extracted_images"].append(img_path)
                    artifacts["image_descriptions"].append(caption)
                    artifacts["image_insights"].append(insights)

                docs.append(
                    Document(
                        text=(
                            "[PDF IMAGE]\n"
                            f"Filename:{filename}\nPage:{page_num + 1}\n"
                            f"ImageIndex:{img_index}\nCaption:{caption}\nInsights:{insights}\nOCR:{ocr_text}"
                        ),
                        metadata={
                            "filename": filename,
                            "type": "pdf-image",
                            "page": page_num + 1,
                            "image_index": img_index,
                            "image_path": img_path,
                            "caption": caption,
                            "insights": insights,
                            "ocr": ocr_text,
                        },
                    )
                )
            except Exception as e:
                print(f"Warning: image extraction failed for {filename} page {page_num + 1} img {img_index}: {e}")
                continue

    if _HAS_CAMELOT and camelot:
        try:
            tables_texts = []
            tlist = camelot.read_pdf(path, pages="all", flavor="lattice")
            for t in tlist:
                if not t.df.empty:
                    tables_texts.append((t.df.to_string(), t.df.to_dict()))
            if not tables_texts:
                tlist = camelot.read_pdf(path, pages="all", flavor="stream")
                for t in tlist:
                    if not t.df.empty:
                        tables_texts.append((t.df.to_string(), t.df.to_dict()))
            for idx, (tbl_text, tbl_dict) in enumerate(tables_texts):
                docs.append(
                    Document(
                        text=f"[PDF TABLE]\nFilename:{filename}\nTableIndex:{idx}\n{tbl_text}",
                        metadata={"filename": filename, "type": "pdf-table", "table_index": idx},
                    )
                )
                if artifacts is not None:
                    artifacts["extracted_tables"].append(tbl_dict)
        except Exception as e:
            print(f"Camelot extraction failed for {filename}: {e}")

    docs.append(
        Document(
            text="".join(full_text_chunks) or "[NO_TEXT]",
            metadata={"filename": filename, "type": "pdf-text"},
        )
    )
    pdf.close()
    return docs
