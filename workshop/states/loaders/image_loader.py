from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image
from llama_index.core import Document

from states.loaders.utils import analyze_image_with_lvm, run_ocr_on_pil, save_pil_image


def load_image(path: str, filename: str, artifacts: Dict[str, List[Any]] | None = None) -> List[Document]:
    try:
        pil_img = Image.open(path).convert("RGB")
        img_path = save_pil_image(pil_img, filename)
        caption, insights = analyze_image_with_lvm(pil_img)
        ocr_text = run_ocr_on_pil(pil_img)
        if artifacts is not None:
            artifacts["extracted_images"].append(img_path)
            artifacts["image_descriptions"].append(caption)
            artifacts["image_insights"].append(insights)
        return [
            Document(
                text=f"[IMAGE]\nFilename:{filename}\nCaption:{caption}\nInsights:{insights}\nOCR:{ocr_text}",
                metadata={"filename": filename, "type": "image", "image_path": img_path},
            )
        ]
    except Exception as e:
        print(f"Failed to load image {filename}: {e}")
        return []

