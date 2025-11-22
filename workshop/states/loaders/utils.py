from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image

try:
    from openai import OpenAI

    client = OpenAI()
except Exception:
    client = None

try:
    from paddleocr import PaddleOCR

    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    _PADDLE_AVAILABLE = True
except Exception:
    paddle_ocr = None
    _PADDLE_AVAILABLE = False

try:
    import pytesseract

    _PYTESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False


__all__ = [
    "client",
    "EXTRACTED_IMG_DIR",
    "save_pil_image",
    "run_ocr_on_pil",
    "extract_numeric_signals",
    "page_to_pil",
    "analyze_image_with_lvm",
    "_PADDLE_AVAILABLE",
    "_PYTESSERACT_AVAILABLE",
]


EXTRACTED_IMG_DIR = os.path.join("uploaded_docs", "extracted_images")
os.makedirs(EXTRACTED_IMG_DIR, exist_ok=True)


def save_pil_image(pil_img: Image.Image, filename_hint: str) -> str:
    os.makedirs(EXTRACTED_IMG_DIR, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z._-]", "_", filename_hint)[:120]
    tmp_path = os.path.join(EXTRACTED_IMG_DIR, f"{safe_name}.png")
    pil_img.save(tmp_path, format="PNG")
    return tmp_path


def run_ocr_on_pil(pil_img: Image.Image) -> str:
    try:
        if _PADDLE_AVAILABLE and paddle_ocr:
            import numpy as np

            arr = np.array(pil_img.convert("RGB"))
            ocr_res = paddle_ocr.ocr(arr, cls=True)
            texts = [line[1][0] for page in ocr_res for line in page]
            return "\n".join(texts).strip()
        if _PYTESSERACT_AVAILABLE and pytesseract:
            text = pytesseract.image_to_string(pil_img)
            return text.strip()
    except Exception:
        pass
    return ""


def extract_numeric_signals(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    perc = re.findall(r"\b\d+(?:\.\d+)?%\b", text)
    currency = re.findall(
        r"(?:[\$\£\€\₹]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|EUR|INR|GBP))",
        text,
    )
    numbers = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
    return {
        "years": years,
        "percentages": perc,
        "currency": currency,
        "numbers": numbers,
    }


def page_to_pil(page: fitz.Page) -> Image.Image | None:
    try:
        pix = page.get_pixmap(alpha=False)
        buf = io.BytesIO(pix.tobytes("png"))
        return Image.open(buf).convert("RGB")
    except Exception:
        return None


def analyze_image_with_lvm(pil_image: Image.Image) -> Tuple[str, str]:
    if client is None:
        return "[Vision client unavailable]", ""

    def _extract_response_text(resp: Any) -> str:
        content = getattr(resp.choices[0].message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts).strip()
        return ""

    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

        try:
            resp1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in one concise sentence."},
                            payload,
                        ],
                    }
                ],
                max_tokens=180,
            )
            caption = _extract_response_text(resp1) or "[No caption generated]"
        except Exception:
            caption = "[No caption generated]"

        try:
            resp2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Give 2-3 short insights or observations about this image."},
                            payload,
                        ],
                    }
                ],
                max_tokens=250,
            )
            insights = _extract_response_text(resp2)
        except Exception:
            insights = ""

        return caption, insights
    except Exception as e:
        return f"[VisionError] {e}", ""

