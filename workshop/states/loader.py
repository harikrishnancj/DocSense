"""
DocSense Loader V3 — Universal loader + visual analyzer + OCR + chart signals.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Any, Dict, Iterable, List

from langsmith import traceable
from llama_index.core import Document

from states.doc_state import DocState
from states.loaders.audio_loader import load_audio
from states.loaders.csv_loader import load_csv
from states.loaders.docx_loader import load_docx
from states.loaders.excel_loader import load_excel
from states.loaders.image_loader import load_image
from states.loaders.pdf_loader import load_pdf
from states.loaders.pptx_loader import load_pptx
from states.loaders.txt_loader import load_txt
from states.loaders.utils import extract_numeric_signals


def _safe_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    return metadata if isinstance(metadata, dict) else {}


def _extend_with_numeric_signals(docs: Iterable[Document]) -> None:
    for d in docs:
        try:
            body = getattr(d, "text", "") or ""
            signals = extract_numeric_signals(body)
            meta = _safe_metadata(d.metadata)
            meta.setdefault("numeric_signals", signals)
            d.metadata = meta
        except Exception:
            continue


def _process_zip(full_path: str, artifacts: Dict[str, List[Any]], all_docs: List[Document]) -> None:
    try:
        with zipfile.ZipFile(full_path, "r") as zf:
            temp_dir = tempfile.mkdtemp()
            zf.extractall(temp_dir)
            for root, _, files in os.walk(temp_dir):
                for sub in sorted(files):
                    sp = os.path.join(root, sub)
                    docs: List[Document]
                    lower = sub.lower()
                    if lower.endswith(".pdf"):
                        docs = load_pdf(sp, sub, artifacts)
                    elif lower.endswith(".docx"):
                        docs = load_docx(sp, sub, artifacts)
                    elif lower.endswith((".pptx", ".ppt")):
                        docs = load_pptx(sp, sub, artifacts)
                    elif lower.endswith(".txt"):
                        docs = load_txt(sp, sub)
                    elif lower.endswith(".csv"):
                        docs = load_csv(sp, sub)
                    elif lower.endswith((".xls", ".xlsx")):
                        docs = load_excel(sp, sub)
                    elif lower.endswith((".png", ".jpg", ".jpeg")):
                        docs = load_image(sp, sub, artifacts)
                    elif lower.endswith((".mp3", ".wav", ".m4a")):
                        docs = load_audio(sp, sub)
                    else:
                        continue
                    _extend_with_numeric_signals(docs)
                    all_docs.extend(docs)
    except Exception as e:
        print(f"Zip processing failed for {full_path}: {e}")


@traceable(name="loader")
def Loader(state: DocState) -> DocState:
    """
    Iterate files in state.folder_path, dispatch to format loaders,
    and enrich each Document with numeric signals.
    """
    folder = state.folder_path or "uploaded_docs"
    os.makedirs(folder, exist_ok=True)

    artifacts: Dict[str, List[Any]] = {
        "extracted_images": [],
        "image_descriptions": [],
        "image_insights": [],
        "extracted_tables": [],
    }
    all_docs: List[Document] = []

    for file in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, file)
        if os.path.isdir(full_path):
            continue

        lower = file.lower()
        try:
            if lower.endswith(".pdf"):
                docs = load_pdf(full_path, file, artifacts)
            elif lower.endswith(".docx"):
                docs = load_docx(full_path, file, artifacts)
            elif lower.endswith((".pptx", ".ppt")):
                docs = load_pptx(full_path, file, artifacts)
            elif lower.endswith(".txt"):
                docs = load_txt(full_path, file)
            elif lower.endswith(".csv"):
                docs = load_csv(full_path, file)
            elif lower.endswith((".xls", ".xlsx")):
                docs = load_excel(full_path, file)
            elif lower.endswith((".png", ".jpg", ".jpeg")):
                docs = load_image(full_path, file, artifacts)
            elif lower.endswith((".mp3", ".wav", ".m4a")):
                docs = load_audio(full_path, file)
            elif lower.endswith(".zip"):
                _process_zip(full_path, artifacts, all_docs)
                docs = []
            else:
                docs = load_txt(full_path, file)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            docs = []

        _extend_with_numeric_signals(docs)
        all_docs.extend(docs)

    state.documents = all_docs
    state.extracted_images = artifacts["extracted_images"]
    state.image_descriptions = artifacts["image_descriptions"]
    state.image_insights = artifacts["image_insights"]
    state.extracted_tables = artifacts["extracted_tables"]
    if not state.folder_path:
        state.folder_path = folder

    return state
