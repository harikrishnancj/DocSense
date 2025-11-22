from __future__ import annotations

from typing import List

from llama_index.core import Document


def load_txt(path: str, filename: str) -> List[Document]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
        return [Document(text=text or "[EMPTY TXT]", metadata={"filename": filename, "type": "txt"})]
    except Exception as e:
        print(f"Failed to load TXT {filename}: {e}")
        return []