from __future__ import annotations

from typing import List

import pandas as pd
from llama_index.core import Document


def load_excel(path: str, filename: str) -> List[Document]:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        docs: List[Document] = []
        for sheet_name, df in sheets.items():
            docs.append(
                Document(text=df.to_string(), metadata={"filename": filename, "type": "excel", "sheet": sheet_name})
            )
        return docs
    
    except Exception as e:
        print(f"Failed to load Excel {filename}: {e}")
        return []
