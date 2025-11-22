from __future__ import annotations

from typing import List

import pandas as pd
from llama_index.core import Document


def load_csv(path: str, filename: str) -> List[Document]:
    try:
        df = pd.read_csv(path)
        txt = df.to_string()
        return [Document(text=txt, metadata={"filename": filename, "type": "csv"})]
    except Exception as e:
        print(f"Failed to load CSV {filename}: {e}")
        return []
