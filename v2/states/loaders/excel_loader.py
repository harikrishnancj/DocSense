import pandas as pd
from llama_index.core import Document

def load_excel(path):
    dfs = pd.read_excel(path, sheet_name=None)
    docs = []

    for sheet, df in dfs.items():
        text = df.to_string()
        docs.append(Document(
            text=f"[EXCEL SHEET: {sheet}]\n{text}",
            metadata={"source": path, "sheet": sheet}
        ))

    return docs
