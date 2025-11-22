import pandas as pd
from llama_index.core import Document

def load_csv(path, name):
    df = pd.read_csv(path)
    return Document(
        text=f"[CSV FILE: {name}]\n{df.to_string()}",
        metadata={"source": path}
    )
