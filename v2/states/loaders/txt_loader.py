from llama_index.core import Document


def load_text(path, name):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    return Document(text=text, metadata={"source": path})