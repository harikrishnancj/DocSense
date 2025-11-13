from llama_index.core import SimpleDirectoryReader
from states.doc_state import DocState
import os

def Loader(state: DocState):
    """
    Load all documents from the folder into the state.
    """
    os.makedirs(state.folder_path, exist_ok=True)
    reader = SimpleDirectoryReader(state.folder_path)
    state.documents = reader.load_data()
    return state
