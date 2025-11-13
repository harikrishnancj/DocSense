from llama_index.core import VectorStoreIndex, StorageContext
from model.model import model2
from states.doc_state import DocState
import os

PERSIST_DIR = "./index_storage"

def build_index(state: DocState) -> DocState:
    """
    Build a vector store index from documents and persist it to disk.
    If a persisted index already exists, it will be loaded instead.
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Check if index exists on disk
    if os.path.exists(os.path.join(PERSIST_DIR, "storage.json")):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = VectorStoreIndex.load_from_storage(storage_context)
        print("✅ Loaded existing index from disk.")
    else:
        # Build and persist new index
        index = VectorStoreIndex.from_documents(state.documents, embed_model=model2)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Built and persisted new index.")

    # Save reference in state
    state.index = index
    return state
