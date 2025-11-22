from langsmith import traceable
from llama_index.core import VectorStoreIndex, StorageContext
from model.model import model2
from states.doc_state import DocState
import os

PERSIST_DIR = "./index_storage"

@traceable(name="indexer")
def build_index(state: DocState) -> DocState:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    if os.path.exists(os.path.join(PERSIST_DIR, "storage.json")):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = VectorStoreIndex.load_from_storage(storage_context)
        print("✅ Loaded existing index.")
    else:
        index = VectorStoreIndex.from_documents(state.documents, embed_model=model2)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Built and persisted new index.")
    state.index = index
    return state
