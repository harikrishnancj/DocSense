from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from pydantic import BaseModel
from typing import List, Dict 
from collections import Counter
import spacy
from model import model1, model2
from langchain_core.prompts import PromptTemplate
from visual_utils import Visualizer


nlp = spacy.load("en_core_web_sm")

'''class DocState(BaseModel):
    folder_path: str = ""
    documents: List = []
    entities: List[Dict] = []
    summary: str = ""
    index: object = None
    visuals: Dict = {}
    user_query: str = ""
    rag_response: str = ""'''
from typing import Any

class DocState(BaseModel):
    folder_path: str = ""
    documents: List = []
    entities: List[Dict] = []
    summary: str = ""
    index_path: str = ""
    visuals: Dict = {}
    user_query: str = ""
    rag_response: str = ""
    index: Any = None   # ✅ add this line


def Loader(state: DocState) -> DocState:
    reader = SimpleDirectoryReader(state.folder_path)
    state.documents = reader.load_data()
    return state

from llama_index.core import StorageContext, load_index_from_storage

def build_index(state: DocState) -> DocState:
    persist_dir = "./index_storage"

    # Build and persist the index
    index = VectorStoreIndex.from_documents(state.documents, embed_model=model2)
    index.storage_context.persist(persist_dir=persist_dir)

    # Load the index from storage
    state.index = persist_dir
    return state



def EntityExtractor(state: DocState):
    text = " ".join([doc.text for doc in state.documents])#for multiple doc
    doc = nlp(text)
    state.entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
    return state


def Summarizer(state: DocState):
    text = " ".join([doc.text for doc in state.documents])
    prompt = PromptTemplate.from_template("Summarize the following document set:\n\n{text}")
    chain = prompt | model1
    result = chain.invoke({"text": text})
    state.summary = getattr(result, "content", str(result))
    return state


def Conditions(state: DocState) -> DocState:
    if state.user_query and state.user_query.strip():
        state.use_rag = True
    else:
        state.use_rag = False
    return state

'''def Visualizer(state: DocState):
    labels = [ent["label"] for ent in state.entities]
    counts = dict(Counter(labels))
    state.visuals = {"entity_distribution": counts}
    return state'''

def Rag(state: DocState):
    if not state.user_query:
        return state

    print(f"💬 Running retrieval for query: {state.user_query}")
    retriever = state.index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(state.user_query)

    context = "\n\n".join([n.text for n in nodes])
    prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {state.user_query}"

    state.rag_response = model1.invoke(prompt).content
    return state



