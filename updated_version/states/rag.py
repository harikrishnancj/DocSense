from states.doc_state import DocState
from model.model import model1


def Rag(state: DocState):
    if not state.user_query or not state.index:
        return state
    retriever = state.index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(state.user_query)
    context = "\n".join([n.text for n in nodes])
    state.rag_response = model1.invoke(f"Answer using context:\n{context}\nQuestion: {state.user_query}").content
    return state