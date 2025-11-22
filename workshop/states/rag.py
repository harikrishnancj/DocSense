from langsmith import traceable
from model.model import model1
from states.doc_state import DocState

@traceable(name="rag",run_type='retriever')
def Rag(state: DocState):
    if not state.user_query or not state.index:
        return state
    retriever = state.index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(state.user_query)
    context = "\n".join([n.text for n in nodes])
    response = model1.invoke(f"Answer using context:\n{context}\nQuestion: {state.user_query}")
    state.rag_response = response.content
    return state
