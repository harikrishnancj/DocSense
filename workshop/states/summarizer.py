from langsmith import traceable
from model.model import model1
from states.doc_state import DocState

@traceable(name="summarizer")
def Summarizer(state: DocState):
    text = " ".join([doc.text for doc in state.documents])
    result = model1.invoke(f"Summarize this text:\n{text}").content
    state.summary = result
    return state
