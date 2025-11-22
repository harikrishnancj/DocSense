from langsmith import traceable
import spacy
from states.doc_state import DocState

nlp = spacy.load("en_core_web_sm")

@traceable(name="entity_extractor")
def EntityExtractor(state: DocState):
    text = state.rag_response or state.summary or " ".join([doc.text for doc in state.documents])
    doc = nlp(text)
    state.entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
    return state
