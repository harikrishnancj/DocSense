from langgraph.graph import StateGraph, START, END
from states.doc_state import DocState
from states.loader import Loader
from states.indexer import build_index
from states.entities import EntityExtractor
from states.rag import Rag
from states.summarizer import Summarizer
from states.visualizer import Visualizer

graph = StateGraph(DocState)

graph.add_node("load_file", Loader)
graph.add_node("build_index", build_index)
graph.add_node("rag", Rag)
graph.add_node("summarize", Summarizer)
graph.add_node("entities", EntityExtractor)
graph.add_node("visualizer", Visualizer)

# START → Load
graph.add_edge(START, "load_file")
def choose_index(state: DocState):
    return "build_index" if state.use_rag else "summarize"

graph.add_conditional_edges(
    "load_file",
    choose_index,
    {
        "build_index": "build_index",
        "summarize": "summarize"
    }
)

graph.add_edge("build_index", "rag")

graph.add_edge("rag", "entities")
graph.add_edge("summarize", "entities")
graph.add_edge("entities", "visualizer")
graph.add_edge("visualizer", END)

app_graph = graph.compile(checkpointer=None)
