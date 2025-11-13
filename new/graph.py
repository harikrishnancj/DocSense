from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from states import DocState, Loader, build_index, EntityExtractor, Summarizer, Visualizer, Rag, Conditions

graph = StateGraph(DocState)

graph.add_node("load_file", Loader)
graph.add_node("build_index", build_index)
graph.add_node("extract_entities", EntityExtractor)
graph.add_node("summarize_text", Summarizer)
graph.add_node("rag", Rag)
graph.add_node("visualize_entities", Visualizer)

graph.add_edge(START, "load_file")
graph.add_edge("load_file", "build_index")
graph.add_edge("build_index", "extract_entities")

# Conditional branching
def choose_path(state: DocState):
    return "use_rag" if getattr(state, "use_rag", False) else "default"

graph.add_conditional_edges("extract_entities", choose_path, {
    "use_rag": "rag",
    "default": "summarize_text"
})

graph.add_edge("rag", "visualize_entities")
graph.add_edge("summarize_text", "visualize_entities")
graph.add_edge("visualize_entities", END)

app = graph.compile(checkpointer=MemorySaver())
