from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from states.doc_state import DocState
from states.loader import Loader
from states.indexer import build_index
from states.entities import EntityExtractor
from states.summarizer import Summarizer
from states.rag import Rag
from states.visualizer import Visualizer
from dotenv import load_dotenv
load_dotenv()

# Initialize graph
graph = StateGraph(DocState)

# Add nodes
graph.add_node("load_file", Loader)
graph.add_node("build_index", build_index)
graph.add_node("summarize_text", Summarizer)
graph.add_node("rag", Rag)
graph.add_node("extract_entities", EntityExtractor)
graph.add_node("visualize_entities", Visualizer)

# Base edges
graph.add_edge(START, "load_file")
graph.add_edge("load_file", "build_index")

# Conditional edge: choose operation based on user selection
def choose_operation(state: DocState):
    return "rag" if getattr(state, "use_rag", False) else "summarize_text"

graph.add_conditional_edges(
    "build_index",
    choose_operation,
    {"rag": "rag", "summarize_text": "summarize_text"}
)

# Entities are extracted after chosen operation
graph.add_edge("summarize_text", "extract_entities")
graph.add_edge("rag", "extract_entities")

# Visualization always comes after entities
graph.add_edge("extract_entities", "visualize_entities")
graph.add_edge("visualize_entities", END)

# Compile graph with memory checkpointing
app_graph = graph.compile(checkpointer=None)


#app_graph = graph.compile(checkpointer=MemorySaver())