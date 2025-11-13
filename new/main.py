from graph import app

if __name__ == "__main__":
    result = app.invoke({
        "folder_path": "docs",
        "user_query": "What are the key topics discussed in the documents?"
    },
    config={"configurable": {"thread_id": "run_1"}})

    print("✅ Pipeline completed.")
    print("Summary:", result["summary"])
    print("RAG Response:", result["rag_response"])
    print("Entities:", result["entities"][:5])
    print("Visuals:", result["visuals"])