import streamlit as st
import requests
import json

st.title("📄 DocSense: Document Summarization & RAG")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
mode = st.radio("Select Operation:", ["Summarization", "RAG"])

user_query = ""
if mode == "RAG":
    user_query = st.text_input("Ask your question:")

if uploaded_file and st.button("Process"):
    files = {"file": uploaded_file}
    data = {"mode": mode, "user_query": user_query}
    
    try:
        response = requests.post("http://localhost:8000/process/", files=files, data=data)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
    else:
        res = response.json()
        
        # Summary or RAG output
        if mode == "RAG":
            st.subheader("RAG Response")
            st.write(res.get("rag_response", "No RAG response"))
        else:
            st.subheader("Summary")
            st.write(res.get("summary", "No summary available"))

        # Entities
        st.subheader("Entities")
        entities = res.get("entities", [])
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except:
                entities = []
        st.write(entities[:10])

        # Visuals
        st.subheader("Visuals")
        visuals = res.get("visuals", {})
        if isinstance(visuals, str):
            try:
                visuals = json.loads(visuals)
            except:
                visuals = {}

        # If visuals is a list (legacy), wrap into dict
        if isinstance(visuals, list):
            visuals = {"charts": visuals}

        charts = visuals.get("charts", [])
        for v in charts:
            if isinstance(v, dict) and "file" in v:
                st.image(v["file"])
