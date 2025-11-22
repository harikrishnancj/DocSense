import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="DocSense", layout="wide")
st.title("📄 DocSense – AI Document Understanding")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx", "xlsx", "csv", "pptx"])
mode = st.radio("Select Operation:", ["Summarization", "RAG"])

user_query = ""
if mode == "RAG":
    user_query = st.text_input("Ask your question:")

if uploaded_file and st.button("Process"):
    files = {"file": uploaded_file}
    data = {"mode": mode, "user_query": user_query}

    with st.spinner("Processing your document..."):
        try:
            response = requests.post("http://localhost:8000/process/", files=files, data=data)
            response.raise_for_status()
            res = response.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # -------------------------------------
    # Summary or RAG Output
    # -------------------------------------
    if mode == "RAG":
        st.subheader("🧠 RAG Response")
        st.write(res.get("rag_response", "No RAG response"))
    else:
        st.subheader("📝 Summary")
        st.write(res.get("summary", "No summary available"))

    # -------------------------------------
    # Entities
    # -------------------------------------
    st.subheader("🔍 Extracted Entities")
    entities = res.get("entities", [])

    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except:
            entities = []

    if entities:
        st.json(entities)
    else:
        st.write("No entities detected.")

    # -------------------------------------
    # Visuals (Charts etc.)
    # -------------------------------------
    st.subheader("📊 Visual Charts")
    visuals = res.get("visuals", {})

    if isinstance(visuals, str):
        try:
            visuals = json.loads(visuals)
        except:
            visuals = {}

    if isinstance(visuals, list):
        visuals = {"charts": visuals}

    charts = visuals.get("charts", [])

    if charts:
        for chart in charts:
            if isinstance(chart, dict) and "file" in chart:
                st.image(chart["file"])
    else:
        st.write("No charts detected.")

    # -------------------------------------
    # Extracted Images + Vision LLM Descriptions
    # -------------------------------------
    st.subheader("🖼 Extracted Images & LLM Insights")

    extracted_images = res.get("extracted_images", [])
    image_descriptions = res.get("image_descriptions", [])
    image_insights = res.get("image_insights", [])

    if extracted_images:
        for i, img_path in enumerate(extracted_images):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(img_path, caption=f"Image {i+1}")

            with col2:
                st.write("**Description:**")
                st.write(image_descriptions[i] if i < len(image_descriptions) else "")

                st.write("**Insights:**")
                st.write(image_insights[i] if i < len(image_insights) else "")
    else:
        st.write("No images extracted.")

    # -------------------------------------
    # Extracted Tables
    # -------------------------------------
    st.subheader("📑 Extracted Tables")

    tables = res.get("extracted_tables", [])

    if isinstance(tables, str):
        try:
            tables = json.loads(tables)
        except:
            tables = []

    if tables:
        for table_index, table_dict in enumerate(tables):
            st.markdown(f"### Table {table_index + 1}")
            df = pd.DataFrame(table_dict)
            st.dataframe(df, use_container_width=True)
    else:
        st.write("No tables found.")
