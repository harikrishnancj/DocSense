import os
import json
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from model.model import model1  

def extract_numeric_insights(text):
    """Extract numeric patterns from text for charts."""
    if not text:
        return {}
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    percentages = re.findall(r"\b\d+(?:\.\d+)?%\b", text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return {"years": years, "percentages": percentages, "numbers": numbers}

def Visualizer(state):
  
    # Decide text to visualize
    text_source = ""
    if getattr(state, "rag_response", None):
        text_source = state.rag_response
    elif getattr(state, "summary", None):
        text_source = state.summary
    else:
        text_source = " ".join([doc.text for doc in state.documents])

    # 2️⃣ Entity counts (if available)
    visuals_data = {}
    if state.entities:
        labels = [ent["label"] for ent in state.entities]
        counts = dict(Counter(labels))
        visuals_data["entity_distribution"] = counts

    # 3️⃣ Summary / RAG text insights
    visuals_data["text_source"] = text_source
    visuals_data["numeric_insights"] = extract_numeric_insights(text_source)

    if not visuals_data:
        print("⚠️ No data available for visualization.")
        return state

    # 4️⃣ Ask LLM for chart suggestions
    prompt = f"""
    You are a visualization reasoning assistant.
    Analyze the following data (entities + text + numeric patterns)
    and propose multiple visualizations in JSON.

    Output strictly in this format:
    {{
        "charts": [
            {{
                "type": "<bar|pie|line|scatter|histogram|wordcloud>",
                "data": {{
                    "labels": [...],
                    "values": [...],
                    "title": "<chart title>"
                }}
            }}
        ]
    }}

    Data:
    {json.dumps(visuals_data, indent=2)}
    """
    decision = model1.invoke(prompt).content
    os.makedirs("visuals", exist_ok=True)
    state.visuals = {"charts": []}

    # 5️⃣ Parse LLM JSON
    try:
        try:
            vis_plan = json.loads(decision)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', decision)
            vis_plan = json.loads(match.group(0)) if match else {}
        charts = vis_plan.get("charts", [])

        # 6️⃣ Plot each chart
        for i, vis in enumerate(charts, 1):
            chart_type = vis.get("type", "bar").lower()
            data = vis.get("data", {})
            labels = data.get("labels", [])
            values = data.get("values", [])
            title = data.get("title", f"Chart {i}")

            if not labels or not values:
                continue

            plt.figure(figsize=(6, 4))
            plt.title(title)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=30, ha="right")

            try:
                if chart_type == "bar":
                    plt.bar(labels, values, color=plt.cm.Paired.colors)
                    for idx, val in enumerate(values):
                        plt.text(idx, val, str(val), ha="center", va="bottom")
                elif chart_type == "line":
                    plt.plot(labels, values, marker="o", color="tab:blue")
                elif chart_type == "pie":
                    plt.pie(values, labels=labels, autopct="%1.1f%%")
                elif chart_type == "scatter":
                    plt.scatter(labels, values, color="tab:green")
                elif chart_type == "histogram":
                    plt.hist(values, bins=10, color="tab:purple")
                elif chart_type == "wordcloud":
                    text = " ".join([f"{k} " * int(v) for k, v in zip(labels, values)])
                    wc = WordCloud(width=800, height=400).generate(text)
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
            except Exception as e:
                print(f"❌ Error plotting {chart_type}: {e}")
                continue

            file_path = f"visuals/chart_{i}_{chart_type}.png"
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            state.visuals["charts"].append({"title": title, "type": chart_type, "file": file_path})

    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        if state.summary:
            wc = WordCloud(width=800, height=400).generate(state.summary)
            fallback_path = "visuals/summary_wordcloud.png"
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(fallback_path)
            plt.close()
            state.visuals["charts"].append({"title": "Summary WordCloud", "type": "wordcloud", "file": fallback_path})

    print(f"✅ Generated {len(state.visuals['charts'])} visualization(s).")
    return state
