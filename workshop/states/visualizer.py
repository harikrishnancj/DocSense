from langsmith import traceable
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import json, os, re
from model.model import model1


# ---------------------------------------
# Helper: Numeric extraction from free text
# ---------------------------------------
@traceable(name="numeric_extractor")
def extract_numeric_insights(text):
    """Extract numeric patterns from text."""
    if not text:
        return {}

    years = re.findall(r"\b(19|20)\d{2}\b", text)
    percentages = re.findall(r"\b\d+(?:\.\d+)?%\b", text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)

    return {
        "years": years,
        "percentages": percentages,
        "numbers": numbers,
    }


# ---------------------------------------
# Helper: Auto-chart for extracted tables
# ---------------------------------------
def auto_chart_from_table(table):
    """Generate a simple default chart from a table (list of row dicts)."""

    if not table or not isinstance(table, list):
        return None

    # Convert rows → columns
    columns = {}
    for row in table:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)

    # Determine numeric columns
    numeric_cols = []
    for col, values in columns.items():
        clean_vals = [v for v in values if v not in ["", None]]
        if clean_vals and all(str(v).replace(".", "", 1).isdigit() for v in clean_vals):
            numeric_cols.append(col)

    categorical_cols = [c for c in columns.keys() if c not in numeric_cols]

    # CASE 1 — categorical + numeric → bar chart
    if categorical_cols and numeric_cols:
        x = categorical_cols[0]
        y = numeric_cols[0]

        return {
            "type": "bar",
            "title": f"{y} by {x}",
            "labels": columns[x],
            "values": [float(v) for v in columns[y] if v not in ["", None]],
        }

    # CASE 2 — two numeric columns → scatter
    if len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]

        return {
            "type": "scatter",
            "title": f"{y} vs {x}",
            "labels": [float(v) for v in columns[x] if v not in ["", None]],
            "values": [float(v) for v in columns[y] if v not in ["", None]],
        }

    # CASE 3 — one numeric → histogram
    if len(numeric_cols) == 1:
        col = numeric_cols[0]
        return {
            "type": "histogram",
            "title": f"Distribution of {col}",
            "labels": [],
            "values": [float(v) for v in columns[col] if v not in ["", None]],
        }

    return None  # No usable chart


# ---------------------------------------
# Main Visualizer Node
# ---------------------------------------
def Visualizer(state):

    # ---------------------------------------
    # 1. Extract text source
    # ---------------------------------------
    text_source = (
        state.rag_response
        or state.summary
        or " ".join([doc.text for doc in state.documents])
    )

    visuals_data = {
        "text_source": text_source,
        "numeric_insights": extract_numeric_insights(text_source),
    }

    # ---------------------------------------
    # 2. Add entity distribution
    # ---------------------------------------
    if getattr(state, "entities", None):
        labels = [ent.get("label") for ent in state.entities if "label" in ent]
        visuals_data["entity_distribution"] = dict(Counter(labels))

    # ---------------------------------------
    # 3. Include extracted tables
    # ---------------------------------------
    table_visual_data = []
    if getattr(state, "extracted_tables", None):
        for tbl in state.extracted_tables:
            if isinstance(tbl, list) and len(tbl) > 0:
                table_visual_data.append(tbl)

    visuals_data["tables"] = table_visual_data

    os.makedirs("visuals", exist_ok=True)
    state.visuals = {"charts": []}

    # ---------------------------------------
    # 4. Auto-generate charts from tables (deterministic)
    # ---------------------------------------
    for idx, tbl in enumerate(table_visual_data, start=1):
        auto_chart = auto_chart_from_table(tbl)
        if auto_chart:
            file_path = f"visuals/table_auto_chart_{idx}.png"
            render_chart(auto_chart, file_path)
            auto_chart["file"] = file_path
            state.visuals["charts"].append(auto_chart)

    # ---------------------------------------
    # 5. Ask LLM for visualization ideas (creative)
    # ---------------------------------------
    prompt = f"""
    You are a visualization reasoning assistant.

    You will analyze:
    - extracted entities
    - extracted numeric patterns
    - extracted tables
    - summary / rag text

    TABLE RULES:
    - If a table has categorical & numeric columns → bar chart.
    - If two numeric columns → scatter.
    - If time column → line chart.
    - If many numeric columns → correlation heatmap.
    - If one numeric column → histogram.
    - If few categories → pie chart.

    OUTPUT STRICT JSON:
    {{
      "charts": [
        {{
          "type": "<bar|pie|line|scatter|histogram|heatmap|wordcloud>",
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

    llm_response = model1.invoke(prompt).content

    # Parse JSON safely
    charts = []
    try:
        try:
            llm_json = json.loads(llm_response)
        except:
            match = re.search(r"\{[\s\S]*\}", llm_response)
            llm_json = json.loads(match.group(0)) if match else {}
        charts = llm_json.get("charts", [])
    except Exception as e:
        print("❌ LLM visualization parsing failed:", e)
        charts = []

    # ---------------------------------------
    # 6. Render LLM-generated charts
    # ---------------------------------------
    for i, chart in enumerate(charts, start=1):
        ctype = chart.get("type")
        data = chart.get("data", {})
        labels = data.get("labels", [])
        values = data.get("values", [])
        title = data.get("title", f"Chart {i}")

        if not values:
            continue

        file_path = f"visuals/chart_{i}_{ctype}.png"
        render_chart({
            "type": ctype,
            "title": title,
            "labels": labels,
            "values": values
        }, file_path)

        state.visuals["charts"].append({
            "type": ctype,
            "title": title,
            "file": file_path
        })

    # ---------------------------------------
    # 7. Fallback: WordCloud
    # ---------------------------------------
    if len(state.visuals["charts"]) == 0:
        wc_path = "visuals/fallback_wordcloud.png"
        wc = WordCloud(width=800, height=400).generate(text_source)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(wc_path)
        plt.close()
        state.visuals["charts"].append({
            "type": "wordcloud",
            "title": "WordCloud",
            "file": wc_path
        })

    return state


# ---------------------------------------
# Helper: Render actual matplotlib chart
# ---------------------------------------
def render_chart(chart, file_path):
    """Render a chart dictionary using Matplotlib."""

    ctype = chart.get("type", "bar")
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    title = chart.get("title", "Chart")

    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    try:
        if ctype == "bar":
            plt.bar(labels, values)
        elif ctype == "line":
            plt.plot(labels, values, marker="o")
        elif ctype == "pie":
            plt.pie(values, labels=labels, autopct="%1.1f%%")
        elif ctype == "scatter":
            plt.scatter(labels, values)
        elif ctype == "histogram":
            plt.hist(values, bins=10)
        elif ctype == "wordcloud":
            wc = WordCloud(width=800, height=400).generate(" ".join(labels))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
    except Exception as e:
        print(f"❌ Error rendering chart {ctype}: {e}")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
