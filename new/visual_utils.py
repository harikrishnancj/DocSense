import os
import re
import json
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from model import model1
'''
def Visualizer(state):
    """
    Generates multiple visualization suggestions from the LLM
    based on entities and/or summary.
    Creates and saves charts to /visuals folder.
    """

    # 1️⃣ Gather data context
    visuals_data = {}

    if state.entities:
        labels = [ent["label"] for ent in state.entities]
        counts = dict(Counter(labels))
        visuals_data["entity_distribution"] = counts

    if state.summary:
        visuals_data["summary_text"] = state.summary

    if not visuals_data:
        print("⚠️ No data available for visualization.")
        return state

    # 2️⃣ Ask LLM to propose multiple charts
    prompt = f"""
    You are a data visualization assistant.

    Given the following document data (entities and/or summary),
    propose **multiple** visualizations (bar, pie, line, scatter, histogram, wordcloud)
    that could help understand the data.

    Return strictly valid JSON in this format:

    {{
        "charts": [
            {{
                "type": "<chart_type>",
                "data": {{
                    "labels": [...],
                    "values": [...],
                    "title": "<chart_title>"
                }}
            }},
            ...
        ]
    }}

    Rules:
    - Categorical counts → bar or pie
    - Numeric/time data → line or histogram
    - Text frequency → wordcloud
    - Avoid duplicate charts
    - Ensure len(labels) == len(values)
    - Output only valid JSON, no text before or after.

    Data:
    {json.dumps(visuals_data, indent=2)}
    """

    decision = model1.invoke(prompt).content
    print("\n🤖 LLM Visualization Plan:\n", decision)

    # 3️⃣ Parse JSON safely
    try:
        try:
            vis_plan = json.loads(decision)
        except json.JSONDecodeError:
            # Try to extract JSON substring if extra text exists
            match = re.search(r'\{[\s\S]*\}', decision)
            if match:
                vis_plan = json.loads(match.group(0))
            else:
                raise

        charts = vis_plan.get("charts", [])
        if not charts:
            print("⚠️ No charts suggested by the LLM.")
            return state

        # 4️⃣ Prepare output folder
        os.makedirs("visuals", exist_ok=True)
        state.visuals = {"charts": []}

        # 5️⃣ Plot each suggested chart
        for i, vis in enumerate(charts, 1):
            chart_type = vis.get("type", "bar").lower()
            data = vis.get("data", {})
            labels = data.get("labels", [])
            values = data.get("values", [])
            title = data.get("title", f"Chart {i}")

            if not labels or not values:
                print(f"⚠️ Skipping invalid chart #{i}: missing labels/values.")
                continue

            plt.figure(figsize=(6, 4))
            plt.title(title)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=30, ha="right")

            # Plot based on chart type
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
                else:
                    plt.text(0.5, 0.5, f"Unknown chart type: {chart_type}", ha="center")
            except Exception as e:
                print(f"❌ Error plotting chart #{i} ({chart_type}):", e)
                continue

            plt.tight_layout()

            # Save chart image
            file_path = f"visuals/chart_{i}_{chart_type}.png"
            plt.savefig(file_path)
            plt.close()

            # Store metadata
            state.visuals["charts"].append({
                "title": title,
                "type": chart_type,
                "file": file_path
            })

        print(f"✅ Generated and saved {len(state.visuals['charts'])} charts to /visuals")

    except Exception as e:
        print("❌ Visualization failed or LLM response invalid:", e)

        # 6️⃣ Fallback basic visualization
        if state.entities:
            print("⚙️ Generating fallback bar chart (entity distribution).")
            labels = [ent["label"] for ent in state.entities]
            counts = dict(Counter(labels))
            plt.bar(counts.keys(), counts.values())
            plt.title("Entity Distribution (Fallback)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            os.makedirs("visuals", exist_ok=True)
            fallback_path = "visuals/fallback_entity_chart.png"
            plt.savefig(fallback_path)
            plt.close()
            state.visuals = {"fallback_chart": fallback_path}

    return state
'''
import os
import re
import json
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from model import model1  # your HuggingFace or OpenAI model


def extract_numeric_insights(text):
    """
    Extract years, percentages, and numeric values from summary text.
    Helps LLM see structured data for chart reasoning.
    """
    if not text:
        return {}
    
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    percentages = re.findall(r"\b\d+(?:\.\d+)?%", text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)

    return {
        "years": years,
        "percentages": percentages,
        "numbers": numbers
    }


def Visualizer(state):
    """
    Smarter visualization generator:
    - Uses entities and summary to generate structured charts
    - LLM suggests multiple chart types
    - Always produces at least one visualization
    """

    visuals_data = {}

    # 1️⃣ Entity counts
    if state.entities:
        labels = [ent["label"] for ent in state.entities]
        counts = dict(Counter(labels))
        visuals_data["entity_distribution"] = counts

    # 2️⃣ Summary-based numeric extraction
    if state.summary:
        visuals_data["summary_text"] = state.summary
        visuals_data["numeric_insights"] = extract_numeric_insights(state.summary)

    if not visuals_data:
        print("⚠️ No data available for visualization.")
        return state

    # 3️⃣ Ask the LLM for chart suggestions
    prompt = f"""
    You are a visualization reasoning assistant.

    Analyze the following data (entities + summary + numeric patterns)
    and propose **multiple** visualizations that help understand it.

    Rules:
    - find best chart types based on data patterns:
    - use common sense:
    - use charts suitable for the data:
    - Use bar or pie for categorical distributions (e.g. entity counts)
    - Use line or histogram for numeric/time patterns (e.g. years, percentages)
    - Use scatter if multiple numeric dimensions appear
    - Use a wordcloud for descriptive summaries
    - Avoid duplicate chart types
    - Output only valid JSON, no commentary.

    Return JSON strictly in this format:
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
    print("\n🤖 LLM Visualization Plan:\n", decision)

    os.makedirs("visuals", exist_ok=True)
    state.visuals = {"charts": []}

    # 4️⃣ Try to parse LLM JSON
    try:
        try:
            vis_plan = json.loads(decision)
        except json.JSONDecodeError:
            # Extract embedded JSON if LLM adds text
            match = re.search(r'\{[\s\S]*\}', decision)
            if match:
                vis_plan = json.loads(match.group(0))
            else:
                raise

        charts = vis_plan.get("charts", [])
        if not charts:
            raise ValueError("No charts found in LLM output")

        # 5️⃣ Plot each suggested chart
        for i, vis in enumerate(charts, 1):
            chart_type = vis.get("type", "bar").lower()
            data = vis.get("data", {})
            labels = data.get("labels", [])
            values = data.get("values", [])
            title = data.get("title", f"Chart {i}")

            if not labels or not values:
                print(f"⚠️ Skipping invalid chart #{i}: missing labels/values.")
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

                else:
                    plt.text(0.5, 0.5, f"Unknown chart type: {chart_type}", ha="center")

                # Save chart image
                file_path = f"visuals/chart_{i}_{chart_type}.png"
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()

                # Save metadata
                state.visuals["charts"].append({
                    "title": title,
                    "type": chart_type,
                    "file": file_path
                })

            except Exception as e:
                print(f"❌ Error plotting {chart_type}: {e}")

    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")

    # 6️⃣ Fallback (wordcloud for summary)
    if not state.visuals["charts"] and state.summary:
        print("⚙️ Generating fallback summary wordcloud.")
        wc = WordCloud(width=800, height=400).generate(state.summary)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        fallback_path = "visuals/summary_wordcloud.png"
        plt.savefig(fallback_path)
        plt.close()
        state.visuals["charts"].append({
            "title": "Summary WordCloud (Fallback)",
            "type": "wordcloud",
            "file": fallback_path
        })

    print(f"✅ Generated {len(state.visuals['charts'])} visualization(s).")
    return state
