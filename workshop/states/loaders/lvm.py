import io
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm_vision = ChatOpenAI(model="gpt-4o", temperature=0)


def analyze_image_with_lvm(pil_image: Image.Image) -> tuple[str, str]:

    try:
        # Convert PIL Image to raw PNG bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Define the image content block (LangChain handles the final encoding)
        image_content_block = {"type": "input_image", "image": image_bytes}

        # First pass — description
        desc_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image clearly and accurately."},
                image_content_block
            ]
        )
        desc = llm_vision.invoke([desc_msg]).content.strip()

        # Second pass — insights
        insight_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Give 2–3 clear insights from the visual information."},
                image_content_block
            ]
        )
        insights = llm_vision.invoke([insight_msg]).content.strip()

        return desc, insights

    except Exception as e:
        return f"[VisionError] {e}", ""
    