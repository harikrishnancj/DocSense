from pydantic import BaseModel
from typing import List, Dict, Any

class DocState(BaseModel):
    folder_path: str = ""
    documents: List = []
    summary: str = ""
    entities: List[Dict] = []
    visuals: Dict = {}
    user_query: str = ""
    rag_response: str = ""
    use_rag: bool = False
    index: Any = None
    ocr_text: str = ""#
    image_previews: list = []#
    chart_candidates: list = []#
