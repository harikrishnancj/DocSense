import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv
load_dotenv()

from backend.app_graph import app_graph
from database.database import SessionLocal, engine
from database.models import Base
from database.crud import save_document #get_document_by_filename

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/process/")
async def process_file(
    file: UploadFile,
    mode: str = Form(...),
    user_query: str = Form(""),
    db: Session = Depends(get_db)
):
    filename = file.filename
    file_path = Path(UPLOAD_DIR) / filename

    # -------------------------------------------
    # 0. Check cache — if file already processed
    # -------------------------------------------
    '''
    existing_doc = get_document_by_filename(db, filename)
    if existing_doc:
        return JSONResponse(content={
            "summary": existing_doc.summary,
            "rag_response": existing_doc.rag_response,
            "entities": existing_doc.entities,
            "visuals": existing_doc.visuals,
            "extracted_images": existing_doc.extracted_images,
            "image_descriptions": existing_doc.image_descriptions,
            "extracted_tables": existing_doc.extracted_tables,
            "image_insights": existing_doc.image_insights
        })
    '''


    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    state = app_graph.invoke({
        "folder_path": UPLOAD_DIR,
        "use_rag": mode.lower() == "rag",
        "user_query": user_query
    })

    summary = state.get("summary", "")
    rag_response = state.get("rag_response", "")
    entities = state.get("entities", [])
    visuals = state.get("visuals", {})

    extracted_images = state.get("extracted_images", [])
    image_descriptions = state.get("image_descriptions", [])
    extracted_tables = state.get("extracted_tables", [])
    image_insights = state.get("image_insights", [])

    save_document(
        db,
        filename=filename,
        summary=summary,
        user_query=user_query,
        rag_response=rag_response,
        entities=entities,
        visuals=visuals,
        extracted_images=extracted_images,
        image_descriptions=image_descriptions,
        extracted_tables=extracted_tables,
        image_insights=image_insights,
    )


    return JSONResponse(content={
        "summary": summary,
        "rag_response": rag_response,
        "entities": entities,
        "visuals": visuals,
        "extracted_images": extracted_images,
        "image_descriptions": image_descriptions,
        "extracted_tables": extracted_tables,
        "image_insights": image_insights
    })
