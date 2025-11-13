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
from database.crud import save_document,get_document_by_filename

# Create database tables
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

    # Step 0: Check if document already exists
    existing_doc = get_document_by_filename(db, filename)
    if existing_doc:
        # Return cached result, don't try to save again
        return JSONResponse(content={
            "summary": existing_doc.summary,
            "rag_response": existing_doc.rag_response,
            "entities": existing_doc.entities,
            "visuals": existing_doc.visuals
        })

    # Step 1: Save file to disk
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 2: Run LangGraph pipeline
    state = app_graph.invoke({
        "folder_path": UPLOAD_DIR,
        "use_rag": mode.lower() == "rag",
        "user_query": user_query
    })

    summary = state.get("summary", "")
    rag_response = state.get("rag_response", "")
    entities = state.get("entities", [])
    visuals = state.get("visuals", {})


    # Step 4: Only save if filename is new
    if not existing_doc:
        save_document(
            db,
            filename=filename,
            summary=summary,
            user_query=user_query,
            rag_response=rag_response,
            entities=str(entities),
            visuals=str(visuals)
        )

    # Step 5: Return result
    return JSONResponse(content={
        "summary": summary,
        "rag_response": rag_response,
        "entities": entities,
        "visuals": visuals
    })