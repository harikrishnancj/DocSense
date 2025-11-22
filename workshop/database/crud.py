import json
from sqlalchemy.orm import Session
from database.models import Document


#def get_document_by_filename(db: Session, filename: str):
    #return db.query(Document).filter(Document.filename == filename).first()


def save_document(
        db: Session,
        filename: str,
        summary: str,
        user_query: str = None,
        rag_response: str = None,
        entities: dict = None,
        visuals: dict = None,
        extracted_images: list = None,
        image_descriptions: list = None,
        extracted_tables: list = None,
        image_insights: list = None,
):
    db_document = Document(
        filename=filename,
        summary=summary,
        user_query=user_query,
        rag_response=rag_response,
        entities=json.dumps(entities) if entities else None,
        visuals=json.dumps(visuals) if visuals else None,

        extracted_images=json.dumps(extracted_images) if extracted_images else None,
        image_descriptions=json.dumps(image_descriptions) if image_descriptions else None,
        extracted_tables=json.dumps(extracted_tables) if extracted_tables else None,
        image_insights=json.dumps(image_insights) if image_insights else None,
    )

    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document
