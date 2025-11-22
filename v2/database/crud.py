 
from sqlalchemy.orm import Session
from database.models import Document


def get_document_by_filename(db: Session, filename: str):
    return db.query(Document).filter(Document.filename == filename).first()

def save_document(db: Session, filename: str, summary: str, user_query: str = None,
                  rag_response: str = None, entities: str = None, visuals: str = None):
    db_document = Document(
        filename=filename,
        summary=summary,
        user_query=user_query,
        rag_response=rag_response,
        entities=entities,
        visuals=visuals
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document
