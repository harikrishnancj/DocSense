from database.database import Base
from sqlalchemy import Column, Integer, String, Text,func
from sqlalchemy.sql.sqltypes import TIMESTAMP

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True, nullable=False)
    summary = Column(Text, nullable=False)
    user_query = Column(Text, nullable=True)
    rag_response = Column(Text, nullable=True)
    entities = Column(Text, nullable=True)
    visuals = Column(Text, nullable=True)
    extracted_images = Column(Text, nullable=True)        # JSON list of paths
    image_descriptions = Column(Text, nullable=True)      # JSON list
    extracted_tables = Column(Text, nullable=True)        # JSON list of dicts
    image_insights = Column(Text, nullable=True) 
    created_at = Column(TIMESTAMP, server_default=func.now())

