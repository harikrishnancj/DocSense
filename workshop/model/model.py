from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM
model1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=512)

# Embeddings
model2 = OpenAIEmbedding(model="text-embedding-3-large")
