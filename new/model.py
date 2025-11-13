
'''from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    task="text-summarization",                        
    max_new_tokens=512,
    top_k=10,
    temperature=0.7,
    repetition_penalty=1.0
)

model1=ChatHuggingFace(llm=llm)

model2 = HuggingFaceEmbedding(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")


'''
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set OpenAI API key (make sure it's in your .env file)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ---- LLM ----
# Equivalent to HuggingFaceEndpoint (Mistral)
# You can use GPT-4 or GPT-3.5 depending on your plan
model1 = ChatOpenAI(
    model="gpt-4o-mini",    # or "gpt-4o" / "gpt-3.5-turbo"
    temperature=0.7,
    max_tokens=512
)

# ---- Embeddings ----
# Equivalent to HuggingFaceEmbedding (for retrieval / similarity)
from llama_index.embeddings.openai import OpenAIEmbedding

model2 = OpenAIEmbedding(model="text-embedding-3-large")
