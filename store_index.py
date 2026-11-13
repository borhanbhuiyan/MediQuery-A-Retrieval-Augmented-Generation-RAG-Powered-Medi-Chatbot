from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from src.helper import download_embedding, load_pdf_files, text_split

load_dotenv()

# API Keys লোড করা
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Loading PDF files from data folder...")
extracted_data = load_pdf_files("data/")

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)

print("Downloading embedding model (Hugging Face)...")
embeddings = download_embedding()

print("Initializing Pinecone index...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if index_name not in [i.name for i in pc.list_indexes()]:
  pc.create_index(
      name=index_name,
      dimension=384,  # all-MiniLM-L6-v2 এর dimension 384
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
  )

print("Storing vectors in Pinecone database...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, embedding=embeddings, index_name=index_name
)

print(
    "SUCCESS: Vector embeddings successfully stored in Pinecone database!"
)