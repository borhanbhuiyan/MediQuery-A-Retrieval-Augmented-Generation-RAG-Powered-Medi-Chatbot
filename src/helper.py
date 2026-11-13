import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# এসএসএল (SSL) সার্টিফিকেট এরর এড়ানোর জন্য
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()


# ১. ডেটা ফোল্ডার থেকে PDF লোড করার ফাংশন
def load_pdf_files(data):
  loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
  documents = loader.load()
  return documents


# ২. লোড করা টেক্সটগুলোকে ছোট ছোট চнке (Chunks) ভাগ করার ফাংশন
def text_split(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=20
  )
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks


# ৩. Hugging Face ফ্রি এমবেডিং মডেল ডাউনলোড করার ফাংশন
def download_embedding():
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  return embeddings