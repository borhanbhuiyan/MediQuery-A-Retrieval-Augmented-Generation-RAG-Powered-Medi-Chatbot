import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embedding
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

# API Key লোড করা (.env থেকে অতিরিক্ত স্পেস বা কোটেশন থাকলে তা পরিষ্কার করা)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
PINECONE_API_KEY = (
    os.environ.get("PINECONE_API_KEY", "").strip().strip('"').strip("'")
)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# এমবেডিং ও Pinecone ইনডেক্স কানেক্ট করা
embeddings = download_embedding()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


# আপনার Groq API Key-তে কোন কোন মডেল সচল আছে তা অটোমেটিক খুঁজে বের করার ফাংশন
def get_active_groq_model(api_key):
  try:
    client = Groq(api_key=api_key)
    models = client.models.list()
    available = [
        m.id
        for m in models.data
        if "whisper" not in m.id.lower()
        and "guard" not in m.id.lower()
        and "tts" not in m.id.lower()
    ]
    print("\n✅ Available Chat Models for your API Key:", available)

    # ভালো মডেলগুলোর অগ্রাধিকার তালিকা
    preferred = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]
    for p in preferred:
      if p in available:
        return p

    # তালিকার কোনোটি না থাকলে আপনার অ্যাকাউন্টের প্রথম সচল চ্যাট মডেলটি নেবে
    if available:
      return available[0]
  except Exception as e:
    print("\n⚠️ Error checking Groq models:", e)
  return "llama-3.1-8b-instant"


ACTIVE_MODEL = get_active_groq_model(GROQ_API_KEY)
print(f"🚀 Using Groq Model: {ACTIVE_MODEL}\n")

# সচল মডেল দিয়ে LLM ইনিশিয়ালাইজ করা
llm = ChatGroq(
    temperature=0.4, model_name=ACTIVE_MODEL, groq_api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


@app.route("/")
def index():
  return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"]
  input_text = msg
  print("User Input:", input_text)

  # Pinecone থেকে প্রাসঙ্গিক তথ্য (Context) খোঁজা
  docs = retriever.invoke(input_text)
  context = "\n".join([doc.page_content for doc in docs])

  # উত্তর জেনারেট করা
  chain = prompt | llm
  response = chain.invoke({"input": input_text, "context": context})

  print("Response:", response.content)
  return str(response.content)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  app.run(host="0.0.0.0", port=port, debug=False)