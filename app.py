from collections import deque
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


def get_working_groq_model(api_key):
  try:
    client = Groq(api_key=api_key)
    models = client.models.list()
    available = [
        m.id
        for m in models.data
        if "whisper" not in m.id.lower() and "guard" not in m.id.lower()
    ]
    print(f"✅ Available Chat Models: {available}")

    preferred_order = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    for pref in preferred_order:
      if pref in available:
        return pref
    if available:
      return available[0]
  except Exception as e:
    print(f"⚠️ Error fetching models: {e}")
  return "llama-3.1-8b-instant"


selected_model = get_working_groq_model(GROQ_API_KEY)
print(f"🚀 Using Groq Model: {selected_model}")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY, model_name=selected_model, temperature=0.3
)

# মেমোরির জন্য ChatPromptTemplate-এ chat_history যুক্ত করা হলো
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# শেষ ৬টি কথোপকথন (১২টি মেসেজ) মনে রাখার জন্য মেমোরি বাফার
chat_history = deque(maxlen=12)


@app.route("/")
def index():
  return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"].strip()
  print(f"User Input: {msg}")

  # ইউজার যদি ফলো-আপ প্রশ্ন করে (যেমন: "ans it shortly"), তবে আগের প্রশ্নটিসহ পাইনকোনে সার্চ করবে
  search_query = msg
  if len(msg.split()) <= 5 and len(chat_history) >= 2:
    last_user_msg = chat_history[-2].content
    search_query = f"{last_user_msg} - {msg}"

  response = rag_chain.invoke(
      {"input": search_query, "chat_history": list(chat_history)}
  )

  answer = response["answer"]
  print("Response :", answer)

  # বর্তমান প্রশ্ন ও উত্তর মেমোরিতে সেভ করা
  chat_history.append(HumanMessage(content=msg))
  chat_history.append(AIMessage(content=answer))

  return str(answer)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  app.run(host="0.0.0.0", port=port, debug=False)