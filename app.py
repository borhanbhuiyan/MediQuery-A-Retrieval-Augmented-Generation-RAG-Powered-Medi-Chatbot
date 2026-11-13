from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embedding
from src.prompt import system_prompt
import os

load_dotenv()

app = Flask(__name__)

# API Key লোড করা
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# এমবেডিং ও Pinecone ইনডেক্স কানেক্ট করা
embeddings = download_embedding()
index_name = "medical-chatbot"

# Pinecone থেকে ভেক্টর স্টোর লোড করা
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# Groq LLM ইনিশিয়ালাইজেশন (Llama 3.3 মডেল)
llm = ChatGroq(temperature=0.4, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

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

  # Groq LLM এর মাধ্যমে উত্তর জেনারেট করা
  chain = prompt | llm
  response = chain.invoke({"input": input_text, "context": context})

  print("Response:", response.content)
  return str(response.content)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080, debug=True)