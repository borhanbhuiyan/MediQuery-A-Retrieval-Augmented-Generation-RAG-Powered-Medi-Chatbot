# MediQuery: A Retrieval-Augmented Generation (RAG) Powered Medical Chatbot

**MediQuery** is an AI-powered clinical assistant built using a Retrieval-Augmented Generation (RAG) architecture. It retrieves verified medical knowledge from a Pinecone vector database and uses Groq's high-speed LLM inference to deliver accurate, context-aware responses through a clean web interface.

## 🌐 Live Demo
**Live Application:** [https://mediquery-a-retrieval-augmented.onrender.com](https://mediquery-a-retrieval-augmented.onrender.com)

*(Note: Hosted on Render's free instance, which spins down during inactivity. Initial requests may take up to 50 seconds to wake the server.)*

---

## ✨ Key Features
* **RAG Pipeline:** Combines semantic vector search with large language model generation to provide reliable clinical answers and minimize hallucinations.
* **Adaptive Response Formatting:** Automatically adjusts answers for concise definitions, detailed medical explanations, or practical patient home-care guidance.
* **Dynamic Model Selection:** Automatically detects and connects to active Groq LLM models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, etc.) to prevent model deprecation failures.
* **Memory-Optimized Deployment:** Configured with CPU-only PyTorch to run smoothly within a 512MB RAM cloud environment.
* **Self-Contained Clinical UI:** Responsive chat interface built with HTML5, CSS, and JavaScript.

---

## 🛠️ Tech Stack
* **Backend Framework:** Flask (Python)
* **LLM Orchestration:** LangChain
* **Vector Database:** Pinecone (Serverless Index)
* **Embeddings Model:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
* **LLM Inference:** Groq API
* **Production Server & Hosting:** Gunicorn on Render

---

## 📂 Project Structure

    MediQuery-A-Retrieval-Augmented-Generation-RAG-Powered-Medi-Chatbot/
    ├── src/
    │   ├── __init__.py
    │   ├── helper.py          # PDF loader, text chunking, and HuggingFace embeddings
    │   └── prompt.py          # System prompt instructions for clinical responses
    ├── templates/
    │   └── chat.html          # Frontend user interface
    ├── static/
    │   └── style.css          # Stylesheet
    ├── app.py                 # Main Flask application and RAG routing
    ├── store_index.py         # Script to process PDFs and upload embeddings to Pinecone
    ├── requirements.txt       # Python dependencies optimized for cloud deployment
    ├── setup.py               # Package configuration
    └── .gitignore             # Excluded sensitive and bulky files (.env, data/)

---

## 🚀 Local Installation & Setup

1. **Clone the Repository:**
   `git clone [https://github.com/borhanbhuiyan/MediQuery-A-Retrieval-Augmented-Generation-RAG-Powered-Medi-Chatbot.git](https://github.com/borhanbhuiyan/MediQuery-A-Retrieval-Augmented-Generation-RAG-Powered-Medi-Chatbot.git)`
   `cd MediQuery-A-Retrieval-Augmented-Generation-RAG-Powered-Medi-Chatbot`

2. **Create and Activate Conda Environment:**
   `conda create -n medbot python=3.10 -y`
   `conda activate medbot`

3. **Install Dependencies:**
   `pip install -r requirements.txt`

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your secret API keys:
   `PINECONE_API_KEY="your_pinecone_api_key_here"`
   `GROQ_API_KEY="your_groq_api_key_here"`

5. **Generate Vector Embeddings (First-Time Setup Only):**
   Place your medical reference PDF inside a `data/` directory and run:
   `python store_index.py`

6. **Run the Application:**
   `python app.py`
   Open your browser and navigate to `http://localhost:8000`.

---

## 👨‍💻 Author
**Omar Borhan Bhuiyan**  
GitHub: [@borhanbhuiyan](https://github.com/borhanbhuiyan)
