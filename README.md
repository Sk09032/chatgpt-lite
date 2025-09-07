# 🧠 ChatGPT-Lite

A **Streamlit-based conversational AI assistant** built with **LangGraph** and **Cohere LLMs**.  
This project demonstrates **multi-path reasoning** with intelligent routing between **RAG (PDF Q&A)**, **real-time search**, and **normal chat**, while maintaining **history-aware memory** across sessions.  

---

## 🚀 Features

- 📄 **Retrieval-Augmented Generation (RAG)**  
  Upload PDFs and query them with context-aware answers.  
  - Uses **FAISS** + **Cohere Embeddings** for semantic search.  
  - Supports chunked document retrieval via `RecursiveCharacterTextSplitter`.

- 🌐 **Real-time Search Agent**  
  Integrates **Tavily Search API** for answering live queries (news, stock, weather, etc.).

- 💬 **Normal Chat Mode**  
  Handles casual or general conversation directly with Cohere LLM.

- 🧾 **History-Aware Memory**  
  Thread-based memory to maintain multi-turn conversational context.

- 🎨 **Streamlit UI**  
  Simple, chat-like interface for user interactions.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Frontend UI  
- [LangGraph](https://www.langchain.com/langgraph) – Node-based state management  
- [Cohere LLM + Embeddings](https://cohere.com/) – Language model & vector embeddings  
- [FAISS](https://faiss.ai/) – Vector database for document retrieval  
- [Tavily Search](https://tavily.com/) – Real-time web search  

---

## 📂 Project Structure  

├── myenv                 # Virtual environment folder  
├── chatbot.py            # Main Streamlit chatbot application  
├── testing_file.ipynb    # Jupyter notebook for testing/debugging  
├── requirements.txt      # Project dependencies  
├── .gitignore            # Git ignore rules  
└── README.md             # Project documentation  



---

## ⚡ Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Sk09032/chatgpt-lite.git
   cd chatgpt-lite


2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Mac/Linux
    venv\Scripts\activate      # On Windows

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt

4. **Set up environment variables**
   Create a .env file with:
   ```ini
    COHERE_API_KEY=your-cohere-api-key
    TAVILY_API_KEY=your-tavily-api-key

5. **Run the app**
   ```bash
   streamlit run app.py