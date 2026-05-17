# 📌 RAG-Based Technical Documentation Assistant using LangGraph

---

## 📌 Project Overview

This project is a Retrieval-Augmented Generation (RAG) system built using **LangGraph** and **FastAPI**.  
It allows users to ask questions about technical documentation and returns grounded answers using a vector database and an LLM.

The system improves response quality using:
- Query rewriting
- Document retrieval
- Document relevance grading
- Conditional routing in LangGraph

---

## 📌 Architecture

User Query → Query Analysis → Retrieval → Document Grading → Conditional Routing → Generation → Response with Sources

---

## 📌 Tech Stack

- Python  
- FastAPI  
- LangGraph  
- LangChain  
- ChromaDB  
- Gemini LLM  

---

## 📌 Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd technical-documentation-assistant

## 2.Install dependencies:
pip install -r requirements.txt

## 3.Create a .env file and add your API key:
Create a `.env` file in the project root and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_google_api_key_here

## 4.Run the FastAPI server:
uvicorn app:app --reload

## 5.Open the API documentation:
http://127.0.0.1:8000/docs

## 📌 API Endpoints
1. Ingest Documents

http
POST /ingest

JSON

{
  "url": "https://docs.langchain.com/"
}


2. Query System
http
POST /query

JSON

{
  "question": "What is LangChain?"
}

3. View Stored Documents

http
GET /documents

4. Submit Feedback

http

POST /feedback

JSON
{
  "question": "What is LangChain?",
  "feedback": "good"
}

📌 Design Decisions
Used LangGraph to structure the RAG workflow as a state machine
Used ChromaDB for local vector storage and fast similarity search
Used Gemini LLM for both embeddings and generation
Added query rewriting to improve retrieval accuracy
Added document grading to filter irrelevant chunks
Added retry mechanism for better robustness
📌 Tradeoffs
Web scraping may fail for some restricted websites
LLM-based grading increases response latency
Local ChromaDB is not scalable for production workloads
📌 Future Improvements
Add streaming responses
Add conversation memory (chat history)
Add web search fallback for missing knowledge
Deploy project on cloud (Render / AWS / Railway)
Improve evaluation using Self-RAG style scoring
📌 Document Corpus

Used:

https://docs.langchain.com/
📌 Status

✔ Working FastAPI application
✔ LangGraph RAG pipeline implemented
✔ Retrieval + grading + routing working
✔ Ready for submission



