from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_graph import graph, ingest_documents, vector_store

load_dotenv()

app = FastAPI(title="RAG LangGraph Assistant")

# -------------------
# MODELS
# -------------------
class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    question: str
    feedback: str

# -------------------
# INGEST
# -------------------
@app.post("/ingest")
def ingest(request: IngestRequest):

    try:
        chunks = ingest_documents(request.url)

        return {
            "message": "success",
            "chunks_added": chunks
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------
# QUERY
# -------------------
@app.post("/query")
def query(request: QueryRequest):

    result = graph.invoke({
        "question": request.question,
        "retry_count": 0
    })

    return {
        "answer": result.get("generation", ""),
        "relevant_docs_count": len(result.get("relevant_docs", []))
    }

# -------------------
# DOCUMENTS
# -------------------
@app.get("/documents")
def get_documents():

    if vector_store is None:
        return {"message": "No documents"}

    docs = vector_store.get()

    return {
        "total_documents": len(docs.get("documents", []))
    }

# -------------------
# FEEDBACK
# -------------------
feedback_store = []

@app.post("/feedback")
def feedback(request: FeedbackRequest):

    feedback_store.append(request.dict())

    return {"message": "saved"}