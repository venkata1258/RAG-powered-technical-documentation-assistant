import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

# -----------------------
# ENV
# -----------------------
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# -----------------------
# LLM
# -----------------------
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key
)

# -----------------------
# EMBEDDINGS
# -----------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_key
)

# -----------------------
# GLOBALS
# -----------------------
vector_store = None
retriever = None

# -----------------------
# INGEST (FIXED)
# -----------------------
def ingest_documents(url: str):
    global vector_store, retriever

    loader = WebBaseLoader(
        url,
        header_template={"User-Agent": "Mozilla/5.0"}
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        return 0

    if vector_store is None:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="docs"
        )
    else:
        vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    return len(chunks)

# -----------------------
# STATE
# -----------------------
class GraphState(TypedDict):
    question: str
    rewritten_query: Optional[str]
    documents: List[Document]
    relevant_docs: List[Document]
    generation: str
    retry_count: int

# -----------------------
# QUERY ANALYSIS
# -----------------------
def query_analysis(state: GraphState):
    res = LLM.invoke(
        f"Rewrite for retrieval: {state['question']}. Return only one query."
    )
    return {"rewritten_query": res.content.strip()}

# -----------------------
# RETRIEVAL
# -----------------------
def retrieve_documents(state: GraphState):
    if retriever is None:
        return {"documents": []}

    query = state.get("rewritten_query") or state["question"]
    docs = retriever.invoke(query)

    return {"documents": docs}

# -----------------------
# GRADING (OPTIMIZED)
# -----------------------
def document_grading(state: GraphState):
    relevant_docs = []
    question = state['question']
    documents = state.get("documents", [])
    
    print(f"\n--- GRADING {len(documents)} RETRIEVED DOCUMENTS ---")

    for i, doc in enumerate(documents):
        # Explicit prompt instructions prevent formatting bugs (e.g., markdown, punctuation)
        res = LLM.invoke(
            f"""
            Analyze if the following document contains information relevant to answer the question.
            
            Question: {question}
            Document: {doc.page_content}

            Respond with exactly one word: 'yes' if relevant, or 'no' if not relevant. Do not include punctuation, markdown, or extra text.
            """
        ).content.strip().lower()

        print(f"Doc {i+1} evaluation raw response: '{res}'")

        # Soft matching validation layer
        if "yes" in res:
            relevant_docs.append(doc)

    print(f"Total relevant documents found: {len(relevant_docs)}\n")
    return {"relevant_docs": relevant_docs}

# -----------------------
# ROUTER
# -----------------------
def route_documents(state: GraphState):
    if len(state.get("relevant_docs", [])) > 0:
        return "generate"
    elif state.get("retry_count", 0) < 2:
        return "rewrite"
    else:
        return "stop"

# -----------------------
# REWRITE
# -----------------------
def rewrite_query(state: GraphState):
    res = LLM.invoke(
        f"Improve query for retrieval: {state['question']}"
    )
    return {
        "rewritten_query": res.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1
    }

# -----------------------
# GENERATION
# -----------------------
def generate_answer(state: GraphState):
    docs = state.get("relevant_docs", [])

    if not docs:
        return {"generation": "No relevant context found."}

    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    res = LLM.invoke(
        f"""
        Answer ONLY using context:

        {context}

        Question: {state['question']}
        """
    ).content

    return {
        "generation": res + "\n\nSources:\n" + "\n".join(sources)
    }

# -----------------------
# STOP
# -----------------------
def stop_generation(state: GraphState):
    return {"generation": "I don't know based on available documentation."}

# -----------------------
# GRAPH
# -----------------------
workflow = StateGraph(GraphState)

workflow.add_node("query_analysis", query_analysis)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", document_grading)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_node("stop", stop_generation)

workflow.add_edge(START, "query_analysis")
workflow.add_edge("query_analysis", "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "stop": "stop"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("stop", END)

graph = workflow.compile()

# -----------------------
# EXPORTS (FIXED FOR API COUPLING)
# -----------------------
def get_vector_store():
    """Returns the live instantiated reference of the vector store."""
    return vector_store