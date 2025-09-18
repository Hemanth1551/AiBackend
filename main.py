from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form

import os
import uuid
import numpy as np
from pymongo import MongoClient
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import List
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import base64
import tempfile
from dotenv import load_dotenv

load_dotenv()


# ===============================
# Configuration
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["rag_chat_db"]
docs_collection = db["documents"]
vectors_collection = db["vectors"]
chat_collection = db["chat_history"]

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, length_function=len
)

search = DuckDuckGoSearchAPIWrapper()

app = FastAPI(title="RAG Chat API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Models
# ===============================
class UploadPDFRequest(BaseModel):
    name: str
    file_base64: str  # PDF content encoded in base64

class AskRequest(BaseModel):
    session_id: str
    question: str

# ===============================
# PDF & DB Utilities
# ===============================
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def upload_document(name: str, text: str):
    doc_id = str(uuid.uuid4())
    docs_collection.insert_one({"_id": doc_id, "name": name, "text": text})
    vector = embedding_model.embed_query(text)
    vectors_collection.insert_one({"doc_id": doc_id, "vector": vector})
    return doc_id

def upload_pdf(pdf_path: str, name: str):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = text_splitter.split_text(raw_text)
    for i, chunk in enumerate(chunks):
        upload_document(f"{name}_part{i+1}", chunk)
    return len(chunks)

def search_similar_docs(query: str, top_k: int = 3):
    query_vec = embedding_model.embed_query(query)
    all_vectors = list(vectors_collection.find({}))
    scored = []
    for item in all_vectors:
        score = np.dot(query_vec, item["vector"]) / (
            np.linalg.norm(query_vec) * np.linalg.norm(item["vector"])
        )
        scored.append((item["doc_id"], score))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in scored:
        doc = docs_collection.find_one({"_id": doc_id})
        results.append({"doc": doc, "score": float(score)})
    return results

def save_message(session_id: str, role: str, content: str):
    chat_collection.insert_one({"session_id": session_id, "role": role, "content": content})

def get_history(session_id: str):
    messages = list(chat_collection.find({"session_id": session_id}))
    return [f"{m['role']}: {m['content']}" for m in messages]

# ===============================
# RAG Graph
# ===============================
class RAGState(BaseModel):
    session_id: str
    question: str
    docs: List[str] = []
    web_results: List[str] = []
    answer: str = ""
    intent: str = ""

# Intent classifier
def classify_intent(state: RAGState):
    prompt = f"""
    You are an intent classifier. Decide if the user question is:
    - "chit_chat"
    - "knowledge"
    Question: {state.question}
    """
    resp = llm.invoke(prompt)
    intent = resp.content.strip().lower()
    if intent not in ["chit_chat", "knowledge"]:
        intent = "knowledge"
    return {"intent": intent}

# Retrieve docs
def retrieve_docs(state: RAGState):
    results = search_similar_docs(state.question)
    if results:
        docs_text = "\n\n".join([r["doc"]["text"] for r in results])
        return {"docs": [docs_text]}
    return {"docs": []}

# Web search
def web_search(state: RAGState):
    results = search.run(state.question)
    return {"web_results": [results] if results else []}

# Generate answer
def generate_answer(state: RAGState):
    session_id = state.session_id
    history = "\n".join(get_history(session_id))
    context_docs = "\n\n".join(state.docs)
    context_web = "\n\n".join(state.web_results)
    context = ""
    if context_docs:
        context += f"Database context:\n{context_docs}\n\n"
    if context_web:
        context += f"Web context:\n{context_web}\n\n"

    prompt = f"""
    Conversation so far:
    {history}

    Question: {state.question}

    Use the conversation history + context to answer naturally.

    {context}

    Answer:
    """
    response = llm.invoke(prompt)
    save_message(session_id, "user", state.question)
    save_message(session_id, "assistant", response.content)
    return {"answer": response.content}

# Build graph
graph = StateGraph(RAGState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate_answer)
graph.set_entry_point("classify_intent")
graph.add_conditional_edges(
    "classify_intent", lambda s: s.intent,
    {"chit_chat": "generate", "knowledge": "retrieve"}
)
def route_after_retrieve(state: RAGState):
    return "generate" if state.docs else "web_search"
graph.add_conditional_edges("retrieve", route_after_retrieve)
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)
rag_app = graph.compile()

# ===============================
# FastAPI Routes (JSON)
# ===============================
# @app.post("/upload-pdf")
# async def api_upload_pdf(request: UploadPDFRequest):
#     try:
#         pdf_data = base64.b64decode(request.file_base64)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid base64 file data")
    
#     pdf_path = f"/tmp/{uuid.uuid4()}.pdf"
#     with open(pdf_path, "wb") as f:
#         f.write(pdf_data)

#     chunks = upload_pdf(pdf_path, request.name)
#     os.remove(pdf_path)
#     return JSONResponse({"message": f"Uploaded {chunks} chunks from PDF {request.name}"})


@app.post("/upload-pdf-file")
async def upload_pdf_file(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")

        # Write uploaded file
        with open(pdf_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty PDF file")
            f.write(content)

        # Process PDF
        chunks = upload_pdf(pdf_path, name)
        if chunks == 0:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        return JSONResponse({"message": f"Uploaded {chunks} chunks from PDF {name}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)



@app.post("/ask")
async def api_ask(request: AskRequest):
    result = rag_app.invoke({"session_id": request.session_id, "question": request.question})
    return JSONResponse({"answer": result["answer"]})


@app.get("/history/{session_id}")
async def api_history(session_id: str):
    history = get_history(session_id)
    return JSONResponse({"history": history})
