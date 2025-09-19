from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os, uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document

UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings & LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
INDEX_PATH = "faiss_index"
vectors = None
documents: List[Document] = []

prompt = ChatPromptTemplate.from_template("""
Answer the question using the context only.
<context>
{context}
</context>
Question: {input}
""")

@app.post("/upload")
async def upload(files: List[UploadFile]):
    global documents
    docs = []
    for file in files:
        file_path = os.path.join(UPLOADS, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # For simplicity, just read as plain text
        txt = open(file_path, "r", encoding="utf-8", errors="ignore").read()
        docs.append(Document(page_content=txt, metadata={"source": file.filename}))
    documents.extend(docs)
    return {"status": "INGESTION_COMPLETE", "count": len(docs)}

@app.post("/build_index")
async def build_index():
    global vectors
    chunks = splitter.split_documents(documents)
    vectors = FAISS.from_documents(chunks, embeddings)
    vectors.save_local(INDEX_PATH)
    return {"status": "INDEX_BUILT", "chunks": len(chunks)}

@app.post("/query")
async def query(question: str = Form(...)):
    global vectors
    if not vectors:
        vectors = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectors.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    ctx = "\n".join([d.page_content for d in docs])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    out = doc_chain.invoke({"input": question, "context": ctx})
    return {
        "status": "FINAL_ANSWER",
        "answer": out["output_text"],
        "sources": [d.metadata for d in docs],
        "context_used": [d.page_content for d in docs]
    }
