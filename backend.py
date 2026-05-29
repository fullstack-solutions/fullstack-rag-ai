import os

from fastapi import (
    FastAPI,
)
import boto3
import os

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fullstack_rag_ai.vector_rag_ai import (
    QAService
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_BUCKET = "vectordb-for-contracts"
SUFIX = "vectordb_for_contracts"
LOCAL_DB_PATH = "./vectordb_for_contracts"

UPLOAD_DIR = "./uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

prompt_template = """
You are a retrieval-based question answering system.

You MUST:
- Use only provided documents as evidence
- Produce a clear final answer
- Cite exact sources (file + page)
- Only quote short relevant spans (max 1-2 sentences)
- Prefer multiple sources if available

Return format:

Answer:
...

Evidence:
1. Source
2. Source
...

Documents:
{context}

Question:
{question}
"""
def download_vectordb():
    s3 = boto3.client("s3")

    if os.path.exists(LOCAL_DB_PATH) and len(os.listdir(LOCAL_DB_PATH)) > 0:
        print("Vector DB already exists locally")
        return

    os.makedirs(LOCAL_DB_PATH, exist_ok=True)

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=SUFIX
    )

    for obj in response.get("Contents", []):
        key = obj["Key"]

        if key.endswith("/"):
            continue

        local_file_path = os.path.join(
            LOCAL_DB_PATH,
            os.path.basename(key)
        )

        print(f"Downloading {key} → {local_file_path}")

        s3.download_file(S3_BUCKET, key, local_file_path)

download_vectordb()

qa = QAService(
    index_path="./vectordb_for_contracts",
    model="llama3",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    k=10,
    prompt_template=prompt_template
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):

    try:

        result = qa.ask(req.question)
        print("QA Result:", result.content)
        return {
            "success": True,
            "answer": result.content
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
