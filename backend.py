import os
import shutil
import uuid

from typing import List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form
)
import boto3
import os

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fullstack_rag_ai.vector_rag_ai import (
    QAService,
    VectorDBSynchronizer
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
        Bucket=S3_BUCKET
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

syncer = VectorDBSynchronizer(
    index_path="./vectordb_for_contracts",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=100,
    chunk_overlap=10,
    chunking_strategy="semantic"
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

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    data_source_ref: str = Form(...)
):

    try:

        if not data_source_ref:

            raise ValueError(
                "data_source_ref is required"
            )

        uploaded_paths = []

        for file in files:

            unique_name = (
                f"{uuid.uuid4()}_{file.filename}"
            )

            save_path = os.path.join(
                UPLOAD_DIR,
                unique_name
            )

            with open(save_path, "wb") as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            uploaded_paths.append(save_path)

        (
            all_docs,
            metadata,
            qa_cache,
            file_metadata,
            message
        ) = syncer.sync(
            manual_file_mode=True,
            uploaded_files=uploaded_paths,
            data_source_ref=data_source_ref
        )

        return {
            "success": True,
            "message": message,
            "uploaded_files": uploaded_paths,
            "total_uploaded": len(uploaded_paths)
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
