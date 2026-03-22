import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from text_loader import load_from_text_list


def load_documents(path: Optional[str] = None, texts: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents either from PDF files in a directory or from a list of text strings.
    Errors in loading individual files are caught and logged without stopping the process.

    Args:
        path (Optional[str]): Path to folder containing PDF files. Defaults to None.
        texts (Optional[List[str]]): List of text strings to convert into Documents. Defaults to None.

    Returns:
        List[Document]: List of LangChain Document objects with metadata for PDFs.
    """
    documents: List[Document] = []

    # Load PDF documents from folder
    if path:
        if not os.path.exists(path):
            print(f"[WARN] PDF directory does not exist: {path}")
        else:
            for file in os.listdir(path):
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(path, file)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        # Add source filename to metadata
                        for d in docs:
                            d.metadata["source"] = file
                        documents.extend(docs)
                        print(f"[INFO] Loaded PDF: {file}")
                    except FileNotFoundError:
                        print(f"[ERROR] File not found: {file_path}")
                    except PermissionError:
                        print(f"[ERROR] Permission denied: {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load PDF {file}: {e}")

    # Load documents from raw text
    if texts:
        try:
            docs = load_from_text_list(texts)
            documents.extend(docs)
            print(f"[INFO] Loaded {len(docs)} text documents")
        except Exception as e:
            print(f"[ERROR] Failed to load documents from text: {e}")

    return documents
