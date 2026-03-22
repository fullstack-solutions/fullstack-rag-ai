import os
from pathlib import Path
from pypdf import PdfReader
from .types import DocumentChunk
from typing import List, Union

def load_pdfs(path: Union[str, Path], chunk_pages: bool = True) -> List[DocumentChunk]:
    """
    Loads PDFs as DocumentChunk objects.
    
    Args:
        path: folder path or single PDF file path
        chunk_pages: 
            True -> each page is a chunk
            False -> entire PDF is a single chunk
    
    Returns:
        List[DocumentChunk]
    """
    documents = []
    path = Path(path)

    pdf_files = []
    if path.is_file() and path.suffix.lower() == ".pdf":
        pdf_files = [path]
    elif path.is_dir():
        pdf_files = [f for f in path.iterdir() if f.suffix.lower() == ".pdf"]
    else:
        raise ValueError(f"Path {path} is not a PDF file or a valid folder.")

    for filepath in pdf_files:
        reader = PdfReader(filepath)
        filename = filepath.name

        if chunk_pages:
            # Each page as a chunk
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        DocumentChunk(
                            id=f"{filename}_{i}",
                            text=text.strip(),
                            metadata={"source": filename, "page": i+1}
                        )
                    )
        else:
            # Entire PDF as a single chunk
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            if full_text.strip():
                documents.append(
                    DocumentChunk(
                        id=filename,
                        text=full_text.strip(),
                        metadata={"source": filename}
                    )
                )

    return documents