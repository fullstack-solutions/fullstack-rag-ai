# document_loader.py

from pathlib import Path
from typing import List, Union
from pypdf import PdfReader
from .types import DocumentChunk

def load_pdfs(path: Union[str, Path], chunk_pages: bool = True) -> List[DocumentChunk]:
    """
    Load PDF documents as DocumentChunk objects.

    Args:
        path: Path to a single PDF file or folder containing PDFs.
        chunk_pages: 
            True -> each page is a separate chunk.
            False -> entire PDF as a single chunk.

    Returns:
        List of DocumentChunk objects.
    """
    path = Path(path)
    documents = []

    # Collect all PDF files
    if path.is_file() and path.suffix.lower() == ".pdf":
        pdf_files = [path]
    elif path.is_dir():
        pdf_files = [f for f in path.iterdir() if f.suffix.lower() == ".pdf"]
    else:
        raise ValueError(f"Invalid PDF path: {path}")

    # Load each PDF
    for filepath in pdf_files:
        reader = PdfReader(filepath)
        filename = filepath.name

        if chunk_pages:
            # Split into pages
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(DocumentChunk(
                        id=f"{filename}_{i}",
                        text=text.strip(),
                        metadata={"source": filename, "page": i+1}
                    ))
        else:
            # Merge all pages into one chunk
            full_text = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            if full_text.strip():
                documents.append(DocumentChunk(
                    id=filename,
                    text=full_text.strip(),
                    metadata={"source": filename}
                ))

    return documents