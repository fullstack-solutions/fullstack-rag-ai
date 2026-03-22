from langchain.schema import Document
from typing import List


def load_from_text_list(texts: List[str]) -> List[Document]:
    """
    Convert a list of text strings into a list of LangChain Document objects.

    Args:
        texts (List[str]): List of text strings to convert.

    Returns:
        List[Document]: List of Document objects with `page_content` set.
    """
    return [Document(page_content=t) for t in texts]