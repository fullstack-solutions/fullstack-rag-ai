from sources.local import LocalSource
from sources.github import GitHubSource
from sources.api import APISource
from core.ingestion import IngestionEngine


# -------- LOCAL --------
def run_local():
    engine = IngestionEngine(
        source=LocalSource("data/docs")
    )
    docs = engine.run()
    print(f"Loaded {len(docs)} documents")


# -------- GITHUB --------
def run_github():
    engine = IngestionEngine(
        source=GitHubSource(token="GITHUB_TOKEN")
    )
    docs = engine.run()
    print(f"Loaded {len(docs)} documents from repos")


# -------- API --------
def fetch_api_data():
    return [
        {"text": "hello world"},
        {"text": "another doc"}
    ]


def run_api():
    engine = IngestionEngine(
        source=APISource(fetch_api_data)
    )
    docs = engine.run()
    print(f"Loaded {len(docs)} API documents")
