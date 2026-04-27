# db_sync_manager.py

from typing import List, Optional, Any


class DBSyncManager:
    """
    Central sync orchestrator:
    - Routes documents automatically
    - Syncs VectorDB + GraphDB
    - Handles repo removal
    - Safe if any DB is missing
    """

    def __init__(self, vector_db=None, graph_db=None, debug: bool = False):
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.debug = debug

        self.vector_db_enabled = vector_db is not None
        self.graph_db_enabled = graph_db is not None

    # -------------------
    # ROUTING LOGIC
    # -------------------
    def _classify_docs(self, external_docs: List[Any]):
        vector_docs = []
        graph_repos = []

        for doc in external_docs:
            # -------------------
            # GRAPH DB RULE
            # -------------------
            if (
                isinstance(doc, dict)
                and "owner" in doc
                and "name" in doc
            ):
                graph_repos.append(doc)

            # -------------------
            # VECTOR DB RULE
            # -------------------
            else:
                vector_docs.append(doc)

        return vector_docs, graph_repos

    # -------------------
    # MAIN SYNC
    # -------------------
    def sync_db(
        self,
        external_docs: Optional[List[Any]] = None,
        remove_repo: bool = False,
        repo_ids: Optional[List[str]] = None,
    ):
        if not external_docs:
            external_docs = []

        # -------------------
        # CLASSIFY INPUT
        # -------------------
        vector_docs, graph_repos = self._classify_docs(external_docs)

        if self.debug:
            print(f"[DEBUG] Vector docs: {len(vector_docs)}")
            print(f"[DEBUG] Graph repos: {len(graph_repos)}")

        # -------------------
        # SYNC VECTOR DB
        # -------------------
        if self.vector_db_enabled:
            if vector_docs:
                try:
                    self.vector_db.sync(external_docs=vector_docs)
                    if self.debug:
                        print("[INFO] VectorDB sync complete.")
                except Exception as e:
                    print(f"[ERROR] VectorDB sync failed: {e}")
            else:
                if self.debug:
                    print("[INFO] No vector docs to sync.")
        else:
            if self.debug:
                print("[INFO] VectorDB not enabled.")

        # -------------------
        # SYNC GRAPH DB
        # -------------------
        if self.graph_db_enabled:
            if graph_repos:
                try:
                    self.graph_db.add_or_update_repos(graph_repos)
                    if self.debug:
                        print("[INFO] GraphDB sync complete.")
                except Exception as e:
                    print(f"[ERROR] GraphDB sync failed: {e}")
            else:
                if self.debug:
                    print("[INFO] No graph repos to sync.")
        else:
            if self.debug:
                print("[INFO] GraphDB not enabled.")

        # -------------------
        # REMOVE REPOS
        # -------------------
        if remove_repo and repo_ids:
            if self.graph_db_enabled:
                try:
                    self.graph_db.remove_repos(repo_ids)
                    if self.debug:
                        print(f"[INFO] Removed repos: {repo_ids}")
                except Exception as e:
                    print(f"[ERROR] Failed to remove repos: {e}")
            else:
                print("[WARN] GraphDB not available for repo removal.")