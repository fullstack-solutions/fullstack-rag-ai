import hashlib
from datetime import datetime
from typing import List, Dict, Any
from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer


class Neo4jGraphDB:
    def __init__(self, uri, user, password, embedding_model='all-MiniLM-L6-v2'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self._init_schema()

    # -------------------
    # SCHEMA
    # -------------------
    def _init_schema(self):
        queries = [
            """
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (r:Repo)
            REQUIRE r.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (f:File)
            REQUIRE f.id IS UNIQUE
            """
        ]
        with self.driver.session() as s:
            for q in queries:
                s.run(q)

    # -------------------
    # HASH
    # -------------------
    def _hash(self, text):
        if isinstance(text, bytes):
            return hashlib.md5(text).hexdigest()
        elif isinstance(text, str):
            return hashlib.md5(text.encode("utf-8")).hexdigest()
        else:
            text = str(text)
            return hashlib.md5(text.encode("utf-8")).hexdigest()

    # -------------------
    # LANGUAGE DETECTION
    # -------------------
    def detect_language(self, filename: str) -> str:
        return filename.split(".")[-1] if "." in filename else "unknown"

    # -------------------
    # EMBEDDINGS
    # -------------------
    def _embed(self, text: str):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        return self.embedding_model.encode(text[:2000]).tolist()  # truncate for performance

    def _cosine_sim(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # -------------------
    # MAIN SYNC
    # -------------------
    def add_or_update_repos(self, repos: List[Dict[str, Any]]):
        with self.driver.session() as session:
            for repo in repos:
                repo_id = f"{repo.get('owner')}/{repo.get('name')}"

                # Create or update repo node
                session.run("""
                MERGE (r:Repo {id:$id})
                SET r.updated_at=$updated
                """, id=repo_id, updated=datetime.utcnow().isoformat())

                files = repo.get("files", [])
                file_nodes = []

                # -------------------
                # CREATE FILE NODES
                # -------------------
                for file in files:
                    path = file.get("path")
                    content = file.get("content", "")
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")
                    if not content:
                        continue  # skip empty

                    file_id = f"{repo_id}:{path}"
                    content_hash = self._hash(content)
                    embedding = self._embed(content)

                    # Check existing file
                    existing = session.run(
                        "MATCH (f:File {id:$id}) RETURN f.hash as hash",
                        id=file_id
                    ).single()

                    if existing and existing["hash"] == content_hash:
                        file_nodes.append({"id": file_id, "content": content, "embedding": embedding})
                        continue

                    # Upsert file node
                    session.run("""
                    MERGE (f:File {id:$id})
                    SET f.path=$path,
                        f.content=$content,
                        f.hash=$hash,
                        f.language=$lang,
                        f.embedding=$embedding,
                        f.updated_at=$updated
                    """,
                        id=file_id,
                        path=path,
                        content=content,
                        hash=content_hash,
                        lang=self.detect_language(path),
                        embedding=embedding,
                        updated=datetime.utcnow().isoformat()
                    )

                    # Link repo → file
                    session.run("""
                    MATCH (r:Repo {id:$repo_id})
                    MATCH (f:File {id:$file_id})
                    MERGE (r)-[:HAS_FILE]->(f)
                    """,
                        repo_id=repo_id,
                        file_id=file_id
                    )

                    file_nodes.append({"id": file_id, "content": content, "embedding": embedding})

                # -------------------
                # LINK RELATED FILES
                # -------------------
                self._link_related_files(session, file_nodes)

                # -------------------
                # LINK RELATED REPOS
                # -------------------
                self._link_related_repos(session, repo_id)

                print(f"[INFO] Synced Repo with files: {repo_id}")

    # -------------------
    # FILE RELATIONSHIPS
    # -------------------
    def _link_related_files(self, session, file_nodes):
        for file in file_nodes:
            emb = file["embedding"]

            # Compare with all existing files (including other repos)
            results = session.run("""
            MATCH (f:File)
            WHERE f.embedding IS NOT NULL
            RETURN f.id AS id, f.embedding AS embedding
            """)
            for r in results:
                other_id = r["id"]
                other_emb = r["embedding"]
                if other_id == file["id"]:
                    continue
                sim = self._cosine_sim(emb, other_emb)
                if sim > 0.80:  # threshold
                    session.run("""
                    MATCH (a:File {id:$id1})
                    MATCH (b:File {id:$id2})
                    MERGE (a)-[:SEMANTICALLY_RELATED {score:$score}]->(b)
                    """, id1=file["id"], id2=other_id, score=sim)

    # -------------------
    # REPO RELATIONSHIPS
    # -------------------
    def _link_related_repos(self, session, current_repo_id):
        # Aggregate embeddings of current repo
        current_files = session.run("""
        MATCH (r:Repo {id:$id})-[:HAS_FILE]->(f)
        WHERE f.embedding IS NOT NULL
        RETURN f.embedding AS embedding
        """, id=current_repo_id)
        current_embeddings = [r["embedding"] for r in current_files]
        if not current_embeddings:
            return
        current_vector = np.mean(np.array(current_embeddings), axis=0)

        # Compare with other repos
        repos = session.run("""
        MATCH (r:Repo)
        WHERE r.id <> $id
        RETURN r.id AS id
        """, id=current_repo_id)
        for repo in repos:
            other_id = repo["id"]
            other_files = session.run("""
            MATCH (r:Repo {id:$id})-[:HAS_FILE]->(f)
            WHERE f.embedding IS NOT NULL
            RETURN f.embedding AS embedding
            """, id=other_id)
            other_embeddings = [r["embedding"] for r in other_files]
            if not other_embeddings:
                continue
            other_vector = np.mean(np.array(other_embeddings), axis=0)
            sim = self._cosine_sim(current_vector, other_vector)
            if sim > 0.75:  # repo similarity threshold
                session.run("""
                MATCH (a:Repo {id:$id1})
                MATCH (b:Repo {id:$id2})
                MERGE (a)-[:SIMILAR_REPO {score:$score}]->(b)
                """, id1=current_repo_id, id2=other_id, score=sim)

    # -------------------
    # SEARCH FILES
    # -------------------
    def search(self, question: str, limit=5):
        query = """
        MATCH (f:File)
        WHERE any(w IN split($q,' ')
        WHERE toLower(f.content) CONTAINS toLower(w))
        RETURN f.content AS content
        LIMIT $limit
        """
        with self.driver.session() as s:
            result = s.run(query, q=question, limit=limit)
            return [r["content"] for r in result]
