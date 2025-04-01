"""Similarity search module for querying document embeddings in SQLite."""

# %%
import logging
import sqlite3

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from numpy.typing import NDArray

from dv.config import embeddings as embeddings
from dv.config import settings
from dv.database import connect_to_db, sanitise_table_name
from dv.logger import setup_logging

# %%
logger = setup_logging(settings.log_level)


# %%
def get_tables(conn: sqlite3.Connection) -> list[str]:
    """
    Get all tables in the SQLite database.

    Args:
        conn: SQLite connection.

    Returns:
        list[str]: List of table names.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

    return [row[0] for row in cursor.fetchall()]


def query_similarity(
    query: str,
    embeddings_model: Embeddings | None = embeddings,
    top_k: int = 5,
    table_name: str | None = None,
) -> list[Document]:
    """
    Perform similarity search across document chunks.

    Args:
        query: The search query.
        top_k: Number of most similar results to return.
        table_name: Optional specific table to search (corresponds to a document).
                    If None, searches across all document tables.
        embeddings_model: Optional embeddings model. If None, uses default.

    Returns:
        list[Document]: List of Document objects containing chunks ordered by similarity.
    """
    # Generate embedding for the query
    if embeddings_model is None:
        raise ValueError("No embeddings model provided")
    query_embedding = embeddings_model.embed_query(query)

    # Connect to database
    conn = connect_to_db()

    try:
        # Determine which tables to search
        if table_name:
            safe_table_name = sanitise_table_name(table_name)
            tables = [safe_table_name]
        else:
            tables = get_tables(conn)

        # Collect results from all relevant tables
        all_results: list[tuple[float, str, str, str]] = []

        for table in tables:
            # Check if this table has the expected columns (to avoid system tables)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]

            if "embeddings" not in columns or "chunk_text" not in columns:
                continue

            # Retrieve all chunks and their embeddings
            cursor.execute(f"SELECT chunk_text, embeddings FROM {table}")
            rows = cursor.fetchall()

            for chunk_text, embedding_bytes in rows:
                if not chunk_text or not embedding_bytes:
                    continue

                # Convert stored embedding from bytes to vector
                stored_embedding = np.array(
                    [float(x) for x in embedding_bytes.decode("utf-8").split(",")]
                )

                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, stored_embedding)

                all_results.append((similarity, chunk_text, table, ""))

        # Sort results by similarity (highest first)
        all_results.sort(reverse=True, key=lambda x: x[0])

        # Get top_k results
        top_results = all_results[:top_k]

        # Convert to LangChain Document objects
        documents = []
        for similarity, text, source_table, _ in top_results:
            metadata = {
                "source": source_table,
                "similarity": similarity,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    finally:
        # Close connection
        conn.close()


def cosine_similarity(vec1: list[float] | NDArray, vec2: list[float] | NDArray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        float: Cosine similarity (-1 to 1, higher is more similar).
    """
    # Convert to numpy arrays if they aren't already
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    similarity = dot_product / (norm_v1 * norm_v2)

    return float(similarity)


class SQLiteVectorStore:
    """Class for performing vector similarity search with SQLite."""

    def __init__(
        self,
        embeddings_model: Embeddings | None = embeddings,
        db_path: str | None = None,
    ):
        """
        Initialize SQLiteVectorStore.

        Args:
            embeddings_model: Model to use for embedding queries.
            db_path: Path to SQLite database. If None, uses settings.SQLITE_DB_PATH.
        """
        self.embeddings_model = embeddings_model or embeddings
        self.db_path = db_path or settings.SQLITE_DB_PATH

    def similarity_search(
        self, query: str, k: int = 5, table_name: str | None = None
    ) -> list[Document]:
        """
        Perform similarity search for the given query.

        Args:
            query: The query to search for.
            k: Number of results to return.
            table_name: Specific table/document to search within.

        Returns:
            list[Document]: Most similar document chunks.
        """
        return query_similarity(
            query=query,
            embeddings_model=self.embeddings_model,
            top_k=k,
            table_name=table_name,
        )

    def similarity_search_with_score(
        self, query: str, k: int = 5, table_name: str | None = None
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search with scores.

        Args:
            query: The query to search for.
            k: Number of results to return.
            table_name: Specific table/document to search within.

        Returns:
            list[tuple[Document, float]]: Document chunks with similarity scores.
        """
        docs = self.similarity_search(query, k, table_name)

        return [(doc, doc.metadata.get("similarity", 0.0)) for doc in docs]


# %%
def create_vector_store(
    embeddings_model: Embeddings | None = embeddings,
    db_path: str | None = settings.SQLITE_DB_PATH,
) -> SQLiteVectorStore:
    """
    Create a vector store from the SQLite database.

    Args:
        embeddings_model: Optional embeddings model. If None, uses default.

    Returns:
        SQLiteVectorStore: An initialized vector store for similarity search.
    """

    return SQLiteVectorStore(embeddings_model=embeddings_model, db_path=db_path)


# %%
if __name__ == "__main__":
    # Example usage
    vectorstore = create_vector_store()
    results = vectorstore.similarity_search(
        query="trigeminal nerve",
        k=3,
        table_name="managing_your_migraine",
    )

    for i, doc in enumerate(results, 1):
        print(f"Result {i} (Similarity: {doc.metadata['similarity']:.4f})")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content}...")
        print("-" * 50)
