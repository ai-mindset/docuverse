"""Database module for saving text from documents and managing embeddings."""

# %%
import glob
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dv.config import embeddings, settings
from dv.logger import setup_logging
from dv.utils import format_docstring

# %%
logger = setup_logging(settings.log_level)


# %%
def ensure_db_directory() -> bool:
    """Create SQLite database directory if it doesn't exist."""
    data_dir = os.path.dirname(settings.SQLITE_DB_PATH)
    os.makedirs(data_dir, exist_ok=True)
    return os.path.exists(data_dir)


# %%
def connect_to_db() -> sqlite3.Connection:
    """
    Connects to SQLite database.

    Returns:
        sqlite3.Connection: Database connection.

    Raises:
        Exception: If connection fails.
    """
    try:
        ensure_db_directory()
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite database: {str(e)}")
        raise


# %%
def create_table_for_document(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Create a table for a specific document if it doesn't exist.

    Args:
        conn: SQLite connection.
        table_name: Name of the table to create.
    """
    # Sanitize table name to avoid SQL injection
    safe_table_name = sanitise_table_name(table_name)

    # Check if table already exists
    table_exists = False
    try:
        cursor = conn.execute(f"""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='{safe_table_name}'
        """)
        table_exists = cursor.fetchone() is not None

        if table_exists:
            logger.info(f"Table '{safe_table_name}' already exists")
        else:
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {safe_table_name} (
                id INTEGER PRIMARY KEY,
                raw_text TEXT NULL,
                chunk_text TEXT,
                embeddings BLOB
            )
            """)
            conn.commit()
            logger.info(f"Table '{safe_table_name}' created successfully")
    except Exception as e:
        logger.error(f"Error working with table '{safe_table_name}': {str(e)}")
        raise


# %%
def sanitise_table_name(table_name: str) -> str:
    """
    Sanitize table name for SQLite.

    Args:
        table_name: Raw table name.

    Returns:
        str: Sanitised table name.
    """
    max_words = 3
    # Split the string on whitespace, hyphens, and underscores
    raw_parts = re.split(r"[\s_-]+", table_name)

    # Take at most max_words, strip and lowercase them
    parts = [part.strip().lower() for part in raw_parts[:max_words] if part.strip()]

    # Join with underscores
    sanitised = "_".join(parts)

    # Ensure it doesn't start with a number
    if sanitised and sanitised[0].isdigit():
        sanitised = "doc_" + sanitised

    return sanitised


# %%
def get_document_files() -> list[tuple[str, str]]:
    """
    Get all document files from the books directory.

    Returns:
        list[tuple[str, str]]: List of tuples containing (file path, table name).
    """
    books_dir = settings.DOCS_DIR
    result = []

    # Get all .txt and .md files
    for ext in ["*.txt", "*.md"]:
        for file_path in glob.glob(os.path.join(books_dir, ext)):
            # Use the filename without extension as the table name
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            result.append((file_path, table_name))

    return result


# %%
def save_document(conn: sqlite3.Connection, file_path: str, table_name: str) -> bool:
    """
    Save document to SQLite database.

    Args:
        conn: SQLite connection.
        file_path: Path to the document file.
        table_name: Name of the table to save document to.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Read the document
        with open(file_path, encoding="utf-8") as file:
            text = file.read()

        # Sanitize table name
        safe_table_name = sanitise_table_name(table_name)

        # Create table if it doesn't exist
        create_table_for_document(conn, safe_table_name)

        # Check if document already exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {safe_table_name}")
        count = cursor.fetchone()[0]

        if count > 0:
            # Document exists, update it
            cursor.execute(f"DELETE FROM {safe_table_name}")
            conn.commit()
            logger.info(f"Existing document in '{safe_table_name}' cleared for update")

        # Insert the raw document text
        cursor.execute(f"INSERT INTO {safe_table_name} (raw_text) VALUES (?)", (text,))
        conn.commit()
        logger.info(f"Document '{file_path}' saved to table '{safe_table_name}'")
        return True
    except Exception as e:
        logger.error(f"Error saving document '{file_path}': {str(e)}")
        return False


# %%
def generate_and_save_embeddings(conn: sqlite3.Connection) -> bool:
    """
    Generate and save embeddings for all documents in the database.

    Args:
        conn: SQLite connection.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.SEPARATORS,
        )

        # Get all tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Get documents without embeddings
            cursor.execute(f"SELECT id, raw_text FROM {table} WHERE embeddings IS NULL")
            docs = cursor.fetchall()

            for doc_id, text in docs:
                if text:
                    # Split text into chunks
                    chunks = text_splitter.split_text(text)

                    # For each chunk, generate embeddings and update the database
                    if not embeddings:
                        raise ValueError("No embeddings models has been passed")
                    for i, chunk in enumerate(chunks):
                        # Generate embedding
                        vector = embeddings.embed_query(chunk)

                        # Convert vector to bytes for storage
                        vector_bytes = bytes(",".join(map(str, vector)), "utf-8")

                        # Insert new row with chunk and embedding, using a calculated ID
                        # based on the original doc_id and chunk index to avoid AUTOINCREMENT
                        # primary key declaration that automatically generates a 'sqlite_sequence' table
                        new_id = (doc_id * 10000) + i + 1
                        raw_text = text if i == 0 else None
                        cursor.execute(
                            f"INSERT INTO {table} (id, raw_text, chunk_text, embeddings) VALUES (?, ?, ?, ?)",
                            (new_id, raw_text, chunk, vector_bytes),
                        )

                    # Remove the original row that had just the raw text
                    cursor.execute(f"DELETE FROM {table} WHERE id = ?", (doc_id,))

                    conn.commit()
                    logger.info(f"Embeddings generated for document in table '{table}'")

        logger.info("Document embeddings complete")
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return False


# %%
def process_all_documents() -> bool:
    """
    Process all documents in the books directory.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Connect to database
        conn = connect_to_db()

        # Get all document files
        document_files = get_document_files()

        if not document_files:
            logger.warning("No document files found in 'books/' directory")
            return False

        # Save each document
        for file_path, table_name in document_files:
            save_document(conn, file_path, table_name)

        # Generate embeddings for all documents
        success = generate_and_save_embeddings(conn)

        # Close connection
        conn.close()

        return success
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return False


# %%
if __name__ == "__main__":
    # Update settings for SQLite
    if not hasattr(settings, "SQLITE_DB_PATH"):
        settings.SQLITE_DB_PATH = "data/books.db"

    # Process all documents
    success = process_all_documents()

    if success:
        logger.info("All documents processed successfully")
    else:
        logger.error("Error processing documents")
