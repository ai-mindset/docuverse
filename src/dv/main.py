"""
Main entry point for the Document Q&A System.

This module orchestrates the document processing pipeline and launches
either the GUI or CLI interface based on user preference.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional

from dv.config import settings
from dv.database import connect_to_db, process_all_documents
from dv.logger import setup_logging

# Set up logging
logger = setup_logging(settings.log_level)


def check_database_exists() -> bool:
    """
    Check if the SQLite database file exists.

    Returns:
        bool: True if the database file exists, False otherwise.
    """
    return os.path.exists(settings.SQLITE_DB_PATH)


def check_database_populated() -> bool:
    """
    Check if the database has tables with data.

    Returns:
        bool: True if the database has populated tables, False otherwise.
    """
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Check if there are any tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        if not tables:
            logger.info("Database exists but has no tables")
            conn.close()
            return False

        # Check if the tables have content
        for table_name in [table[0] for table in tables]:
            # Skip system tables
            if table_name.startswith("sqlite_"):
                continue

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]

            if count > 0:
                conn.close()
                return True

        logger.info("Database has tables but they are empty")
        conn.close()
        return False

    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return False


def check_docs_directory() -> bool:
    """
    Check if the docs directory exists and contains documents.

    Returns:
        bool: True if the docs directory exists and contains documents, False otherwise.
    """
    docs_dir = Path(settings.DOCS_DIR)

    # Check if directory exists
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.info(f"Docs directory '{docs_dir}' does not exist")
        return False

    # Check if directory contains .txt or .md files
    txt_files = list(docs_dir.glob("*.txt"))
    md_files = list(docs_dir.glob("*.md"))

    if not txt_files and not md_files:
        logger.info(
            f"Docs directory '{docs_dir}' exists but contains no .txt or .md files"
        )
        return False

    return True


def initialize_database() -> bool:
    """
    Initialize the database by processing all documents.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    logger.info("Initializing database...")

    # Ensure docs directory exists
    docs_dir = Path(settings.DOCS_DIR)
    docs_dir.mkdir(exist_ok=True)

    # Process all documents
    success = process_all_documents()

    if success:
        logger.info("Database initialized successfully")
    else:
        logger.error("Failed to initialize database")

    return success


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Document Q&A System")

    parser.add_argument(
        "--cli", action="store_true", help="Use command-line interface instead of GUI"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=settings.LLM_MODEL,
        help=f"Ollama model to use (default: {settings.LLM_MODEL})",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM (0-1) (default: 0.1)",
    )

    parser.add_argument(
        "--results",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)",
    )

    parser.add_argument(
        "--reindex", action="store_true", help="Force reindexing of all documents"
    )

    parser.add_argument(
        "--light-mode",
        action="store_true",
        help="Use light mode for GUI (ignored in CLI mode)",
    )

    return parser.parse_args()


def exit_with_error(message: str, exit_code: int = 1) -> NoReturn:
    """
    Print error message and exit the program with the specified exit code.

    Args:
        message: Error message to display.
        exit_code: Exit code to return to the OS.
    """
    logger.error(message)
    print(f"Error: {message}")
    sys.exit(exit_code)


def run_cli(model: str, temperature: float, results: int) -> None:
    """
    Run the CLI interface.

    Args:
        model: Name of the Ollama model to use.
        temperature: Temperature for the LLM (0-1).
        results: Number of documents to retrieve.
    """
    from dv.qa import create_qa_chain

    # Create the QA chain
    qa = create_qa_chain(model_name=model, temperature=temperature, k=results)

    print(f"Q&A system initialized with model: {model}")
    print("Type 'exit' to quit or 'reset' to start a new conversation")
    print("-" * 50)

    # Interactive loop
    while True:
        try:
            # Get user input
            question = input("\nQuestion: ").strip()

            # Check for exit command
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Check for reset command
            if question.lower() == "reset":
                qa.reset_chat_history()
                print("Conversation has been reset.")
                continue

            # Skip empty questions
            if not question:
                continue

            # Process the question
            answer = qa.query(question)
            print("\nAnswer:", answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in CLI: {str(e)}")
            print(f"An error occurred: {str(e)}")


def run_gui(model: str, light_mode: bool) -> None:
    """
    Run the GUI interface.

    Args:
        model: Name of the Ollama model to use.
        light_mode: Whether to use light mode instead of dark mode.
    """
    import customtkinter as ctk

    from dv.gui import QAApplication

    # Update model in settings
    settings.LLM_MODEL = model

    # Set appearance mode
    ctk.set_appearance_mode("light" if light_mode else "dark")

    # Create and run the application
    root = ctk.CTk()
    app = QAApplication(root)
    root.mainloop()


def main() -> int:
    """
    Main entry point for the application.

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # Parse command line arguments
    args = parse_arguments()

    # Update settings based on arguments
    if args.model:
        settings.LLM_MODEL = args.model

    # Check if docs directory exists and has documents
    docs_exist = check_docs_directory()
    if not docs_exist:
        print("No documents found in docs directory.")
        print(f"Please add .txt or .md files to '{settings.DOCS_DIR}' directory.")
        return 1

    # Check database status
    db_exists = check_database_exists()
    db_populated = db_exists and check_database_populated()

    # Handle database initialization
    if args.reindex or not db_populated:
        if args.reindex:
            logger.info("Forced reindexing of documents")
        elif not db_exists:
            logger.info("Database does not exist, creating new database")
        elif not db_populated:
            logger.info("Database exists but is empty or outdated")

        # Initialize/reindex the database
        success = initialize_database()
        if not success:
            return exit_with_error("Failed to initialize database")

    # Launch the appropriate interface
    try:
        if args.cli:
            run_cli(model=args.model, temperature=args.temperature, results=args.results)
        else:
            run_gui(model=args.model, light_mode=args.light_mode)

        return 0

    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        return 0

    except Exception as e:
        return exit_with_error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    sys.exit(main())
