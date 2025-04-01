"""
Main entry point for the Document Q&A System.

This module orchestrates the document processing pipeline and launches
either the GUI or CLI interface based on user preference.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from dv.cli import main as cli_main
from dv.config import settings
from dv.database import connect_to_db, process_all_documents
from dv.gui import main as gui_main
from dv.logger import setup_logging
from dv.qa import create_qa_chain

# %%
# Ensure EXIT_KEYWORDS is in settings
if not hasattr(settings, "EXIT_KEYWORDS"):
    settings.EXIT_KEYWORDS = ["bye", "exit", "goodbye", "quit"]

# Set up logging
logger = setup_logging(settings.log_level)


# %%
def check_database_exists() -> bool:
    """
    Check if the SQLite database file exists.

    Returns:
        bool: True if the database file exists, False otherwise.
    """
    return os.path.exists(settings.SQLITE_DB_PATH)


# %%
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


# %%
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


# %%
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


# %%
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Q&A System")

    # Common arguments for both CLI and GUI
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

    # Interface selection
    parser.add_argument(
        "--cli", action="store_true", help="Use command-line interface instead of GUI"
    )

    # GUI-specific arguments
    parser.add_argument(
        "--light-mode",
        action="store_true",
        help="Use light mode for GUI (ignored in CLI mode)",
    )

    # Store args in settings for use throughout the application
    args = parser.parse_args()

    # Update settings with the parsed arguments
    settings.TEMP = args.temperature
    settings.RESULTS = args.results

    return args


# %%
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


# %%
def gui_prompt_loop(args: argparse.Namespace) -> int:
    """
    Custom implementation of GUI prompt loop with exit keyword handling.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    import customtkinter as ctk

    from dv.gui import QAApplication

    # Set appearance mode and default colour theme
    ctk.set_appearance_mode("light" if args.light_mode else "dark")
    ctk.set_default_color_theme("blue")

    try:
        # Create the root window
        root = ctk.CTk()

        # Custom QAApplication with exit keyword handling
        class EnhancedQAApplication(QAApplication):
            def __init__(self, root):
                """Initialize with added exit keyword handling."""
                super().__init__(root)

            def on_send_click(self, event=None):
                """Override to handle exit keywords."""
                question = self.question_entry.get().strip()

                if not question:
                    # Flash the entry widget to indicate it's empty
                    original_fg = self.question_entry.cget("fg_color")
                    self.question_entry.configure(fg_color="#3d1a1a")
                    self.root.after(
                        100, lambda: self.question_entry.configure(fg_color=original_fg)
                    )
                    return

                # Check for exit keywords
                if question.lower() in (
                    keyword.lower() for keyword in settings.EXIT_KEYWORDS
                ):
                    print("Goodbye!")
                    self.root.quit()
                    return

                # Clear the entry
                self.question_entry.delete(0, "end")

                # Update chat with user's question
                self.update_chat("User", question)

                # Disable UI elements during processing
                self.toggle_ui(False)

                # Start processing animation
                self.start_processing_animation()

                # Process query in a separate thread
                import threading

                threading.Thread(
                    target=self.process_query, args=(question,), daemon=True
                ).start()

        # Create and run the application
        app = EnhancedQAApplication(root)

        # Add window close event handling
        def on_closing():
            """Handle window closing event."""
            print("Goodbye!")
            root.quit()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Start the GUI main loop
        root.mainloop()
        return 0

    except Exception as e:
        logger.error(f"Error in GUI: {str(e)}")
        return exit_with_error(f"GUI error: {str(e)}")


# %%
def cli_prompt_loop() -> int:
    """
    Custom implementation of CLI prompt loop with exit keyword handling.

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # Parse the CLI-specific arguments
    args = parse_arguments()

    # Create the QA chain
    qa = create_qa_chain(
        model_name=args.model, temperature=args.temperature, k=args.results
    )

    print(f"Q&A system initialized with model: {args.model}")
    print("Type 'exit' to quit or 'reset' to start a new conversation")
    print("-" * 50)

    # Interactive loop
    while True:
        try:
            # Get user input
            question = input("\nQuestion: ").strip()

            # Check for exit keywords
            if question.lower() in (
                keyword.lower() for keyword in settings.EXIT_KEYWORDS
            ):
                print("Goodbye!")
                return 0

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
            return 0
        except Exception as e:
            logger.error(f"Error in CLI: {str(e)}")
            print(f"An error occurred: {str(e)}")
            return 1


# %%
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
            # Use our custom CLI implementation that handles exit keywords properly
            return cli_main(args=args)
        else:
            # Use our custom GUI implementation that handles exit keywords properly
            return gui_prompt_loop(args)

    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        return 0

    except Exception as e:
        return exit_with_error(f"An unexpected error occurred: {str(e)}")


# %%
if __name__ == "__main__":
    sys.exit(main())
