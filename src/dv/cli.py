"""Command-line interface for interacting with the document Q&A system."""

import sys

from dv.config import settings
from dv.database import process_all_documents
from dv.logger import setup_logging
from dv.qa import create_qa_chain

# %%
logger = setup_logging(settings.log_level)


# %%
def main(args=None) -> int:
    """Run the interactive Q&A system."""
    # Use provided args or parse them if not provided
    # Check if no arguments were passed
    if args is None:
        raise ValueError(
            "No arguments provided to main(). Please provide the necessary arguments."
        )

    # Reindex documents if requested
    if args.reindex:
        print("Indexing documents...")
        success = process_all_documents()
        if not success:
            print("Error indexing documents. Exiting.")
            sys.exit(1)
        print("Indexing complete.")

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

            # Check for exit command
            if question.lower() in settings.EXIT_KEYWORDS:
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
            return 0
        except Exception as e:
            logger.error(f"Error in CLI: {str(e)}")
            print(f"An error occurred: {str(e)}")


# %%
if __name__ == "__main__":
    main()
