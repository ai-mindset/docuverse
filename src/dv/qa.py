"""Q&A chain module to enable conversational retrieval over document embeddings."""

from typing import Any

import ipdb
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import AddableDict
from langchain_ollama import ChatOllama

from dv.config import settings
from dv.logger import setup_logging
from dv.similarity_search import create_vector_store

# %%
logger = setup_logging(settings.log_level)


# %%
class QAChain:
    """Conversational Q&A chain using document similarity search and LLM."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        k: int = 3,
    ):
        """
        Initialize the QA chain.

        Args:
            model_name: Name of the Ollama model to use.
            temperature: Temperature for the LLM (0-1).
            k: Number of similar documents to retrieve for context.
        """
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature
        self.k = k
        self.vectorstore = create_vector_store()
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            client=settings.CLIENT,
        )
        self.chat_history = []
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> Any:
        """
        Create the Q&A chain with memory.

        Returns:
            A runnable chain for conversational Q&A.
        """
        # Define the context retriever
        retriever = self._get_retriever()

        # Define the system prompt template
        system_template = settings.SYS_PROMPT
        system_template += """
            Context: {context} 
            Chat History: {chat_history}
            Question: {question}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        # Build the chain
        qa_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self.chat_history,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return qa_chain

    def _get_retriever(self) -> Any:
        """
        Create a retriever function to get relevant document chunks.

        Returns:
            Function that retrieves and formats related documents.
        """

        def retrieve_and_format(query: str) -> str:
            """Retrieve similar documents and format them as context."""
            docs = self.vectorstore.similarity_search(query=query, k=self.k)
            formatted_docs = []

            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                similarity = doc.metadata.get("similarity", 0.0)
                text = doc.page_content.strip()
                formatted = f"Document {i} (Source: {source}, Relevance: {similarity:.2f}):\n{text}\n"
                formatted_docs.append(formatted)

            return "\n".join(formatted_docs)

        return retrieve_and_format

    def add_message(self, message: str, is_human: bool = True) -> None:
        """
        Add a message to the chat history.

        Args:
            message: The message text.
            is_human: Whether the message is from the human (True) or AI (False).
        """
        if is_human:
            self.chat_history.append(HumanMessage(content=message))
        else:
            self.chat_history.append(AIMessage(content=message))

    def query(self, question: str) -> str:
        """
        Process a query and return the answer.

        Args:
            question: The question to answer.

        Returns:
            str: The answer to the question.
        """
        try:
            logger.info(f"Processing query: {question}")

            # Run the chain
            answer = self.qa_chain.invoke(question)

            # Add the Q&A pair to chat history
            self.add_message(question, is_human=True)
            self.add_message(answer, is_human=False)

            return answer
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, an error occurred: {error_msg}"

    def reset_chat_history(self) -> None:
        """Reset the chat history."""
        self.chat_history = []
        logger.info("Chat history has been reset")

    def get_context_for_query(self, query: str) -> list[Document]:
        """
        Get the retrieved documents for a specific query.

        Args:
            query: The query to retrieve context for.

        Returns:
            List[Document]: The retrieved documents.
        """
        docs = self.vectorstore.similarity_search(query=query, k=self.k)
        return docs


def create_qa_chain(
    model_name: str | None = None,
    temperature: float = 0.1,
    k: int = 3,
) -> QAChain:
    """
    Create a new QA chain instance.

    Args:
        model_name: Name of the Ollama model to use.
        temperature: Temperature for the LLM (0-1).
        k: Number of similar documents to retrieve for context.

    Returns:
        QAChain: An initialized QA chain.
    """
    return QAChain(model_name=model_name, temperature=temperature, k=k)


if __name__ == "__main__":
    # Example usage
    qa = create_qa_chain()
    questions = {
        "q1": "What treatments are recommended for migraines?",
        "q2": "Are there any preventive measures mentioned? If so, list all of them",
    }

    # First question
    q1 = questions["q1"]
    answer1 = qa.query(question=q1)
    print(f"Q: {questions['q1']}")
    print(f"A: {answer1}")
    print("-" * 50)

    # Follow-up question (uses conversation context)
    q2 = questions["q2"]
    answer2 = qa.query(question=q2)
    retrieved_docs = qa.get_context_for_query(query=q2)
    ipdb.set_trace()
    print(f"Q: {questions['q2']}")
    print(f"A: {answer2}")
