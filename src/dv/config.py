"""Configuration file for the application."""

# %% [markdown]
# ## Imports

# %%
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from os import getenv, path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from ollama import Client
from pydantic import BaseModel, ConfigDict, Field, computed_field

# %%
load_dotenv()
home = path.expanduser("~")


# %% [markdown]
# ## Pydantic data validation


# %%
class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    log_level: str = Field(default="INFO")
    log_file: str = "app.log"

    CHUNK_SIZE: int = Field(default=1000)

    @computed_field
    @property
    def CHUNK_OVERLAP(self) -> int:  # noqa: N802
        return int(self.CHUNK_SIZE * 0.2)

    SEPARATORS: list[str] = Field(default=["\n\n", "\n", ".", " ", ""])
    SQLITE_DB_PATH: str = Field(default="db/books.db")
    DOCS_DIR: str = Field(default="docs")
    CLIENT: Client = Field(default=Client(host="http://localhost:11434"))
    LLM_MODEL: str = Field(default="mistral")

    PROMPTS_DIR: str = Field(default="prompts/")

    @computed_field
    @property
    def SYS_PROMPT(self) -> str:  # noqa: N802
        sys_prompt_content = ""
        for filename in os.listdir(self.PROMPTS_DIR):
            if filename.startswith(("sys_", "system_")) and (
                filename.endswith(".md") or filename.endswith(".txt")
            ):
                with open(
                    os.path.join(self.PROMPTS_DIR, filename), encoding="utf-8"
                ) as file:
                    sys_prompt_content += file.read() + "\n\n"
        return sys_prompt_content.strip()

    GUI_FONT: Mapping[str, int | str] = Field(
        default={
            "size": 18,
            "weight": "bold",
        }
    )
    EXIT_KEYWORDS: list[str] = ["bye", "exit", "goodbye", "quit"]


# %%
class Document(TypedDict):
    _id: str
    content: str
    created_at: datetime
    metadata: dict[str, Any]


# %%
settings = Settings()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
