"""Configuration file for the application."""

# %% [markdown]
# ## Imports

# %%
from datetime import datetime
from os import path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, ConfigDict, Field, computed_field

import dv.prompts as prompts

# %%
load_dotenv()
home = path.expanduser("~")


# %% [markdown]
# ## Pydantic data validation

# %%
# Define weight literals
WeightLiteral = Literal["normal", "bold"]


# %%
class FontSettings(TypedDict):
    size: int
    weight: WeightLiteral


# %%
# Use correctly typed constants for the default font settings
_DEFAULT_FONT_SETTINGS: FontSettings = {
    "size": 22,
    "weight": "bold",
}


# %%
# Create a function that returns the properly typed default
def get_default_font() -> FontSettings:
    return _DEFAULT_FONT_SETTINGS


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

    SEPARATORS: list[str] = Field(
        default=["\n\n", "\n", ".", " ", ""],
        description="A list of separator strings used for splitting text into segments.",
    )
    SQLITE_DB_PATH: str = Field(
        default="db/books.db",
        description="The file path to the SQLite database where book data is stored.",
    )
    DOCS_DIR: str = Field(
        default="docs",
        description="The directory path where documentation files are located.",
    )
    CLIENT: str = Field(
        default="http://localhost:11434",
        description="Ollama localhost URL",
    )
    LLM_MODEL: str = Field(
        default="mistral-nemo",
        description="The identifier for the language model used in processing text.",
    )
    TEMP: float = Field(
        default=0.3,
        description="The temperature value used in the language model to control randomness.",
    )
    RESULTS: int = Field(
        default=3,
        description="The number of results (k) returned by the query or search function.",
    )

    PROMPTS_DICT: dict[str, str] = prompts.prompts_dict

    @computed_field
    @property
    def SYS_PROMPT(self) -> str:  # noqa: N802
        key = next(
            (k for k in self.PROMPTS_DICT if "sys_prompt" in k and "info_retrieve" in k),
            "",
        )
        sys_prompt: str = self.PROMPTS_DICT.get(key, "")
        if not sys_prompt:
            raise KeyError(
                f"Required system prompt key '{key}' not found in PROMPTS_DICT"
            )
        return sys_prompt

    EXIT_KEYWORDS: list[str] = Field(default=["bye", "exit", "goodbye", "quit"])
    GUI_FONT: FontSettings = Field(default_factory=get_default_font)

    def __init__(self, **data: Any) -> None:
        # Handle GUI_FONT validation before initializing the model
        if "GUI_FONT" in data and isinstance(data["GUI_FONT"], dict):
            font_data = data["GUI_FONT"]
            if "weight" in font_data:
                weight = font_data["weight"]
                if weight not in ("normal", "bold"):
                    raise ValueError(
                        f"Font weight must be 'normal' or 'bold', got {weight}"
                    )

        super().__init__(**data)


# %%
class Document(TypedDict):
    _id: str
    content: str
    created_at: datetime
    metadata: dict[str, Any]


# %%
settings = Settings()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
