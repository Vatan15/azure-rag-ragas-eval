"""Settings shared by every module: constants and access to the .env secrets.

Importing this module never fails and never contacts Azure. Secrets are only
read (and checked) at the moment a client is actually created, via require_env().
"""
import os

from dotenv import load_dotenv

# Reads key=value lines from a local .env file into environment variables.
# The .env file is git-ignored; .env.example shows which names it must contain.
load_dotenv()

# ---- Azure OpenAI -----------------------------------------------------------
API_VERSION = "2024-12-01-preview"   # a preview api_version; see README limitations
TEMPERATURE = 0                      # deterministic-as-possible answers
MAX_TOKENS = 300                     # cap on the length of one answer

# ---- Azure AI Search / retrieval --------------------------------------------
SEARCH_INDEX_NAME = "rag-chunks"
SEMANTIC_CONFIG_NAME = "my-semantic-config"
TOP_K = 3                            # chunks passed to the model

# ---- Embeddings and chunking ------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
CHUNK_SIZE = 512                     # characters
CHUNK_OVERLAP = 50                   # characters

# ---- Retry with exponential backoff -----------------------------------------
MAX_RETRIES = 3
BASE_DELAY = 1.0                     # seconds; doubles each attempt (1, 2, 4)

# Names that must exist in .env (values are never printed anywhere).
REQUIRED_ENV = (
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_ADMIN_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
)


def require_env(name):
    """Return the value of an environment variable, or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
