"""Hybrid search (BM25 keywords + vector similarity, fused with RRF) on Azure AI Search,
with an optional semantic ranker that re-reads the top candidates and reorders them."""
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from .config import (
    EMBEDDING_MODEL_NAME,
    SEARCH_INDEX_NAME,
    SEMANTIC_CONFIG_NAME,
    TOP_K,
    require_env,
)


@lru_cache(maxsize=None)
def get_embed_model():
    # Imported here because it is slow (it pulls in PyTorch); only paid when first needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=None)
def get_search_client():
    return SearchClient(
        endpoint=require_env("AZURE_SEARCH_ENDPOINT"),
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(require_env("AZURE_SEARCH_ADMIN_KEY")),
    )


def retrieve(question, use_semantic=False, top=TOP_K):
    """Return the top chunks for a question as [{"id": ..., "content": ...}, ...]."""
    q_vector = get_embed_model().encode(question).tolist()
    vector_query = VectorizedQuery(
        vector=q_vector, k_nearest_neighbors=top, fields="content_vector"
    )
    extra = {}
    if use_semantic:
        extra = {
            "query_type": "semantic",
            "semantic_configuration_name": SEMANTIC_CONFIG_NAME,
        }
    # search_text drives the keyword (BM25) half, vector_queries the vector half.
    results = get_search_client().search(
        search_text=question, vector_queries=[vector_query], top=top, **extra
    )
    # The result is a lazy iterator: the list is built HERE, inside the function, so the
    # network work finishes inside with_retry and a failure is caught and retried.
    return [{"id": r["id"], "content": r["content"]} for r in results]
