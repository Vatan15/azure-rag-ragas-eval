"""Ingestion: split the source document into chunks, embed them locally, and load them
into an Azure AI Search index (vector + keyword + semantic configuration).

Command line:
    python -m src.ingest --dry-run   chunk + embed locally only (no Azure calls)
    python -m src.ingest             also (re)create the index and upload the chunks
"""
import argparse
import json
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONS,
    SEARCH_INDEX_NAME,
    SEMANTIC_CONFIG_NAME,
    require_env,
)
from .search import get_embed_model

DOCUMENT_PATH = Path("data/raw_documents/drug_info_00.json")
CHUNKS_PATH = Path("data/processed/drug_info_00_chunks.json")


def load_document(path=DOCUMENT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text):
    """Split text into ~512-character chunks that overlap by 50 characters.

    The splitter tries the separators in order (paragraph, line, sentence, word,
    character), so chunks break at natural boundaries where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_chunks(chunks):
    """One 384-number vector per chunk, computed locally (no Azure call)."""
    return get_embed_model().encode(chunks)


def build_index_definition():
    """The final index: id, text, a 384-dim vector field, HNSW/cosine, semantic config."""
    fields = [
        SearchField(name="id", type=SearchFieldDataType.STRING, key=True),
        SearchField(name="content", type=SearchFieldDataType.STRING, searchable=True, retrievable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.SINGLE),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="my-vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw",
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw")
        ],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )
    return SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def build_documents(chunks, embeddings):
    """The records uploaded to the index. The id is the chunk's position, as a string."""
    return [
        {"id": str(i), "content": chunks[i], "content_vector": embeddings[i].tolist()}
        for i in range(len(chunks))
    ]


def create_index_and_upload(chunks, embeddings):
    """The only function here that talks to Azure (creates/updates the index, uploads chunks)."""
    endpoint = require_env("AZURE_SEARCH_ENDPOINT")
    credential = AzureKeyCredential(require_env("AZURE_SEARCH_ADMIN_KEY"))

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    index_client.create_or_update_index(build_index_definition())

    search_client = SearchClient(endpoint=endpoint, index_name=SEARCH_INDEX_NAME, credential=credential)
    results = search_client.upload_documents(documents=build_documents(chunks, embeddings))
    ok = sum(1 for r in results if r.succeeded)
    print(f"Uploaded OK: {ok} of {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk, embed and index the source document.")
    parser.add_argument("--dry-run", action="store_true", help="chunk and embed only; no Azure calls")
    args = parser.parse_args()

    doc = load_document()
    chunks = chunk_text(doc["markdown"])
    embeddings = embed_chunks(chunks)
    print(f"chunks: {len(chunks)} | embeddings shape: {embeddings.shape}")

    if args.dry_run:
        print("dry run: nothing written, no Azure calls made")
    else:
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        create_index_and_upload(chunks, embeddings)
