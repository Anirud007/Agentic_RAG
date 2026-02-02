"""
Ingestion via Milvus MCP: chunk → embed → create collections → insert (all via MCP).

Uses doc_chunker, doc_embeddings, and milvus_mcp_client so retrieval and ingestion
are handled by the Milvus MCP server.

On Milvus Cloud, newly created collections can take a few seconds to appear; we retry
load_collection and treat it as best-effort so insert can still proceed.
"""

import logging
from pathlib import Path

try:
    from ingest.doc_chunker import Chunk, chunk_files
    from ingest.doc_embeddings import (
        COLLECTION_IMAGE,
        COLLECTION_TEXT,
        BGE_DIM,
        EmbeddingResult,
        EmbeddingService,
        SIGLIP_DIM,
        embed_chunks,
        get_collection_names,
    )
    from ingest.milvus_direct_client import MilvusDirectClient
except ImportError:
    from doc_chunker import Chunk, chunk_files
    from doc_embeddings import (
        COLLECTION_IMAGE,
        COLLECTION_TEXT,
        BGE_DIM,
        EmbeddingResult,
        EmbeddingService,
        SIGLIP_DIM,
        embed_chunks,
        get_collection_names,
    )
    from milvus_direct_client import MilvusDirectClient

logger = logging.getLogger(__name__)

# Default scalar fields we store (with quick-setup we only have id + vector; optional custom schema)
TEXT_COLLECTION_FIELDS = (
    "id", "vector", "content", "content_length", "token_count",
    "location", "section_title", "source_file", "document_type",
    "page_number", "slide_number", "sheet_name",
    "table_id", "column_names", "chunk_index"
)
IMAGE_COLLECTION_FIELDS = (
    "id", "vector", "source_file", "document_type",
    "location", "page_number", "slide_number"
)


def _ensure_collections(
    client: MilvusDirectClient,
    use_custom_schema: bool = False,
    text_dim: int = BGE_DIM,
    image_dim: int = SIGLIP_DIM,
    thread_id: str | None = None,
) -> tuple[str, str]:
    """
    Create session-specific collections if they do not exist.
    Returns (text_collection_name, image_collection_name).
    """
    text_collection, image_collection = get_collection_names(thread_id)
    existing = client.list_collections()

    if text_collection not in existing:
        if use_custom_schema:
            field_schema = [
                # Identity
                {"name": "id", "type": "VARCHAR", "max_length": 64},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": text_dim},

                # Content
                {"name": "content", "type": "VARCHAR", "max_length": 65535},
                {"name": "content_length", "type": "INT32"},
                {"name": "token_count", "type": "INT32"},

                # Source
                {"name": "source_file", "type": "VARCHAR", "max_length": 512},
                {"name": "document_type", "type": "VARCHAR", "max_length": 16},
                {"name": "document_role", "type": "VARCHAR", "max_length": 128},

                # Location
                {"name": "location", "type": "VARCHAR", "max_length": 256},
                {"name": "page_number", "type": "INT32"},
                {"name": "slide_number", "type": "INT32"},
                {"name": "sheet_name", "type": "VARCHAR", "max_length": 128},

                # Section
                {"name": "section_title", "type": "VARCHAR", "max_length": 512},
                {"name": "chunk_index", "type": "INT32"},

                # Table
                {"name": "table_id", "type": "VARCHAR", "max_length": 128},
                {"name": "column_names", "type": "VARCHAR", "max_length": 2048},
            ]
            client.create_collection(
                text_collection,
                dimension=text_dim,
                auto_id=False,
                field_schema=field_schema,
            )
        else:
            client.create_collection(text_collection, dimension=text_dim, auto_id=True)
        logger.info("Created collection %s (dim=%s)", text_collection, text_dim)

    if image_collection not in existing:
        if use_custom_schema:
            field_schema = [
                # Identity
                {"name": "id", "type": "VARCHAR", "max_length": 64},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": image_dim},

                # Source
                {"name": "source_file", "type": "VARCHAR", "max_length": 512},
                {"name": "document_type", "type": "VARCHAR", "max_length": 16},
                {"name": "document_role", "type": "VARCHAR", "max_length": 128},

                # Location
                {"name": "location", "type": "VARCHAR", "max_length": 256},
                {"name": "page_number", "type": "INT32"},
                {"name": "slide_number", "type": "INT32"},
            ]
            client.create_collection(
                image_collection,
                dimension=image_dim,
                auto_id=False,
                field_schema=field_schema,
            )
        else:
            client.create_collection(image_collection, dimension=image_dim, auto_id=True)
        logger.info("Created collection %s", image_collection)

    return text_collection, image_collection


def _results_by_collection(results: list[EmbeddingResult]) -> dict[str, list[EmbeddingResult]]:
    out: dict[str, list[EmbeddingResult]] = {}
    for r in results:
        out.setdefault(r.collection, []).append(r)
    return out


def index_embedding_results(
    client: MilvusDirectClient,
    results: list[EmbeddingResult],
    *,
    batch_size: int = 32,
    use_custom_schema: bool = False,
    thread_id: str | None = None,
) -> None:
    """
    Insert EmbeddingResults into Milvus (session-specific collections).
    Ensures collections exist, then inserts in batches (no load before insert on serverless).
    """
    if not results:
        return
    text_dim = next((r.embedding_dim for r in results if r.collection == COLLECTION_TEXT), BGE_DIM)
    image_dim = next((r.embedding_dim for r in results if r.collection == COLLECTION_IMAGE), SIGLIP_DIM)
    text_collection, image_collection = _ensure_collections(
        client, use_custom_schema=use_custom_schema, text_dim=text_dim, image_dim=image_dim, thread_id=thread_id
    )

    # Map default collection names to session-specific ones
    collection_map = {
        COLLECTION_TEXT: text_collection,
        COLLECTION_IMAGE: image_collection,
    }

    by_coll = _results_by_collection(results)
    for collection_name, items in by_coll.items():
        # Initialize data lists
        ids: list[str] = []
        vectors: list[list[float]] = []
        contents: list[str] = []
        content_lengths: list[int] = []
        token_counts: list[int] = []
        locations: list[str] = []
        section_titles: list[str] = []
        source_files: list[str] = []
        document_types: list[str] = []
        document_roles: list[str] = []
        page_numbers: list[int] = []
        slide_numbers: list[int] = []
        sheet_names: list[str] = []
        table_ids: list[str] = []
        column_names_list: list[str] = []
        chunk_indices: list[int] = []

        for r in items:
            chunk = r.chunk
            ids.append(chunk.chunk_id or str(len(ids)))
            emb = r.embedding
            if getattr(emb, "ndim", 1) > 1:
                emb = emb.ravel()
            vectors.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
            contents.append(chunk.content[:65535] if chunk.content else "")
            content_lengths.append(chunk.content_length or len(chunk.content))
            token_counts.append(chunk.token_count or 0)
            locations.append(chunk.location or "")
            section_titles.append(chunk.section_title or "")
            source_files.append(chunk.source_file or "")
            document_types.append(chunk.document_type or "")
            document_roles.append(getattr(chunk, "document_role", "") or "")
            page_numbers.append(chunk.page_number or 0)
            slide_numbers.append(chunk.slide_number or 0)
            sheet_names.append(chunk.sheet_name or "")
            table_ids.append(chunk.table_id or "")
            # Serialize column names as comma-separated string
            col_names = chunk.column_names or []
            column_names_list.append(",".join(col_names) if col_names else "")
            chunk_indices.append(chunk.chunk_index or 0)

        data: dict[str, list] = {"vector": vectors}

        # Always include essential fields for retrieval (content, location, source_file, section_title)
        if collection_name == COLLECTION_TEXT:
            data["content"] = contents
            data["location"] = locations
            data["section_title"] = section_titles
            data["source_file"] = source_files
        else:
            # Image collection
            data["source_file"] = source_files
            data["location"] = locations

        # Add additional fields only for custom schema
        if use_custom_schema:
            data["id"] = ids
            if collection_name == COLLECTION_TEXT:
                data["content_length"] = content_lengths
                data["token_count"] = token_counts
                data["document_type"] = document_types
                data["document_role"] = document_roles
                data["page_number"] = page_numbers
                data["slide_number"] = slide_numbers
                data["sheet_name"] = sheet_names
                data["table_id"] = table_ids
                data["column_names"] = column_names_list
                data["chunk_index"] = chunk_indices
            else:
                data["document_type"] = document_types
                data["document_role"] = document_roles
                data["page_number"] = page_numbers
                data["slide_number"] = slide_numbers

        # Use session-specific collection name
        target_collection = collection_map.get(collection_name, collection_name)

        n = len(vectors)
        for i in range(0, n, batch_size):
            batch = {k: v[i : i + batch_size] for k, v in data.items()}
            client.insert_data(target_collection, batch)
        logger.info("Inserted %d items into %s", len(ids), target_collection)


def run_ingestion(
    file_paths: list[str] | list[Path],
    *,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
    max_workers_embed: int = 4,
    batch_size: int = 32,
    use_custom_schema: bool = False,
    use_light_text_model: bool = False,
    thread_id: str | None = None,
    document_role: str | None = None,
) -> list[EmbeddingResult]:
    """
    Full pipeline: chunk files → embed → create/load collections → insert via direct Milvus client.
    use_light_text_model: use BGE-small (384 dim, ~130MB) instead of BGE-M3 (1024 dim, large download).
    thread_id: creates session-specific collections (docs_text_{thread_id}).
    document_role: optional role (e.g. resume, paper, manual) applied to all chunks from this ingest.
    Returns the list of EmbeddingResults (for logging or downstream use).
    """
    paths = [Path(p) for p in file_paths]
    chunks = chunk_files(paths)
    if document_role:
        for c in chunks:
            c.document_role = document_role
    if not chunks:
        logger.warning("No chunks produced from %s", file_paths)
        return []

    svc = EmbeddingService(use_light_text_model=use_light_text_model)
    results = svc.embed_chunks(chunks, max_workers=max_workers_embed)

    client = MilvusDirectClient(uri=milvus_uri, token=milvus_token)
    index_embedding_results(
        client,
        results,
        batch_size=batch_size,
        use_custom_schema=use_custom_schema,
        thread_id=thread_id,
    )
    return results
