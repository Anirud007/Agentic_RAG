"""
Retrieval via Milvus: embed query → vector search.

Uses doc_embeddings for query embedding and direct Milvus client for search.
"""

import logging
import time
from typing import Any

try:
    from ingest.doc_embeddings import (
        COLLECTION_IMAGE,
        COLLECTION_TEXT,
        EmbeddingService,
        embed_chunks,
        get_collection_names,
    )
    from ingest.milvus_direct_client import MilvusDirectClient
except ImportError:
    from doc_embeddings import (
        COLLECTION_IMAGE,
        COLLECTION_TEXT,
        EmbeddingService,
        embed_chunks,
        get_collection_names,
    )
    from milvus_direct_client import MilvusDirectClient

logger = logging.getLogger(__name__)

# Retry when Milvus Cloud reports collection not found (eventual consistency)
SEARCH_RETRY_ATTEMPTS = 3
SEARCH_RETRY_DELAY = 2.0


def _is_collection_not_found(result: Any, exc: Exception | None = None) -> bool:
    """True if the response indicates collection not found (don't treat doc content as error)."""
    if exc and "not found" in str(exc).lower():
        return True
    if result is None:
        return False
    if isinstance(result, str):
        # Only treat as error if it looks like an error message, not successful "Vector search results..."
        lower = result.lower()
        if "collection not found" in lower and (lower.strip().startswith("error") or "error:" in lower[:80]):
            return True
        return False
    if isinstance(result, dict):
        msg = (result.get("message") or result.get("error") or result.get("reason") or "").lower()
        return "collection not found" in msg or "not found" in msg
    return False


class MilvusRetriever:
    """
    Retrieve from Milvus: embed query with BGE (text) or SigLIP (image),
    then run vector search through the direct Milvus client.
    """

    def __init__(
        self,
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
    ):
        self._client = MilvusDirectClient(uri=milvus_uri, token=milvus_token)
        self._embedding_svc = EmbeddingService()

    def retrieve_text(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        thread_id: str | None = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> Any:
        """
        Embed query and run vector search on session-specific collection.
        Uses thread_id to determine collection name if not explicitly provided.
        """
        # Derive collection name from thread_id if not provided
        if collection_name is None:
            collection_name, _ = get_collection_names(thread_id)

        dense, _ = self._embedding_svc.embed_text(query)
        out_fields = output_fields or [
            "content", "location", "section_title", "source_file",
            "document_type", "document_role",
        ]
        # Best-effort load (serverless may not need it or may fail)
        try:
            self._client.load_collection(collection_name)
        except Exception as e:
            logger.debug("Load collection %s (best-effort): %s", collection_name, e)
        last_result, last_exc = None, None
        for attempt in range(SEARCH_RETRY_ATTEMPTS):
            try:
                result = self._client.vector_search(
                    collection_name,
                    dense.tolist(),
                    limit=limit,
                    output_fields=out_fields,
                    filter_expr=filter_expr,
                    metric_type="COSINE",
                )
                if _is_collection_not_found(result, None):
                    last_result = result
                    preview = str(result)[:400] if result is not None else "None"
                    logger.info(
                        "Vector search attempt %d: MCP returned (collection not found?). Preview: %s",
                        attempt + 1,
                        preview,
                    )
                    if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                        time.sleep(SEARCH_RETRY_DELAY)
                    continue
                # MCP returns string; if it's an error message, don't use as context
                if isinstance(result, str) and result.strip().lower().startswith("error:"):
                    last_result = result
                    logger.warning(
                        "Vector search attempt %d: MCP returned error (e.g. dimension mismatch). %s",
                        attempt + 1,
                        result.strip()[:350],
                    )
                    if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                        time.sleep(SEARCH_RETRY_DELAY)
                    continue
                # MCP returned None (empty/unparseable response) – retry
                if result is None:
                    last_result = None
                    logger.warning(
                        "Vector search attempt %d: MCP returned no content (None). Retrying.",
                        attempt + 1,
                    )
                    if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                        time.sleep(SEARCH_RETRY_DELAY)
                    continue
                # Success: real results (string or list)
                if isinstance(result, list) and len(result) == 0:
                    logger.info("Vector search returned no hits for collection=%s", collection_name)
                return result
            except Exception as e:
                last_exc = e
                logger.warning("Vector search attempt %d raised: %s", attempt + 1, e)
                last_result = None
                # Don't re-raise: retry (timeout/MCP errors) then return [] so app shows "no docs" not crash
            if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                time.sleep(SEARCH_RETRY_DELAY)
        logger.warning(
            "Vector search failed after %d attempts. Last MCP response: %s; last error: %s. Check MILVUS_DB matches ingest DB.",
            SEARCH_RETRY_ATTEMPTS,
            (str(last_result)[:500] if last_result else "None"),
            last_exc,
        )
        return []  # empty so RAG can continue with no context

    def retrieve_image(
        self,
        image: Any,
        *,
        collection_name: str = COLLECTION_IMAGE,
        limit: int = 10,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> Any:
        """
        Embed image with SigLIP and run vector search on docs_image (via MCP).
        Load is best-effort; retries search on collection not found.
        """
        dense = self._embedding_svc.embed_image(image)
        try:
            self._client.load_collection(collection_name)
        except Exception as e:
            logger.debug("Load collection %s (best-effort): %s", collection_name, e)
        out_fields = output_fields or ["source_file", "location"]
        for attempt in range(SEARCH_RETRY_ATTEMPTS):
            try:
                result = self._client.vector_search(
                    collection_name,
                    dense.tolist(),
                    limit=limit,
                    output_fields=out_fields,
                    filter_expr=filter_expr,
                    metric_type="COSINE",
                )
                if not _is_collection_not_found(result, None):
                    return result
            except Exception as e:
                if not _is_collection_not_found(None, e):
                    raise
                if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                    time.sleep(SEARCH_RETRY_DELAY)
        return []  # empty so caller can continue

    def retrieve_image_by_text(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        thread_id: str | None = None,
        limit: int = 5,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> Any:
        """
        Embed text query with SigLIP and search image collection (cross-modal search).
        Uses same embedding space for text-to-image retrieval.
        """
        # Derive image collection name from thread_id if not provided
        if collection_name is None:
            _, collection_name = get_collection_names(thread_id)

        dense = self._embedding_svc.embed_text_for_image_search(query)
        try:
            self._client.load_collection(collection_name)
        except Exception as e:
            logger.debug("Load image collection %s (best-effort): %s", collection_name, e)

        # Image collection has no "content" field (only vector + metadata)
        out_fields = output_fields or ["source_file", "location", "document_type", "document_role"]
        for attempt in range(SEARCH_RETRY_ATTEMPTS):
            try:
                result = self._client.vector_search(
                    collection_name,
                    dense.tolist(),
                    limit=limit,
                    output_fields=out_fields,
                    filter_expr=filter_expr,
                    metric_type="COSINE",
                )
                if not _is_collection_not_found(result, None):
                    logger.info("Image search returned %d results from %s",
                               len(result) if isinstance(result, list) else 0, collection_name)
                    return result
            except Exception as e:
                if not _is_collection_not_found(None, e):
                    logger.warning("Image search failed: %s", e)
                    return []
                if attempt < SEARCH_RETRY_ATTEMPTS - 1:
                    time.sleep(SEARCH_RETRY_DELAY)
        return []  # empty so caller can continue

    def list_collections(self) -> list[str]:
        """List collection names (via MCP)."""
        return self._client.list_collections()


def retrieve_text(
    query: str,
    *,
    collection_name: str | None = None,
    thread_id: str | None = None,
    limit: int = 10,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
) -> Any:
    """One-off text retrieval from session-specific collection."""
    retriever = MilvusRetriever(milvus_uri=milvus_uri, milvus_token=milvus_token)
    return retriever.retrieve_text(query, collection_name=collection_name, thread_id=thread_id, limit=limit)


def retrieve_image(
    image: Any,
    *,
    collection_name: str = COLLECTION_IMAGE,
    limit: int = 10,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
) -> Any:
    """One-off image retrieval."""
    retriever = MilvusRetriever(milvus_uri=milvus_uri, milvus_token=milvus_token)
    return retriever.retrieve_image(image, collection_name=collection_name, limit=limit)
