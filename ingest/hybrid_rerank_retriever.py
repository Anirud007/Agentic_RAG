"""
Hybrid retriever with reranking: Milvus (vector search) → rerank → top_k.

Uses the existing MilvusRetriever to get more candidates,
then reranks with BGE reranker and returns the top_k for adaptive_rag.
"""

import logging
from typing import Any

try:
    from ingest.doc_embeddings import COLLECTION_TEXT
    from ingest.milvus_retriever import MilvusRetriever
except ImportError:
    from doc_embeddings import COLLECTION_TEXT
    from milvus_retriever import MilvusRetriever

logger = logging.getLogger(__name__)

# Reranker model (lightweight; use bge-reranker-large for better quality)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _hits_to_docs(hits: Any) -> list[dict[str, Any]]:
    """Normalize MCP/search result into list of dicts with 'content' and optional metadata."""
    if hits is None:
        return []
    if isinstance(hits, list):
        docs = []
        for item in hits:
            if isinstance(item, dict):
                content = item.get("content") or (item.get("entity") or {}).get("content")
                if content is None and "entity" in item:
                    content = (item["entity"] or {}).get("content")
                if content is not None:
                    ent = item.get("entity") or {}
                    docs.append({
                        "content": content,
                        "location": item.get("location") or ent.get("location"),
                        "section_title": item.get("section_title") or ent.get("section_title"),
                        "source_file": item.get("source_file") or ent.get("source_file"),
                        "document_type": item.get("document_type") or ent.get("document_type"),
                        "document_role": item.get("document_role") or ent.get("document_role"),
                    })
            else:
                docs.append({
                    "content": str(item),
                    "location": None,
                    "section_title": None,
                    "source_file": None,
                    "document_type": None,
                    "document_role": None,
                })
        return docs
    if isinstance(hits, dict) and "results" in hits:
        return _hits_to_docs(hits["results"])
    return []


class HybridRerankRetriever:
    """
    Retrieve from Milvus, then rerank with BGE reranker and return top_k.

    Flow: vector search (retrieve_k candidates) → rerank (query + doc pairs) → return top_k.
    """

    def __init__(
        self,
        *,
        top_k: int = 6,
        retrieve_k: int = 20,
        reranker_model: str | None = None,
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
    ):
        self._milvus = MilvusRetriever(milvus_uri=milvus_uri, milvus_token=milvus_token)
        self._top_k = max(1, top_k)
        self._retrieve_k = max(self._top_k, retrieve_k)
        self._reranker_model = reranker_model or DEFAULT_RERANKER_MODEL
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load the BGE reranker (avoids loading at import time)."""
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    self._reranker_model,
                    use_fp16=True,
                )
                logger.info("Loaded reranker: %s", self._reranker_model)
            except Exception as e:
                logger.warning("Reranker load failed (%s); returning un-reranked results.", e)
        return self._reranker

    def retrieve_text(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        thread_id: str | None = None,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> Any:
        """
        Hybrid retrieve: vector search on session-specific collection (retrieve_k) → rerank → top_k.
        Returns list of doc dicts (content, location, section_title, source_file) in relevance order.
        """
        hits = self._milvus.retrieve_text(
            query,
            collection_name=collection_name,
            thread_id=thread_id,
            limit=self._retrieve_k,
            output_fields=output_fields,
            filter_expr=filter_expr,
        )
        docs = _hits_to_docs(hits)
        if not docs:
            return []
        if len(docs) <= self._top_k:
            return docs

        reranker = self._get_reranker()
        if reranker is None:
            return docs[: self._top_k]

        # Include metadata in doc string so reranker can use document identity/role
        def _doc_text(d: dict[str, Any]) -> str:
            sf = d.get("source_file") or "unknown"
            dt = d.get("document_type") or ""
            dr = d.get("document_role") or ""
            meta = f"[Document: {sf}"
            if dt:
                meta += f"; Type: {dt}"
            if dr:
                meta += f"; Role: {dr}"
            meta += "] "
            return meta + (d.get("content") or "")
        pairs = [[query, _doc_text(d)] for d in docs]
        try:
            scores = reranker.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            elif not isinstance(scores, list):
                scores = list(scores)
        except Exception as e:
            logger.warning("Rerank compute_score failed: %s; using original order.", e)
            return docs[: self._top_k]

        # Sort by score descending; keep same length as docs
        indexed = list(zip(scores, docs))
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in indexed[: self._top_k]]


def retrieve_text_hybrid_rerank(
    query: str,
    *,
    collection_name: str | None = None,
    thread_id: str | None = None,
    top_k: int = 6,
    retrieve_k: int = 20,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
) -> Any:
    """One-off hybrid rerank retrieval from session-specific collection."""
    retriever = HybridRerankRetriever(
        top_k=top_k,
        retrieve_k=retrieve_k,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
    )
    return retriever.retrieve_text(query, collection_name=collection_name, thread_id=thread_id)
