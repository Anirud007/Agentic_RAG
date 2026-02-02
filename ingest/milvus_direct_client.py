"""
Direct Milvus client using pymilvus - bypasses MCP subprocess issues.

This client connects directly to Milvus/Zilliz Cloud and provides the same
interface as MilvusMCPClient but without needing the MCP server subprocess.

Priority: Zilliz Cloud (if MILVUS_URI + MILVUS_TOKEN set) > Milvus Lite (local)
"""

import logging
import os
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

# Get project root for local Milvus Lite database
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MILVUS_DB = str(PROJECT_ROOT / "milvus_data" / "milvus.db")

# Environment configuration
ENV_MILVUS_URI = os.environ.get("MILVUS_URI", "").strip()
ENV_MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "").strip()


class MilvusDirectClient:
    """
    Direct Milvus client using pymilvus.
    Prioritizes Zilliz Cloud when credentials are available, falls back to local Milvus Lite.
    """

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        **kwargs,  # Accept extra args for compatibility
    ):
        # Determine configuration: explicit params > env vars > local fallback
        self._token = token or ENV_MILVUS_TOKEN

        if uri:
            self._uri = uri
        elif ENV_MILVUS_URI and self._token:
            # Cloud credentials available
            self._uri = ENV_MILVUS_URI
        else:
            # Fallback to local Milvus Lite
            self._uri = LOCAL_MILVUS_DB

        self._client = None
        self._is_cloud = bool(self._uri and not self._uri.endswith(".db"))

        # Ensure local database directory exists for Milvus Lite
        if not self._is_cloud:
            db_dir = Path(self._uri).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Using Milvus Lite at %s", self._uri)
        else:
            logger.info("Using Zilliz Cloud at %s", self._uri)

    def _get_client(self) -> MilvusClient:
        """Get or create Milvus client connection with auth error handling."""
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self._uri, token=self._token)
                logger.info("Connected to Milvus at %s", self._uri)
            except Exception as e:
                error_msg = str(e).lower()
                if any(kw in error_msg for kw in ["auth", "unauthorized", "403", "401", "forbidden"]):
                    raise ConnectionError(
                        f"Authentication failed for Milvus at {self._uri}. "
                        "Check your MILVUS_TOKEN in .env file."
                    ) from e
                if any(kw in error_msg for kw in ["connection", "timeout", "refused", "unreachable"]):
                    raise ConnectionError(
                        f"Failed to connect to Milvus at {self._uri}. "
                        "Check your MILVUS_URI and network connectivity."
                    ) from e
                raise
        return self._client

    def list_collections(self) -> list[str]:
        """List all collection names in Milvus."""
        client = self._get_client()
        return client.list_collections()

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        *,
        primary_field_name: str = "id",
        vector_field_name: str = "vector",
        metric_type: str = "COSINE",
        auto_id: bool = False,
        field_schema: list[dict] | None = None,
    ) -> Any:
        """Create a collection with optional custom schema."""
        client = self._get_client()

        # Check if collection exists
        if collection_name in client.list_collections():
            logger.info("Collection %s already exists", collection_name)
            return {"status": "exists"}

        if field_schema:
            # Create with custom schema
            from pymilvus import CollectionSchema, FieldSchema

            fields = []
            for f in field_schema:
                name = f["name"]
                ftype = f["type"].upper()

                if ftype == "VARCHAR":
                    max_len = f.get("max_length", 65535)
                    field = FieldSchema(
                        name=name,
                        dtype=DataType.VARCHAR,
                        max_length=max_len,
                        is_primary=(name == primary_field_name),
                        auto_id=(name == primary_field_name and auto_id),
                    )
                elif ftype == "FLOAT_VECTOR":
                    dim = f.get("dim", dimension)
                    field = FieldSchema(
                        name=name,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=dim,
                    )
                elif ftype == "INT32":
                    field = FieldSchema(name=name, dtype=DataType.INT32)
                elif ftype == "INT64":
                    field = FieldSchema(
                        name=name,
                        dtype=DataType.INT64,
                        is_primary=(name == primary_field_name),
                        auto_id=(name == primary_field_name and auto_id),
                    )
                else:
                    field = FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=256)

                fields.append(field)

            schema = CollectionSchema(fields=fields)
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
            # Create index on vector field (MilvusClient expects IndexParams, not dict)
            index_params = client.prepare_index_params(
                vector_field_name,
                index_type="AUTOINDEX",
                metric_type=metric_type,
            )
            client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )
        else:
            # Quick setup with auto schema
            client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                auto_id=auto_id,
                primary_field_name=primary_field_name,
                vector_field_name=vector_field_name,
            )

        logger.info("Created collection %s (dim=%d)", collection_name, dimension)
        return {"status": "created"}

    def load_collection(self, collection_name: str, replica_number: int = 1) -> Any:
        """Load collection into memory (serverless handles this automatically)."""
        client = self._get_client()
        client.load_collection(collection_name)
        return {"status": "loaded"}

    def insert_data(self, collection_name: str, data: dict[str, list]) -> Any:
        """Insert data into a collection."""
        client = self._get_client()

        # Convert dict format to list of dicts for pymilvus
        if not data:
            return {"insert_count": 0}

        keys = list(data.keys())
        n = len(data[keys[0]])
        rows = []
        for i in range(n):
            row = {k: data[k][i] for k in keys}
            rows.append(row)

        result = client.insert(collection_name=collection_name, data=rows)
        logger.info("Inserted %d rows into %s", n, collection_name)
        return {"insert_count": n, "ids": result.get("ids", [])}

    def vector_search(
        self,
        collection_name: str,
        vector: list[float],
        *,
        limit: int = 10,
        vector_field: str = "vector",
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
        metric_type: str = "COSINE",
    ) -> Any:
        """Run vector similarity search. Returns list of dicts with entity fields."""
        client = self._get_client()

        try:
            results = client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                limit=limit,
                output_fields=output_fields or ["*"],
                filter=filter_expr or "",
            )
        except Exception as e:
            logger.warning("Vector search failed for %s: %s", collection_name, e)
            return []

        # Parse pymilvus search results into list of dicts
        parsed = []
        logger.info("Raw search results type: %s, value: %s", type(results), str(results)[:500])

        if results and len(results) > 0:
            first_batch = results[0]
            logger.info("First batch type: %s, len: %d", type(first_batch), len(first_batch) if hasattr(first_batch, '__len__') else -1)

            for hit in first_batch:
                entity = {}
                logger.debug("Hit type: %s, attrs: %s", type(hit), dir(hit)[:10] if hasattr(hit, '__dir__') else 'N/A')

                # Extract entity fields - pymilvus returns different formats
                if hasattr(hit, 'entity'):
                    # MilvusClient search returns hits with .entity attribute
                    entity = dict(hit.entity) if hasattr(hit.entity, 'items') else {}
                    for key in hit.entity.keys() if hasattr(hit.entity, 'keys') else []:
                        entity[key] = hit.entity.get(key)
                elif hasattr(hit, 'fields'):
                    entity = dict(hit.fields)
                elif isinstance(hit, dict):
                    entity = hit.get('entity', hit)
                else:
                    # Try to extract as dict-like
                    try:
                        entity = dict(hit)
                    except:
                        entity = {"content": str(hit)}

                # Add distance/score if available
                if hasattr(hit, 'distance'):
                    entity['_distance'] = hit.distance
                elif hasattr(hit, 'score'):
                    entity['_score'] = hit.score

                if entity:
                    parsed.append(entity)

        logger.info("Vector search on %s returned %d parsed hits", collection_name, len(parsed))
        return parsed

    def get_collection_info(self, collection_name: str) -> Any:
        """Get collection schema and metadata."""
        client = self._get_client()
        return client.describe_collection(collection_name)

    def drop_collection(self, collection_name: str) -> Any:
        """Drop a collection if it exists."""
        client = self._get_client()
        if collection_name in client.list_collections():
            client.drop_collection(collection_name)
            logger.info("Dropped collection %s", collection_name)
            return {"status": "dropped"}
        return {"status": "not_found"}

    def drop_session_collections(self, thread_id: str) -> dict:
        """Drop all collections for a session (text and image)."""
        from ingest.doc_embeddings import get_collection_names
        text_col, image_col = get_collection_names(thread_id)
        results = {}
        results["text"] = self.drop_collection(text_col)
        results["image"] = self.drop_collection(image_col)
        logger.info("Cleaned up session collections for thread_id=%s", thread_id)
        return results
