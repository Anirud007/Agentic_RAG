"""
Milvus MCP client: run the Milvus MCP server (stdio) and call its tools from Python.

Used by the ingest indexer and retriever so ingestion and retrieval are handled via MCP.
Requires: mcp-server-milvus cloned and configured in .cursor/mcp.json (or env).
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default server config (matches .cursor/mcp.json); override with env or constructor
DEFAULT_MCP_COMMAND = os.environ.get("MILVUS_MCP_COMMAND", "/opt/homebrew/bin/uv")
DEFAULT_MCP_DIR = os.environ.get(
    "MILVUS_MCP_DIR",
    "/Users/user/Documents/Agentic RAG/mcp-server-milvus/src/mcp_server_milvus",
)
DEFAULT_MILVUS_URI = os.environ.get("MILVUS_URI", "https://in03-4851ec6903bf5b9.serverless.aws-eu-central-1.cloud.zilliz.com")
DEFAULT_MILVUS_DB = os.environ.get("MILVUS_DB", "db_4851ec6903bf5b9")


def _load_mcp_config_from_cursor() -> tuple[str, list[str], dict[str, str] | None]:
    """Load command, args, and env from project .cursor/mcp.json. Prefer project root so DB/args match ingest."""
    project_root = Path(__file__).resolve().parent.parent
    for root in [project_root, Path.cwd()]:
        mcp_path = root / ".cursor" / "mcp.json"
        if mcp_path.exists():
            try:
                with open(mcp_path) as f:
                    cfg = json.load(f)
                servers = cfg.get("mcpServers", {}).get("milvus", {})
                cmd = servers.get("command", DEFAULT_MCP_COMMAND)
                args = servers.get("args", [])
                env = servers.get("env")
                if isinstance(env, dict):
                    env = {k: str(v) for k, v in env.items() if v is not None and str(v).strip()}
                else:
                    env = None
                if cmd and args:
                    return cmd, args, env
            except Exception as e:
                logger.warning("Could not read %s: %s", mcp_path, e)
    # Fallback: use defaults and set MILVUS_DB in env so server uses same DB as ingest
    fallback_args = [
        "--directory",
        DEFAULT_MCP_DIR,
        "run",
        "server.py",
        "--milvus-uri",
        DEFAULT_MILVUS_URI,
        "--milvus-db",
        DEFAULT_MILVUS_DB,
    ]
    fallback_env = dict(os.environ)
    fallback_env["MILVUS_DB"] = DEFAULT_MILVUS_DB
    return DEFAULT_MCP_COMMAND, fallback_args, fallback_env


def _check_mcp_server_path(cmd: str, args: list[str]) -> None:
    """Raise a clear error if the MCP server directory or command is missing."""
    if not cmd or not os.path.isfile(cmd):
        raise FileNotFoundError(
            f"Milvus MCP command not found: {cmd!r}. "
            "Set MILVUS_MCP_COMMAND to your 'uv' (or python) path, e.g. export MILVUS_MCP_COMMAND=$(which uv)."
        )
    a = args or []
    try:
        i = a.index("--directory")
        dir_path = a[i + 1] if i + 1 < len(a) else None
    except ValueError:
        return
    if not dir_path:
        return
    d = Path(dir_path)
    if not d.is_dir():
        raise FileNotFoundError(
            f"Milvus MCP server directory not found: {d!s}. "
            "Clone the server: git clone https://github.com/zilliztech/mcp-server-milvus.git "
            "into this project (or set MILVUS_MCP_DIR to its src/mcp_server_milvus path)."
        )
    if not (d / "server.py").is_file():
        raise FileNotFoundError(
            f"Milvus MCP server.py not found in {d!s}. "
            "Ensure MILVUS_MCP_DIR points to the mcp_server_milvus package (contains server.py)."
        )


@asynccontextmanager
async def milvus_mcp_session(
    command: str | None = None,
    args: list[str] | None = None,
):
    """
    Async context manager: spawn Milvus MCP server (stdio) and yield a session that can call tools.
    Use with: async with milvus_mcp_session() as session: ...
    """
    from contextlib import AsyncExitStack

    cmd, default_args, env = _load_mcp_config_from_cursor()
    cmd = command or cmd
    args = args if args is not None else default_args
    _check_mcp_server_path(cmd, args)

    # Log so we can verify same DB as ingest (helps when retrieval returns "no documents")
    a = args or []
    try:
        i = a.index("--milvus-db")
        db_arg = a[i + 1] if i + 1 < len(a) else None
    except ValueError:
        db_arg = None
    db_env = (env or {}).get("MILVUS_DB") if isinstance(env, dict) else None
    logger.info("MCP config: MILVUS_DB from args=%s, from env=%s", db_arg, db_env)

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise ImportError(
            "Install the MCP Python SDK: pip install mcp. Then ensure mcp-server-milvus is cloned and .cursor/mcp.json is configured."
        ) from e

    # Merge mcp.json env with process env so subprocess has PATH + MILVUS_DB etc.
    proc_env = dict(os.environ)
    if isinstance(env, dict):
        proc_env.update(env)
        env = proc_env
    params = StdioServerParameters(command=cmd, args=args, env=env)
    async with AsyncExitStack() as stack:
        stdio = await stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        yield session


async def _call_tool(session: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool and return parsed result (text or JSON)."""
    result = await session.call_tool(tool_name, arguments)
    if not result.content:
        logger.warning("MCP tool %s returned no content", tool_name)
        return None
    for part in result.content:
        if hasattr(part, "text") and part.text:
            try:
                return json.loads(part.text)
            except json.JSONDecodeError:
                return part.text
    logger.warning("MCP tool %s returned content but no part.text", tool_name)
    return None


# --- Sync helpers that run the async client ---


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class MilvusMCPClient:
    """
    Synchronous wrapper: runs the Milvus MCP server in a subprocess and calls its tools.
    Ingestion and retrieval should use this (or the async session) so all Milvus access goes via MCP.
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
    ):
        self._command = command
        self._args = args

    def _with_session(self, fn):
        async def _run():
            async with milvus_mcp_session(self._command, self._args) as session:
                return await fn(session)

        return _run_async(_run())

    def list_collections(self) -> list[str]:
        """List all collection names in Milvus (via MCP)."""
        out = self._with_session(lambda s: _call_tool(s, "milvus_list_collections", {}))
        if out is None:
            return []
        if isinstance(out, list):
            return out
        if isinstance(out, dict) and "collection_names" in out:
            return out["collection_names"]
        if isinstance(out, str):
            try:
                data = json.loads(out)
                return data.get("collection_names", data) if isinstance(data, dict) else []
            except json.JSONDecodeError:
                pass
        return []

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
        """Create a collection (via MCP). Use field_schema for custom fields (content, location, etc.)."""
        args: dict[str, Any] = {
            "collection_name": collection_name,
            "dimension": dimension,
            "primary_field_name": primary_field_name,
            "vector_field_name": vector_field_name,
            "metric_type": metric_type,
            "auto_id": auto_id,
        }
        if field_schema is not None:
            args["field_schema"] = field_schema
        return self._with_session(lambda s: _call_tool(s, "milvus_create_collection", args))

    def load_collection(self, collection_name: str, replica_number: int = 1) -> Any:
        """Load collection into memory for search (via MCP)."""
        return self._with_session(
            lambda s: _call_tool(
                s,
                "milvus_load_collection",
                {"collection_name": collection_name, "replica_number": replica_number},
            )
        )

    def insert_data(self, collection_name: str, data: dict[str, list]) -> Any:
        """
        Insert data into a collection (via MCP).
        data: dict mapping field names to lists, e.g. {"id": [...], "vector": [...], "content": [...]}.
        """
        return self._with_session(
            lambda s: _call_tool(
                s,
                "milvus_insert_data",
                {"collection_name": collection_name, "data": data},
            )
        )

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
        """Run vector similarity search (via MCP)."""
        args: dict[str, Any] = {
            "collection_name": collection_name,
            "vector": vector,
            "limit": limit,
            "vector_field": vector_field,
            "metric_type": metric_type,
        }
        if output_fields is not None:
            args["output_fields"] = output_fields
        if filter_expr is not None:
            args["filter_expr"] = filter_expr
        return self._with_session(lambda s: _call_tool(s, "milvus_vector_search", args))

    def get_collection_info(self, collection_name: str) -> Any:
        """Get collection schema and metadata (via MCP)."""
        return self._with_session(
            lambda s: _call_tool(
                s,
                "milvus_get_collection_info",
                {"collection_name": collection_name},
            )
        )
