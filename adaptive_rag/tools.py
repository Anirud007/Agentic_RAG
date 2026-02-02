"""Retrieval and web search tools for adaptive RAG."""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool, StructuredTool
import asyncio
import logging
import os
import threading
from typing import Any

from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)

# Cache retriever instances to avoid reloading models on each request
_cached_retriever = None
_cached_image_retriever = None


def format_milvus_hits(hits: Any) -> str:
    """Turn Milvus MCP search result into a single context string."""
    if hits is None:
        return ""
    if isinstance(hits, str):
        return hits
    if isinstance(hits, list):
        parts = []
        for i, item in enumerate(hits):
            if isinstance(item, dict):
                content = item.get("content") or item.get("entity", {}).get("content")
                location = item.get("location") or item.get("entity", {}).get("location")
                section = item.get("section_title") or item.get("entity", {}).get("section_title")
                if content:
                    parts.append(f"[{i+1}] {content}")
                    if location:
                        parts[-1] += f" (source: {location})"
                    if section:
                        parts[-1] += f" [section: {section}]"
            else:
                parts.append(str(item))
        return "\n\n".join(parts) if parts else ""
    if isinstance(hits, dict) and "results" in hits:
        return format_milvus_hits(hits["results"])
    return str(hits)


def vectorstore_retrieve(
    question: str,
    limit: int = 6,
    thread_id: str | None = None,
    filter_expr: str | None = None,
) -> tuple[str, Any]:
    """
    Retrieve from session-specific Milvus collection via hybrid rerank retriever.
    Returns (context_str, raw_hits).
    Uses cached retriever to avoid reloading embedding models on each request.
    filter_expr: optional Milvus filter (e.g. document_role == "resume", source_file == "x.pdf").
    """
    global _cached_retriever
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ingest.hybrid_rerank_retriever import HybridRerankRetriever
        from ingest.doc_embeddings import get_collection_names

        collection_name, _ = get_collection_names(thread_id)
        logger.info("Retrieving from collection=%s for thread_id=%s, question=%r",
                    collection_name, thread_id, question[:100])

        # Use cached retriever to avoid model reloading
        if _cached_retriever is None:
            _cached_retriever = HybridRerankRetriever(top_k=limit, retrieve_k=20)
            logger.info("Created new HybridRerankRetriever (will be cached)")

        hits = _cached_retriever.retrieve_text(
            question, thread_id=thread_id, filter_expr=filter_expr
        )

        context = format_milvus_hits(hits)
        logger.info("Retrieved %d hits, context length=%d chars",
                    len(hits) if hits else 0, len(context))
        return context, hits
    except Exception as e:
        logger.warning("Vectorstore retrieve failed: %s", e, exc_info=True)
        return "", None


def image_retrieve(question: str, limit: int = 5, thread_id: str | None = None) -> tuple[str, Any]:
    """
    Retrieve images from session-specific Milvus image collection via text-to-image search.
    Returns (context_str describing images, raw_hits).
    Uses SigLIP cross-modal embeddings for text-to-image retrieval.
    """
    global _cached_image_retriever
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ingest.milvus_retriever import MilvusRetriever
        from ingest.doc_embeddings import get_collection_names

        _, image_collection = get_collection_names(thread_id)
        logger.info("Retrieving images from collection=%s for thread_id=%s, question=%r",
                    image_collection, thread_id, question[:100])

        # Use cached retriever to avoid model reloading
        if _cached_image_retriever is None:
            _cached_image_retriever = MilvusRetriever()
            logger.info("Created new MilvusRetriever for image search (will be cached)")

        hits = _cached_image_retriever.retrieve_image_by_text(
            question, thread_id=thread_id, limit=limit
        )

        # Format image results as context
        if not hits:
            logger.info("No images found in collection %s", image_collection)
            return "", None

        parts = []
        for i, item in enumerate(hits):
            if isinstance(item, dict):
                source = item.get("source_file") or item.get("entity", {}).get("source_file", "unknown")
                location = item.get("location") or item.get("entity", {}).get("location", "")
                # Include image reference in context
                parts.append(f"[Image {i+1}] From file: {source}" + (f" at {location}" if location else ""))

        context = "\n".join(parts) if parts else ""
        logger.info("Retrieved %d images, context: %s", len(hits), context[:200])
        return context, hits
    except Exception as e:
        logger.warning("Image retrieve failed: %s", e, exc_info=True)
        return "", None


# def web_search(query: str, num_results: int = 5, search_depth: str = "basic") -> str:
#     """Web search using Tavily MCP server (https://docs.tavily.com/documentation/mcp).
#     Connects to remote Tavily MCP via SSE and calls tavily-search tool.
#     Returns concatenated snippets (summary + results).

#     Args:
#         query: Search query string
#         num_results: Maximum number of results to return
#         search_depth: "basic" for faster results, "advanced" for more comprehensive
#     """
#     import asyncio
#     import json
#     import os

#     api_key = os.getenv("TAVILY_API_KEY")
#     if not api_key:
#         logger.warning("TAVILY_API_KEY not set; web search disabled")
#         return ""

#     async def _search_via_mcp() -> str:
#         from mcp import ClientSession, types as mcp_types
#         from mcp.client.sse import sse_client

#         url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
#         try:
#             async with sse_client(
#                 url=url,
#                 timeout=15,
#                 sse_read_timeout=90,
#             ) as (read, write):
#                 async with ClientSession(
#                     read_stream=read,
#                     write_stream=write,
#                     client_info=mcp_types.Implementation(
#                         name="adaptive_rag.tools",
#                         version="1.0",
#                     ),
#                 ) as session:
#                     await session.initialize()
#                     result = await session.call_tool(
#                         "tavily-search",
#                         arguments={
#                             "query": query,
#                             "max_results": num_results,
#                             "search_depth": search_depth,
#                             "include_answer": True,
#                         },
#                     )
#                     if not result.content:
#                         return ""
#                     text_parts = []
#                     for item in result.content:
#                         if getattr(item, "type", None) == "text" and getattr(item, "text", None):
#                             text_parts.append(item.text)
#                     raw = "\n".join(text_parts)
#                     if not raw:
#                         return ""
#                     # Parse JSON response (Tavily MCP returns JSON with answer + results)
#                     try:
#                         data = json.loads(raw)
#                     except json.JSONDecodeError:
#                         return raw
#                     parts = []
#                     if data.get("answer"):
#                         parts.append(f"**Summary:** {data['answer']}\n")
#                     for i, r in enumerate(data.get("results", [])):
#                         title = r.get("title", "")
#                         content = r.get("content", "")
#                         url = r.get("url", "")
#                         parts.append(f"[{i+1}] {title}\n{content}\nSource: {url}")
#                     if not parts:
#                         return data.get("answer", "") or raw
#                     logger.info(
#                         "Tavily MCP search returned %d results for: %s",
#                         len(data.get("results", [])),
#                         query[:50],
#                     )
#                     return "\n\n".join(parts)
#         except Exception as e:
#             logger.warning("Tavily MCP web search failed: %s", e)
#             return ""

#     try:
#         return asyncio.run(_search_via_mcp())
#     except Exception as e:
#         logger.warning("Web search error: %s", e)
#         return ""



_tavily_mcp_tools_cache: list[Any] | None = None


TAVILY_TIME_RANGE_VALID = ("day", "week", "month", "year")


def _normalize_tavily_search_input(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure Tavily search gets valid args: time_range, and max_results (not top_k/topn)."""
    inp = dict(input_dict)
    # Map top_k or topn -> max_results (agent may send either; Tavily API expects max_results)
    for key in ("top_k", "topn"):
        if key in inp:
            if "max_results" not in inp:
                inp["max_results"] = inp[key]
            del inp[key]
    # Fix time_range
    tr = inp.get("time_range")
    if not tr or (isinstance(tr, str) and tr.strip() not in TAVILY_TIME_RANGE_VALID):
        inp["time_range"] = "year"
    return inp


def _run_async_tool_sync(tool: BaseTool, input_dict: dict[str, Any]) -> Any:
    """Run tool.ainvoke() in a new thread so sync callers (e.g. agent) can use it."""
    name = getattr(tool, "name", "")
    if name in ("tavily-search", "tavily_search"):
        input_dict = _normalize_tavily_search_input(input_dict)
    result_holder: list[Any] = []

    def _run():
        try:
            out = asyncio.run(tool.ainvoke(input_dict))
            result_holder.append(out)
        except Exception as e:
            result_holder.append(e)

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    if not result_holder:
        return ""
    r = result_holder[0]
    if isinstance(r, Exception):
        raise r
    return r


def _wrap_async_tool_for_sync(tool: BaseTool) -> BaseTool:
    """Wrap an async-only MCP tool so invoke() runs ainvoke() in a thread."""
    name = getattr(tool, "name", "mcp_tool")
    description = getattr(tool, "description", "")
    args_schema = getattr(tool, "args_schema", None)

    def sync_func(**kwargs: Any) -> Any:
        return _run_async_tool_sync(tool, kwargs)

    wrapped = StructuredTool.from_function(
        func=sync_func,
        name=name,
        description=description or f"Async tool: {name}",
    )
    if args_schema is not None:
        wrapped.args_schema = args_schema
    return wrapped


async def _get_tavily_mcp_tools_async() -> list[Any]:
    """Return LangChain tools from Tavily MCP server (tavily-search, tavily-extract, etc.)."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set; Tavily MCP tools disabled")
        return []
    client = MultiServerMCPClient(
        {
            "tavily-remote": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "mcp-remote",
                    f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                ],
            }
        }
    )
    raw_tools = await client.get_tools()
    if not raw_tools:
        return []
    # Wrap async-only MCP tools so agent can call them with sync invoke()
    return [_wrap_async_tool_for_sync(t) for t in raw_tools]


def get_tavily_mcp_tools() -> list[Any]:
    """Return Tavily MCP tools (cached). Use in graph RAG_TOOLS for web search.
    Runs async fetcher in a new thread when called from a running event loop (e.g. FastAPI).
    """
    global _tavily_mcp_tools_cache
    if _tavily_mcp_tools_cache is not None:
        return _tavily_mcp_tools_cache
    result_holder: list[Any] = []

    def _run_in_thread():
        try:
            result_holder.append(asyncio.run(_get_tavily_mcp_tools_async()))
        except Exception as e:
            logger.warning("Failed to load Tavily MCP tools: %s", e)
            result_holder.append([])

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to use asyncio.run()
        try:
            _tavily_mcp_tools_cache = asyncio.run(_get_tavily_mcp_tools_async())
            return _tavily_mcp_tools_cache
        except Exception as e:
            logger.warning("Failed to load Tavily MCP tools: %s", e)
            return []

    # Already inside an event loop (e.g. uvicorn): run in a separate thread
    t = threading.Thread(target=_run_in_thread)
    t.start()
    t.join()
    _tavily_mcp_tools_cache = result_holder[0] if result_holder else []
    return _tavily_mcp_tools_cache


def web_search(query: str, num_results: int = 5, search_depth: str = "basic") -> str:
    """Sync web search for graph nodes: invokes Tavily MCP search tool and returns result string."""
    tools_list = get_tavily_mcp_tools()
    search_tool = next(
        (t for t in tools_list if getattr(t, "name", "") in ("tavily-search", "tavily_search")),
        None,
    )
    if not search_tool:
        logger.warning("Tavily search tool not found; web search disabled")
        return ""
    try:
        result = search_tool.invoke(
            {"query": query, "max_results": num_results, "search_depth": search_depth}
        )
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return ""