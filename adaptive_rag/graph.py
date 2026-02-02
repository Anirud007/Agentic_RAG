"""
LangGraph adaptive RAG with ReAct-style self-reflection.

Flow: Query Analysis → [Vectorstore | Web search | Direct LLM] → RAG + self-reflection:
  - Relevant to question? → Generate → Hallucinations? → Answers question? → Answer with RAG
  - Loops: not relevant → retry web; hallucinations / doesn't answer → re-check relevance or retry
  - Fallback: direct LLM (no RAG).

Memory: short-term (InMemorySaver) or long-term (SqliteSaver) checkpointer; messages list in state.
"""

import concurrent.futures
import logging
from pathlib import Path
from typing import Literal, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from adaptive_rag.state import AdaptiveRAGState
from adaptive_rag.tools import vectorstore_retrieve, image_retrieve, web_search, get_tavily_mcp_tools
import os 
from dotenv import load_dotenv
load_dotenv()   

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
logger = logging.getLogger(__name__)

# Optional long-term checkpointer (persists to SQLite)
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None  # pip install langgraph-checkpoint-sqlite

# Align with API defaults: env OLLAMA_MODEL / MAX_RETRIES
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
MEMORY_HISTORY_LIMIT = 10  # last N messages to include as context
LLM_TIMEOUT_SEC = int(os.environ.get("LLM_TIMEOUT_SEC", "90"))
CONTEXT_SUMMARY_MAX_CHARS = 2000  # max chars before summarizing for judge/generation
MEMORY_SNIPPET_MAX_CHARS = 400  # max chars per message in memory context before summarizing


def _build_retrieval_filter(state: AdaptiveRAGState) -> str | None:
    """Build Milvus filter expr from state source_file and document_role (query-time filter)."""
    parts: list[str] = []
    source_file = state.get("source_file")
    document_role = state.get("document_role")
    if source_file and str(source_file).strip():
        escaped = str(source_file).replace('\\', '\\\\').replace('"', '\\"')
        parts.append(f'source_file == "{escaped}"')
    if document_role and str(document_role).strip():
        escaped = str(document_role).replace('\\', '\\\\').replace('"', '\\"')
        parts.append(f'document_role == "{escaped}"')
    return " and ".join(parts) if parts else None


# Tools from adaptive_rag.tools wrapped for the agent (thread_id=None when called by agent)
def _tool_vectorstore(question: str, limit: int = 6) -> str:
    """Retrieve relevant document chunks from the knowledge base. Use for questions about uploaded documents."""
    context, _ = vectorstore_retrieve(question, limit=limit, thread_id=None, filter_expr=None)
    return context or "No document context found."


def _tool_image_retrieve(question: str, limit: int = 5) -> str:
    """Retrieve relevant images from uploaded documents. Use when the user asks about figures, charts, or images."""
    context, _ = image_retrieve(question, limit=limit, thread_id=None)
    return context or "No images found."


# RAG tools: vectorstore + image retrieve (StructuredTool) + Tavily MCP tools (web search, extract, etc.)
RAG_TOOLS = [
    StructuredTool.from_function(
        func=_tool_vectorstore,
        name="vectorstore_retrieve",
        description="Retrieve relevant text chunks from the knowledge base. Call this for questions about uploaded documents.",
    ),
    StructuredTool.from_function(
        func=_tool_image_retrieve,
        name="image_retrieve",
        description="Retrieve relevant images from uploaded documents. Call when the user asks about figures, charts, or images.",
    ),
] + get_tavily_mcp_tools()


def _get_llm(model: str = OLLAMA_MODEL, temperature: float = 0.2):
    """Return a ReAct agent with Ollama and RAG tools (vectorstore, web search, image retrieve)."""
    llm = ChatOllama(model=model, temperature=temperature)
    return create_agent(llm, tools=RAG_TOOLS, debug=True)


def _format_messages_for_context(messages: list[BaseMessage] | list, last_n: int = MEMORY_HISTORY_LIMIT) -> str:
    """Format last N messages as a string for LLM context (short-term conversation memory)."""
    if not messages:
        return ""
    msgs = messages[-last_n:] if len(messages) > last_n else messages
    lines = []
    for m in msgs:
        role = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else "unknown")
        content = getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        label = "User" if role == "human" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _invoke_llm(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    timeout_sec: int | None = None,
) -> str:
    """Invoke the agent (LLM + tools) with prompt and optional system message. Uses timeout to avoid hanging."""
    timeout_sec = timeout_sec if timeout_sec is not None else LLM_TIMEOUT_SEC
    agent = _get_llm(model=model)
    msgs: list[BaseMessage] = [HumanMessage(content=prompt)]
    if system:
        msgs.insert(0, SystemMessage(content=system))

    def _run():
        result = agent.invoke({"messages": msgs})
        messages = result.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        return last_ai.content if (last_ai and hasattr(last_ai, "content")) else str(result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout_sec}s")


def _summarize_context(text: str, max_chars: int = CONTEXT_SUMMARY_MAX_CHARS, model: str = OLLAMA_MODEL) -> str:
    """Summarize long text so we pass coherent context instead of truncation. Returns text unchanged if short."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    system = """Summarize the following document/context concisely. Preserve key facts, numbers, names, and conclusions. Output only the summary, no preamble. Keep under 400 words."""
    try:
        summary = _invoke_llm(f"Context to summarize:\n\n{text}", system=system, model=model)
        return (summary or text[:max_chars]).strip()
    except Exception as e:
        logger.warning("Summarization failed, using truncation fallback: %s", e)
        return text[:max_chars] + "..."


def _conversation_context(state: AdaptiveRAGState) -> str:
    return _format_messages_for_context(state.get("messages") or [])


def _extract_memory_context(state: AdaptiveRAGState) -> str:
    """Extract relevant memory context from conversation history for follow-ups.

    Includes previous turn(s) when the user query looks like a follow-up (pronouns,
    explicit refs, or short follow-up phrases) so the model can relate "second point",
    "elaborate", "that one", etc. to the last answer.
    """
    question = state.get("question", "")
    messages = state.get("messages", [])
    last_source = state.get("source", "")

    if not messages or len(messages) < 2:
        return ""

    q_lower = question.lower().strip()
    q_words = set(q_lower.split())

    # Explicit memory references
    memory_refs = ("you said", "you mentioned", "earlier", "we discussed", "before", "previously")
    has_explicit_memory_ref = any(ref in q_lower for ref in memory_refs)

    # Pronouns that reference previous context
    pronouns = ("it", "that", "this", "them", "those", "these", "the same")
    has_pronoun_ref = any(p in q_words for p in pronouns)

    # Follow-up phrases that imply reference to previous answer
    follow_up_phrases = (
        "elaborate", "explain more", "second", "third", "first", "point", "same",
        "that one", "this one", "more", "again", "continue", "what about", "and the",
        "also", "other", "another", "expand", "detail", "clarify", "how", "why",
    )
    has_follow_up_phrase = any(phrase in q_lower for phrase in follow_up_phrases)

    # Short query often means follow-up (e.g. "and?", "the second one?")
    is_short_follow_up = len(question) <= 80 and len(messages) >= 2

    should_include_memory = (
        has_explicit_memory_ref
        or has_pronoun_ref
        or has_follow_up_phrase
        or is_short_follow_up
    )

    if not should_include_memory:
        return ""

    # When only pronouns (no explicit ref, no follow-up phrase), require last answer
    # to be doc-related so we don't inject noise for unrelated pronoun use
    if has_pronoun_ref and not has_explicit_memory_ref and not has_follow_up_phrase and not is_short_follow_up:
        last_assistant_msg = None
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "unknown")
            if role == "ai" or role == "assistant":
                last_assistant_msg = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
                break
        if last_assistant_msg:
            doc_indicators = ("document", "file", "pdf", "uploaded", "indexed", "page", "slide",
                              "chunks", "report", "from the", "according to")
            is_doc_related = any(ind in last_assistant_msg.lower() for ind in doc_indicators)
            if last_source in ("rag", "vectorstore", "hybrid"):
                is_doc_related = True
            if not is_doc_related:
                return ""

    # Last 2 messages (previous user + assistant) for minimal context
    recent_messages = messages[-2:] if len(messages) > 2 else messages

    memory_parts = []
    for msg in recent_messages:
        role = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "unknown")
        content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if len(content) > MEMORY_SNIPPET_MAX_CHARS:
            content = _summarize_context(content, MEMORY_SNIPPET_MAX_CHARS)
        label = "User asked" if role == "human" else "You answered"
        memory_parts.append(f"{label}: {content}")

    if memory_parts:
        return "## Previous Context:\n" + "\n".join(memory_parts)
    return ""


def query_analysis(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """
    2-stage query routing for a general-purpose GenAI bot.
    
    Stage A: "Does this question need uploaded docs?" (LLM classifier)
    Stage B: If no docs needed, use deterministic time-sensitive heuristic
    
    This prevents over-routing to vectorstore and makes decisions predictable.
    """
    question = state.get("question", "")
    q_lower = question.lower().strip()
    messages = state.get("messages", [])
    last_source = state.get("source", "")  # Track previous answer source
    
    # ===== STAGE 0: Handle obvious greetings (pattern matching) =====
    greeting_patterns = (
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "whats up", "sup", "yo", "greetings",
        "thank you", "thanks", "bye", "goodbye", "see you", "later",
    )
    if q_lower in greeting_patterns or any(q_lower.startswith(p + " ") for p in ("hi", "hello", "hey", "thanks")):
        logger.info("Query routed to direct_llm - greeting detected")
        return {**state, "route": "direct_llm", "query_intent": "greeting", "routing_confidence": 1.0}
    
    # ===== STAGE A: "Does this need docs?" - LLM binary classifier =====
    # Only include last 3 messages for minimal context
    ctx = _format_messages_for_context(messages, last_n=3)
    
    system = """You are a classifier. Decide if the user is asking about uploaded documents in THIS chat.

Return JSON only:
{"needs_docs": true/false, "confidence": 0.0-1.0, "reason": "short"}

Rules:
- needs_docs=true if the user explicitly refers to files/docs/pdf/ppt/excel OR
  the question clearly depends on content from uploaded documents (e.g., "in the document", "from the file", "summarize the PDF", "what does slide 5 say", "analyze the scope").
- If the PREVIOUS assistant message was based on uploaded documents (e.g. candidate profile, report, resume) and the user is asking a FOLLOW-UP about a specific aspect (e.g. education, educational background, experience, skills, qualifications, summary), set needs_docs=true so we retrieve from the documents.
- Pronouns like "it/this/that" count as needs_docs=true if the immediately previous message was about a document/file.
- If the user asks general knowledge, coding help, how-to, brainstorming, or anything not dependent on uploaded content => needs_docs=false.
- Default to needs_docs=false if uncertain."""

    prompt = f"Conversation:\n{ctx}\n\nCurrent question: {question}" if ctx else question
    reply = _invoke_llm(prompt, system=system).strip()
    
    # Parse JSON response
    needs_docs = False
    confidence = 0.5
    try:
        import json
        # Handle markdown code blocks
        if "```" in reply:
            reply = reply.split("```")[1].replace("json", "").strip()
        parsed = json.loads(reply)
        needs_docs = parsed.get("needs_docs", False)
        confidence = parsed.get("confidence", 0.5)
        logger.info("Doc classifier: needs_docs=%s, confidence=%.2f, reason=%s", 
                    needs_docs, confidence, parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: check for explicit doc keywords
        logger.warning("Failed to parse classifier response: %s. Falling back to keyword check.", e)
        doc_keywords = ("document", "file", "pdf", "ppt", "excel", "uploaded", "the report", "the slide")
        needs_docs = any(kw in q_lower for kw in doc_keywords)
    
    # ===== STAGE B: If not doc-needed, decide web vs direct_llm =====
    if needs_docs:
        route = "vectorstore"
        intent = "question"
        logger.info("Query routed to vectorstore - doc content needed")
    else:
        # Time-sensitive heuristic for web search
        TIME_SENSITIVE = (
            "today", "current", "latest", "recent", "this week", "now",
            "breaking", "news", "price", "pricing", "weather", "stock",
            "live", "right now", "2024", "2025", "2026"
        )
        is_time_sensitive = any(kw in q_lower for kw in TIME_SENSITIVE)
        
        if is_time_sensitive:
            route = "web_search"
            intent = "question"
            logger.info("Query routed to web_search - time-sensitive detected")
        else:
            route = "direct_llm"
            intent = "question"
            logger.info("Query routed to direct_llm - general question")

    # ===== STAGE C: If doc-related, check if question is ambiguous in context =====
    clarification_question = ""
    if route == "vectorstore" and messages and len(messages) >= 2:
        ctx_amb = _format_messages_for_context(messages, last_n=3)
        system_amb = """You are a disambiguator. Only set ambiguous=true when the conversation clearly involves MULTIPLE DISTINCT ENTITIES (e.g. both a job description AND a candidate resume, or multiple candidates/documents) and the user's question could refer to EITHER.

Rules:
- If there is only ONE uploaded document or one clear referent (e.g. one resume = one candidate), set ambiguous=false. Do NOT ask "which candidate?" when there is only one.
- For broad questions about the uploaded content (e.g. "overview of the candidate", "what do you have?", "tell me about the document", "educational background") when only one document or one candidate is in context, set ambiguous=false.
- Only set ambiguous=true when the user could mean one of several distinct things (e.g. "education" when we have both job requirements and candidate resume) and clarification is needed.

Return JSON only:
{"ambiguous": true/false, "clarification": "One short question to ask the user, or empty string if ambiguous=false"}
If ambiguous=false, set clarification="". """
        prompt_amb = f"Conversation:\n{ctx_amb}\n\nCurrent question: {question}"
        try:
            reply_amb = _invoke_llm(prompt_amb, system=system_amb).strip()
            if "```" in reply_amb:
                reply_amb = reply_amb.split("```")[1].replace("json", "").strip()
            import json as _json
            parsed_amb = _json.loads(reply_amb)
            if parsed_amb.get("ambiguous") and parsed_amb.get("clarification"):
                clarification_question = (parsed_amb.get("clarification") or "").strip()
                if clarification_question:
                    route = "clarify"
                    logger.info("Query ambiguous - asking for clarification: %s", clarification_question[:80])
        except Exception as e:
            logger.debug("Ambiguity check failed, continuing: %s", e)

    return {
        **state,
        "route": route,
        "query_intent": intent,
        "routing_confidence": confidence,
        "clarification_question": clarification_question,
    }


def route_after_analysis(state: AdaptiveRAGState) -> Literal["retrieve_vectorstore", "retrieve_web", "retrieve_hybrid", "answer_llm", "clarify"]:
    route = state.get("route")
    if route == "clarify":
        return "clarify"
    if route == "vectorstore":
        return "retrieve_vectorstore"
    if route == "web_search":
        return "retrieve_web"
    if route == "hybrid":
        return "retrieve_hybrid"
    return "answer_llm"


def clarify(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Return a clarification question when the user's intent is ambiguous."""
    clarification_question = state.get("clarification_question", "")
    if not clarification_question:
        clarification_question = "Could you clarify what you'd like to know?"
    return {
        **state,
        "final_answer": clarification_question,
        "source": "clarify",
        "messages": [AIMessage(content=clarification_question)],
    }


def retrieve_vectorstore(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Retrieve context from session-specific Milvus collection (text + images)."""
    question = state.get("question", "")
    thread_id = state.get("thread_id")
    filter_expr = _build_retrieval_filter(state)

    # Retrieve text context
    context, hits = vectorstore_retrieve(
        question, limit=6, thread_id=thread_id, filter_expr=filter_expr
    )

    # Also retrieve images using cross-modal search
    image_context, image_hits = image_retrieve(question, limit=3, thread_id=thread_id)

    # Combine contexts if we have images
    combined_context = context
    if image_context:
        if combined_context:
            combined_context = f"{context}\n\n--- Images in Documents ---\n{image_context}"
        else:
            combined_context = f"--- Images in Documents ---\n{image_context}"
        logger.info("Added image context: %s", image_context[:200])

    return {
        **state,
        "context": combined_context,
        "image_context": image_context,
        "retrieved_docs": hits,
        "retrieved_images": image_hits,
    }


def check_info_sufficient(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """
    After retrieval: check if memory + retrieved chunks contain enough to answer.
    If yes → answer_from_memory_and_chunks; if no → full RAG flow (check_relevant → generate).
    """
    context = (state.get("context") or "").strip()
    question = state.get("question", "")
    messages = state.get("messages", [])

    # No or very little context → not sufficient, go to full RAG flow
    if len(context) < 100:
        logger.info("check_info_sufficient: context too short, continuing to check_relevant")
        return {**state, "info_sufficient": False}

    ctx = _format_messages_for_context(messages, last_n=3)
    context_for_judge = _summarize_context(context, CONTEXT_SUMMARY_MAX_CHARS)
    system = """You are a judge. Given the conversation and the retrieved document context below, can we answer the user's current question with this information?
Return JSON only:
{"sufficient": true/false, "reason": "one short sentence"}
- sufficient=true only if the retrieved context (and conversation) clearly contain the answer or enough to answer.
- sufficient=false if the context is irrelevant, too vague, or the question asks for something not in the context."""
    prompt = f"Conversation:\n{ctx}\n\nRetrieved context:\n{context_for_judge}\n\nUser question: {question}\n\nCan we answer with this information?"
    try:
        import json as _json
        reply = _invoke_llm(prompt, system=system).strip()
        if "```" in reply:
            reply = reply.split("```")[1].replace("json", "").strip()
        parsed = _json.loads(reply)
        sufficient = bool(parsed.get("sufficient", False))
        logger.info("check_info_sufficient: sufficient=%s, reason=%s", sufficient, parsed.get("reason", ""))
        return {**state, "info_sufficient": sufficient}
    except Exception as e:
        logger.warning("check_info_sufficient failed: %s, defaulting to not sufficient", e)
        return {**state, "info_sufficient": False}


def route_after_info_check(state: AdaptiveRAGState) -> Literal["answer_from_memory_and_chunks", "check_relevant"]:
    """If memory + chunks are sufficient, answer from them; else continue full RAG flow."""
    if state.get("info_sufficient"):
        return "answer_from_memory_and_chunks"
    return "check_relevant"


def answer_from_memory_and_chunks(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Answer using conversation memory + already-retrieved chunks (no extra retrieval)."""
    question = state.get("question", "")
    context = state.get("context", "")
    memory_context = _extract_memory_context(state)
    ctx = _conversation_context(state)

    full_context = context
    if memory_context:
        full_context = f"{memory_context}\n\n{full_context}"
    if ctx:
        full_context = f"Conversation:\n{ctx}\n\n{full_context}"
    if len(full_context) > CONTEXT_SUMMARY_MAX_CHARS * 2:
        full_context = _summarize_context(full_context, CONTEXT_SUMMARY_MAX_CHARS * 2)

    system = """You are a helpful AI assistant. Answer the user's question using ONLY the conversation and retrieved context below. Be concise. Cite the context when relevant. If the information is not in the context, say so briefly."""
    prompt = f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"
    answer = _invoke_llm(prompt, system=system)

    return {
        **state,
        "final_answer": answer,
        "answer": answer,
        "source": "rag",
        "messages": [AIMessage(content=answer)],
    }


def retrieve_web(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Retrieve context from web search."""
    question = state.get("question", "")
    context = web_search(question, num_results=5)
    return {
        **state,
        "context": context,
        "web_context": context,  # Also store separately for hybrid
        "retrieved_docs": None,
    }


def retrieve_hybrid(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Retrieve context from BOTH vectorstore AND web search, combining results.
    
    Used when a query needs document context enriched with current information.
    Example: "Compare my document's findings with current industry trends"
    """
    question = state.get("question", "")
    thread_id = state.get("thread_id")
    filter_expr = _build_retrieval_filter(state)
    logger.info("Hybrid retrieval - combining vectorstore and web search")
    
    # 1. Retrieve from vectorstore (documents)
    doc_context, doc_hits = vectorstore_retrieve(
        question, limit=4, thread_id=thread_id, filter_expr=filter_expr
    )
    image_context, image_hits = image_retrieve(question, limit=2, thread_id=thread_id)
    
    # 2. Retrieve from web
    web_context = web_search(question, num_results=3)
    
    # 3. Combine all contexts with clear sections
    combined_parts = []
    
    if doc_context:
        combined_parts.append(f"## From Your Documents:\n{doc_context}")
    
    if image_context:
        combined_parts.append(f"## Images in Documents:\n{image_context}")
    
    if web_context:
        combined_parts.append(f"## From Web Search (Current Information):\n{web_context}")
    
    combined_context = "\n\n".join(combined_parts) if combined_parts else ""
    
    logger.info("Hybrid context: doc=%d chars, web=%d chars", 
                len(doc_context or ""), len(web_context or ""))
    
    return {
        **state,
        "context": combined_context,
        "web_context": web_context,
        "image_context": image_context,
        "retrieved_docs": doc_hits,
        "retrieved_images": image_hits,
    }


NO_DOCS_MESSAGE = (
    "I don't have any documents for this chat session yet. Each conversation has its own "
    "private document collection. Please upload your files using the 📎 attachment button "
    "and wait for the upload to complete before asking questions about them."
)


def check_relevant(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Self-reflection: is the retrieved context relevant to the question?"""
    question = state.get("question", "")
    context = state.get("context", "")
    image_context = state.get("image_context", "")
    retrieved_docs = state.get("retrieved_docs", [])
    q_lower = question.lower()
    
    # Check if we have any content (text OR images)
    has_any_content = (context and context.strip()) or (image_context and image_context.strip())
    has_images = bool(image_context and image_context.strip())
    
    # Check if question is about visual content
    visual_keywords = ("pie chart", "chart", "graph", "diagram", "figure", "table", "image", 
                       "picture", "screenshot", "visual", "illustration", "plot", "bar chart",
                       "line graph", "what does it show", "what does it look like")
    is_visual_question = any(keyword in q_lower for keyword in visual_keywords)
    
    if not has_any_content:
        # User asked about uploaded docs but vector store returned nothing → don't fall back to web
        if state.get("route") == "vectorstore":
            return {
                **state,
                "relevant": False,
                "answer": NO_DOCS_MESSAGE,
                "no_docs_indexed": True,
            }
        return {**state, "relevant": False, "next_after_retry": "retrieve_web"}
    
    # If asking about visual content and we have images, mark as relevant
    if has_images and is_visual_question:
        logger.info("Visual question with images available - marking as relevant for image-handling")
        return {**state, "relevant": True}
    
    # If we only have image context (no text), mark as relevant to proceed to generate
    if image_context and (not context or context.strip().startswith("--- Images in Documents ---")):
        logger.info("Only image context available - marking as relevant for image-handling")
        return {**state, "relevant": True}
    
    # ===== SCORE-BASED RELEVANCE CHECK (no LLM) =====
    # If retrieval returned results, trust it and proceed to generate
    # This eliminates oscillation from LLM YES/NO judgments
    if retrieved_docs and len(retrieved_docs) > 0:
        logger.info("Score-based relevance: %d docs retrieved, marking as relevant", len(retrieved_docs))
        return {**state, "relevant": True}
    
    # If we have context text (from any source), proceed
    if context and len(context.strip()) > 100:
        logger.info("Context available (%d chars), marking as relevant", len(context))
        return {**state, "relevant": True}
    
    # No meaningful content
    logger.info("No meaningful content found, marking as not relevant")
    out = {**state, "relevant": False}
    out["next_after_retry"] = "retrieve_web"
    return out


def route_after_relevant(state: AdaptiveRAGState) -> Literal["generate", "increment_retry", "answer_llm", "answer_rag"]:
    if state.get("no_docs_indexed"):
        return "answer_rag"
    if state.get("relevant"):
        return "generate"
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if retry >= max_retries:
        return "answer_llm"
    return "increment_retry"


def generate(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Generate answer from context using LLM, enriched with memory when needed."""
    question = state.get("question", "")
    context = state.get("context", "")
    image_context = state.get("image_context", "")
    retry = state.get("retry_count", 0)
    ctx = _conversation_context(state)
    q_lower = question.lower()
    
    # Extract memory context for follow-up questions
    memory_context = _extract_memory_context(state)
    
    # Check if we only have image metadata but no actual text content
    has_text_content = context and not context.strip().startswith("--- Images in Documents ---")
    has_images = bool(image_context and image_context.strip())
    
    # Check if question is about visual content (charts, graphs, diagrams, etc.)
    visual_keywords = ("pie chart", "chart", "graph", "diagram", "figure", "table", "image", 
                       "picture", "screenshot", "visual", "illustration", "plot", "bar chart",
                       "line graph", "what does it show", "what does it look like")
    is_visual_question = any(keyword in q_lower for keyword in visual_keywords)
    
    if has_images and is_visual_question:
        # User is asking about images/charts - explain limitations
        num_images = len(image_context.split('[Image')) - 1
        
        if has_text_content:
            # We have both images and text - provide text context but explain image limitation
            answer = f"""I found **{num_images} images** in your document (including charts, figures, or diagrams), but I cannot directly view or interpret visual content.

                            **Note:** I can see that images exist on the following pages, but I cannot read their visual content:
                            {image_context}

                            **However, based on the text content in your document, here's what I found:**

"""
            # Let the LLM generate based on text context
            system = """Answer the question using only the provided text context. 
                        Note: The user is asking about a visual element (chart/graph/image) in their document. 
                        You cannot see the visual, but provide any relevant information from the text that might describe or relate to it.
                        If the text doesn't describe the visual element, say so and suggest they describe what they see."""
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (note you cannot see the actual image/chart, only describe what the text says about it):"
            text_answer = _invoke_llm(prompt, system=system)
            answer += text_answer
        else:
            # Only images, no text
            answer = f"""I found **{num_images} images** in your uploaded document, but I cannot directly read or interpret visual content like charts, graphs, or diagrams.

                            **Images found:**
                            {image_context}

                            **What you can do:**
                            - Describe what you see in the pie chart, and I can help analyze or explain it
                            - If the document has text near the chart (like captions or data labels), I may have captured that in the text content
                            - For scanned documents, consider using OCR to extract text first

                            Would you like to describe what you see in the chart?"""
        
        return {**state, "answer": answer, "image_only_response": True, "memory_context": memory_context}
    
    if has_images and not has_text_content:
        # Only image metadata available - the PDF likely contains only images
        answer = f"""I found **{len(image_context.split('[Image'))-1} images** in your uploaded document, but I cannot directly read the visual content of images.

                        **Images found:**
                        {image_context}

                        **What you can do:**
                        - If the document contains text shown in images, try uploading a text-based version of the document (like a Word doc or text file)
                        - For scanned PDFs, use OCR tools to convert images to searchable text first
                        - Ask specific questions about the document, and I'll do my best to help based on available information

                        Would you like me to help with anything else?"""
        return {**state, "answer": answer, "image_only_response": True, "memory_context": memory_context}
    
    # Build the full context including memory if available
    full_context = context
    if memory_context:
        full_context = f"{memory_context}\n\n{context}"
        logger.info("Added memory context to generation prompt")
    
    system = """You are a helpful AI assistant. Answer the question based on the provided context.

RESPONSE GUIDELINES:
1. Be direct and helpful - give the user what they need
2. When citing from documents, use human-friendly citations like (filename.pdf, page 5) or (presentation.pptx, slide 3)
3. If information is not found in context, say "not found in the document" 
4. Keep responses concise but complete
5. If there's ambiguity, ask a clarifying question OR provide the most likely answer

FORMAT GUIDELINES:
- Use **bold** for emphasis on key terms
- Use bullet points for lists
- Use headers (##) to organize longer responses

GROUNDING:
- Base your answer ONLY on the provided context
- If the context doesn't contain the answer, say so clearly
- Do not make up information"""
    if retry > 0:
        system += "\n\nIMPORTANT: Be extra careful to only state facts that are clearly supported. Do not speculate."
    prompt = f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"
    answer = _invoke_llm(prompt, system=system)
    return {**state, "answer": answer, "memory_context": memory_context}


def check_hallucinations(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Self-reflection: does the answer contain hallucinations?

    SIMPLIFIED: For RAG responses, we trust the retrieval and skip this check.
    This prevents oscillation and speeds up response time significantly.
    """
    answer = state.get("answer", "")
    route = state.get("route", "")

    if not answer:
        return {**state, "has_hallucinations": True, "next_after_retry": "check_relevant"}

    # Skip check for image-only responses
    if state.get("image_only_response"):
        logger.info("Skipping hallucination check - image-only response")
        return {**state, "has_hallucinations": False}

    # Skip for RAG/web routes - we trust the retrieved context
    if route in ("vectorstore", "web_search", "hybrid"):
        logger.info("Skipping hallucination check - trusting RAG retrieval")
        return {**state, "has_hallucinations": False}

    # Only check for direct_llm responses (no context grounding)
    return {**state, "has_hallucinations": False}


def route_after_hallucinations(state: AdaptiveRAGState) -> Literal["check_answers_question", "check_relevant"]:
    if state.get("has_hallucinations"):
        return "check_relevant"
    return "check_answers_question"


def check_answers_question(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Self-reflection: does the answer fully address the question?

    SIMPLIFIED: For RAG responses with retrieved docs, we trust the answer.
    This prevents oscillation and speeds up response time significantly.
    """
    answer = state.get("answer", "")
    route = state.get("route", "")
    retrieved_docs = state.get("retrieved_docs", [])

    # Skip check for image-only responses
    if state.get("image_only_response"):
        logger.info("Skipping completeness check - image-only response")
        return {**state, "answers_question": True}

    # If we have retrieved docs, trust the answer
    if retrieved_docs or route in ("vectorstore", "web_search", "hybrid"):
        logger.info("Skipping completeness check - trusting RAG answer")
        return {**state, "answers_question": True}

    # For direct LLM, trust it too (no retrieval to verify against)
    return {**state, "answers_question": True}


def route_after_answers_question(
    state: AdaptiveRAGState,
) -> Literal["answer_rag", "increment_retry", "answer_llm"]:
    if state.get("answers_question"):
        return "answer_rag"
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if retry >= max_retries:
        return "answer_llm"
    return "increment_retry"


def answer_rag(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Final RAG answer; append assistant message for memory."""
    final_answer = state.get("answer", "")
    route = state.get("route", "vectorstore")
    
    # Track the actual source based on routing decision
    if route == "hybrid":
        source = "hybrid"
    elif route == "web_search":
        source = "web"
    else:
        source = "rag"
    
    return {
        **state,
        "final_answer": final_answer,
        "source": source,
        "messages": [AIMessage(content=final_answer)],
    }


def answer_llm(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Fallback: answer with LLM only (no RAG context); append assistant message for memory."""
    question = state.get("question", "")
    ctx = _conversation_context(state)
    prompt = f"Conversation:\n{ctx}\n\nCurrent question: {question}" if ctx else question
    answer = _invoke_llm(
        prompt,
        system="""You are a helpful AI assistant. Answer questions directly and helpfully.
Use the Conversation above to interpret follow-ups (e.g. "elaborate", "what about the second point?", "that one") in context.

GUIDELINES:
- Be concise but complete
- If you don't know something, say so briefly
- For ambiguous questions, make a reasonable assumption or ask for clarification
- Never mention "I don't have access to..." - just help with what you can

FORMAT:
- Use **bold** for key terms
- Use bullet points for lists
- Use headers (##) for longer responses""",
    )
    return {
        **state,
        "final_answer": answer,
        "source": "llm",
        "messages": [AIMessage(content=answer)],
    }


def increment_retry(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Increment retry count when looping back."""
    retry = state.get("retry_count", 0)
    return {**state, "retry_count": retry + 1}


def route_after_increment_retry(state: AdaptiveRAGState) -> Literal["retrieve_web", "check_relevant"]:
    """After incrementing retry: go to retrieve_web (when context was not relevant) or check_relevant (when refining)."""
    next_after = state.get("next_after_retry", "check_relevant")
    return next_after


def _get_checkpointer(use_long_term: bool = False, db_path: str | Path | None = None) -> Any:
    """Short-term: MemorySaver (in-memory). Long-term: SqliteSaver (persists to file)."""
    if use_long_term and SqliteSaver is not None:
        path = db_path or Path(__file__).resolve().parent.parent / "data" / "langgraph_memory.sqlite"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(path))
    return MemorySaver()


def create_adaptive_rag_graph(
    ollama_model: str = OLLAMA_MODEL,
    max_retries: int = MAX_RETRIES,
    checkpointer: Any = None,
    use_long_term_memory: bool = False,
    memory_db_path: str | Path | None = None,
):
    """Build and compile the adaptive RAG LangGraph with optional memory checkpointer.

    - Short-term memory: checkpointer=None and use_long_term_memory=False → MemorySaver (in-memory, lost on restart).
    - Long-term memory: use_long_term_memory=True (and optionally memory_db_path) → SqliteSaver (persists to SQLite).
    - Or pass your own checkpointer instance.
    """
    global OLLAMA_MODEL, MAX_RETRIES
    OLLAMA_MODEL = ollama_model
    MAX_RETRIES = max_retries

    if checkpointer is None:
        checkpointer = _get_checkpointer(use_long_term=use_long_term_memory, db_path=memory_db_path)

    builder = StateGraph(AdaptiveRAGState)

    builder.add_node("query_analysis", query_analysis)
    builder.add_node("clarify", clarify)
    builder.add_node("retrieve_vectorstore", retrieve_vectorstore)
    builder.add_node("check_info_sufficient", check_info_sufficient)
    builder.add_node("answer_from_memory_and_chunks", answer_from_memory_and_chunks)
    builder.add_node("retrieve_web", retrieve_web)
    builder.add_node("retrieve_hybrid", retrieve_hybrid)
    builder.add_node("check_relevant", check_relevant)
    builder.add_node("generate", generate)
    builder.add_node("check_hallucinations", check_hallucinations)
    builder.add_node("check_answers_question", check_answers_question)
    builder.add_node("answer_rag", answer_rag)
    builder.add_node("answer_llm", answer_llm)
    builder.add_node("increment_retry", increment_retry)

    builder.set_entry_point("query_analysis")
    builder.add_conditional_edges("query_analysis", route_after_analysis)

    builder.add_edge("clarify", END)
    builder.add_edge("retrieve_vectorstore", "check_info_sufficient")
    builder.add_conditional_edges("check_info_sufficient", route_after_info_check)
    builder.add_edge("answer_from_memory_and_chunks", END)
    builder.add_edge("retrieve_web", "check_relevant")
    builder.add_edge("retrieve_hybrid", "check_relevant")
    builder.add_conditional_edges("check_relevant", route_after_relevant)

    builder.add_edge("generate", "check_hallucinations")
    builder.add_conditional_edges(
        "check_hallucinations",
        lambda s: "increment_retry" if s.get("has_hallucinations") else "check_answers_question",
    )
    builder.add_conditional_edges("check_answers_question", route_after_answers_question)

    builder.add_conditional_edges("increment_retry", route_after_increment_retry)

    builder.add_edge("answer_rag", END)
    builder.add_edge("answer_llm", END)

    return builder.compile(checkpointer=checkpointer)


# Re-export for streamlit and FastAPI
def get_graph(
    ollama_model: str = OLLAMA_MODEL,
    max_retries: int = MAX_RETRIES,
    checkpointer: Any = None,
    use_long_term_memory: bool = False,
    memory_db_path: str | Path | None = None,
):
    return create_adaptive_rag_graph(
        ollama_model=ollama_model,
        max_retries=max_retries,
        checkpointer=checkpointer,
        use_long_term_memory=use_long_term_memory,
        memory_db_path=memory_db_path,
    )


# LangGraph Studio compatible function (requires RunnableConfig parameter)
def studio_graph(config: dict) -> Any:
    """Graph factory for LangGraph Studio. Takes RunnableConfig as required."""
    return create_adaptive_rag_graph()
