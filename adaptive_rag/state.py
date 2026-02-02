"""State schema for the adaptive RAG LangGraph."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class AdaptiveRAGState(TypedDict, total=False):
    """State for adaptive RAG: query analysis → retrieve → self-reflection → answer."""

    question: str
    messages: Annotated[list, add_messages]
    thread_id: str  # Session ID for document isolation
    source_file: str  # Optional: filter retrieval to this document
    document_role: str  # Optional: filter retrieval to this role (e.g. resume, paper)

    # Enhanced routing fields
    route: Literal["vectorstore", "web_search", "direct_llm", "hybrid", "clarify"]
    routing_confidence: float  # 0.0-1.0 confidence in route decision
    query_intent: Literal["question", "command", "clarification", "greeting", "follow_up"]
    clarification_question: str  # When route=clarify: question to ask the user
    info_sufficient: bool  # After check_info_sufficient: can we answer from memory + chunks?
    
    # Context and retrieval
    context: str
    web_context: str  # Context from web search (for hybrid mode)
    image_context: str  # Context from image retrieval (descriptions/locations)
    memory_context: str  # Relevant past conversation context
    retrieved_docs: list
    retrieved_images: list  # Raw image search hits
    
    # Response
    answer: str
    relevant: bool
    has_hallucinations: bool
    answers_question: bool
    image_only_response: bool  # Flag to skip hallucination checks for image-only responses
    
    # Flow control
    retry_count: int
    max_retries: int
    next_after_retry: Literal["retrieve_web", "check_relevant"]
    no_docs_indexed: bool  # true when user asked about docs but vector store returned nothing
    
    # Final output
    final_answer: str
    source: Literal["rag", "llm", "web", "hybrid", "clarify"]
    error: str
