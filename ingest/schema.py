"""
Unified Ingestion Schema for Multi-Agentic RAG Chatbot.

This module defines the comprehensive data schema for document ingestion,
supporting: PDF, DOCX, PPTX, TXT, MD, XLSX

Schema Design Principles:
- Consistent field naming across all document types
- Complete metadata for citation and retrieval
- Type-safe with Pydantic validation
- Extensible for future document types
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    MD = "md"


class PartType(str, Enum):
    """Types of extracted content parts."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    HEADER = "header"
    FOOTER = "footer"


class ChunkType(str, Enum):
    """Types of processed chunks for embedding."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


# =============================================================================
# DOCUMENT METADATA SCHEMA
# =============================================================================

class DocumentMetadata(BaseModel):
    """
    Comprehensive document-level metadata.
    Captured once per document during ingestion.
    """
    # File Information
    file_name: str = Field(..., description="Original filename with extension")
    file_path: str = Field(..., description="Full path to the source file")
    file_size_bytes: int = Field(0, description="File size in bytes")
    file_extension: str = Field(..., description="File extension (e.g., '.pdf')")
    document_type: DocumentType = Field(..., description="Classified document type")

    # Document Properties
    title: Optional[str] = Field(None, description="Document title if available")
    author: Optional[str] = Field(None, description="Document author if available")
    created_date: Optional[datetime] = Field(None, description="Document creation date")
    modified_date: Optional[datetime] = Field(None, description="Last modification date")

    # Structure Information
    total_pages: Optional[int] = Field(None, description="Total pages (PDF/DOCX)")
    total_slides: Optional[int] = Field(None, description="Total slides (PPTX)")
    total_sheets: Optional[int] = Field(None, description="Total sheets (XLSX)")
    total_paragraphs: Optional[int] = Field(None, description="Total paragraphs")
    total_tables: Optional[int] = Field(None, description="Total tables in document")
    total_images: Optional[int] = Field(None, description="Total images in document")

    # Ingestion Tracking
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    ingestion_version: str = Field("1.0.0", description="Schema version")

    class Config:
        use_enum_values = True


# =============================================================================
# LOCATION SCHEMA (Where content appears in document)
# =============================================================================

class ContentLocation(BaseModel):
    """
    Precise location of content within a document.
    Enables accurate citations and source references.
    """
    # Page-based (PDF, DOCX)
    page_number: Optional[int] = Field(None, description="1-indexed page number")
    page_total: Optional[int] = Field(None, description="Total pages for context")

    # Slide-based (PPTX)
    slide_number: Optional[int] = Field(None, description="1-indexed slide number")
    slide_total: Optional[int] = Field(None, description="Total slides for context")
    slide_title: Optional[str] = Field(None, description="Slide title if available")

    # Sheet-based (XLSX)
    sheet_name: Optional[str] = Field(None, description="Excel sheet name")
    sheet_index: Optional[int] = Field(None, description="0-indexed sheet index")
    cell_range: Optional[str] = Field(None, description="Cell range (e.g., 'A1:D10')")

    # Paragraph/Section (DOCX, TXT, MD)
    paragraph_index: Optional[int] = Field(None, description="0-indexed paragraph")
    section_name: Optional[str] = Field(None, description="Document section/heading")
    heading_level: Optional[int] = Field(None, description="Heading level (1-6)")

    # Table-specific
    table_index: Optional[int] = Field(None, description="0-indexed table number")
    table_name: Optional[str] = Field(None, description="Table caption/name")
    row_index: Optional[int] = Field(None, description="0-indexed row number")
    row_range: Optional[str] = Field(None, description="Row range (e.g., '1-5')")

    # Image-specific
    image_index: Optional[int] = Field(None, description="0-indexed image number")
    image_caption: Optional[str] = Field(None, description="Image caption if available")

    def to_citation(self) -> str:
        """Generate human-readable citation string."""
        parts = []
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.slide_number:
            title = f" ({self.slide_title})" if self.slide_title else ""
            parts.append(f"Slide {self.slide_number}{title}")
        if self.sheet_name:
            parts.append(f"Sheet '{self.sheet_name}'")
        if self.cell_range:
            parts.append(f"Cells {self.cell_range}")
        if self.table_index is not None:
            parts.append(f"Table {self.table_index + 1}")
        if self.row_index is not None:
            parts.append(f"Row {self.row_index + 1}")
        if self.section_name:
            parts.append(f"Section: {self.section_name}")
        return ", ".join(parts) if parts else "Document"


# =============================================================================
# TABLE SCHEMA
# =============================================================================

class TableSchema(BaseModel):
    """
    Detailed table structure information.
    Enables structured table querying and row-level retrieval.
    """
    table_id: str = Field(..., description="Unique table identifier")
    column_names: list[str] = Field(default_factory=list, description="Column headers")
    column_count: int = Field(0, description="Number of columns")
    row_count: int = Field(0, description="Number of data rows (excluding header)")
    has_header: bool = Field(True, description="Whether first row is header")
    column_types: Optional[list[str]] = Field(None, description="Inferred column types")

    # For XLSX specific
    is_pivot_table: bool = Field(False, description="Is this a pivot table")
    has_merged_cells: bool = Field(False, description="Contains merged cells")
    has_formulas: bool = Field(False, description="Contains formulas")


# =============================================================================
# EXTRACTED PART SCHEMA (Raw extraction output)
# =============================================================================

class ExtractedPart(BaseModel):
    """
    Single extracted content part from a document.
    Output of the preprocessing stage before chunking.
    """
    # Identity
    part_id: str = Field(..., description="Unique part identifier")
    part_type: PartType = Field(..., description="Type of content")

    # Content
    content: str = Field(..., description="Text content or base64 image")
    content_length: int = Field(0, description="Character count for text")

    # Source
    source_file: str = Field(..., description="Source filename")
    document_type: DocumentType = Field(..., description="Source document type")

    # Location
    location: ContentLocation = Field(default_factory=ContentLocation)

    # Table-specific (when part_type == TABLE)
    table_schema: Optional[TableSchema] = Field(None, description="Table structure")

    # Image-specific (when part_type == IMAGE)
    image_format: Optional[str] = Field(None, description="Image format (png, jpg, etc.)")
    image_width: Optional[int] = Field(None, description="Image width in pixels")
    image_height: Optional[int] = Field(None, description="Image height in pixels")

    # Extraction metadata
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# =============================================================================
# CHUNK SCHEMA (After chunking, ready for embedding)
# =============================================================================

class Chunk(BaseModel):
    """
    Processed chunk ready for embedding and indexing.
    This is the final schema stored in Milvus.
    """
    # Identity
    chunk_id: str = Field(..., description="Unique chunk ID (SHA256 hash)")
    chunk_index: int = Field(0, description="Position within document")
    chunk_type: ChunkType = Field(..., description="Type of chunk content")

    # Content
    content: str = Field(..., description="Chunk text or base64 image")
    content_length: int = Field(0, description="Character count")
    token_count: int = Field(0, description="Estimated token count")

    # Source Document
    source_file: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Document type")

    # Location (for citations)
    location: ContentLocation = Field(default_factory=ContentLocation)
    location_string: str = Field("", description="Human-readable location")

    # Section Context
    section_title: str = Field("", description="Section/heading for context")
    section_hierarchy: list[str] = Field(default_factory=list, description="Heading path")

    # Table Context (when chunk_type == TABLE)
    table_schema: Optional[TableSchema] = Field(None)
    table_context: Optional[str] = Field(None, description="Table description")

    # Chunking Metadata
    is_continuation: bool = Field(False, description="Continues from previous chunk")
    has_continuation: bool = Field(False, description="Continues in next chunk")
    parent_part_id: Optional[str] = Field(None, description="Source part ID")

    # Processing Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump()


# =============================================================================
# EMBEDDING RESULT SCHEMA
# =============================================================================

class EmbeddingResult(BaseModel):
    """
    Chunk with computed embedding, ready for Milvus insertion.
    """
    # Chunk data
    chunk: Chunk = Field(..., description="Source chunk")

    # Embedding
    embedding: list[float] = Field(..., description="Dense embedding vector")
    embedding_dim: int = Field(..., description="Embedding dimension")

    # Sparse embedding (for hybrid search)
    sparse_embedding: Optional[dict[str, float]] = Field(
        None, description="Lexical weights for sparse retrieval"
    )

    # Model info
    model_name: str = Field(..., description="Embedding model used")
    modality: str = Field(..., description="text or image")

    # Collection routing
    collection: str = Field(..., description="Target Milvus collection")

    class Config:
        use_enum_values = True


# =============================================================================
# MILVUS COLLECTION SCHEMAS
# =============================================================================

# Text Collection Schema (for Milvus)
TEXT_COLLECTION_SCHEMA = {
    "collection_name": "docs_text",
    "fields": [
        {"name": "id", "type": "VARCHAR", "max_length": 64, "is_primary": True},
        {"name": "vector", "type": "FLOAT_VECTOR", "dim": 384},  # BGE-small default

        # Content
        {"name": "content", "type": "VARCHAR", "max_length": 65535},
        {"name": "content_length", "type": "INT32"},
        {"name": "token_count", "type": "INT32"},

        # Source
        {"name": "source_file", "type": "VARCHAR", "max_length": 512},
        {"name": "document_type", "type": "VARCHAR", "max_length": 16},

        # Location
        {"name": "location_string", "type": "VARCHAR", "max_length": 256},
        {"name": "page_number", "type": "INT32"},
        {"name": "slide_number", "type": "INT32"},
        {"name": "sheet_name", "type": "VARCHAR", "max_length": 128},

        # Section
        {"name": "section_title", "type": "VARCHAR", "max_length": 512},
        {"name": "chunk_type", "type": "VARCHAR", "max_length": 16},

        # Table
        {"name": "table_id", "type": "VARCHAR", "max_length": 128},
        {"name": "column_names", "type": "VARCHAR", "max_length": 2048},

        # Metadata
        {"name": "chunk_index", "type": "INT32"},
        {"name": "created_at", "type": "INT64"},  # Unix timestamp
    ],
    "indexes": [
        {"field": "vector", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 128}},
        {"field": "document_type", "type": "STL_SORT"},
        {"field": "source_file", "type": "STL_SORT"},
    ]
}

# Image Collection Schema (for Milvus)
IMAGE_COLLECTION_SCHEMA = {
    "collection_name": "docs_image",
    "fields": [
        {"name": "id", "type": "VARCHAR", "max_length": 64, "is_primary": True},
        {"name": "vector", "type": "FLOAT_VECTOR", "dim": 768},  # SigLIP default

        # Source
        {"name": "source_file", "type": "VARCHAR", "max_length": 512},
        {"name": "document_type", "type": "VARCHAR", "max_length": 16},

        # Location
        {"name": "location_string", "type": "VARCHAR", "max_length": 256},
        {"name": "page_number", "type": "INT32"},
        {"name": "slide_number", "type": "INT32"},

        # Image info
        {"name": "image_caption", "type": "VARCHAR", "max_length": 1024},
        {"name": "image_format", "type": "VARCHAR", "max_length": 16},

        # Metadata
        {"name": "created_at", "type": "INT64"},
    ],
    "indexes": [
        {"field": "vector", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 64}},
        {"field": "source_file", "type": "STL_SORT"},
    ]
}


# =============================================================================
# SCHEMA SUMMARY
# =============================================================================

SCHEMA_SUMMARY = """
================================================================================
                    DOCUMENT INGESTION SCHEMA v1.0.0
================================================================================

SUPPORTED DOCUMENT TYPES:
  - PDF  : Text, Tables, Images (per page)
  - DOCX : Text, Tables, Images (paragraphs + media)
  - PPTX : Text, Tables, Images (per slide)
  - XLSX : Tables, Cell data (per sheet)
  - TXT  : Plain text
  - MD   : Markdown text

EXTRACTION PIPELINE:
  Document → ExtractedPart[] → Chunk[] → EmbeddingResult[] → Milvus

MILVUS COLLECTIONS:
  1. docs_text  : Text/Table chunks (BGE embeddings, 384/1024 dim)
  2. docs_image : Image chunks (SigLIP embeddings, 768 dim)

KEY FIELDS FOR RETRIEVAL:
  - chunk_id       : Unique identifier for deduplication
  - content        : Searchable text content
  - location_string: Human-readable citation
  - section_title  : Context for retrieval ranking
  - document_type  : Filter by source type
  - source_file    : Filter by specific document

CITATION FORMAT:
  "[{source_file}] {location_string} - {section_title}"
  Example: "[report.pdf] Page 5, Table 2 - Financial Summary"

================================================================================
"""


def print_schema_summary():
    """Print the schema summary."""
    print(SCHEMA_SUMMARY)


if __name__ == "__main__":
    print_schema_summary()
