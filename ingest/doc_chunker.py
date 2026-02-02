"""
Structure-aware document chunker for enterprise QA.

Uses pre_processor for extraction. Chunking follows document structure first;
token limits are a safety cap (Stage B). Supports PDF, DOCX, PPT (Excel later).

- Text (PDF/DOCX): Section/paragraph-aware; merge until ~350–450 tokens; section_title.
- Tables: Row-level chunks; "In table X (columns: A,B,C), row states: …"
- PPT: 1 slide = 1 chunk (title + bullets).
- Every chunk: chunk_id, location, section_title for citations.
"""

import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from ingest.pre_processor import (
        FilePreprocessor,
        Part,
        PathResolver,
        is_supported,
        preprocess,
    )
except ImportError:
    from pre_processor import (
        FilePreprocessor,
        Part,
        PathResolver,
        is_supported,
        preprocess,
    )

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults (enterprise QA)
# -----------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # English approx
TEXT_TOKEN_TARGET = 400  # 350–450
TEXT_TOKEN_MAX = 500    # hard cap before Stage B split
TEXT_TOKEN_OVERLAP_HARD = 30  # overlap only when forced to split
TABLE_ROW_GROUP_SIZE = 1  # 1 row per chunk; use 3–5 for tiny rows
MIN_PARAGRAPH_TOKENS = 20  # merge paragraphs below this

# -----------------------------------------------------------------------------
# Chunk model (with citation fields)
# -----------------------------------------------------------------------------


class Chunk(BaseModel):
    """
    A single chunk for ingestion with comprehensive metadata.
    Supports all document types: PDF, DOCX, PPTX, XLSX, TXT, MD
    """

    # Content
    content: str = Field(..., description="Chunk text or base64 image")
    content_length: int = Field(0, description="Character count")
    token_count: int = Field(0, description="Estimated token count")

    # Identity
    chunk_id: str = Field("", description="Stable SHA256 id for dedup and citations")
    chunk_index: int = Field(0, description="Position within document")
    source_part_type: str = Field(..., description="text, table, or image")

    # Source Document
    source_file: str = Field("", description="Original filename")
    document_type: str = Field("", description="pdf, docx, pptx, xlsx, txt, md")
    document_role: str = Field("", description="Optional role: resume, paper, manual, etc. (set at ingest)")

    # Location (for citations)
    location: str = Field("", description="Human-readable location string")
    page_number: int | None = Field(None, description="1-indexed page number (PDF/DOCX)")
    slide_number: int | None = Field(None, description="1-indexed slide number (PPTX)")
    sheet_name: str | None = Field(None, description="Sheet name (XLSX)")
    paragraph_index: int | None = Field(None, description="Paragraph index (DOCX/TXT)")

    # Section Context
    section_title: str = Field("", description="Heading/section for retrieval context")
    section_hierarchy: list[str] = Field(default_factory=list, description="Heading path")

    # Table-specific
    table_id: str | None = Field(None, description="Unique table identifier")
    column_names: list[str] | None = Field(None, description="Table column headers")
    row_index: int | None = Field(None, description="Row index within table")

    # Chunking metadata
    is_continuation: bool = Field(False, description="Continues from previous chunk")
    has_continuation: bool = Field(False, description="Continues in next chunk")
    total_chunks: int = Field(0, description="Total chunks from this part")

    # Additional metadata (for extensibility)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# -----------------------------------------------------------------------------
# Token and ID helpers
# -----------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _make_chunk_id(doc_id: str, location: str, content: str) -> str:
    raw = f"{doc_id}|{location}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _format_location(meta: dict[str, Any], row_index: int | None = None) -> str:
    """Generate human-readable location string from metadata."""
    parts = []

    # Page-based (PDF, DOCX)
    if "page" in meta:
        parts.append(f"Page {meta['page'] + 1}")

    # Slide-based (PPTX)
    if "slide" in meta:
        parts.append(f"Slide {meta['slide'] + 1}")

    # Sheet-based (XLSX)
    if "sheet_name" in meta:
        parts.append(f"Sheet '{meta['sheet_name']}'")

    # Table reference
    if "table_index" in meta or "table_id" in meta:
        table_num = meta.get("table_index", 0) + 1
        parts.append(f"Table {table_num}")

    # Row reference
    if row_index is not None:
        parts.append(f"Row {row_index + 1}")

    # Paragraph (DOCX, TXT)
    if "paragraph" in meta and not parts:
        parts.append(f"Paragraph {meta['paragraph'] + 1}")

    return ", ".join(parts) if parts else "Document"


def _get_document_type(source_file: str) -> str:
    """Extract document type from file extension."""
    ext = os.path.splitext(source_file)[1].lower()
    type_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".txt": "txt",
        ".md": "md",
    }
    return type_map.get(ext, "unknown")


def _doc_id(source_file: str) -> str:
    return os.path.basename(source_file) or "doc"


def _infer_section_title(text: str, max_len: int = 80) -> str:
    first_line = (text.split("\n")[0] or "").strip()
    if not first_line:
        return ""
    if len(first_line) <= max_len and (first_line.endswith(":") or first_line.isupper() or not first_line.endswith(".")):
        return first_line
    return first_line[:max_len] + "…" if len(first_line) > max_len else first_line


# -----------------------------------------------------------------------------
# Stage B: token budget (split long / merge short)
# -----------------------------------------------------------------------------


def _split_long_text_by_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text by sentence boundaries when over max_tokens; add small overlap."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return [text]
    target = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if current_len + len(s) > target and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            overlap_count = 0
            overlap_len = 0
            for i in range(len(current) - 1, -1, -1):
                overlap_len += len(current[i]) + 1
                overlap_count += 1
                if overlap_len >= overlap_chars:
                    break
            current = current[-overlap_count:] if overlap_count < len(current) else []
            current_len = sum(len(x) + 1 for x in current) - 1
        current.append(s)
        current_len += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _merge_short_segments(segments: list[tuple[str, str]], min_tokens: int) -> list[tuple[str, str]]:
    """Merge (content, section_title) segments that are too short, within same section."""
    if not segments:
        return []
    merged: list[tuple[str, str]] = []
    buf_text: list[str] = []
    buf_title = ""
    for text, title in segments:
        if _estimate_tokens(text) < min_tokens and buf_text and (not buf_title or buf_title == title):
            buf_text.append(text)
            if title and not buf_title:
                buf_title = title
        else:
            if buf_text:
                merged.append(("\n\n".join(buf_text), buf_title or "Section"))
                buf_text = []
                buf_title = ""
            if _estimate_tokens(text) < min_tokens:
                buf_text.append(text)
                buf_title = title or buf_title
            else:
                merged.append((text, title or "Section"))
    if buf_text:
        merged.append(("\n\n".join(buf_text), buf_title or "Section"))
    return merged


# -----------------------------------------------------------------------------
# Structure-first: text (PDF/DOCX)
# -----------------------------------------------------------------------------


def _chunk_text_structure_first(
    content: str,
    metadata: dict[str, Any],
    source_file: str,
    token_target: int,
    token_max: int,
    overlap_hard: int,
) -> list[Chunk]:
    """
    Paragraph/section-aware text chunking. Merge paragraphs until token target;
    Stage B: split long by sentence (overlap), merge short.
    """
    is_slide = "slide" in metadata
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    target_chars = token_target * CHARS_PER_TOKEN
    max_chars = token_max * CHARS_PER_TOKEN

    doc_type = _get_document_type(source_file)

    if is_slide:
        # PPT: 1 slide = 1 chunk unless over cap
        full = "\n\n".join(paragraphs)
        if _estimate_tokens(full) <= token_max:
            doc_id = _doc_id(source_file)
            loc = _format_location(metadata)
            section_title = _infer_section_title(full) or f"Slide {metadata.get('slide', 0) + 1}"
            chunk_id = _make_chunk_id(doc_id, loc, full)
            return [
                Chunk(
                    content=full,
                    content_length=len(full),
                    token_count=_estimate_tokens(full),
                    metadata=metadata,
                    chunk_index=0,
                    source_part_type="text",
                    source_file=source_file,
                    document_type=doc_type,
                    chunk_id=chunk_id,
                    location=loc,
                    slide_number=metadata.get("slide", 0) + 1,
                    section_title=section_title,
                    total_chunks=1,
                )
            ]
        # Stage B: split by sentence
        split_parts = _split_long_text_by_sentences(full, token_max, overlap_hard)
        doc_id = _doc_id(source_file)
        loc = _format_location(metadata)
        section_title = _infer_section_title(full) or f"Slide {metadata.get('slide', 0) + 1}"
        return [
            Chunk(
                content=p,
                content_length=len(p),
                token_count=_estimate_tokens(p),
                metadata={**metadata, "chunk_index": i, "total_chunks": len(split_parts)},
                chunk_index=i,
                source_part_type="text",
                source_file=source_file,
                document_type=doc_type,
                chunk_id=_make_chunk_id(doc_id, f"{loc} part {i+1}", p),
                location=f"{loc} (part {i+1})",
                slide_number=metadata.get("slide", 0) + 1,
                section_title=section_title,
                is_continuation=i > 0,
                has_continuation=i < len(split_parts) - 1,
                total_chunks=len(split_parts),
            )
            for i, p in enumerate(split_parts)
        ]

    # PDF/DOCX: merge paragraphs until target
    segments: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    section_title = ""
    for p in paragraphs:
        title = _infer_section_title(p) if not current else section_title
        if current_len + len(p) > target_chars and current:
            merged_text = "\n\n".join(current)
            segments.append((merged_text, section_title or _infer_section_title(merged_text)))
            current = [p]
            current_len = len(p)
            section_title = title
        else:
            current.append(p)
            current_len += len(p)
            if not section_title:
                section_title = title
    if current:
        segments.append(("\n\n".join(current), section_title or "Section"))

    segments = _merge_short_segments(segments, MIN_PARAGRAPH_TOKENS)

    chunks: list[Chunk] = []
    doc_id = _doc_id(source_file)
    for i, (text, title) in enumerate(segments):
        if _estimate_tokens(text) > token_max:
            split_parts = _split_long_text_by_sentences(text, token_max, overlap_hard)
            for j, part in enumerate(split_parts):
                loc = _format_location(metadata)
                loc_label = f"{loc} (part {j+1})" if len(segments) > 1 or j > 0 else loc
                chunks.append(
                    Chunk(
                        content=part,
                        content_length=len(part),
                        token_count=_estimate_tokens(part),
                        metadata={**metadata, "chunk_index": len(chunks), "total_chunks": len(split_parts)},
                        chunk_index=len(chunks),
                        source_part_type="text",
                        source_file=source_file,
                        document_type=doc_type,
                        chunk_id=_make_chunk_id(doc_id, loc_label, part),
                        location=loc_label,
                        page_number=metadata.get("page", -1) + 1 if "page" in metadata else None,
                        paragraph_index=metadata.get("paragraph"),
                        section_title=title,
                        is_continuation=j > 0,
                        has_continuation=j < len(split_parts) - 1,
                        total_chunks=len(split_parts),
                    )
                )
        else:
            loc = _format_location(metadata)
            chunks.append(
                Chunk(
                    content=text,
                    content_length=len(text),
                    token_count=_estimate_tokens(text),
                    metadata={**metadata, "chunk_index": i, "total_chunks": len(segments)},
                    chunk_index=i,
                    source_part_type="text",
                    source_file=source_file,
                    document_type=doc_type,
                    chunk_id=_make_chunk_id(doc_id, loc, text),
                    location=loc,
                    page_number=metadata.get("page", -1) + 1 if "page" in metadata else None,
                    paragraph_index=metadata.get("paragraph"),
                    section_title=title,
                    total_chunks=len(segments),
                )
            )
    return chunks


# -----------------------------------------------------------------------------
# Structure-first: table row-level
# -----------------------------------------------------------------------------


def _chunk_table_row_level(
    content: str,
    metadata: dict[str, Any],
    source_file: str,
    row_group_size: int = 1,
) -> list[Chunk]:
    """
    Row-level table chunking. Format: "In table X (columns: A,B,C), row states: A=…, B=…, C=…"
    """
    doc_type = _get_document_type(source_file)
    doc_id = _doc_id(source_file)

    if "Table:\n" not in content and not content.strip().startswith("Table:"):
        loc = _format_location(metadata)
        return [
            Chunk(
                content=content,
                content_length=len(content),
                token_count=_estimate_tokens(content),
                metadata=metadata,
                chunk_index=0,
                source_part_type="table",
                source_file=source_file,
                document_type=doc_type,
                chunk_id=_make_chunk_id(doc_id, loc, content),
                location=loc,
                page_number=metadata.get("page", -1) + 1 if "page" in metadata else None,
                slide_number=metadata.get("slide", -1) + 1 if "slide" in metadata else None,
                sheet_name=metadata.get("sheet_name"),
                table_id=metadata.get("table_id", "Table"),
                column_names=metadata.get("columns"),
                section_title=metadata.get("table_id", "Table"),
                total_chunks=1,
            )
        ]

    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if not lines:
        return []
    rows_raw = lines[1:] if lines[0].lower().startswith("table") else lines
    columns = metadata.get("columns")
    if isinstance(columns, list):
        col_names = [str(c).strip() for c in columns]
    else:
        col_names = [f"col_{i}" for i in range(len(rows_raw[0].split(" | ")))] if rows_raw else []

    chunks: list[Chunk] = []
    table_label = metadata.get("table_id", "table")
    total_row_chunks = (len(rows_raw) + row_group_size - 1) // row_group_size

    for g in range(0, len(rows_raw), row_group_size):
        group = rows_raw[g : g + row_group_size]
        cells_per_row = [r.split(" | ") for r in group]
        row_texts: list[str] = []
        for cells in cells_per_row:
            pairs = []
            for idx, cell in enumerate(cells):
                name = col_names[idx] if idx < len(col_names) else f"col_{idx}"
                pairs.append(f"{name}={cell}")
            row_texts.append(", ".join(pairs))
        segment = "In table " + table_label + " (columns: " + ", ".join(col_names) + "), row states: " + "; ".join(row_texts)
        loc = _format_location(metadata, row_index=g)
        chunk_idx = g // row_group_size

        chunks.append(
            Chunk(
                content=segment,
                content_length=len(segment),
                token_count=_estimate_tokens(segment),
                metadata={**metadata, "row_index": g, "row_count": len(group)},
                chunk_index=chunk_idx,
                source_part_type="table",
                source_file=source_file,
                document_type=doc_type,
                chunk_id=_make_chunk_id(doc_id, loc, segment),
                location=loc,
                page_number=metadata.get("page", -1) + 1 if "page" in metadata else None,
                slide_number=metadata.get("slide", -1) + 1 if "slide" in metadata else None,
                sheet_name=metadata.get("sheet_name"),
                table_id=table_label,
                column_names=col_names,
                row_index=g,
                section_title=table_label,
                total_chunks=total_row_chunks,
            )
        )
    return chunks


# -----------------------------------------------------------------------------
# Main: chunk_parts (dispatch by type)
# -----------------------------------------------------------------------------


def chunk_parts(
    parts: list[Part],
    *,
    source_file: str | None = None,
    text_token_target: int = TEXT_TOKEN_TARGET,
    text_token_max: int = TEXT_TOKEN_MAX,
    text_token_overlap_hard: int = TEXT_TOKEN_OVERLAP_HARD,
    table_row_group_size: int = TABLE_ROW_GROUP_SIZE,
) -> list[Chunk]:
    """
    Structure-aware chunking of preprocessor parts.

    - Text: Section/paragraph merge → token cap (Stage B). PPT = 1 slide = 1 chunk.
    - Table: Row-level; "In table X (columns: …), row states: …"
    - Image: 1 chunk with chunk_id, location, section_title.

    Args:
        parts: From pre_processor.preprocess().
        source_file: Override source file path.
        text_token_target: Target size for text chunks (tokens).
        text_token_max: Hard cap before sentence split.
        text_token_overlap_hard: Overlap when forced to split (tokens).
        table_row_group_size: Rows per table chunk (1 or 3–5 for tiny rows).

    Returns:
        List of Chunk with chunk_id, location, section_title.
    """
    chunks: list[Chunk] = []
    for part in parts:
        src_file = str(source_file or part.metadata.get("source_file", "") or "")
        if part.type == "text":
            chunks.extend(
                _chunk_text_structure_first(
                    part.content,
                    part.metadata,
                    src_file,
                    token_target=text_token_target,
                    token_max=text_token_max,
                    overlap_hard=text_token_overlap_hard,
                )
            )
        elif part.type == "table":
            chunks.extend(
                _chunk_table_row_level(
                    part.content,
                    part.metadata,
                    src_file,
                    row_group_size=table_row_group_size,
                )
            )
        elif part.type == "image":
            loc = _format_location(part.metadata)
            chunk_id = _make_chunk_id(_doc_id(src_file), loc, part.content[:500])
            section = part.metadata.get("image_id", "Image") or "Image"
            doc_type = _get_document_type(src_file)
            chunks.append(
                Chunk(
                    content=part.content,
                    content_length=len(part.content),
                    token_count=0,  # Images don't have tokens
                    metadata=part.metadata,
                    chunk_index=0,
                    source_part_type="image",
                    source_file=src_file,
                    document_type=doc_type,
                    chunk_id=chunk_id,
                    location=loc,
                    page_number=part.metadata.get("page", -1) + 1 if "page" in part.metadata else None,
                    slide_number=part.metadata.get("slide", -1) + 1 if "slide" in part.metadata else None,
                    section_title=section,
                    total_chunks=1,
                )
            )
        else:
            loc = _format_location(part.metadata)
            doc_type = _get_document_type(src_file)
            chunks.append(
                Chunk(
                    content=part.content,
                    content_length=len(part.content),
                    token_count=_estimate_tokens(part.content),
                    metadata=part.metadata,
                    chunk_index=0,
                    source_part_type=part.type,
                    source_file=src_file,
                    document_type=doc_type,
                    chunk_id=_make_chunk_id(_doc_id(src_file), loc, part.content[:500]),
                    location=loc,
                    section_title=part.metadata.get("source_file", ""),
                    total_chunks=1,
                )
            )
    return chunks


# -----------------------------------------------------------------------------
# File / files / directory (unchanged API, new defaults)
# -----------------------------------------------------------------------------


def chunk_file(
    file_path: str,
    *,
    resolve: bool = True,
    preprocessor: FilePreprocessor | None = None,
    text_token_target: int = TEXT_TOKEN_TARGET,
    text_token_max: int = TEXT_TOKEN_MAX,
    text_token_overlap_hard: int = TEXT_TOKEN_OVERLAP_HARD,
    table_row_group_size: int = TABLE_ROW_GROUP_SIZE,
) -> list[Chunk]:
    """Preprocess one file and return structure-aware chunks."""
    prep = preprocessor or FilePreprocessor()
    try:
        parts = prep.preprocess(file_path, resolve=resolve)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Skip %s: %s", file_path, e)
        return []
    path = PathResolver.resolve(file_path) if resolve else os.path.abspath(file_path)
    return chunk_parts(
        parts,
        source_file=str(path),
        text_token_target=text_token_target,
        text_token_max=text_token_max,
        text_token_overlap_hard=text_token_overlap_hard,
        table_row_group_size=table_row_group_size,
    )


def chunk_files(
    file_paths: list[str] | list[Path],
    *,
    resolve: bool = True,
    max_workers: int | None = None,
    preprocessor: FilePreprocessor | None = None,
    text_token_target: int = TEXT_TOKEN_TARGET,
    text_token_max: int = TEXT_TOKEN_MAX,
    text_token_overlap_hard: int = TEXT_TOKEN_OVERLAP_HARD,
    table_row_group_size: int = TABLE_ROW_GROUP_SIZE,
) -> list[Chunk]:
    """Preprocess and chunk multiple files concurrently (structure-aware)."""
    if not file_paths:
        return []
    # Normalize to str so Chunk.source_file is always str
    paths_str = [str(p) for p in file_paths]
    workers = max_workers if max_workers is not None else min(32, len(paths_str) + 4)
    prep = preprocessor or FilePreprocessor()
    all_chunks: list[Chunk] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(
                _chunk_one_file,
                path,
                resolve=resolve,
                preprocessor=prep,
                text_token_target=text_token_target,
                text_token_max=text_token_max,
                text_token_overlap_hard=text_token_overlap_hard,
                table_row_group_size=table_row_group_size,
            ): path
            for path in paths_str
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                logger.warning("Chunk file %s failed: %s", path, e, exc_info=False)
    logger.info("Total chunks from %d files: %d", len(paths_str), len(all_chunks))
    return all_chunks


def _chunk_one_file(
    file_path: str,
    *,
    resolve: bool,
    preprocessor: FilePreprocessor,
    text_token_target: int,
    text_token_max: int,
    text_token_overlap_hard: int,
    table_row_group_size: int,
) -> list[Chunk]:
    return chunk_file(
        file_path,
        resolve=resolve,
        preprocessor=preprocessor,
        text_token_target=text_token_target,
        text_token_max=text_token_max,
        text_token_overlap_hard=text_token_overlap_hard,
        table_row_group_size=table_row_group_size,
    )


def chunk_directory(
    dir_path: str,
    *,
    max_workers: int | None = None,
    extensions: tuple[str, ...] = (".docx", ".pdf", ".txt", ".md", ".pptx"),
    text_token_target: int = TEXT_TOKEN_TARGET,
    text_token_max: int = TEXT_TOKEN_MAX,
    text_token_overlap_hard: int = TEXT_TOKEN_OVERLAP_HARD,
    table_row_group_size: int = TABLE_ROW_GROUP_SIZE,
) -> list[Chunk]:
    """Discover supported files and chunk them concurrently (structure-aware)."""
    path = Path(dir_path)
    if not path.is_dir():
        logger.warning("Not a directory: %s", dir_path)
        return []
    files = [
        str(p)
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return chunk_files(
        files,
        resolve=False,
        max_workers=max_workers,
        text_token_target=text_token_target,
        text_token_max=text_token_max,
        text_token_overlap_hard=text_token_overlap_hard,
        table_row_group_size=table_row_group_size,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Structure-aware chunker (pre_processor)")
    parser.add_argument("paths", nargs="+", help="File or directory paths")
    parser.add_argument("--text-tokens", type=int, default=TEXT_TOKEN_TARGET)
    parser.add_argument("--text-max", type=int, default=TEXT_TOKEN_MAX)
    parser.add_argument("--overlap", type=int, default=TEXT_TOKEN_OVERLAP_HARD)
    parser.add_argument("--table-rows", type=int, default=TABLE_ROW_GROUP_SIZE)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", "-o", help="Write JSONL")
    args = parser.parse_args()

    all_paths: list[str] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            all_paths.extend(
                str(x) for x in path.iterdir()
                if x.is_file() and x.suffix.lower() in (".docx", ".pdf", ".txt", ".md", ".pptx")
            )
        elif path.is_file() and is_supported(str(path)):
            all_paths.append(str(path))
        else:
            logger.warning("Skip: %s", p)

    if not all_paths:
        logger.error("No supported files")
        return 1

    chunks = chunk_files(
        all_paths,
        max_workers=args.workers,
        text_token_target=args.text_tokens,
        text_token_max=args.text_max,
        text_token_overlap_hard=args.overlap,
        table_row_group_size=args.table_rows,
    )
    if args.output:
        import json
        with open(args.output, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        logger.info("Wrote %d chunks to %s", len(chunks), args.output)
    else:
        for i, c in enumerate(chunks):
            preview = (c.content[:50] + "…") if len(c.content) > 50 else c.content
            print(f"  [{i}] {c.source_part_type} | {c.location} | {c.section_title} | {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
