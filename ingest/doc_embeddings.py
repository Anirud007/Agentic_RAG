"""
Document embedding service: BGE-M3 (dense + sparse) for text/table, SigLIP for images.

- Text, table, slide chunks → BGE-M3 dense + sparse (lexical_weights) → collection docs_text.
- Image chunks → SigLIP dense only → collection docs_image.
- Each result includes collection, embedding_dim, model_name, modality, and optional sparse_embedding.
"""

import base64
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, NamedTuple

# Workaround: some transformers/torch version combos lack is_torch_fx_available
try:
    import transformers.utils.import_utils as _transformers_import_utils
    if not hasattr(_transformers_import_utils, "is_torch_fx_available"):
        _transformers_import_utils.is_torch_fx_available = lambda: False
except Exception:
    pass

if TYPE_CHECKING:
    from PIL import Image

import numpy as np

try:
    from ingest.doc_chunker import Chunk
except ImportError:
    from doc_chunker import Chunk

logger = logging.getLogger(__name__)

# Model IDs
TEXT_MODEL_ID = "BAAI/bge-m3"
TEXT_MODEL_LIGHT = "BAAI/bge-small-en-v1.5"  # ~130MB, 384 dim, dense only (no sparse)
IMAGE_MODEL_ID = "google/siglip-base-patch16-224"

# Collection routing (Option 1: two collections)
COLLECTION_TEXT = "docs_text"
COLLECTION_IMAGE = "docs_image"
BGE_DIM = 1024
BGE_DIM_LIGHT = 384
SIGLIP_DIM = 768


def get_collection_names(thread_id: str | None = None) -> tuple[str, str]:
    """
    Return (text_collection, image_collection) names for a thread.

    Each chat session gets its own collections to isolate uploaded documents.
    Collection names can only contain letters, numbers, and underscores.
    """
    if thread_id:
        # Sanitize thread_id: replace hyphens with underscores (Milvus requirement)
        safe_id = thread_id.replace("-", "_")
        return f"docs_text_{safe_id}", f"docs_image_{safe_id}"
    return COLLECTION_TEXT, COLLECTION_IMAGE


class EmbeddingResult(NamedTuple):
    """One embedded chunk plus indexer metadata (where to insert, model info)."""

    chunk: Chunk
    embedding: np.ndarray  # dense vector
    collection: str  # "docs_text" or "docs_image"
    embedding_dim: int
    model_name: str
    modality: str  # "text" or "image"
    sparse_embedding: dict[str, float] | None = None  # BGE-M3 lexical_weights (text only); None for images


def _use_light_text_default() -> bool:
    import os
    return os.environ.get("USE_LIGHT_TEXT_MODEL", "").lower() in ("1", "true", "yes")


class EmbeddingService:
    """Loads BGE-M3 (or light BGE-small) for text and SigLIP for images; embeds chunks with optional parallelism."""

    def __init__(
        self,
        text_model_id: str | None = None,
        image_model_id: str = IMAGE_MODEL_ID,
        device: str | None = None,
        use_light_text_model: bool | None = None,
    ):
        if use_light_text_model is None:
            use_light_text_model = _use_light_text_default()
        self._use_light_text = bool(use_light_text_model)
        self._text_model_id = text_model_id or (TEXT_MODEL_LIGHT if self._use_light_text else TEXT_MODEL_ID)
        self._image_model_id = image_model_id
        self._device = device
        self._text_model = None
        self._text_embedding_dim = BGE_DIM_LIGHT if self._use_light_text else BGE_DIM
        self._image_processor = None
        self._image_model = None
        self._text_lock = threading.Lock()
        self._image_lock = threading.Lock()

    def _ensure_text_model(self):
        if self._text_model is None:
            if self._use_light_text:
                from sentence_transformers import SentenceTransformer
                # Use local_files_only if model is already cached to avoid HuggingFace network calls
                try:
                    self._text_model = SentenceTransformer(
                        self._text_model_id, device=self._device, local_files_only=True
                    )
                except Exception:
                    # Fallback to downloading if not cached
                    self._text_model = SentenceTransformer(self._text_model_id, device=self._device)
                self._text_embedding_dim = BGE_DIM_LIGHT
                logger.info("Loaded light text embedding model (dense only): %s", self._text_model_id)
            else:
                from FlagEmbedding import BGEM3FlagModel
                self._text_model = BGEM3FlagModel(
                    self._text_model_id,
                    use_fp16=(self._device != "cpu"),
                )
                self._text_embedding_dim = BGE_DIM
                logger.info("Loaded text embedding model (dense+sparse): %s", self._text_model_id)

    def _ensure_image_model(self):
        if self._image_model is None:
            import torch
            from transformers import AutoProcessor, AutoModel

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._image_processor = AutoProcessor.from_pretrained(self._image_model_id)
            self._image_model = AutoModel.from_pretrained(self._image_model_id).to(dev)
            self._image_model.eval()
            self._image_device = dev
            logger.info("Loaded image embedding model: %s on %s", self._image_model_id, dev)

    def embed_text(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        """Embed a single text. Returns (dense vector, sparse lexical_weights dict; sparse is {} for light model)."""
        self._ensure_text_model()
        with self._text_lock:
            if self._use_light_text:
                vec = self._text_model.encode(text, normalize_embeddings=True)
                if isinstance(vec, np.ndarray) and vec.ndim == 2:
                    vec = vec[0]
                return np.asarray(vec, dtype=np.float32), {}
            out = self._text_model.encode(
                text,
                return_dense=True,
                return_sparse=True,
                max_length=512,
            )
        dense = out["dense_vecs"]
        if isinstance(dense, np.ndarray) and dense.ndim == 2:
            dense = dense[0]
        dense = np.asarray(dense, dtype=np.float32)
        sparse = out.get("lexical_weights")
        if sparse is None:
            sparse = {}
        elif isinstance(sparse, list) and len(sparse) > 0:
            sparse = sparse[0] if isinstance(sparse[0], dict) else {}
        else:
            sparse = sparse if isinstance(sparse, dict) else {}
        return dense, sparse

    def embed_image(self, image: "Image.Image") -> np.ndarray:
        """Embed a PIL Image with SigLIP. Returns L2-normalized vector."""
        from PIL import Image as PILImage
        import torch

        self._ensure_image_model()
        if not isinstance(image, PILImage.Image):
            image = PILImage.open(image).convert("RGB")
        inputs = self._image_processor(images=image, return_tensors="pt").to(self._image_device)
        with self._image_lock:
            with torch.no_grad():
                out = self._image_model.get_image_features(**inputs)
        # Use pooler_output (one vector per image), not last_hidden_state (patch tokens).
        pooled = getattr(out, "pooler_output", None)
        if pooled is None and isinstance(out, (tuple, list)):
            pooled = out[1]  # BaseModelOutputWithPooling as tuple: (last_hidden_state, pooler_output)
        if pooled is None:
            pooled = getattr(out, "last_hidden_state", None)
            if pooled is not None:
                pooled = pooled[:, 0]
        vec = pooled[0].cpu().float().numpy()
        if vec.ndim > 1:
            vec = np.asarray(vec).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_text_for_image_search(self, text: str) -> np.ndarray:
        """Embed text with SigLIP for cross-modal image search. Returns L2-normalized 768-dim vector."""
        import torch

        self._ensure_image_model()
        # SigLIP processor handles text tokenization
        inputs = self._image_processor(text=[text], return_tensors="pt", padding=True).to(self._image_device)
        with self._image_lock:
            with torch.no_grad():
                # Use text_model directly to get text embeddings in same space as images
                text_outputs = self._image_model.text_model(**inputs)
                pooler = text_outputs.pooler_output
        vec = pooler[0].cpu().float().numpy()
        if vec.ndim > 1:
            vec = np.asarray(vec).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _collection_and_meta(self, source_part_type: str, embedding: np.ndarray) -> tuple[str, int, str, str]:
        """Return (collection, embedding_dim, model_name, modality) for indexer."""
        if source_part_type == "image":
            return COLLECTION_IMAGE, SIGLIP_DIM, IMAGE_MODEL_ID, "image"
        return COLLECTION_TEXT, self._text_embedding_dim, self._text_model_id, "text"

    def embed_chunks(
        self,
        chunks: list[Chunk],
        max_workers: int | None = 4,
    ) -> list[EmbeddingResult]:
        """
        Embed all chunks; route to docs_text (text/table/slide) or docs_image (image).
        Returns list of EmbeddingResult with chunk, embedding, collection, embedding_dim, model_name, modality.
        """
        from PIL import Image

        if not chunks:
            return []

        def do_one(
            idx: int, chunk: Chunk
        ) -> tuple[int, Chunk, np.ndarray, dict[str, float] | None]:
            if chunk.source_part_type == "image":
                raw = chunk.content
                try:
                    if raw.startswith("data:"):
                        raw = raw.split(",", 1)[-1]
                    img_bytes = base64.b64decode(raw)
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    logger.warning("Skip image chunk %s: decode failed: %s", chunk.chunk_id, e)
                    return idx, chunk, np.zeros(SIGLIP_DIM, dtype=np.float32), None
                emb = self.embed_image(image)
                return idx, chunk, emb, None
            else:
                dense, sparse = self.embed_text(chunk.content)
                return idx, chunk, dense, sparse

        workers = max(1, max_workers) if max_workers is not None else 4
        results: list[tuple[int, Chunk, np.ndarray, dict[str, float] | None]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(do_one, i, c): i for i, c in enumerate(chunks)}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    i = futures[fut]
                    logger.warning("Embedding failed for chunk index %s: %s", i, e)
                    results.append((i, chunks[i], np.zeros(self._text_embedding_dim, dtype=np.float32), None))

        results.sort(key=lambda x: x[0])
        out: list[EmbeddingResult] = []
        for _, chunk, emb, sparse in results:
            collection, embedding_dim, model_name, modality = self._collection_and_meta(
                chunk.source_part_type, emb
            )
            out.append(
                EmbeddingResult(
                    chunk=chunk,
                    embedding=emb,
                    collection=collection,
                    embedding_dim=embedding_dim,
                    model_name=model_name,
                    modality=modality,
                    sparse_embedding=sparse,
                )
            )
        return out

    def embed_texts_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Embed a list of texts. Returns (dense (N,dim), list of lexical_weights; empty list for light model)."""
        self._ensure_text_model()
        with self._text_lock:
            if self._use_light_text:
                dense = self._text_model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
                return np.asarray(dense, dtype=np.float32), []
            out = self._text_model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                batch_size=batch_size,
                max_length=512,
            )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = out.get("lexical_weights") or []
        if isinstance(sparse, dict):
            sparse = [sparse]
        return dense, sparse


# Optional: convenience for single-file use
def embed_chunks(
    chunks: list[Chunk], max_workers: int | None = 4
) -> list[EmbeddingResult]:
    """One-off embed with default EmbeddingService. Returns list of EmbeddingResult for indexer."""
    svc = EmbeddingService()
    return svc.embed_chunks(chunks, max_workers=max_workers)
