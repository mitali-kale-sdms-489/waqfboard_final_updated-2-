"""
Vector search POC — semantic document retrieval.

Builds one searchable text from extracted fields, generates a single
embedding, stores it in document_embeddings, and (in later phases) finds
top-K similar documents. Property ID from extracted_fields is the final
arbiter — vector search only retrieves candidates.

pgvector is used when the Postgres extension is available; otherwise
similarity runs in Python over the stored float arrays so the demo never
hard-fails on a missing extension.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentEmbedding, ExtractedField, FieldName

logger = logging.getLogger(__name__)
settings = get_settings()

# Set at startup by app.main._init_pgvector() — False on SQLite or when the
# extension can't be created.
pgvector_available: bool = False

# Display order for the combined searchable text (matches demo example).
_SEARCH_FIELD_ORDER: list[tuple[FieldName, str]] = [
    (FieldName.property_id, "Property ID"),
    (FieldName.survey_number, "Survey Number"),
    (FieldName.village, "Village"),
    (FieldName.registration_date, "Registration Date"),
    (FieldName.extent, "Extent"),
    (FieldName.mutawalli_name, "Mutawalli"),
]

GEMINI_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_TIMEOUT = 15.0


def build_searchable_text(fields: list[ExtractedField]) -> str:
    """Combine extracted fields into one searchable document string."""
    by_name = {f.field_name: f for f in fields}
    lines: list[str] = []
    for field_name, label in _SEARCH_FIELD_ORDER:
        row = by_name.get(field_name)
        if row is None:
            continue
        value = (row.field_value_en or row.field_value or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def generate_embedding(text: str) -> list[float] | None:
    """Generate one embedding vector for the combined searchable text."""
    if not text.strip():
        logger.info("vector_search: empty searchable text — skipping embedding.")
        return None
    if not settings.gemini_configured:
        logger.info("vector_search: Gemini not configured — skipping embedding.")
        return None

    logger.info("vector_search: requesting embedding with model=%s", settings.embedding_model)
    url = (
        f"{GEMINI_EMBED_BASE}/{settings.embedding_model}:embedContent"
        f"?key={settings.gemini_api_key}"
    )
    payload = {
        "model": f"models/{settings.embedding_model}",
        "content": {"parts": [{"text": text}]},
    }
    try:
        with httpx.Client(timeout=EMBED_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding") or {}
            values = None
            if isinstance(embedding, dict):
                values = embedding.get("values") or embedding.get("value")
            elif isinstance(embedding, list):
                values = embedding
            if not values:
                logger.warning("vector_search: Gemini embedding response had no values. response=%s", data)
                return None
            logger.info(
                "vector_search: embedding generated model=%s dimension=%d",
                settings.embedding_model,
                len(values),
            )
            return [float(v) for v in values]
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text if exc.response is not None else ""
        logger.warning(
            "vector_search: embedding unavailable (model=%s status=%s response=%s); continuing without an embedding.",
            settings.embedding_model,
            exc.response.status_code if exc.response is not None else None,
            response_text,
        )
        return None
    except Exception as exc:
        logger.warning(
            "vector_search: embedding unavailable (model=%s error=%s); continuing without an embedding.",
            settings.embedding_model,
            exc,
        )
        return None


def store_embedding(
    db: Session,
    document_id: str,
    searchable_text: str,
    embedding: list[float] | None,
) -> DocumentEmbedding | None:
    """Persist (or replace) the document's embedding row."""
    if not searchable_text.strip():
        return None

    row = db.get(DocumentEmbedding, document_id)
    if row is None:
        row = DocumentEmbedding(
            document_id=document_id,
            searchable_text=searchable_text,
            embedding=embedding,
        )
        db.add(row)
        logger.info(
            "vector_search: creating embedding row for document=%s dim=%s",
            document_id,
            len(embedding) if embedding is not None else None,
        )
    else:
        row.searchable_text = searchable_text
        row.embedding = embedding
        logger.info(
            "vector_search: updating embedding row for document=%s dim=%s",
            document_id,
            len(embedding) if embedding is not None else None,
        )
    db.flush()
    return row


def index_document_fields(db: Session, document_id: str, fields: list[ExtractedField]) -> DocumentEmbedding | None:
    """Phase 1 entry point: build text, embed, store. Never raises."""
    try:
        searchable_text = build_searchable_text(fields)
        if not searchable_text.strip():
            logger.info("vector_search: no field values to index for %s.", document_id)
            return None
        logger.info(
            "vector_search: built searchable_text for document=%s length=%d",
            document_id,
            len(searchable_text),
        )
        embedding = generate_embedding(searchable_text)
        if embedding is None:
            logger.warning("vector_search: no embedding generated for document=%s.", document_id)
        return store_embedding(db, document_id, searchable_text, embedding)
    except Exception:
        logger.exception("vector_search: failed to index document %s.", document_id)
        return None


# ---------------------------------------------------------------------------
# Phase 2: similarity retrieval with a Python cosine fallback.
# ---------------------------------------------------------------------------
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_to_pgvector_literal(embedding: list[float]) -> str:
    """Format a float list for Postgres pgvector literal cast."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


@dataclass
class SimilarDocument:
    document_id: str
    similarity: float
    searchable_text: str


def find_similar_documents(
    db: Session,
    query_embedding: list[float],
    *,
    exclude_document_id: str | None = None,
    top_k: int | None = None,
) -> list[SimilarDocument]:
    """Retrieve top-K similar documents by embedding distance."""
    limit = top_k or settings.vector_search_top_k
    if not query_embedding:
        return []

    if pgvector_available and settings.database_url.startswith("postgres"):
        return _find_similar_pgvector(db, query_embedding, exclude_document_id, limit)
    return _find_similar_python(db, query_embedding, exclude_document_id, limit)


def _find_similar_pgvector(
    db: Session,
    query_embedding: list[float],
    exclude_document_id: str | None,
    limit: int,
) -> list[SimilarDocument]:
    from sqlalchemy import text

    query_literal = _embedding_to_pgvector_literal(query_embedding)
    sql = """
        SELECT document_id, searchable_text,
               1 - (embedding::vector <=> CAST(:query AS vector)) AS similarity
        FROM document_embeddings
        WHERE embedding IS NOT NULL
    """
    params: dict = {"query": query_literal, "limit": limit}
    if exclude_document_id:
        sql += " AND document_id != :exclude_id"
        params["exclude_id"] = exclude_document_id
    sql += " ORDER BY embedding::vector <=> CAST(:query AS vector) LIMIT :limit"

    try:
        # A failed pgvector cast/operator must not leave the upload
        # transaction unusable before the Python fallback runs.
        with db.begin_nested():
            rows = db.execute(text(sql), params).mappings().all()
        return [
            SimilarDocument(
                document_id=row["document_id"],
                similarity=float(row["similarity"]),
                searchable_text=row["searchable_text"],
            )
            for row in rows
        ]
    except Exception:
        logger.exception("vector_search: pgvector query failed — falling back to Python.")
        return _find_similar_python(db, query_embedding, exclude_document_id, limit)


def _find_similar_python(
    db: Session,
    query_embedding: list[float],
    exclude_document_id: str | None,
    limit: int,
) -> list[SimilarDocument]:
    rows = db.query(DocumentEmbedding).filter(DocumentEmbedding.embedding.isnot(None)).all()
    scored: list[SimilarDocument] = []
    for row in rows:
        if exclude_document_id and row.document_id == exclude_document_id:
            continue
        if not row.embedding:
            continue
        sim = _cosine_similarity(query_embedding, list(row.embedding))
        scored.append(
            SimilarDocument(
                document_id=row.document_id,
                similarity=sim,
                searchable_text=row.searchable_text,
            )
        )
    scored.sort(key=lambda s: s.similarity, reverse=True)
    return scored[:limit]


def _dedupe_similarity_candidates(candidates: list[SimilarDocument]) -> list[SimilarDocument]:
    """Return the best similarity match for each document_id, preserving
    the highest-score order."""
    best_by_doc: dict[str, SimilarDocument] = {}
    for candidate in candidates:
        existing = best_by_doc.get(candidate.document_id)
        if existing is None or candidate.similarity > existing.similarity:
            best_by_doc[candidate.document_id] = candidate
    return sorted(best_by_doc.values(), key=lambda c: c.similarity, reverse=True)


def get_property_id(db: Session, document_id: str) -> str | None:
    """Return the extracted Property ID for the final verification step."""
    row = (
        db.query(ExtractedField)
        .filter(
            ExtractedField.document_id == document_id,
            ExtractedField.field_name == FieldName.property_id,
        )
        .first()
    )
    if row is None:
        return None
    return ((row.field_value or row.field_value_en) or "").strip() or None


def verify_property_match(
    db: Session,
    current_document_id: str,
    candidates: list[SimilarDocument],
) -> tuple[str, str | None]:
    """Make the final decision using Property ID, never vector similarity.

    Similarity search supplies only the candidate set. A document is an
    existing property only when its extracted Property ID exactly matches a
    candidate's ID after trimming whitespace and normalizing case.
    """
    current_property_id = get_property_id(db, current_document_id)
    if current_property_id is None:
        return "new_property", None

    normalized_current_id = current_property_id.casefold()
    for candidate in candidates:
        candidate_property_id = get_property_id(db, candidate.document_id)
        if candidate_property_id and candidate_property_id.casefold() == normalized_current_id:
            return "existing_property", candidate.document_id
    return "new_property", None


def run_similarity_search_pipeline(
    db: Session,
    document_id: str,
    fields: list[ExtractedField],
) -> dict:
    """Index one document, retrieve candidates, then verify Property ID.

    Returns a dict of internal diagnostics (logged by the router, not
    exposed on the API response). Vector similarity never determines the
    final property status. Never raises.
    """
    result: dict = {
        "indexed": False,
        "similar_documents": [],
        "property_match_status": None,
        "matched_document_id": None,
    }
    try:
        row = index_document_fields(db, document_id, fields)
        if row is None or not row.embedding:
            return result

        result["indexed"] = True
        candidates = find_similar_documents(
            db,
            list(row.embedding),
            exclude_document_id=document_id,
        )
        deduped_candidates = _dedupe_similarity_candidates(candidates)
        logger.info(
            "vector_search: found %d similarity candidates for document=%s, %d deduplicated by document_id.",
            len(candidates),
            document_id,
            len(deduped_candidates),
        )
        result["similar_documents"] = [
            {
                "document_id": c.document_id,
                "similarity": round(c.similarity, 4),
            }
            for c in deduped_candidates
        ]
        match_status, matched_document_id = verify_property_match(db, document_id, candidates)
        logger.info(
            "vector_search: property match status=%s document=%s matched=%s",
            match_status,
            document_id,
            matched_document_id,
        )
        result["property_match_status"] = match_status
        result["matched_document_id"] = matched_document_id
        return result
    except Exception:
        logger.exception("vector_search: pipeline failed for %s.", document_id)
        return result
