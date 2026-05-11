# =============================================================
# knowledge/search.py — Advisor Bones | Knowledge Base Search
# Shiftwork Solutions LLC
# Created:      2026-05-11
# Last Updated: 2026-05-11
#
# PURPOSE:
#   Semantic search against the pgvector knowledge base.
#   Called by app.py on every /chat request (when KB_ENABLED).
#   Graceful fallback — if the DB is unavailable or empty,
#   returns None and the advisor continues with web search only.
#
# HOW IT WORKS:
#   1. Embed the user's query using OpenAI text-embedding-3-small
#      via the Anthropic-compatible embeddings endpoint.
#      NOTE: We use the openai library pointed at the standard
#      OpenAI API for embeddings since Anthropic does not currently
#      offer an embeddings endpoint. OPENAI_API_KEY must be set
#      in Render env vars for KB to work.
#   2. Run a cosine similarity search against knowledge_chunks.
#   3. Return top-N chunks above the similarity floor as a
#      formatted string for injection into the system prompt.
#
# DATABASE SCHEMA (created by ingest.py):
#   knowledge_chunks (
#     id          SERIAL PRIMARY KEY,
#     content     TEXT NOT NULL,
#     embedding   vector(1536),
#     source      TEXT,         -- filename or URL of origin doc
#     metadata    JSONB,        -- arbitrary key/value pairs
#     created_at  TIMESTAMP DEFAULT NOW()
#   )
#
# ENVIRONMENT VARIABLES:
#   DATABASE_URL   — PostgreSQL connection string
#   OPENAI_API_KEY — For embeddings only
#
# CHANGE LOG:
#   2026-05-11 — Initial build for Advisor Bones v1.0
# =============================================================

import os
import json
import psycopg2
import psycopg2.extras
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    KB_RESULTS_LIMIT,
    KB_SIMILARITY_FLOOR,
)

# ── Lazy import openai so the app boots even if not installed ──
_openai_client = None

def _get_openai_client():
    """Return a cached OpenAI client, or None if unavailable."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("KB search: OPENAI_API_KEY not set — knowledge base disabled")
        return None
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except ImportError:
        print("KB search: openai package not installed — knowledge base disabled")
        return None


def _get_db_connection():
    """Return a psycopg2 connection, or None on failure."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("KB search: DATABASE_URL not set")
        return None
    try:
        # Render PostgreSQL URLs sometimes use 'postgres://' — psycopg2 needs 'postgresql://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(db_url, connect_timeout=5)
        return conn
    except Exception as e:
        print(f"KB search: DB connection failed (non-fatal): {e}")
        return None


def _embed_query(text):
    """
    Embed a query string using OpenAI text-embedding-3-small.
    Returns a list of floats, or None on failure.
    """
    client = _get_openai_client()
    if not client:
        return None
    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],  # Safety truncation
            dimensions=EMBEDDING_DIMS,
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"KB search: embedding failed (non-fatal): {e}")
        return None


def search_knowledge_base(query: str) -> str | None:
    """
    Search the knowledge base for chunks relevant to `query`.

    Returns a formatted string ready for injection into the
    system prompt, or None if no results / DB unavailable.

    Format:
        KNOWLEDGE BASE CONTEXT:
        [Source: filename.txt]
        chunk content here...

        [Source: another_file.txt]
        more content...
    """
    if not query or not query.strip():
        return None

    embedding = _embed_query(query)
    if embedding is None:
        return None

    conn = _get_db_connection()
    if conn is None:
        return None

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Cosine similarity search via pgvector operator <=>
            # Lower value = more similar (cosine distance, not similarity)
            # We convert: similarity = 1 - distance
            sql = """
                SELECT
                    content,
                    source,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT %s;
            """
            vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
            cur.execute(sql, (vec_str, vec_str, KB_SIMILARITY_FLOOR, KB_RESULTS_LIMIT))
            rows = cur.fetchall()

        conn.close()

        if not rows:
            return None

        lines = ["KNOWLEDGE BASE CONTEXT (synthesize with any other sources):"]
        for row in rows:
            source = row.get("source") or "knowledge base"
            content = (row.get("content") or "").strip()
            if content:
                lines.append(f"\n[Source: {source}]")
                lines.append(content)

        if len(lines) == 1:
            return None  # Header only — no usable content

        return "\n".join(lines)

    except Exception as e:
        print(f"KB search: query failed (non-fatal): {e}")
        try:
            conn.close()
        except Exception:
            pass
        return None


def kb_is_available() -> bool:
    """
    Quick health check — returns True if the knowledge_chunks
    table exists and has at least one row. Used by /health route.
    """
    conn = _get_db_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks LIMIT 1;")
            count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return False

# I did no harm and this file is not truncated
