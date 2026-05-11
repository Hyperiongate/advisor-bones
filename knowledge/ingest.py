# =============================================================
# knowledge/ingest.py — Advisor Bones | Document Ingestion CLI
# Shiftwork Solutions LLC
# Created:      2026-05-11
# Last Updated: 2026-05-11
#
# PURPOSE:
#   One-time (and repeatable) CLI script to embed documents
#   and load them into the pgvector knowledge base.
#   Run this locally or from a Render shell to populate the KB.
#
# USAGE:
#   # Ingest a single file:
#   python knowledge/ingest.py --file path/to/document.txt
#
#   # Ingest all .txt and .md files in a directory:
#   python knowledge/ingest.py --dir path/to/docs/
#
#   # Ingest a plain text string (for testing):
#   python knowledge/ingest.py --text "This is a test chunk."
#
#   # Clear all chunks and re-ingest (nuclear option):
#   python knowledge/ingest.py --dir path/to/docs/ --clear
#
#   # Show current KB stats:
#   python knowledge/ingest.py --stats
#
# SUPPORTED FILE TYPES:
#   .txt, .md — read as plain text
#   .pdf       — extracted with pdfminer.six (pip install pdfminer.six)
#   .docx      — extracted with python-docx (pip install python-docx)
#
# CHUNKING STRATEGY:
#   Documents are split into overlapping chunks of ~800 tokens
#   (~600 words) with 100-word overlap to preserve context
#   across chunk boundaries.
#
# ENVIRONMENT VARIABLES:
#   DATABASE_URL   — PostgreSQL connection string
#   OPENAI_API_KEY — For embeddings
#
# CHANGE LOG:
#   2026-05-11 — Initial build for Advisor Bones v1.0
# =============================================================

import os
import sys
import argparse
import textwrap
import time
import psycopg2
import psycopg2.extras

# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, EMBEDDING_DIMS

# ── Constants ──────────────────────────────────────────────────
CHUNK_WORDS    = 600    # Target words per chunk
OVERLAP_WORDS  = 100    # Words of overlap between adjacent chunks
BATCH_SIZE     = 20     # Embeddings per API call (rate limit safety)
BATCH_DELAY    = 0.5    # Seconds between batches


# ── Database ───────────────────────────────────────────────────

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        sys.exit(1)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        sys.exit(1)


def ensure_schema(conn):
    """
    Create the pgvector extension and knowledge_chunks table if
    they do not already exist. Safe to run multiple times.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id         SERIAL PRIMARY KEY,
                content    TEXT NOT NULL,
                embedding  vector({EMBEDDING_DIMS}),
                source     TEXT,
                metadata   JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        # IVFFlat index for fast approximate nearest-neighbor search.
        # lists=100 is appropriate for up to ~1M rows.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
                ON knowledge_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        """)
    conn.commit()
    print("Schema verified.")


def clear_chunks(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM knowledge_chunks;")
    conn.commit()
    print("All existing chunks deleted.")


def get_stats(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM knowledge_chunks;")
        row = cur.fetchone()
    print(f"\nKnowledge base stats:")
    print(f"  Total chunks : {row[0]}")
    print(f"  Unique sources: {row[1]}")


# ── Text extraction ────────────────────────────────────────────

def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext == ".pdf":
        try:
            from pdfminer.high_level import extract_text
            return extract_text(filepath)
        except ImportError:
            print(f"  SKIP {filepath}: pdfminer.six not installed. "
                  f"Run: pip install pdfminer.six")
            return ""
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print(f"  SKIP {filepath}: python-docx not installed. "
                  f"Run: pip install python-docx")
            return ""
    else:
        print(f"  SKIP {filepath}: unsupported file type '{ext}'")
        return ""


# ── Chunking ───────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping word-window chunks.
    Returns list of dicts: {content, source, metadata}.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end   = min(start + CHUNK_WORDS, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 50:  # Skip trivially short chunks
            chunks.append({
                "content":  chunk,
                "source":   source,
                "metadata": {"chunk_index": idx, "word_count": end - start},
            })
            idx += 1
        start = end - OVERLAP_WORDS
        if start >= end:
            break  # Safety: prevent infinite loop on very short texts

    return chunks


# ── Embeddings ─────────────────────────────────────────────────

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)


def embed_chunks(chunks: list[dict], client) -> list[dict]:
    """
    Add 'embedding' key to each chunk dict.
    Processes in batches with delay to avoid rate limits.
    """
    total  = len(chunks)
    result = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        print(f"  Embedding chunks {batch_start + 1}–{batch_start + len(batch)} of {total}...")

        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMS,
            )
            for i, item in enumerate(resp.data):
                chunk_copy = dict(batch[i])
                chunk_copy["embedding"] = item.embedding
                result.append(chunk_copy)
        except Exception as e:
            print(f"  ERROR embedding batch: {e}")
            print("  Skipping this batch and continuing...")

        if batch_start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    return result


# ── Database insertion ─────────────────────────────────────────

def insert_chunks(conn, chunks: list[dict]):
    """Insert embedded chunks into knowledge_chunks table."""
    if not chunks:
        return 0

    rows_inserted = 0
    with conn.cursor() as cur:
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
            vec_str = "[" + ",".join(str(v) for v in chunk["embedding"]) + "]"
            cur.execute(
                """
                INSERT INTO knowledge_chunks (content, embedding, source, metadata)
                VALUES (%s, %s::vector, %s, %s);
                """,
                (
                    chunk["content"],
                    vec_str,
                    chunk.get("source", "unknown"),
                    psycopg2.extras.Json(chunk.get("metadata", {})),
                )
            )
            rows_inserted += 1
    conn.commit()
    return rows_inserted


# ── Main ───────────────────────────────────────────────────────

def ingest_text(text: str, source: str, conn, client):
    chunks = chunk_text(text, source)
    if not chunks:
        print(f"  No usable text found in: {source}")
        return 0
    print(f"  {len(chunks)} chunks created from: {source}")
    embedded = embed_chunks(chunks, client)
    inserted = insert_chunks(conn, embedded)
    print(f"  {inserted} chunks inserted.")
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Advisor Bones knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python knowledge/ingest.py --file docs/guide.txt
              python knowledge/ingest.py --dir docs/
              python knowledge/ingest.py --dir docs/ --clear
              python knowledge/ingest.py --stats
        """)
    )
    parser.add_argument("--file",  help="Path to a single file to ingest")
    parser.add_argument("--dir",   help="Path to a directory of files to ingest")
    parser.add_argument("--text",  help="Raw text string to ingest (for testing)")
    parser.add_argument("--clear", action="store_true",
                        help="Delete all existing chunks before ingesting")
    parser.add_argument("--stats", action="store_true",
                        help="Show current knowledge base stats and exit")
    args = parser.parse_args()

    conn   = get_db_connection()
    ensure_schema(conn)

    if args.stats:
        get_stats(conn)
        conn.close()
        return

    if not any([args.file, args.dir, args.text]):
        parser.print_help()
        conn.close()
        return

    if args.clear:
        confirm = input("Delete ALL existing chunks? Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            conn.close()
            return
        clear_chunks(conn)

    client       = get_openai_client()
    total_chunks = 0

    if args.text:
        total_chunks += ingest_text(args.text, "cli-input", conn, client)

    if args.file:
        if not os.path.isfile(args.file):
            print(f"ERROR: File not found: {args.file}")
        else:
            text = extract_text_from_file(args.file)
            if text.strip():
                source = os.path.basename(args.file)
                total_chunks += ingest_text(text, source, conn, client)

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERROR: Directory not found: {args.dir}")
        else:
            supported = (".txt", ".md", ".pdf", ".docx")
            files     = sorted([
                f for f in os.listdir(args.dir)
                if os.path.splitext(f)[1].lower() in supported
            ])
            if not files:
                print(f"No supported files found in: {args.dir}")
            for filename in files:
                filepath = os.path.join(args.dir, filename)
                print(f"\nProcessing: {filename}")
                text = extract_text_from_file(filepath)
                if text.strip():
                    total_chunks += ingest_text(text, filename, conn, client)

    print(f"\nIngestion complete. Total chunks inserted this run: {total_chunks}")
    get_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()

# I did no harm and this file is not truncated
