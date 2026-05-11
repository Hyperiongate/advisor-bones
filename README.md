# Advisor Bones

A reusable, cloneable AI advisor skeleton built by Shiftwork Solutions LLC.

Deploy it as-is, or clone it and customize `config.py` to create a domain-specific AI advisor with a distinct persona, knowledge base, and brand identity — in minutes, not days.

---

## What It Does

Advisor Bones is a Flask-based AI advisor that answers questions by drawing on two sources simultaneously:

1. **A curated knowledge base** — documents you embed into a PostgreSQL/pgvector database using the included ingestion script
2. **Current public information** — live web search via Anthropic's built-in tool use

Claude Sonnet acts as the orchestrator, deciding when to search the web, synthesizing both sources into a single coherent response, and delivering it via voice (ElevenLabs TTS) and text.

---

## AI Stack

| Service | Role | Why |
|---|---|---|
| **Claude Sonnet** (`claude-sonnet-4-20250514`) | Orchestrator + synthesizer | Fast enough for conversation, capable enough for multi-source synthesis and autonomous tool use |
| **Anthropic web_search tool** | Live public information | Built into the Anthropic API — no extra service, no extra key |
| **OpenAI text-embedding-3-small** | Document embedding for pgvector KB | 1536-dim vectors, fast, cost-effective. Anthropic does not currently offer an embeddings endpoint |
| **ElevenLabs Scribe v1** | Speech-to-text (voice input) | Same STT service used by Thomas |
| **ElevenLabs TTS** | Text-to-speech (voice output) | Same TTS service used by Thomas |

---

## How to Clone for a New Use Case

1. Fork or copy this repo to a new GitHub repository
2. Open `config.py` — **this is the only file you need to edit** for a standard clone
3. Change: persona name, voice ID, brand colors, system prompt, opening message, contact info, overlay copy, and symptom chips
4. Populate the knowledge base with domain-specific documents (see below)
5. Deploy to Render

---

## Repo Structure

```
advisor-bones/
├── app.py                  # Flask application — routes, consensus engine, TTS/STT, PDF
├── config.py               # ← EDIT THIS to customize per clone
├── requirements.txt
├── render.yaml             # Render deployment config
├── knowledge/
│   ├── search.py           # pgvector semantic search (called by app.py on every chat turn)
│   └── ingest.py           # CLI script to embed and load documents into the KB
├── templates/
│   └── index.html          # Chat UI — Thomas-style, config-driven
└── static/
    └── advisor.jpg         # Sidebar avatar image — replace per clone
```

---

## Environment Variables

Set these in Render. Never commit them to GitHub.

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs TTS + STT |
| `OPENAI_API_KEY` | Yes (if KB enabled) | Embeddings only |
| `DATABASE_URL` | Yes (if KB enabled) | Render PostgreSQL internal URL |
| `FORMSPREE_ID` | Optional | Formspree form ID for transcript email |
| `KB_ENABLED` | Optional | `true` / `false` — defaults to `true` |
| `WEB_SEARCH_ENABLED` | Optional | `true` / `false` — defaults to `true` |

---

## First-Time Deployment

### 1. GitHub
Push all files to a new public (or private) GitHub repository.

### 2. Render — Database
- Create a new **PostgreSQL** database in the Render dashboard
- Copy its **Internal Database URL**
- pgvector will be initialized automatically on first use (the ingestion script runs `CREATE EXTENSION IF NOT EXISTS vector`)

### 3. Render — Web Service
- Create a new **Web Service** pointing to your GitHub repo
- Runtime: **Python**
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Add all environment variables listed above
- Health check path: `/health`

### 4. Static image
Drop an image at `static/advisor.jpg` — this is the sidebar avatar. Any square-ish image works. Replace per clone.

---

## Populating the Knowledge Base

The knowledge base starts empty. The advisor works immediately via web search only. Add documents when you have domain-specific content to ground it.

```bash
# Install dependencies locally first
pip install -r requirements.txt
pip install pdfminer.six python-docx   # if ingesting PDFs or Word docs

# Set env vars locally
export DATABASE_URL="your-render-postgresql-url"
export OPENAI_API_KEY="your-openai-key"

# Ingest a directory of documents
python knowledge/ingest.py --dir path/to/your/docs/

# Check what's in the KB
python knowledge/ingest.py --stats

# Nuke and re-ingest (use with care)
python knowledge/ingest.py --dir path/to/your/docs/ --clear
```

**Supported file types:** `.txt`, `.md`, `.pdf` (requires `pdfminer.six`), `.docx` (requires `python-docx`)

---

## Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/opening` | Opening message + audio |
| `POST` | `/chat` | Consensus response + audio |
| `POST` | `/transcribe` | Audio → text (ElevenLabs STT) |
| `POST` | `/transcript` | Download PDF transcript |
| `POST` | `/api/tts` | TTS proxy for external pages |
| `GET` | `/health` | Health check (KB status, feature flags) |

---

## Health Check

```
GET /health
```

Returns:
```json
{
  "status": "ok",
  "service": "advisor-bones",
  "persona": "Advisor",
  "tts_enabled": true,
  "kb_enabled": true,
  "web_search_enabled": true,
  "kb_populated": false
}
```

`kb_populated: false` is normal on a fresh deployment before ingestion.

---

## Security Notes

- API keys are never exposed to the browser — TTS is proxied through `/api/tts`
- Session history is in-memory only — resets on dyno restart
- No user data is persisted unless a visitor downloads a transcript
- Transcripts are emailed via Formspree (fire-and-forget, never blocks the visitor)

---

## Built By

[Shiftwork Solutions LLC](https://shift-work.com) — Management consulting for 24/7 shift operations.
(415) 265-1621
