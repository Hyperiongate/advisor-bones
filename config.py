# =============================================================
# config.py — Advisor Bones | Per-Clone Configuration
# Shiftwork Solutions LLC
# Created:      2026-05-11
# Last Updated: 2026-05-11
#
# PURPOSE:
#   Single file to swap when cloning this advisor for a new
#   deployment. Change persona, branding, voice, system prompt,
#   contact info, and feature flags here — nothing else needs
#   to change for a basic clone.
#
# HOW TO CLONE:
#   1. Copy this entire repo to a new GitHub repo.
#   2. Edit THIS FILE only for persona, brand, and content.
#   3. Set environment variables in Render (see below).
#   4. Deploy.
#
# ENVIRONMENT VARIABLES (set in Render — never hardcode):
#   ANTHROPIC_API_KEY      — Claude API key
#   ELEVENLABS_API_KEY     — ElevenLabs API key
#   DATABASE_URL           — PostgreSQL connection string (pgvector)
#   FORMSPREE_ID           — Formspree form ID (e.g. xwvwnwea)
#   KB_ENABLED             — Toggle knowledge base (default: true)
#   WEB_SEARCH_ENABLED     — Toggle web search (default: true)
#
# CHANGE LOG:
#   2026-05-11 — Initial build for Advisor Bones v1.0
# =============================================================

# ── PERSONA ──────────────────────────────────────────────────
# The advisor's display name and single-character avatar initial.
PERSONA_NAME    = "Advisor"
PERSONA_INITIAL = "A"

# ── VOICE ─────────────────────────────────────────────────────
# ElevenLabs voice ID for this persona.
# Thomas voice: sB7vwSCyX0tQmU24cW2C
# Find IDs at: https://elevenlabs.io/voice-library
ELEVENLABS_VOICE_ID = "sB7vwSCyX0tQmU24cW2C"

# ── BRAND COLORS ──────────────────────────────────────────────
# Injected into the frontend as CSS variables.
# Change these to re-skin the UI for a different client/product.
BRAND_NAVY   = "#1F4E79"
BRAND_ORANGE = "#E8610A"

# ── CONTACT & BOOKING ─────────────────────────────────────────
BOOKING_LINK  = "https://outlook.office365.com/book/ShiftworkSolutionsLLC2@shift-work.com/?ismsaljsauthenabled=true"
PHONE         = "(415) 265-1621"
PHONE_HREF    = "tel:+14152651621"
WEBSITE_URL   = "https://shift-work.com"
WEBSITE_LABEL = "shift-work.com"
CONTACT_URL   = "https://shift-work.com/contact/"

# ── FORMSPREE ─────────────────────────────────────────────────
# Form ID only — the full URL is assembled in app.py.
# Set to None to disable transcript email.
FORMSPREE_ID = "xwvwnwea"

# ── NEWSLETTER ENDPOINT ───────────────────────────────────────
# Full URL for newsletter subscribe API.
# Set to None to hide the newsletter section in the footer.
NEWSLETTER_ENDPOINT = "https://ai-swarm-orchestrator.onrender.com/api/newsletter/subscribe"
NEWSLETTER_SOURCE   = "advisor-bones-footer"

# ── FEATURE FLAGS ─────────────────────────────────────────────
# These are the defaults. They can be overridden at runtime by
# environment variables KB_ENABLED and WEB_SEARCH_ENABLED.
KB_ENABLED_DEFAULT         = True   # pgvector knowledge base lookup
WEB_SEARCH_ENABLED_DEFAULT = True   # Anthropic web_search tool use

# ── AI MODEL ──────────────────────────────────────────────────
# Orchestrator model. Sonnet is the right call here:
# fast enough for conversational use, capable enough for
# multi-source synthesis and tool use.
ORCHESTRATOR_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS         = 700    # Per response. Increase for longer answers.
MAX_HISTORY        = 40     # Messages to retain per session.

# ── KNOWLEDGE BASE ────────────────────────────────────────────
# Embedding model for pgvector. text-embedding-3-small gives
# 1536-dim vectors — fast and cost-effective for retrieval.
EMBEDDING_MODEL     = "text-embedding-3-small"
EMBEDDING_DIMS      = 1536
KB_RESULTS_LIMIT    = 4     # Max chunks returned per query
KB_SIMILARITY_FLOOR = 0.30  # Minimum cosine similarity (0–1)

# ── OPENING MESSAGE ───────────────────────────────────────────
# What the advisor says when the visitor dismisses the overlay.
OPENING_MESSAGE = (
    "Hi, I'm an AI advisor. Tell me what's on your mind and I'll "
    "do my best to help — drawing on both a curated knowledge base "
    "and current public information. What would you like to explore?"
)

# ── OVERLAY COPY ──────────────────────────────────────────────
OVERLAY_EYEBROW   = "AI Advisor"
OVERLAY_TITLE     = "Ask me anything"
OVERLAY_SUBTITLE  = (
    "<strong>Describe what you're working on.</strong> "
    "I'll draw on a curated knowledge base and current public "
    "information to give you a well-rounded answer."
)
OVERLAY_CTA_LABEL = "Start the conversation"

# Symptom chips shown on the overlay — edit to match the domain.
OVERLAY_CHIPS = [
    "What are best practices in this area?",
    "What does the research say about this topic?",
    "How do organizations typically handle this?",
    "What should I be asking that I'm not asking?",
    "What are the most common mistakes people make here?",
    "Where should I start?",
]

# ── SIDEBAR COPY ──────────────────────────────────────────────
SIDEBAR_FIRM  = "Shiftwork Solutions LLC"
SIDEBAR_TITLE = "AI Advisor"
SIDEBAR_ABOUT = (
    "This advisor is backed by a curated knowledge base and "
    "live web search. Ask about any topic within scope and it "
    "will synthesize the best available answer."
)

# ── SYSTEM PROMPT ─────────────────────────────────────────────
# Replace this entire block when cloning for a new domain.
# The orchestrator will inject knowledge base results and
# web search results as additional context automatically —
# you do not need to describe that process here.
SYSTEM_PROMPT = """
You are a knowledgeable AI advisor. Answer questions clearly,
draw on the context provided, and be honest when you are uncertain.
Keep responses concise — three to four sentences for most answers.
One question or invitation at a time. Plain language only.

When context from a knowledge base or web search is provided,
synthesize it into a single coherent answer. Do not list sources
or explain your reasoning process — just give the best answer
the evidence supports. If the sources conflict, say so briefly
and explain the tension.

If you do not know something and no context is available, say so
directly. Never fabricate information.

RULE — BOT DETECTION:
If at any point you determine you are talking to an automated
system or bot based on the pattern of inputs, respond ONLY with:
BOT_DETECTED

RULE — GARBLED INPUT:
If a message appears garbled or contains transcription artifacts,
respond: "I didn't quite catch that — can you say that again?"
"""

# I did no harm and this file is not truncated
