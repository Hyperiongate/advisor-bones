# =============================================================
# app.py — Advisor Bones | Core Flask Application
# Shiftwork Solutions LLC
# Created:      2026-05-11
# Last Updated: 2026-05-11
#
# PURPOSE:
#   Standalone Flask backend for a cloneable AI advisor.
#   No dependency on the AI Swarm Orchestrator or any other
#   external Shiftwork Solutions service.
#
#   The advisor handles queries by:
#     1. Searching the pgvector knowledge base (if populated)
#     2. Using Anthropic web_search tool use for current public info
#     3. Synthesizing both into a single consensus response via
#        Claude Sonnet as the orchestrator
#
#   All persona, brand, content, and feature settings are in
#   config.py — this file should not need to change per clone.
#
# ROUTES:
#   GET  /              — Serves advisor chat UI
#   POST /chat          — Advisor response + audio (consensus engine)
#   POST /opening       — Opening message + audio
#   POST /transcribe    — Audio blob -> text via ElevenLabs STT
#   POST /transcript    — Download PDF transcript
#   POST /api/tts       — TTS proxy for external pages
#   GET  /health        — Render health check
#
# ENVIRONMENT VARIABLES (set in Render — never hardcode here):
#   ANTHROPIC_API_KEY   — Claude API key
#   ELEVENLABS_API_KEY  — ElevenLabs API key
#   OPENAI_API_KEY      — OpenAI API key (for embeddings only)
#   DATABASE_URL        — PostgreSQL + pgvector connection string
#   FORMSPREE_ID        — Formspree form ID
#   KB_ENABLED          — Override config default (true/false)
#   WEB_SEARCH_ENABLED  — Override config default (true/false)
#
# DEPLOYMENT:
#   GitHub -> Render web service (advisor-bones)
#   Start command: gunicorn app:app
#
# CHANGE LOG:
#   2026-05-11 — Initial build for Advisor Bones v1.0
#                Full consensus engine: pgvector KB + Anthropic
#                web_search tool use + Sonnet orchestrator.
#                Thomas-compatible routes, PDF transcript,
#                ElevenLabs TTS/STT, Formspree transcript email.
#                All config-driven — no hardcoded persona content.
# =============================================================

import os
import re
import base64
import requests
import io
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_file, Response
from flask_cors import CORS
import anthropic
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdf_canvas

import config
from knowledge.search import search_knowledge_base, kb_is_available

app = Flask(__name__)
CORS(app)

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Feature flags — env vars override config defaults
KB_ENABLED         = os.environ.get("KB_ENABLED", str(config.KB_ENABLED_DEFAULT)).lower() == "true"
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", str(config.WEB_SEARCH_ENABLED_DEFAULT)).lower() == "true"

# Formspree
FORMSPREE_ENDPOINT = (
    f"https://formspree.io/f/{config.FORMSPREE_ID}"
    if config.FORMSPREE_ID else None
)

# In-memory session store (resets on dyno restart — acceptable for this use case)
conversation_histories = {}


# =============================================================
# TTS URL STRIPPING
#
# Removes URLs from text before sending to ElevenLabs so the
# advisor does not speak raw URLs aloud. The frontend renders
# URLs as clickable links independently via linkifyText().
#
# Inherited pattern from Thomas — 2026-04-21
# =============================================================

def strip_urls_for_tts(text):
    """Remove URLs gracefully, replacing intro-word + URL with
    'via the link in the chat' so spoken sentences stay natural."""
    text = re.sub(
        r'\s+(?:here|at|there)\s*:?\s*https?://[^\s,;)"\'<>]+',
        ' via the link in the chat',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'https?://[^\s,;)"\'<>]+', 'via the link in the chat', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text


# =============================================================
# CONSENSUS ENGINE
#
# For each /chat turn:
#   1. Query pgvector KB for semantically similar chunks
#   2. Run Claude Sonnet with web_search tool enabled
#   3. Claude decides autonomously whether to search the web
#   4. KB context is injected into the system prompt as
#      additional grounding — Claude synthesizes both
#
# The agentic tool loop runs up to MAX_TOOL_ROUNDS to handle
# cases where Claude issues multiple web search calls.
# =============================================================

MAX_TOOL_ROUNDS = 4  # Max web search rounds per response

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}


def build_system_prompt_with_context(kb_context: str | None) -> str:
    """
    Assemble the full system prompt for this turn.
    Injects KB context beneath the base prompt if available.
    """
    prompt = config.SYSTEM_PROMPT.strip()

    if kb_context:
        prompt += f"\n\n{kb_context}"

    return prompt


def run_consensus_chat(messages: list, system_prompt: str) -> tuple[str, list]:
    """
    Run the Claude Sonnet consensus engine with web_search tool use.

    Handles the agentic tool loop:
      - If Claude returns tool_use blocks, execute them and feed
        results back until Claude produces a final text response.
      - Returns (final_text, updated_messages).
      - Raises on unrecoverable errors.

    The messages list passed in is the conversation history.
    We build a separate working_messages list for the API calls
    within this turn, then return only the final assistant reply
    for appending to conversation_histories.
    """
    tools = [WEB_SEARCH_TOOL] if WEB_SEARCH_ENABLED else []

    # Working copy for the tool loop — starts with full history
    working_messages = list(messages)

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        kwargs = dict(
            model=config.ORCHESTRATOR_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=working_messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = anthropic_client.messages.create(**kwargs)

        # Collect text blocks and tool_use blocks
        text_blocks    = [b for b in response.content if b.type == "text"]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # If Claude gave us a text response (possibly alongside tool use), we're done
        if text_blocks and not tool_use_blocks:
            return text_blocks[0].text, working_messages

        # If Claude ONLY produced tool_use blocks, execute them
        if tool_use_blocks:
            if round_num >= MAX_TOOL_ROUNDS:
                # Safety: force a final answer without tools
                break

            # Append the assistant's tool_use turn to working_messages
            working_messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type":  "tool_use",
                        "id":    b.id,
                        "name":  b.name,
                        "input": b.input,
                    }
                    for b in tool_use_blocks
                ]
            })

            # Build tool_result blocks — web_search results come back
            # directly from the Anthropic API as part of the response
            # content when type == "tool_result". We need to pass them
            # back as a user turn.
            tool_results = []
            for b in tool_use_blocks:
                # Find matching tool_result in response.content
                result_content = ""
                for block in response.content:
                    if hasattr(block, "type") and block.type == "tool_result":
                        if getattr(block, "tool_use_id", None) == b.id:
                            result_content = getattr(block, "content", "")
                            break
                # If no embedded result (some API versions), use empty string
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": b.id,
                    "content":     result_content if result_content else "No results returned.",
                })

            working_messages.append({
                "role":    "user",
                "content": tool_results,
            })
            continue  # Next round

        # Edge case: response has neither text nor tool_use
        break

    # Fallback: ask Claude for a final answer without tools
    working_messages.append({
        "role":    "user",
        "content": "Please provide your final answer based on what you know.",
    })
    final_resp = anthropic_client.messages.create(
        model=config.ORCHESTRATOR_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system_prompt,
        messages=working_messages,
    )
    final_text_blocks = [b for b in final_resp.content if b.type == "text"]
    if final_text_blocks:
        return final_text_blocks[0].text, working_messages

    return "I wasn't able to generate a response. Please try again.", working_messages


# =============================================================
# ElevenLabs TTS
# =============================================================

def generate_speech(text: str) -> str | None:
    """Call ElevenLabs TTS, return base64 MP3. Returns None on failure."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        tts_text = strip_urls_for_tts(text)
        if not tts_text:
            return None
        headers = {
            "xi-api-key":   ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        }
        payload = {
            "text":     tts_text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability":        0.55,
                "similarity_boost": 0.80,
                "style":            0.20,
                "use_speaker_boost": True,
            },
        }
        resp = requests.post(ELEVENLABS_TTS_URL, headers=headers,
                             json=payload, timeout=15)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
        print(f"ElevenLabs TTS error {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"ElevenLabs TTS exception: {e}")
        return None


# =============================================================
# PDF TRANSCRIPT
# =============================================================

def generate_transcript_pdf(session_id, messages, lead_info=None):
    """Generate branded PDF transcript. Returns BytesIO buffer."""
    buffer = io.BytesIO()
    c      = pdf_canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    navy   = HexColor(config.BRAND_NAVY)
    gold   = HexColor(config.BRAND_ORANGE)
    gray   = HexColor("#6b7280")
    dark   = HexColor("#1f2937")
    margin = inch

    def check_page(y, needed=1.5):
        if y < needed * inch:
            c.showPage()
            return height - margin
        return y

    # Header
    c.setFillColor(navy)
    c.rect(0, height - 1.4 * inch, width, 1.4 * inch, fill=1, stroke=0)
    c.setFillColor(gold)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, height - 0.65 * inch, config.SIDEBAR_FIRM)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 11)
    c.drawRightString(width - margin, height - 0.55 * inch, "Conversation Transcript")
    c.drawRightString(width - margin, height - 0.85 * inch,
                      datetime.now().strftime("%B %d, %Y"))

    y = height - 1.9 * inch
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Conversation Transcript")
    y -= 0.1 * inch
    c.setStrokeColor(gold)
    c.setLineWidth(1.5)
    c.line(margin, y, width - margin, y)
    y -= 0.35 * inch

    text_indent = 0.25 * inch
    max_w       = width - 2 * margin - text_indent - 0.5 * inch

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue  # Skip tool_use/tool_result turns
        if content in ("__INIT__", "BOT_DETECTED"):
            continue
        speaker = config.PERSONA_NAME if role == "assistant" else "Visitor"
        c.setFillColor(navy if role == "assistant" else gray)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, speaker + ":")
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        c.setFillColor(dark)
        words = content.split()
        line  = ""
        for word in words:
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 10) < max_w:
                line = test
            else:
                c.drawString(margin + text_indent, y, line)
                y -= 0.18 * inch
                y  = check_page(y)
                line = word
        if line:
            c.drawString(margin + text_indent, y, line)
            y -= 0.18 * inch
        y -= 0.18 * inch
        y = check_page(y)

    if lead_info:
        y = check_page(y, needed=3)
        y -= 0.2 * inch
        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Contact Information Provided")
        y -= 0.1 * inch
        c.setStrokeColor(gold)
        c.setLineWidth(1.5)
        c.line(margin, y, width - margin, y)
        y -= 0.3 * inch
        c.setFont("Helvetica", 11)
        c.setFillColor(dark)
        for key, val in lead_info.items():
            if val:
                c.drawString(margin, y, f"{key}:  {val}")
                y -= 0.28 * inch

    # Footer
    c.setFillColor(navy)
    c.rect(0, 0, width, 0.65 * inch, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 9)
    c.drawString(margin, 0.38 * inch,
                 f"{config.SIDEBAR_FIRM}  |  {config.WEBSITE_LABEL}  |  {config.PHONE}")
    c.drawRightString(width - margin, 0.38 * inch, "Confidential")

    c.save()
    buffer.seek(0)
    return buffer


# =============================================================
# FORMSPREE TRANSCRIPT EMAIL
# =============================================================

def email_transcript_via_formspree(session_id, messages, lead_info=None):
    """Fire-and-forget transcript email. Never raises."""
    if not FORMSPREE_ENDPOINT:
        return
    try:
        lines = [
            f"=== {config.PERSONA_NAME.upper()} CONVERSATION TRANSCRIPT ===",
            f"Session: {session_id}",
            f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}",
            "",
        ]
        if lead_info:
            lines.append("--- Contact Information ---")
            for key, val in lead_info.items():
                if val:
                    lines.append(f"{key}: {val}")
            lines.append("")
        lines.append("--- Conversation ---")
        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if content in ("__INIT__", "BOT_DETECTED"):
                continue
            speaker = config.PERSONA_NAME if role == "assistant" else "Visitor"
            lines.append(f"\n{speaker}:")
            lines.append(content)
        lines.append(f"\n=== END TRANSCRIPT ===")

        payload = {
            "_subject": (f"{config.PERSONA_NAME} Transcript — "
                         f"{datetime.now().strftime('%B %d, %Y %I:%M %p')}"),
            "message":  "\n".join(lines),
            "_replyto": f"noreply@{config.WEBSITE_URL.replace('https://', '').replace('http://', '')}",
        }
        resp = requests.post(
            FORMSPREE_ENDPOINT,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"Transcript email sent for session {session_id}")
        else:
            print(f"Formspree transcript email failed {resp.status_code}")
    except requests.exceptions.Timeout:
        print("Formspree transcript email timed out — continuing")
    except Exception as e:
        print(f"Formspree transcript email error (non-fatal): {e}")


# =============================================================
# ROUTES
# =============================================================

@app.route("/health")
def health():
    return jsonify({
        "status":             "ok",
        "service":            "advisor-bones",
        "persona":            config.PERSONA_NAME,
        "tts_enabled":        bool(ELEVENLABS_API_KEY),
        "kb_enabled":         KB_ENABLED,
        "web_search_enabled": WEB_SEARCH_ENABLED,
        "kb_populated":       kb_is_available() if KB_ENABLED else False,
    }), 200


@app.route("/")
def index():
    return render_template_string(
        open("templates/index.html").read(),
        persona_name        = config.PERSONA_NAME,
        persona_initial     = config.PERSONA_INITIAL,
        brand_navy          = config.BRAND_NAVY,
        brand_orange        = config.BRAND_ORANGE,
        booking_link        = config.BOOKING_LINK,
        phone               = config.PHONE,
        phone_href          = config.PHONE_HREF,
        website_url         = config.WEBSITE_URL,
        website_label       = config.WEBSITE_LABEL,
        contact_url         = config.CONTACT_URL,
        sidebar_firm        = config.SIDEBAR_FIRM,
        sidebar_title       = config.SIDEBAR_TITLE,
        sidebar_about       = config.SIDEBAR_ABOUT,
        overlay_eyebrow     = config.OVERLAY_EYEBROW,
        overlay_title       = config.OVERLAY_TITLE,
        overlay_subtitle    = config.OVERLAY_SUBTITLE,
        overlay_cta_label   = config.OVERLAY_CTA_LABEL,
        overlay_chips       = config.OVERLAY_CHIPS,
        newsletter_endpoint = config.NEWSLETTER_ENDPOINT or "",
        newsletter_source   = config.NEWSLETTER_SOURCE,
    )


@app.route("/opening", methods=["POST"])
def opening():
    """
    Return the opening message and audio.
    Called when the visitor dismisses the instructional overlay.
    Accepts: { session_id (optional) }
    """
    data       = request.get_json() or {}
    session_id = data.get("session_id") or _new_session_id()

    conversation_histories[session_id] = [{
        "role":    "assistant",
        "content": config.OPENING_MESSAGE,
    }]

    audio_b64 = generate_speech(config.OPENING_MESSAGE)
    return jsonify({
        "reply":      config.OPENING_MESSAGE,
        "audio":      audio_b64,
        "session_id": session_id,
    }), 200


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receive audio blob, send to ElevenLabs STT, return text.
    Handles all browser audio formats (webm, ogg, mp4, wav).
    """
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "STT not configured"}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_data = audio_file.read()
    if not audio_data:
        return jsonify({"error": "Empty audio file"}), 400

    raw_mime  = audio_file.content_type or "audio/webm"
    base_mime = raw_mime.split(";")[0].strip().lower()
    mime_map  = {
        "audio/webm":  ("audio.webm", "audio/webm"),
        "audio/ogg":   ("audio.ogg",  "audio/ogg"),
        "audio/mp4":   ("audio.mp4",  "audio/mp4"),
        "audio/mpeg":  ("audio.mp3",  "audio/mpeg"),
        "audio/wav":   ("audio.wav",  "audio/wav"),
        "audio/x-wav": ("audio.wav",  "audio/wav"),
    }
    filename, content_type = mime_map.get(base_mime, ("audio.webm", "audio/webm"))
    print(f"STT: raw_mime={raw_mime} base_mime={base_mime} "
          f"filename={filename} size={len(audio_data)}")

    try:
        headers  = {"xi-api-key": ELEVENLABS_API_KEY}
        files    = {"file": (filename, audio_data, content_type)}
        data     = {"model_id": "scribe_v1", "language_code": "en"}
        response = requests.post(ELEVENLABS_STT_URL, headers=headers,
                                 files=files, data=data, timeout=20)
        if response.status_code == 200:
            text = response.json().get("text", "").strip()
            print(f"STT result: {repr(text)}")
            return jsonify({"text": text}), 200
        print(f"ElevenLabs STT error {response.status_code}: {response.text}")
        return jsonify({"error": f"STT failed: {response.status_code}"}), 500
    except Exception as e:
        print(f"ElevenLabs STT exception: {e}")
        return jsonify({"error": f"STT exception: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main conversation route — consensus engine.

    Flow per turn:
      1. Search pgvector KB for relevant context
      2. Build system prompt with KB context injected
      3. Run Claude Sonnet with web_search tool use enabled
      4. Return synthesized response + TTS audio

    Accepts: { message, session_id }
    Returns: { reply, audio, session_id } or { bot_detected: true }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    session_id   = data.get("session_id", "default")
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    conversation_histories[session_id].append({
        "role": "user", "content": user_message,
    })

    # Keep last N messages to manage context window
    if len(conversation_histories[session_id]) > config.MAX_HISTORY:
        conversation_histories[session_id] = \
            conversation_histories[session_id][-config.MAX_HISTORY:]

    # Step 1: KB search (graceful if unavailable or empty)
    kb_context = None
    if KB_ENABLED:
        try:
            kb_context = search_knowledge_base(user_message)
        except Exception as e:
            print(f"KB search error (non-fatal): {e}")

    # Step 2: Build system prompt with KB context
    system_prompt = build_system_prompt_with_context(kb_context)

    # Step 3: Run consensus engine
    try:
        # Pass only simple string-content messages to the API
        # (filter out any tool_use/tool_result turns in history)
        api_messages = [
            m for m in conversation_histories[session_id]
            if isinstance(m.get("content"), str)
        ]

        advisor_reply, _ = run_consensus_chat(api_messages, system_prompt)

        # Bot detection
        if advisor_reply.strip() == "BOT_DETECTED":
            conversation_histories.pop(session_id, None)
            return jsonify({"bot_detected": True}), 200

        conversation_histories[session_id].append({
            "role": "assistant", "content": advisor_reply,
        })

        audio_b64 = generate_speech(advisor_reply)
        return jsonify({
            "reply":      advisor_reply,
            "audio":      audio_b64,
            "session_id": session_id,
        }), 200

    except anthropic.APIError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/transcript", methods=["POST"])
def download_transcript():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    session_id = data.get("session_id", "default")
    lead_info  = data.get("lead_info", None)
    messages   = conversation_histories.get(session_id, [])
    if not messages:
        return jsonify({"error": "No conversation found for this session"}), 404
    try:
        email_transcript_via_formspree(session_id, messages, lead_info)
        pdf_buffer = generate_transcript_pdf(session_id, messages, lead_info)
        filename   = f"{config.PERSONA_NAME}-Transcript-{datetime.now().strftime('%Y-%m-%d')}.pdf"
        return send_file(pdf_buffer, mimetype="application/pdf",
                         as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Transcript PDF error: {e}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


@app.route("/api/tts", methods=["POST"])
def tts_proxy():
    """
    TTS proxy for external pages — keeps ElevenLabs API key server-side.
    Accepts JSON: { "text": "...", "voice_id": "..." (optional) }
    Returns: audio/mpeg stream
    Max text: 4500 characters per request.
    """
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "TTS not configured"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 4500:
        return jsonify({"error": "Text exceeds 4500 character limit"}), 400

    voice_id = data.get("voice_id", config.ELEVENLABS_VOICE_ID).strip() or config.ELEVENLABS_VOICE_ID
    tts_url  = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        headers = {
            "xi-api-key":   ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        }
        payload = {
            "text":     text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability":        0.5,
                "similarity_boost": 0.75,
                "style":            0.0,
                "use_speaker_boost": True,
            },
        }
        el_resp = requests.post(tts_url, headers=headers, json=payload, timeout=30)
        if el_resp.status_code == 200:
            return Response(
                el_resp.content,
                status=200,
                mimetype="audio/mpeg",
                headers={
                    "Cache-Control":  "no-store",
                    "Content-Length": str(len(el_resp.content)),
                }
            )
        print(f"ElevenLabs TTS proxy error {el_resp.status_code}")
        return jsonify({"error": f"ElevenLabs error {el_resp.status_code}"}), el_resp.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "TTS request timed out"}), 504
    except Exception as e:
        print(f"ElevenLabs TTS proxy exception: {e}")
        return jsonify({"error": f"TTS proxy error: {str(e)}"}), 500


@app.route("/booking-link")
def booking_link():
    return jsonify({"url": config.BOOKING_LINK}), 200


# =============================================================
# HELPERS
# =============================================================

def _new_session_id() -> str:
    """Generate a simple session ID."""
    import uuid
    return str(uuid.uuid4())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# I did no harm and this file is not truncated
