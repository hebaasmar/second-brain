"""
Coaching layer: question detection, beat navigation, glanceable display.
"""

import re
import os
from typing import Optional

from dotenv import load_dotenv
import anthropic

load_dotenv()

_anthropic_client = None
_response_cache: dict = {}  # (question, beat_text_key) → generated response

# ── Persona & methodology ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a real-time interview coach for Heba Asmar. Your job is to generate exactly what she should say out loud, right now, based on her actual experience and the question just asked.

Who Heba is:
Principal PM, 12 years in product, 5 in AI/ML. Two-time founder. Most recently Founder in Residence at Temasek's venture studio where she built Kunik from scratch — a two-sided marketplace connecting AI developers with licensed financial data. Took RAG accuracy from 78% to 96% by building an evaluation framework that diagnosed retrieval as the core failure, not the model. Before that, Director of PM at Warner Chappell managing a $2B royalty platform across 40 territories — ML matching from 85% to 95%, writer portal adoption from 30% to 75%. Co-founded Imbr, cut songwriter payments from 730 days to 30 days. No CS degree. Self-taught. Builds things herself.

Question type methodology:
BEHAVIORAL ("tell me about a time..."): One sentence of context. Then the tension — what made it hard, who disagreed, what was uncertain. Then her specific reasoning and action. Then the result with a real number. The decision and reasoning are the interesting part, not the setup. 3–4 sentences.
DISAGREEMENT/PUSHBACK: Who she disagreed with, what they wanted, why she thought they were wrong, what data or reasoning she used to make the case, what happened. Don't soften it. She was right and proved it.
PRODUCT SENSE: Clarify the goal in one sentence. Show you understand the technical constraint, not just the user need. Give a concrete recommendation with one real tradeoff. No framework names. Just sharp thinking out loud.
0-TO-1/AMBIGUITY: How she figured out what to build when nothing existed. User interviews, what she learned, what she killed, what she shipped. Specific.
TECHNICAL/AI: One level deeper than most PMs. Not "I worked on RAG" but the specific decision and why. Use real numbers and stack details from the notes.
ESTIMATION: One key assumption stated out loud. Rough math. Sanity check.

Voice rules:
Sound like a senior PM talking to a peer, not presenting to a board. Use her actual numbers — if the notes say 78% to 96%, say that. If they say she pushed back on engineering, say that. Contractions always. No corporate language. No "I'm passionate about." No "throughout my career." Never generalize what the notes make specific. If the notes are thin, say what she'd logically say given her background — don't invent facts.

For research lab interviews specifically:
Anthropic and similar labs want to see: technical depth (do you understand what's actually hard), judgment under ambiguity (how you decide with incomplete information), and that you build things yourself. Weight answers toward showing reasoning and technical tradeoffs, not just outcomes.

Output format:
3–4 sentences. What she says out loud right now. First person. Direct. No preamble. No labels. No bullet points. Just the answer."""


# ── Anthropic client ──────────────────────────────────────────────────────────
def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── Text helpers ──────────────────────────────────────────────────────────────
def _clean_beat(beat_text: str) -> str:
    """Strip header, prefix, and probe lines; return the core narrative."""
    clean = []
    for line in beat_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.lower().startswith("probe:"):
            continue
        if "|" in line and len(line.split("|")) >= 2:
            continue
        clean.append(line)
    return " ".join(clean)


def _extract_fallback(beat_text: str) -> str:
    """Fallback: first substantive sentence from the beat."""
    for line in beat_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("probe:"):
            continue
        if "|" in line and len(line.split("|")) >= 2:
            continue
        if len(line) > 20:
            end = re.search(r'[.!?]', line)
            return line[:end.end()].strip() if end else line
    return ""


def _beat_label(beat_str: str) -> str:
    """Strip 'Beat N: ' prefix, return just the descriptive label."""
    return re.sub(r'^Beat \d+:\s*', '', beat_str).strip()


def _fit_reason(question: str, chunk: dict) -> str:
    """Return 1–2 tags that explain why this story matches the question."""
    tags = chunk.get("tags", [])
    q_lower = question.lower()
    matched = [t for t in tags if t.lower() in q_lower]
    if matched:
        return " · ".join(matched[:2])
    if tags:
        return " · ".join(tags[:2])
    return "semantic match"


# ── LLM generation ────────────────────────────────────────────────────────────
def generate_response(question: str, chunk: dict, all_beats: list, beat_index: int) -> str:
    """
    Call Claude with the full coaching persona to generate exactly what
    Heba should say out loud right now. Cached per (question, beat).
    Falls back to extracted text on API error.
    """
    idx = min(beat_index, len(all_beats) - 1)
    current_beat = all_beats[idx]
    beat_text = current_beat.get("text", "")

    cache_key = (question[:80], beat_text[:120])
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    # Build full story context: current beat + remaining beats as notes
    story_context = _clean_beat(beat_text)
    remaining = [_clean_beat(b.get("text", "")) for b in all_beats[idx + 1:]]
    if remaining:
        story_context += "\n\nRemainder of story (later beats):\n" + "\n".join(
            f"- {_beat_label(all_beats[idx + 1 + i].get('beat', ''))}: {t[:120]}"
            for i, t in enumerate(remaining)
        )

    user_message = (
        f"Interview question: \"{question}\"\n\n"
        f"Story: {chunk.get('company', '')} — {chunk.get('story', '')}\n"
        f"Current beat ({beat_index + 1}/{len(all_beats)}): "
        f"{_beat_label(current_beat.get('beat', ''))}\n\n"
        f"Notes:\n{story_context}"
    )

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        result = message.content[0].text.strip()
        _response_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  (LLM error: {e} — using extracted text)")
        return _extract_fallback(beat_text)


# ── Question detection ────────────────────────────────────────────────────────
QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "who", "which",
    "can you", "could you", "tell me", "describe", "explain",
    "walk me through", "have you", "did you", "do you",
    "would you", "was there", "talk me through", "give me",
)


def is_question(text: str) -> bool:
    """Return True if the transcription looks like a real interview question."""
    text = text.strip()
    if len(text) < 10:
        return False
    if text.endswith("?"):
        return True
    lower = text.lower()
    if any(lower.startswith(s) for s in QUESTION_STARTERS):
        return True
    return False


def is_followup(text: str, current_story: Optional[dict]) -> bool:
    """
    Return True if this question continues the current story thread
    rather than opening a new topic.
    """
    if not current_story:
        return False

    text_lower = text.lower()

    # Very short clarifying questions are almost always follow-ups
    if len(text.split()) <= 7:
        return True

    # Tag overlap
    for tag in current_story.get("tags", []):
        if tag.lower() in text_lower:
            return True

    # Company or story name overlap
    if current_story.get("company", "").lower() in text_lower:
        return True
    if current_story.get("story", "").lower() in text_lower:
        return True

    return False


# ── Display ───────────────────────────────────────────────────────────────────
def format_display(question: str, chunk: dict, all_beats: list, beat_index: int) -> str:
    """
    Render the glanceable output:
      1. Cue line  — story + why it fits + beat position
      2. Response  — exactly what to say out loud (3–4 sentences)
      3. Beat nav  — what's next
    """
    company = chunk.get("company", "")
    story = chunk.get("story", "")
    total = len(all_beats)
    idx = min(beat_index, total - 1)
    current_beat = all_beats[idx]

    fit = _fit_reason(question, chunk)
    beat_title = _beat_label(current_beat.get("beat", ""))
    beat_num = idx + 1

    response = generate_response(question, chunk, all_beats, beat_index)

    sep = "─" * 52

    lines = [
        "",
        sep,
        f"  {company.upper()}: {story}",
        f"  ● {beat_num}/{total}  {beat_title}  [{fit}]",
        sep,
        "",
    ]

    # One sentence per line, blank line between each
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
    for sentence in sentences:
        lines.append(f"  {sentence}")
        lines.append("")

    if idx + 1 < total:
        next_label = _beat_label(all_beats[idx + 1].get("beat", ""))
        lines.append(f"  ↓ next: {next_label}")

    lines += [sep, ""]

    return "\n".join(lines)
