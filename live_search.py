"""
Live interview coach — push-to-talk.

Press SPACE or ENTER to record 8 seconds, then get a story cue.
Press Ctrl-C to quit.
"""

import os
import sys
import tty
import termios
import whisper

from audio_capture import record_clip
from embeddings import create_embeddings, search_full, save_chunks, load_chunks, get_story_beats
from notion_sync import sync_story_bank
from coach import is_question, is_followup, format_display

CACHE_FILE = "story_bank_with_embeddings.json"
RECORD_SECONDS = 8

# ── Conversation state ────────────────────────────────────────────────────────
state = {
    "story":      None,   # best-matching chunk for the active story
    "beats":      [],     # all beats for the active story, in order
    "beat_index": 0,      # which beat we're on right now
}


# ── Data loading ──────────────────────────────────────────────────────────────
def sync_and_embed():
    """Pull fresh Story Bank from Notion, embed, cache. Fall back to cache."""
    print("Syncing Story Bank from Notion...")
    try:
        chunks = sync_story_bank()
        if chunks:
            chunks = create_embeddings(chunks)
            save_chunks(chunks, CACHE_FILE)
            print(f"Synced and embedded {len(chunks)} chunks.\n")
            return chunks
        print("No chunks returned from Notion.")
    except Exception as e:
        print(f"Notion sync failed: {e}")

    if os.path.exists(CACHE_FILE):
        print(f"Loading from cache: {CACHE_FILE}")
        return load_chunks(CACHE_FILE)

    print("No cache available. Exiting.")
    exit(1)


# ── Keyboard ──────────────────────────────────────────────────────────────────
def _read_key() -> str:
    """Block until a single keypress and return it (raw mode, no echo)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def wait_for_trigger():
    """
    Block until the user presses SPACE or ENTER.
    Returns False if they pressed Ctrl-C or 'q'.
    """
    print("\n  [ SPACE or ENTER to record · q to quit ]\n")
    while True:
        ch = _read_key()
        if ch in (" ", "\r", "\n"):
            return True
        if ch in ("q", "Q", "\x03"):   # q or Ctrl-C
            return False


# ── Core logic ────────────────────────────────────────────────────────────────
def handle_question(question: str, chunks: list):
    """Pick or advance a story and print the display."""
    if is_followup(question, state["story"]) and state["beats"]:
        if state["beat_index"] + 1 < len(state["beats"]):
            state["beat_index"] += 1
        chunk = state["story"]
        all_beats = state["beats"]
    else:
        results = search_full(question, chunks, top_k=1)
        if not results:
            print("  (no match found in story bank)\n")
            return
        chunk = results[0]
        all_beats = get_story_beats(chunk["company"], chunk["story"], chunks)
        state["story"]      = chunk
        state["beats"]      = all_beats
        state["beat_index"] = 0

    print(format_display(question, state["story"], state["beats"], state["beat_index"]))


def record_and_process(whisper_model, chunks: list):
    """Record one clip → transcribe → route to story engine."""
    print(f"  ● Recording {RECORD_SECONDS}s…", flush=True)
    filepath = record_clip("live_query.wav", seconds=RECORD_SECONDS)

    print("  Transcribing…", flush=True)
    result = whisper_model.transcribe(filepath)
    text = result["text"].strip()

    if not text:
        print("  (nothing heard)\n")
        return

    print(f"\n  Heard: {text}\n")

    if not is_question(text):
        print("  (not a question — try again)\n")
        return

    handle_question(text, chunks)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Whisper model…")
    whisper_model = whisper.load_model("base")

    chunks = sync_and_embed()

    print("Ready.")
    print("─" * 52)

    while True:
        try:
            if not wait_for_trigger():
                print("\nDone.")
                break
            record_and_process(whisper_model, chunks)
        except KeyboardInterrupt:
            print("\nDone.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue
