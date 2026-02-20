"""
Interview coach — browser overlay.

Run:  python3 overlay.py
Then: browser opens automatically at http://localhost:5055

SPACE  — start / stop recording
The page updates in real time via Server-Sent Events (SSE).
"""

import json
import os
import queue
import re
import threading
import webbrowser

from flask import Flask, Response, jsonify, render_template_string, request

import whisper

from audio_capture import Recorder
from embeddings import (
    create_embeddings, load_chunks, save_chunks,
    search_full, get_story_beats,
)
from notion_sync import sync_story_bank
from coach import is_question, is_followup, generate_response, _beat_label

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_FILE = "story_bank_with_embeddings.json"
PORT       = 5055

# ── App state ─────────────────────────────────────────────────────────────────
app       = Flask(__name__)
recorder  = Recorder()
recording = False
state     = {"story": None, "beats": [], "beat_index": 0}
_clients: list[queue.Queue] = []
_clients_lock = threading.Lock()

# Loaded at startup
whisper_model = None
chunks: list  = []


# ── SSE broadcast ─────────────────────────────────────────────────────────────
def broadcast(payload: dict):
    with _clients_lock:
        for q in list(_clients):
            q.put(payload)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    """SSE endpoint — one persistent connection per browser tab."""
    q: queue.Queue = queue.Queue()
    with _clients_lock:
        _clients.append(q)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=20)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/toggle", methods=["POST"])
def toggle():
    global recording
    if not recording:
        recording = True
        broadcast({"type": "recording"})
        recorder.start()
    else:
        recording = False
        broadcast({"type": "status", "message": "Transcribing…"})
        threading.Thread(target=_process, daemon=True).start()
    return jsonify({"recording": recording})


# ── Processing pipeline ───────────────────────────────────────────────────────
def _process():
    try:
        filepath = recorder.stop()

        result   = whisper_model.transcribe(filepath)
        question = result["text"].strip()

        if not question:
            broadcast({"type": "idle"})
            return

        if not is_question(question):
            broadcast({"type": "not_question", "heard": question})
            return

        broadcast({"type": "status", "message": "Thinking…"})

        s = state
        if is_followup(question, s["story"]) and s["beats"]:
            if s["beat_index"] + 1 < len(s["beats"]):
                s["beat_index"] += 1
        else:
            results = search_full(question, chunks, top_k=1)
            if not results:
                broadcast({"type": "status", "message": "No match found in story bank."})
                return
            chunk = results[0]
            beats = get_story_beats(chunk["company"], chunk["story"], chunks)
            s["story"]      = chunk
            s["beats"]      = beats
            s["beat_index"] = 0

        idx      = s["beat_index"]
        chunk    = s["story"]
        beats    = s["beats"]
        total    = len(beats)
        company  = chunk.get("company", "")
        story    = chunk.get("story", "")
        beat_lbl = _beat_label(beats[idx].get("beat", ""))
        next_lbl = _beat_label(beats[idx + 1].get("beat", "")) if idx + 1 < total else ""

        response  = generate_response(question, chunk, beats, idx)
        sentences = [t.strip() for t in re.split(r"(?<=[.!?])\s+", response) if t.strip()]

        broadcast({
            "type":       "coaching",
            "company":    company,
            "story":      story,
            "beat_label": beat_lbl,
            "beat_num":   idx + 1,
            "total":      total,
            "sentences":  sentences,
            "next_beat":  next_lbl,
            "heard":      question,
        })

    except Exception as exc:
        broadcast({"type": "error", "message": str(exc)})


# ── HTML / JS / CSS ───────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Coach</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0f0f0f;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    font-size: 18px;
    line-height: 1.55;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Top bar ── */
  #bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 22px;
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
    font-size: 12px;
    color: #555;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    user-select: none;
  }
  #bar-story { color: #999; font-size: 12px; }
  #bar-hint  { color: #444; }

  /* ── Main content ── */
  #main {
    flex: 1;
    padding: 36px 40px 28px;
    max-width: 760px;
    width: 100%;
  }

  #beat-label {
    font-size: 12px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 28px;
    min-height: 18px;
  }

  #sentences {
    display: flex;
    flex-direction: column;
    gap: 22px;
  }

  .sentence {
    font-size: 22px;
    font-weight: 400;
    color: #f0f0f0;
    line-height: 1.5;
    opacity: 0;
    transform: translateY(6px);
    animation: fadeUp 0.35s ease forwards;
  }

  @keyframes fadeUp {
    to { opacity: 1; transform: translateY(0); }
  }

  .sentence:nth-child(1) { animation-delay: 0.0s; }
  .sentence:nth-child(2) { animation-delay: 0.12s; }
  .sentence:nth-child(3) { animation-delay: 0.24s; }
  .sentence:nth-child(4) { animation-delay: 0.36s; }

  #next-beat {
    margin-top: 36px;
    font-size: 13px;
    color: #444;
  }
  #next-beat span { color: #666; }

  /* ── Status messages ── */
  .status-msg {
    font-size: 16px;
    color: #555;
    font-style: italic;
  }

  .recording-msg {
    font-size: 20px;
    color: #ff5555;
    font-weight: 500;
  }

  .not-q-heard {
    font-size: 18px;
    color: #888;
    margin-bottom: 14px;
  }
  .not-q-label {
    font-size: 13px;
    color: #444;
  }

  /* ── Footer ── */
  #footer {
    padding: 14px 40px;
    border-top: 1px solid #1e1e1e;
    font-size: 12px;
    color: #333;
    display: flex;
    gap: 28px;
  }
  #footer kbd {
    background: #222;
    color: #666;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 11px;
    font-family: inherit;
  }
  #rec-indicator {
    margin-left: auto;
    color: #ff5555;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
  }
  #rec-indicator.active { opacity: 1; }
</style>
</head>
<body>

<div id="bar">
  <span id="bar-story">COACH</span>
  <span id="bar-hint">click here · then press space</span>
</div>

<div id="main">
  <div id="beat-label"></div>
  <div id="sentences">
    <div class="sentence status-msg">Ready. Press <strong>SPACE</strong> to start recording.</div>
  </div>
  <div id="next-beat"></div>
</div>

<div id="footer">
  <span><kbd>space</kbd> start / stop</span>
  <span><kbd>↓</kbd> follow-up advances the beat</span>
  <span id="rec-indicator">● REC</span>
</div>

<script>
  // ── SSE connection ──────────────────────────────────────────────────────────
  const es = new EventSource('/stream');

  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    handle(d);
  };

  // ── State ───────────────────────────────────────────────────────────────────
  let isRecording = false;

  // ── Keyboard ────────────────────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat) {
      e.preventDefault();
      toggle();
    }
  });

  function toggle() {
    fetch('/toggle', { method: 'POST' }).catch(console.error);
  }

  // ── DOM helpers ─────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  function setSentences(html) {
    $('sentences').innerHTML = html;
  }

  function setBar(story, hint) {
    $('bar-story').textContent = story || 'COACH';
    $('bar-hint').textContent  = hint  || '';
  }

  // ── Event handler ────────────────────────────────────────────────────────────
  function handle(d) {
    switch (d.type) {

      case 'recording':
        isRecording = true;
        $('rec-indicator').classList.add('active');
        $('beat-label').textContent = '';
        $('next-beat').innerHTML    = '';
        setSentences('<div class="recording-msg">● Recording…<br><br>Speak now.</div>');
        setBar('COACH', '● recording — press SPACE to stop');
        break;

      case 'status':
        isRecording = false;
        $('rec-indicator').classList.remove('active');
        $('beat-label').textContent = '';
        $('next-beat').innerHTML    = '';
        setSentences(`<div class="status-msg">${d.message}</div>`);
        setBar('COACH', '');
        break;

      case 'idle':
        isRecording = false;
        $('rec-indicator').classList.remove('active');
        setSentences('<div class="status-msg">Nothing heard. Press SPACE to try again.</div>');
        setBar('COACH', 'press SPACE to record');
        break;

      case 'not_question':
        isRecording = false;
        $('rec-indicator').classList.remove('active');
        $('beat-label').textContent = '';
        $('next-beat').innerHTML    = '';
        setSentences(`
          <div class="not-q-heard">${escHtml(d.heard)}</div>
          <div class="not-q-label">Didn't sound like a question — press SPACE to try again.</div>
        `);
        setBar('COACH', 'press SPACE to record');
        break;

      case 'coaching':
        isRecording = false;
        $('rec-indicator').classList.remove('active');
        $('beat-label').textContent =
          `Beat ${d.beat_num}/${d.total}  ·  ${d.beat_label}`;

        const html = d.sentences
          .map(s => `<div class="sentence">${escHtml(s)}</div>`)
          .join('');
        setSentences(html);

        $('next-beat').innerHTML = d.next_beat
          ? `<span>↓ next:</span> ${escHtml(d.next_beat)}`
          : '';

        setBar(
          `${d.company.toUpperCase()}: ${d.story}`,
          `beat ${d.beat_num}/${d.total}  ·  press SPACE to record`
        );
        break;

      case 'error':
        setSentences(`<div class="status-msg" style="color:#ff5555">Error: ${escHtml(d.message)}</div>`);
        break;
    }
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
</script>
</body>
</html>"""


# ── Data loading ──────────────────────────────────────────────────────────────
def sync_and_embed():
    print("Syncing Story Bank from Notion…")
    try:
        c = sync_story_bank()
        if c:
            c = create_embeddings(c)
            save_chunks(c, CACHE_FILE)
            print(f"Synced {len(c)} chunks.")
            return c
        print("No chunks returned.")
    except Exception as e:
        print(f"Notion sync failed: {e}")

    if os.path.exists(CACHE_FILE):
        print("Loading from cache.")
        return load_chunks(CACHE_FILE)

    print("No data available. Exiting.")
    exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Whisper model…")
    whisper_model = whisper.load_model("base")

    chunks = sync_and_embed()

    print(f"\nStarting coach at http://localhost:{PORT}")
    print("Opening browser…\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
